import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, String, Int16, Empty
from pathlib import Path
from typing import List, Optional
import serial  # Replacing pigpio with pyserial
import time

from .protocol import (
    SOF,
    build_packet,
    expected_packet_length,
    parse_packet,
    validate_packet,
    CMD_INIT,
    CMD_SPIKE,
    CMD_DOPAMINE,
    CMD_STOP,
    CMD_OUT,
    CMD_WEIGHT,
    CMD_ERR,
)

from .spike_codec import pack_input_spikes, unpack_output_spikes

STATE_READY = "READY"
STATE_WAIT_OUT = "WAIT_OUT"
STATE_WAIT_DOPAMINE = "WAIT_DOPAMINE"
STATE_WAIT_WEIGHT = "WAIT_WEIGHT"

class UartBridgeNode(Node):
    def __init__(self):
        super().__init__("uart_bridge_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        _default_weights = "/opt/robot_ws/src/ros2/weights_logs/weights_current.mem"

        self.declare_parameter("port", "/dev/ttyAMA3")
        self.declare_parameter("baudrate", 250000)
        # Path read once on startup and pushed to the FPGA via CMD_INIT.
        self.declare_parameter("initial_weights_file", _default_weights)
        # Path written to when the FPGA returns updated weights at episode end.
        # In the side-by-side comparison launch this is pointed at an
        # FPGA-only directory so it never overwrites the shared seed file.
        self.declare_parameter("save_weights_file", _default_weights)
        self.declare_parameter("response_timeout_sec", 1.0)
        self.declare_parameter("max_retry_count", 2)
        self.declare_parameter("poll_period_sec", 0.01)
        self.declare_parameter("dopamine_timeout_sec", 1.0)

        self.port_name = self.get_parameter("port").value
        self.baudrate = self.get_parameter("baudrate").value
        self.initial_weights_file = self.get_parameter("initial_weights_file").value
        self.save_weights_file = self.get_parameter("save_weights_file").value
        self.response_timeout_sec = self.get_parameter("response_timeout_sec").value
        self.max_retry_count = self.get_parameter("max_retry_count").value
        self.poll_period_sec = self.get_parameter("poll_period_sec").value
        self.dopamine_timeout_sec = self.get_parameter("dopamine_timeout_sec").value

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.status_pub = self.create_publisher(String, "/uart/status", 10)
        self.error_pub = self.create_publisher(String, "/uart/error", 10)
        self.fpga_action_pub = self.create_publisher(UInt8MultiArray, "/fpga/action_spikes", 10)
        self.fpga_input_echo_pub = self.create_publisher(UInt8MultiArray, "/snn/fpga/input_echo", 10)
        self.episode_reset_pub = self.create_publisher(Empty, "/episode_reset", 10)

        self.create_subscription(UInt8MultiArray, "/snn/input", self.snn_input_callback, 10)
        self.create_subscription(Int16, "/reward/dopamine", self.dopamine_callback, 10)
        # /episode_complete is the only trigger for the CMD_STOP + weight save
        # round-trip. uart_node also *publishes* /episode_reset at the end of
        # handle_update — subscribing here would re-trigger itself and create
        # an infinite CMD_STOP loop with weights_logger in the chain.
        self.create_subscription(Empty, "/episode_complete", self.episode_complete_callback, 10)

        # -----------------------------
        # Internal state
        # -----------------------------
        self.ser: Optional[serial.Serial] = None
        self.weights: List[int] = []
        self.last_packet_sent: bytes = b""
        self.wait_start_time = None
        self.retry_count = 0
        self.initialized = False
        self.rx_buffer = bytearray()
        self.state = STATE_READY
        self.waiting_for_dopamine = False
        self._pending_input: list = []  # input sent with last CMD_SPIKE, echoed on response

        self.episode_counter = 1

        # Episode archives live next to the saved file so each FPGA run keeps
        # its own history. The shared initial_weights_file is read-only.
        self.episode_log_dir = Path(self.save_weights_file).parent / "episode_logs"
        self.episode_log_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Startup
        # -----------------------------
        self.open_serial_port()
        self.load_weights()
        self.try_send_init()

        # Timers
        self.create_timer(self.poll_period_sec, self.poll_serial)
        self.create_timer(0.05, self.check_for_timeouts)

        self.publish_status("UART Hardware Node started.")

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def publish_error(self, text: str):
        msg = String()
        msg.data = text
        self.error_pub.publish(msg)
        self.get_logger().error(text)

    def open_serial_port(self):
        try:
            # timeout=0 makes read() non-blocking
            self.ser = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,
                write_timeout=None
            )

            self.ser.setDTR(False)
            self.ser.setRTS(False)

            self.publish_status(f"Opened Hardware UART: {self.port_name} @ {self.baudrate}")
        except Exception as e:
            self.publish_error(f"Serial open failed: {e}")
            self.ser = None

    def snn_input_callback(self, msg: UInt8MultiArray):
        if not self.initialized:
            return

        if self.state != STATE_READY:
            return

        raw_spikes = list(msg.data)
        payload = pack_input_spikes(raw_spikes)

        self.publish_status(f"Raw /snn/input length: {len(raw_spikes)}")
        self.publish_status(f"Raw /snn/input data: {raw_spikes}")
        self.publish_status(f"Packed SPIKE payload length: {len(payload)}")
        self.publish_status(f"Packed SPIKE payload: {payload}")

        self._pending_input = raw_spikes  # echoed alongside FPGA response in handle_out
        self.send_packet(CMD_SPIKE, payload)
        self.state = STATE_WAIT_OUT

    def dopamine_callback(self, msg: Int16):
        if self.state != STATE_WAIT_DOPAMINE or not self.waiting_for_dopamine:
            return

        dopamine_value = int(msg.data)

        if dopamine_value < -128 or dopamine_value > 127:
            self.publish_error(f"Dopamine value out of Int8 range: {dopamine_value}")
            return

        dopamine_byte = dopamine_value & 0xFF

        self.send_packet(CMD_DOPAMINE, [dopamine_byte])
        self.publish_status(f"Sent dopamine reward: {dopamine_value}")

        self.waiting_for_dopamine = False
        self.state = STATE_READY

    def send_packet(self, cmd: int, payload: List[int]):
        if not self.ser or not self.ser.is_open:
            self.publish_error("Serial unavailable")
            return

        try:
            packet = build_packet(cmd, payload)

            self.publish_status(f"TX CMD={cmd}, payload_len={len(payload)}, payload={payload}")
            self.publish_status(f"TX bytes: {list(packet)}")

            self.last_packet_sent = packet
            self.wait_start_time = self.get_clock().now()
            self.retry_count = 0

            self.ser.write(packet)
            self.ser.flush()
            self.publish_status(f"Sent CMD={cmd}")

        except Exception as e:
            self.publish_error(f"Send error: {e}")

    def poll_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        # Check if hardware has bytes waiting in the FIFO
        if self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)

            self.publish_status(f"RX raw bytes: {list(data)}")

            self.rx_buffer.extend(data)
            self.process_buffer()

    def process_buffer(self):
        while True:
            # 1. Sync: Find the Start of Frame (SOF)
            while self.rx_buffer and self.rx_buffer[0] != SOF:
                self.rx_buffer.pop(0)

            # 2. Check if we have enough for header (SOF, CMD, LEN)
            if len(self.rx_buffer) < 3:
                break

            # 3. Check if full packet is present
            total_len = expected_packet_length(self.rx_buffer[2])
            if len(self.rx_buffer) < total_len:
                break

            # 4. Extract and handle
            pkt = bytes(self.rx_buffer[:total_len])
            del self.rx_buffer[:total_len]
            
            self.publish_status(f"RX packet candidate: {list(pkt)}")

            FPGA_sum_1, PI_sum_1, FPGA_sum_2, PI_sum_2 = validate_packet(pkt)
            if FPGA_sum_1 == PI_sum_1 and FPGA_sum_2 == PI_sum_2:
                cmd, payload = parse_packet(pkt)
                self.dispatch_command(cmd, payload)
            else:
                self.publish_error(f"Checksum failed\nPI sums: ({PI_sum_1}, {PI_sum_2}) -- FPGA sums: ({FPGA_sum_1}, {FPGA_sum_2})")

    def dispatch_command(self, cmd, payload):
        self.publish_status(f"Parsed CMD={cmd}, payload={payload}")

        if cmd == CMD_OUT:
            self.handle_out(payload)
        elif cmd == CMD_WEIGHT:
            self.handle_update(payload)
        elif cmd == CMD_ERR:
            self.publish_error(f"FPGA Error Code: {payload}")
            self.wait_start_time = None
            self.state = STATE_READY
        else:
            self.publish_error(f"Unknown FPGA command: {cmd}")

    def save_weights(self):
        if not self.save_weights_file:
            return

        try:
            current_path = Path(self.save_weights_file)
            current_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Update current weights
            tmp_file = current_path.with_suffix(current_path.suffix + ".tmp")
            with open(tmp_file, "w") as f:
                for w in self.weights:
                    f.write(f"{w & 0xFF:02X}\n")
            tmp_file.replace(current_path)

            # 2. Save episode archive
            episode_file = self.episode_log_dir / f"weights_ep_{self.episode_counter:04d}.mem"
            with open(episode_file, "w") as f:
                for w in self.weights:
                    f.write(f"{w & 0xFF:02X}\n")

            self.publish_status(
                f"Saved updated weights to {current_path} and episode log {episode_file}"
            )

            self.episode_counter += 1

        except Exception as e:
            self.publish_error(f"Failed to save weights: {e}")

    def handle_update(self, payload):
        self.publish_status(f"Received {len(payload)} weights from FPGA")

        self.weights = list(payload)
        self.save_weights()

        self.wait_start_time = None
        self.state = STATE_READY

        self.episode_reset_pub.publish(Empty())
        self.publish_status("Published /episode_reset")

        self.publish_status("Weight update complete, node returned to READY")

    def handle_out(self, payload):
        if not payload:
            self.publish_error("OUT packet had empty payload")
            self.state = STATE_READY
            self.wait_start_time = None
            return

        action_spikes = unpack_output_spikes(payload[0], expected_len=4)

        # Publish the input that produced these spikes before the spikes themselves
        # so snn_comparator receives the echo before the derived /snn/fpga/winner.
        echo_msg = UInt8MultiArray()
        echo_msg.data = list(self._pending_input)
        self.fpga_input_echo_pub.publish(echo_msg)

        msg = UInt8MultiArray()
        msg.data = action_spikes
        self.fpga_action_pub.publish(msg)

        self.publish_status(f"Published action spikes: {action_spikes}")

        if not any(action_spikes):
            self.publish_status("No FPGA output spikes — sending neutral dopamine 0")
            self.send_packet(CMD_DOPAMINE, [0])
            self.state = STATE_READY
            return

        self.waiting_for_dopamine = True
        self.state = STATE_WAIT_DOPAMINE
        self.wait_start_time = self.get_clock().now()

    def check_for_timeouts(self):
        if self.state == STATE_WAIT_DOPAMINE and self.wait_start_time is not None:
            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9

            if elapsed > self.dopamine_timeout_sec:
                self.publish_error("Timed out waiting for dopamine reward")
                self.waiting_for_dopamine = False
                self.state = STATE_READY
                self.wait_start_time = None

            return

        if self.wait_start_time is None:
            return

        elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
        if elapsed > self.response_timeout_sec:
            if self.retry_count < self.max_retry_count:
                self.retry_count += 1
                self.ser.write(self.last_packet_sent)
                self.wait_start_time = self.get_clock().now()
                self.publish_status(f"Retry {self.retry_count} sent")
            else:
                self.publish_error("Max retries reached - clearing wait state")
                self.wait_start_time = None

    def try_send_init(self):
        if not self.ser or not self.ser.is_open:
            self.publish_error("INIT not sent: serial port is not open")
            return

        if not self.weights:
            self.publish_error("INIT not sent: no weights loaded")
            return

        packet = build_packet(CMD_INIT, self.weights)

        self.ser.write(packet)
        self.ser.flush()

        time.sleep(1.0)

        self.initialized = True

        self.initialized = True
        self.publish_status(
            f"Sent CMD_INIT with {len(self.weights)} weights from {self.initial_weights_file}"
        )

    def load_weights(self):
        if not self.initial_weights_file:
            self.publish_error("No initial_weights_file parameter set")
            self.weights = []
            return

        path = Path(self.initial_weights_file)

        if not path.is_file():
            self.publish_error(f"Weights file not found: {path}")
            self.weights = []
            return

        try:
            weights = []

            with open(path, "r") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    if line.startswith("#") or line.startswith("//"):
                        continue

                    line = line.split("#")[0]
                    line = line.split("//")[0]
                    line = line.strip()

                    if not line:
                        continue

                    for token in line.split():
                        weights.append(int(token, 16) & 0xFF)

            self.weights = weights
            self.publish_status(f"Loaded {len(self.weights)} weights from {path}")

        except Exception as e:
            self.weights = []
            self.publish_error(f"Failed to load weights from {path}: {e}")

    def episode_complete_callback(self, msg: Empty):
        self.publish_status(f"Received /episode_complete while state={self.state}")

        if not self.initialized:
            self.publish_error("Ignoring episode complete: UART node not initialized")
            return

        if self.state != STATE_READY:
            self.publish_error(f"Ignoring episode complete: node is busy in state {self.state}")
            return

        self.publish_status("Episode complete: sending CMD_STOP to request weights")
        self.send_packet(CMD_STOP, [])
        self.state = STATE_WAIT_WEIGHT

def main():
    rclpy.init()
    node = UartBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()