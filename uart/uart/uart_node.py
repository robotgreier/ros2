import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, String, Int16
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

class UartBridgeNode(Node):
    def __init__(self):
        super().__init__("uart_bridge_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("port", "/dev/ttyAMA3")
        self.declare_parameter("baudrate", 250000)
        self.declare_parameter("weights_file",str(Path.home() / "/opt/robot_ws/src/ros2/weights_logs/weights_current.mem"))
        self.declare_parameter("response_timeout_sec", 1.0)
        self.declare_parameter("max_retry_count", 2)
        self.declare_parameter("poll_period_sec", 0.01)
        self.declare_parameter("dopamine_timeout_sec", 1.0)

        self.port_name = self.get_parameter("port").value
        self.baudrate = self.get_parameter("baudrate").value
        self.weights_file = self.get_parameter("weights_file").value
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
        self.create_subscription(UInt8MultiArray, "/snn/input", self.snn_input_callback, 10)
        self.create_subscription(Int16, "/reward/dopamine", self.dopamine_callback, 10)

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
        else:
            self.publish_error(f"Unknown FPGA command: {cmd}")

    def save_weights(self):
        if not self.weights_file:
            return
        try:
            with open(self.weights_file, "w") as f:
                for w in self.weights:
                    f.write(f"{w:02X}\n")
            self.publish_status("Saved updated weights")
        except Exception as e:
            self.publish_error(f"Failed to save weights: {e}")

    def handle_update(self, payload):
        self.weights = list(payload)
        self.save_weights()
        self.wait_start_time = None

    def handle_out(self, payload):
        if not payload:
            self.publish_error("OUT packet had empty payload")
            self.state = STATE_READY
            self.wait_start_time = None
            return

        action_spikes = unpack_output_spikes(payload[0], expected_len=4)

        msg = UInt8MultiArray()
        msg.data = action_spikes
        self.fpga_action_pub.publish(msg)

        self.publish_status(f"Published action spikes: {action_spikes}")

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
            f"Sent CMD_INIT with {len(self.weights)} weights from {self.weights_file}"
        )

    def load_weights(self):
        if not self.weights_file:
            self.publish_error("No weights_file parameter set")
            self.weights = []
            return

        path = Path(self.weights_file)

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

def main():
    rclpy.init()
    node = UartBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()