import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, String, Int16
from pathlib import Path
from typing import List, Optional
import serial  # Replacing pigpio with pyserial

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
        self.declare_parameter("weights_file", "")
        self.declare_parameter("response_timeout_sec", 1.0)
        self.declare_parameter("max_retry_count", 2)
        self.declare_parameter("poll_period_sec", 0.01)

        self.port_name = self.get_parameter("port").value
        self.baudrate = self.get_parameter("baudrate").value
        self.weights_file = self.get_parameter("weights_file").value
        self.response_timeout_sec = self.get_parameter("response_timeout_sec").value
        self.max_retry_count = self.get_parameter("max_retry_count").value
        self.poll_period_sec = self.get_parameter("poll_period_sec").value

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
                timeout=0,
                write_timeout=None
            )
            self.publish_status(f"Opened Hardware UART: {self.port_name} @ {self.baudrate}")
        except Exception as e:
            self.publish_error(f"Serial open failed: {e}")
            self.ser = None

    def snn_input_callback(self, msg: UInt8MultiArray):
        if not self.initialized:
            return

        if self.state != STATE_READY:
            return

        payload = pack_input_spikes(list(msg.data))
        self.send_packet(CMD_SPIKE, payload)
        self.state = STATE_WAIT_OUT

    def dopamine_callback(self, msg: Int16):
        if self.state != STATE_WAIT_DOPAMINE:
            return

        dopamine_value = int(msg.data)

        if dopamine_value < -128 or dopamine_value > 127:
            self.publish_error(f"Dopamine value out of Int8 range: {dopamine_value}")
            return

        dopamine_byte = dopamine_value & 0xFF

        self.send_packet(CMD_DOPAMINE, [dopamine_byte])
        self.publish_status(f"Sent dopamine reward: {dopamine_value}")

        self.state = STATE_READY

    def send_packet(self, cmd: int, payload: List[int]):
        if not self.ser or not self.ser.is_open:
            self.publish_error("Serial unavailable")
            return
        
        try:
            packet = build_packet(cmd, payload)
            self.last_packet_sent = packet
            self.wait_start_time = self.get_clock().now()
            self.retry_count = 0

            self.ser.write(packet)
            self.ser.flush() # Forces immediate physical transmission
            self.publish_status(f"Sent CMD={cmd}")
        except Exception as e:
            self.publish_error(f"Send error: {e}")

    def poll_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        # Check if hardware has bytes waiting in the FIFO
        if self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
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
            
            if validate_packet(pkt):
                cmd, payload = parse_packet(pkt)
                self.dispatch_command(cmd, payload)
            else:
                self.publish_error("Checksum failed")

    def dispatch_command(self, cmd, payload):
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

        self.state = STATE_WAIT_DOPAMINE
        self.wait_start_time = None

    def check_for_timeouts(self):
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
        if self.weights and self.ser:
            packet = build_packet(CMD_INIT, self.weights)
            self.ser.write(packet)
            self.ser.flush()
            self.initialized = True
            self.publish_status("Sent CMD_INIT")

    def load_weights(self):
        path = Path(self.weights_file)
        if path.is_file():
            with open(path) as f:
                self.weights = [int(line.strip(), 16) for line in f if line.strip()]

def main():
    rclpy.init()
    node = UartBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()