"""
ROS 2 UART bridge node.

This node is responsible for:
- subscribing to /snn/input
- sending UART packets to the FPGA
- receiving UART packets from the FPGA
- publishing status/debug information
- publishing output spikes
"""

# uart_node.py

from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8MultiArray, String

import serial

from .protocol import (
    SOF,
    build_packet,
    expected_packet_length,
    parse_packet,
    validate_packet,
    CMD_INIT,
    CMD_SPIKE,
    CMD_STOP,
)

from .spike_codec import pack_input_spikes, unpack_output_spikes

class UartBridgeNode(Node):
    def __init__(self):
        super().__init__("uart_bridge_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("serial_port", "/dev/ttyAMA5")
        self.declare_parameter("baudrate", 250000)
        self.declare_parameter("timeout", 1.0)
        self.declare_parameter("weights_file", "")
        self.declare_parameter("response_timeout_sec", 1.0)
        self.declare_parameter("max_retry_count", 2)
        self.declare_parameter("poll_period_sec", 0.01)

        self.serial_port_name = self.get_parameter("serial_port").value
        self.baudrate = self.get_parameter("baudrate").value
        self.timeout = self.get_parameter("timeout").value
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

        # -----------------------------
        # Internal state
        # -----------------------------
        self.ser = None
        self.weights: List[int] = []

        self.last_packet_sent: bytes = b""
        self.last_command_sent: Optional[int] = None

        self.waiting_for_affirm = False
        self.waiting_for_out = False
        self.waiting_for_update = False

        self.wait_start_time = None
        self.retry_count = 0

        self.initialized = False
        self.rx_buffer = bytearray()

        # -----------------------------
        # Startup
        # -----------------------------
        self.open_serial_port()
        self.load_weights()
        self.try_send_init()

        self.create_timer(self.poll_period_sec, self.poll_serial)
        self.create_timer(0.05, self.check_for_timeouts)

        self.publish_status("UART node started.")

    # =================================================
    # Logging helpers
    # =================================================

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

    # =================================================
    # Serial setup
    # =================================================

    def open_serial_port(self):
        try:
            self.ser = serial.Serial(
                port=self.serial_port_name,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.publish_status(f"Opened serial port {self.serial_port_name}")
        except Exception as e:
            self.publish_error(f"Serial open failed: {e}")
            self.ser = None

    # =================================================
    # Weights handling
    # =================================================

    def load_weights(self):
        try:
            path = Path(self.weights_file)
            if not path.exists():
                self.publish_error(f"Weights file not found: {self.weights_file}")
                return
            with open(path) as f:
                self.weights = [int(line.strip(), 16) for line in f if line.strip()]
            self.publish_status(f"Loaded {len(self.weights)} weights from {path.name}")
        except Exception as e:
            self.publish_error(f"Weight load error: {e}")
            self.weights = []

    def save_weights(self):
        try:
            with open(self.weights_file, "w") as f:
                for w in self.weights:
                    f.write(f"{w:02X}\n")
            self.publish_status("Saved updated weights")
        except Exception as e:
            self.publish_error(f"Failed to save weights: {e}")

    # =================================================
    # ROS callback
    # =================================================

    def snn_input_callback(self, msg: UInt8MultiArray):
        if not self.initialized:
            self.publish_status("Ignoring input: not initialized")
            return

        if self.waiting_for_out or self.waiting_for_update:
            self.publish_status("Busy, ignoring input")
            return

        raw_spikes = list(msg.data)
        payload = pack_input_spikes(raw_spikes)
        self.send_spike(payload)

    # =================================================
    # Sending
    # =================================================

    def send_spike(self, payload: List[int]):
        self.send_packet(CMD_SPIKE, payload)

    def send_packet(self, cmd: int, payload: List[int]):
        try:
            packet = build_packet(cmd, payload)

            self.last_packet_sent = packet
            self.last_command_sent = cmd
            self.retry_count = 0

            self.waiting_for_affirm = False
            self.waiting_for_update = False
            self.wait_start_time = self.get_clock().now()

            if cmd == CMD_SPIKE:
                self.waiting_for_out = True
            elif cmd == CMD_STOP:
                self.waiting_for_out = False
                self.waiting_for_update = True
            else:
                self.waiting_for_out = False

            self.publish_status(f"Sending CMD={cmd}, LEN={len(payload)}")

            if self.ser:
                self.ser.write(packet)
            else:
                self.publish_status("Serial unavailable, packet not sent")

        except Exception as e:
            self.publish_error(f"Send error: {e}")

    def try_send_init(self):
        if not self.weights:
            self.publish_error("INIT skipped, no weights")
            return
        if not self.ser:
            self.publish_error("Serial unavailable, INIT not sent")
            return
        try:
            packet = build_packet(CMD_INIT, self.weights)
            self.ser.write(packet)
            self.initialized = True
            self.publish_status("INIT sent, initialized")
        except Exception as e:
            self.publish_error(f"INIT send error: {e}")

    # =================================================
    # RX processing
    # =================================================

    def poll_serial(self):
        if not self.ser:
            return

        if self.ser.in_waiting:
            data = self.ser.read(self.ser.in_waiting)
            self.rx_buffer.extend(data)
            self.process_buffer()

    def process_buffer(self):
        while True:
            packet = self.extract_packet()
            if packet is None:
                return
            self.handle_packet(packet)

    def extract_packet(self) -> Optional[bytes]:
        while self.rx_buffer and self.rx_buffer[0] != SOF:
            discarded = self.rx_buffer.pop(0)
            self.publish_error(f"Discarded byte: {discarded}")

        if len(self.rx_buffer) < 3:
            return None

        length = self.rx_buffer[2]
        total = expected_packet_length(length)

        if len(self.rx_buffer) < total:
            return None

        pkt = bytes(self.rx_buffer[:total])
        del self.rx_buffer[:total]
        return pkt

    def handle_packet(self, pkt: bytes):
        self.publish_status(f"RX: {list(pkt)}")

        if not validate_packet(pkt):
            self.publish_error("Invalid checksum")
            return

        cmd, payload = parse_packet(pkt)
        self.dispatch_command(cmd, payload)

    # =================================================
    # Command dispatch
    # =================================================

    def dispatch_command(self, cmd, payload):
        if cmd == 0:    # spike output (after SPIKE)
            self.handle_out(payload)
        elif cmd == 1:  # weight update (after STOP)
            self.handle_update(payload)
        elif cmd == 2:  # error from FPGA
            self.publish_error(f"FPGA ERR: {payload}")
        else:
            self.publish_error(f"Unknown CMD {cmd}")

    # =================================================
    # Command handlers
    # =================================================

    def handle_out(self, payload):
        if len(payload) != 1:
            self.publish_error("Invalid OUT payload")
            return

        out_byte = int(payload[0])
        action_spikes = unpack_output_spikes(out_byte, expected_len=4)

        msg = UInt8MultiArray()
        msg.data = action_spikes
        self.fpga_action_pub.publish(msg)

        self.publish_status(
            f"OUT byte={out_byte}, action_spikes={action_spikes}"
        )

        self.clear_wait_state()

    def handle_update(self, payload):
        self.weights = payload
        self.save_weights()
        self.clear_wait_state()

    # =================================================
    # Timeout + retry
    # =================================================

    def check_for_timeouts(self):
        if not self.wait_start_time:
            return

        elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9

        if elapsed < self.response_timeout_sec:
            return

        if self.retry_count < self.max_retry_count:
            self.retry_count += 1
            self.publish_error(f"Timeout → retry {self.retry_count}")

            self.resend_last_packet()
            self.wait_start_time = self.get_clock().now()
        else:
            self.publish_error("Max retries reached")
            self.clear_wait_state()

    def resend_last_packet(self):
        if self.ser and self.last_packet_sent:
            self.ser.write(self.last_packet_sent)
            self.publish_status("Resent last packet")

    # =================================================
    # State helpers
    # =================================================

    def clear_wait_state(self):
        self.waiting_for_affirm = False
        self.waiting_for_out = False
        self.waiting_for_update = False
        self.wait_start_time = None
        self.retry_count = 0


def main():
    rclpy.init()
    node = UartBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()