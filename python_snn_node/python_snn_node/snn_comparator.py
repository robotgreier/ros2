"""
SNN comparator.

Subscribed topics:
  - /snn/input              UInt8MultiArray   (latest input vector)
  - /snn/winner             Int32             (Python SNN winner)
  - /snn/spikes             Int32MultiArray   (Python SNN output spikes)
  - /snn/decision           String            (Python SNN action name)
  - /snn/fpga/winner        Int32             (FPGA decoded winner)
  - /snn/fpga/decision      String            (FPGA decoded action name)
  - /fpga/action_spikes     UInt8MultiArray   (FPGA raw output spike vector)

Published topics:
  (none — this node writes a CSV only)

One CSV row is appended on every Python /snn/winner event. The row snapshots
the latest input and latest FPGA outputs at that moment, together with their
arrival timestamps so offline analysis can compute per-side lag and
agreement statistics.

Parameters live under the snn_comparator section in
my_ros2_bringup/config/params.yaml.
"""

import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, Int32MultiArray, String, UInt8MultiArray

from .csv_logger import CsvAsyncLogger


class SnnComparator(Node):
    def __init__(self):
        super().__init__('snn_comparator')

        self._declare_parameters()
        self._load_parameters()
        self._init_state()
        self._setup_logger()
        self._setup_io()

        self.get_logger().info(
            f"snn_comparator started: input_size={self.input_size}, "
            f"num_actions={self.num_actions}"
        )

    # --- Setup helpers ---

    def _declare_parameters(self):
        self.declare_parameter('input_topic', '/snn/input')
        self.declare_parameter('py_winner_topic', '/snn/winner')
        self.declare_parameter('py_spikes_topic', '/snn/spikes')
        self.declare_parameter('py_decision_topic', '/snn/decision')
        self.declare_parameter('fpga_winner_topic', '/snn/fpga/winner')
        self.declare_parameter('fpga_decision_topic', '/snn/fpga/decision')
        self.declare_parameter('fpga_spikes_topic', '/fpga/action_spikes')
        self.declare_parameter('input_size', 26)
        self.declare_parameter('num_actions', 4)
        self.declare_parameter('log_dir', '')
        self.declare_parameter('log_queue_size', 5000)
        self.declare_parameter('log_flush_hz', 10.0)

    def _load_parameters(self):
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.py_winner_topic = str(self.get_parameter('py_winner_topic').value)
        self.py_spikes_topic = str(self.get_parameter('py_spikes_topic').value)
        self.py_decision_topic = str(self.get_parameter('py_decision_topic').value)
        self.fpga_winner_topic = str(self.get_parameter('fpga_winner_topic').value)
        self.fpga_decision_topic = str(self.get_parameter('fpga_decision_topic').value)
        self.fpga_spikes_topic = str(self.get_parameter('fpga_spikes_topic').value)
        self.input_size = int(self.get_parameter('input_size').value)
        self.num_actions = int(self.get_parameter('num_actions').value)
        self.log_dir = str(self.get_parameter('log_dir').value).strip()
        self.log_queue_size = int(self.get_parameter('log_queue_size').value)
        self.log_flush_hz = float(self.get_parameter('log_flush_hz').value)

    def _init_state(self):
        self.t_input_ns: int = 0
        self.last_input: list[int] = [-1] * self.input_size
        self.t_py_ns: int = 0
        self.py_winner: int = -1
        self.py_decision: str = ''
        self.py_spikes: list[int] = [-1] * self.num_actions
        self.t_fpga_ns: int = 0
        self.fpga_winner: int = -1
        self.fpga_decision: str = ''
        self.fpga_spikes: list[int] = [-1] * self.num_actions

    def _setup_logger(self):
        log_dir = self.log_dir or os.path.expanduser('~/.ros/snn_comparison_logs')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(log_dir, f'snn_comparison_{ts}.csv')

        header = ['t_row_ns', 't_input_ns']
        header += [f'input_{i}' for i in range(self.input_size)]
        header += ['t_py_ns', 'py_winner', 'py_decision']
        header += [f'py_spike_{i}' for i in range(self.num_actions)]
        header += ['t_fpga_ns', 'fpga_winner', 'fpga_decision']
        header += [f'fpga_spike_{i}' for i in range(self.num_actions)]
        header += ['agree']

        self.logger_csv = CsvAsyncLogger(
            filepath=filepath,
            header=header,
            queue_size=self.log_queue_size,
            flush_hz=self.log_flush_hz,
        )
        self.get_logger().info(f"Logging comparison rows to {filepath}")

    def _setup_io(self):
        # Match python_snn_node's RELIABLE input QoS so we see every encoder tick.
        qos_input = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            UInt8MultiArray, self.input_topic, self._on_input, qos_input
        )
        self.create_subscription(
            Int32, self.py_winner_topic, self._on_py_winner, 10
        )
        self.create_subscription(
            Int32MultiArray, self.py_spikes_topic, self._on_py_spikes, 10
        )
        self.create_subscription(
            String, self.py_decision_topic, self._on_py_decision, 10
        )
        self.create_subscription(
            Int32, self.fpga_winner_topic, self._on_fpga_winner, 10
        )
        self.create_subscription(
            String, self.fpga_decision_topic, self._on_fpga_decision, 10
        )
        self.create_subscription(
            UInt8MultiArray, self.fpga_spikes_topic, self._on_fpga_spikes, 10
        )

    # --- Callbacks ---

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _on_input(self, msg: UInt8MultiArray):
        data = [int(x) for x in msg.data]
        if len(data) != self.input_size:
            self.get_logger().warn(
                f"{self.input_topic} length {len(data)} != expected {self.input_size}"
            )
            return
        self.t_input_ns = self._now_ns()
        self.last_input = data

    def _on_py_winner(self, msg: Int32):
        self.t_py_ns = self._now_ns()
        self.py_winner = int(msg.data)
        # Python's tick is the row trigger.
        self._emit_row()

    def _on_py_spikes(self, msg: Int32MultiArray):
        data = [int(x) for x in msg.data]
        if len(data) >= self.num_actions:
            self.py_spikes = data[: self.num_actions]

    def _on_py_decision(self, msg: String):
        self.py_decision = str(msg.data)

    def _on_fpga_winner(self, msg: Int32):
        self.t_fpga_ns = self._now_ns()
        self.fpga_winner = int(msg.data)

    def _on_fpga_decision(self, msg: String):
        self.fpga_decision = str(msg.data)

    def _on_fpga_spikes(self, msg: UInt8MultiArray):
        data = [int(x) for x in msg.data]
        if len(data) >= self.num_actions:
            self.fpga_spikes = data[: self.num_actions]

    def _emit_row(self):
        agree = int(self.py_winner == self.fpga_winner)
        row = [self._now_ns(), self.t_input_ns]
        row += list(self.last_input)
        row += [self.t_py_ns, self.py_winner, self.py_decision]
        row += list(self.py_spikes)
        row += [self.t_fpga_ns, self.fpga_winner, self.fpga_decision]
        row += list(self.fpga_spikes)
        row += [agree]
        self.logger_csv.push(row)

    def destroy_node(self):
        try:
            self.logger_csv.close()
            self.get_logger().info(
                f"CSV logger closed (dropped_rows={self.logger_csv.dropped})"
            )
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = SnnComparator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
