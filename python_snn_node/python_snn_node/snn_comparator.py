"""
SNN comparator.

Subscribed topics:
  - /snn/input              UInt8MultiArray   (latest input vector)
  - /snn/winner             Int32             (Python SNN winner)
  - /snn/spikes             Int32MultiArray   (Python SNN output spikes)
  - /snn/decision           String            (Python SNN action name)
  - /snn/fpga/input_echo    UInt8MultiArray   (input that produced the FPGA response)
  - /snn/fpga/winner        Int32             (FPGA decoded winner)
  - /snn/fpga/decision      String            (FPGA decoded action name)
  - /fpga/action_spikes     UInt8MultiArray   (FPGA raw output spike vector)

Published topics:
  (none — this node writes a CSV only)

Row-emit strategy:
  Each CSV row is triggered by the FPGA response arriving for a specific input
  vector. uart_node publishes that input vector on /snn/fpga/input_echo
  immediately before publishing /fpga/action_spikes, so by the time
  /snn/fpga/winner arrives at this node, the echo is already stored.

  The Python SNN results are buffered in _py_buffer keyed by the input vector
  tuple. The Python SNN has no UART latency so it always processes an input
  before the FPGA response for that same input arrives (~107 ms later). When
  the FPGA response arrives the comparator looks up the Python result for the
  matching input and emits a row — guaranteeing both sides processed the same
  input vector.

  Rows where the FPGA responds but no Python result is buffered for that input
  (e.g. the input was never processed by the Python SNN in time) are logged
  with Python fields set to -1 and flagged with matched=0.

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

_PY_BUFFER_MAX = 32  # max buffered Python results; oldest evicted when full


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
        self.declare_parameter('fpga_echo_topic', '/snn/fpga/input_echo')
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
        self.fpga_echo_topic = str(self.get_parameter('fpga_echo_topic').value)
        self.fpga_winner_topic = str(self.get_parameter('fpga_winner_topic').value)
        self.fpga_decision_topic = str(self.get_parameter('fpga_decision_topic').value)
        self.fpga_spikes_topic = str(self.get_parameter('fpga_spikes_topic').value)
        self.input_size = int(self.get_parameter('input_size').value)
        self.num_actions = int(self.get_parameter('num_actions').value)
        self.log_dir = str(self.get_parameter('log_dir').value).strip()
        self.log_queue_size = int(self.get_parameter('log_queue_size').value)
        self.log_flush_hz = float(self.get_parameter('log_flush_hz').value)

    def _init_state(self):
        # Latest input snapshot (used only as fallback if echo is missing)
        self.t_input_ns: int = 0
        self.last_input: list[int] = [-1] * self.input_size

        # Python results: input_tuple → {t_py_ns, winner, decision, spikes, t_input_ns}
        # Ordered dict so we can evict oldest when full.
        self._py_buffer: dict[tuple, dict] = {}

        # Staging area for current Python tick (populated across several callbacks)
        self._py_staged: dict = {
            't_py_ns': 0,
            'winner': -1,
            'decision': '',
            'spikes': [-1] * self.num_actions,
            't_input_ns': 0,
            'input': [-1] * self.input_size,
        }

        # FPGA side — echo sets the key; winner/decision/spikes fill the values
        self._fpga_echo: tuple | None = None   # input tuple from the latest echo
        self.t_fpga_ns: int = 0
        self.fpga_winner: int = -1
        self.fpga_decision: str = ''
        self.fpga_spikes: list[int] = [-1] * self.num_actions

    def _setup_logger(self):
        log_dir = self.log_dir or '/opt/robot_ws/src/ros2/logs/snn_comparison'
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(log_dir, f'snn_comparison_{ts}.csv')

        header = ['t_row_ns', 't_input_ns']
        header += [f'input_{i}' for i in range(self.input_size)]
        header += ['t_py_ns', 'py_winner', 'py_decision']
        header += [f'py_spike_{i}' for i in range(self.num_actions)]
        header += ['t_fpga_ns', 'fpga_winner', 'fpga_decision']
        header += [f'fpga_spike_{i}' for i in range(self.num_actions)]
        header += ['agree', 'matched']  # matched=1 means same input confirmed

        self.logger_csv = CsvAsyncLogger(
            filepath=filepath,
            header=header,
            queue_size=self.log_queue_size,
            flush_hz=self.log_flush_hz,
        )
        self.get_logger().info(f"Logging comparison rows to {filepath}")

    def _setup_io(self):
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
            UInt8MultiArray, self.fpga_echo_topic, self._on_fpga_echo, qos_input
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
        t = self._now_ns()
        self._py_staged['t_py_ns'] = t
        self._py_staged['winner'] = int(msg.data)
        # Snapshot the input that was active at this tick.
        self._py_staged['t_input_ns'] = self.t_input_ns
        self._py_staged['input'] = list(self.last_input)
        self._buffer_py_result()

    def _on_py_spikes(self, msg: Int32MultiArray):
        data = [int(x) for x in msg.data]
        if len(data) >= self.num_actions:
            self._py_staged['spikes'] = data[: self.num_actions]

    def _on_py_decision(self, msg: String):
        self._py_staged['decision'] = str(msg.data)

    def _buffer_py_result(self):
        key = tuple(self._py_staged['input'])
        # Evict oldest entry when buffer is full (keeps memory bounded).
        if len(self._py_buffer) >= _PY_BUFFER_MAX and key not in self._py_buffer:
            oldest_key = next(iter(self._py_buffer))
            del self._py_buffer[oldest_key]
        self._py_buffer[key] = {
            't_py_ns': self._py_staged['t_py_ns'],
            'winner': self._py_staged['winner'],
            'decision': self._py_staged['decision'],
            'spikes': list(self._py_staged['spikes']),
            't_input_ns': self._py_staged['t_input_ns'],
        }

    def _on_fpga_echo(self, msg: UInt8MultiArray):
        data = [int(x) for x in msg.data]
        self._fpga_echo = tuple(data)

    def _on_fpga_winner(self, msg: Int32):
        self.t_fpga_ns = self._now_ns()
        self.fpga_winner = int(msg.data)
        # FPGA winner is the row trigger: pair with the Python result for the
        # same input that was echoed by uart_node.
        self._emit_row()

    def _on_fpga_decision(self, msg: String):
        self.fpga_decision = str(msg.data)

    def _on_fpga_spikes(self, msg: UInt8MultiArray):
        data = [int(x) for x in msg.data]
        if len(data) >= self.num_actions:
            self.fpga_spikes = data[: self.num_actions]

    def _emit_row(self):
        echo_key = self._fpga_echo

        if echo_key is not None and echo_key in self._py_buffer:
            py = self._py_buffer.pop(echo_key)
            matched = 1
            input_vec = list(echo_key)
            t_input_ns = py['t_input_ns']
            t_py_ns = py['t_py_ns']
            py_winner = py['winner']
            py_decision = py['decision']
            py_spikes = py['spikes']
        else:
            # Fallback: echo missing or no Python result for this input.
            # Log what we have; marked unmatched so it can be filtered in analysis.
            matched = 0
            input_vec = list(echo_key) if echo_key is not None else self.last_input
            t_input_ns = self.t_input_ns
            t_py_ns = -1
            py_winner = -1
            py_decision = ''
            py_spikes = [-1] * self.num_actions
            if echo_key is None:
                self.get_logger().warn('FPGA winner arrived without input echo')
            else:
                self.get_logger().warn(
                    'No Python result buffered for FPGA input — input may have been skipped'
                )

        agree = int(py_winner == self.fpga_winner) if matched else 0

        row = [self._now_ns(), t_input_ns]
        row += input_vec
        row += [t_py_ns, py_winner, py_decision]
        row += py_spikes
        row += [self.t_fpga_ns, self.fpga_winner, self.fpga_decision]
        row += list(self.fpga_spikes)
        row += [agree, matched]
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
