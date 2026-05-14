
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import csv
import os
from datetime import datetime


class SnnLogger(Node):
    """
    Streaming power logger for SNN experiments.

    Logs:
    - time (relative)
    - phase
    - power (system, FPGA, total)
    - cumulative energy
    """

    def __init__(self):
        super().__init__('snn_logger_node')

        # ---- File setup ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/opt/robot_ws/src/ros2/power_monitor/analysis/csv_logs/snn"
        os.makedirs(log_dir, exist_ok=True)

        self.filename = f"{log_dir}/snn_logger_{timestamp}.csv"

        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        # Header
        self.writer.writerow([
            "time_s",
            "phase",
            "P_total_W",
            "P_system_W",
            "P_fpga_W",
            "E_total_Wh",
            "E_system_Wh",
            "E_fpga_Wh"
        ])
        self.file.flush()

        self.get_logger().info(f"Logging to {self.filename}")

        # ---- State ----
        self.system = None
        self.fpga = None
        self.phase = "idle_pre"

        self.last_time = None
        self.t0 = self.now()

        # ---- Energy accumulators ----
        self.E_total = 0.0
        self.E_system = 0.0
        self.E_fpga = 0.0

        # ---- Subscriptions ----
        self.create_subscription(Vector3, "/system/power", self.cb_system, 10)
        self.create_subscription(Vector3, "/fpga/power", self.cb_fpga, 10)
        self.create_subscription(String, "/experiment/phase", self.cb_phase, 10)

        # ---- Logging timer ----
        self.timer = self.create_timer(0.1, self.log_row)  # 10 Hz

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cb_phase(self, msg: String):
        self.phase = msg.data

    def cb_system(self, msg: Vector3):
        self.system = msg

    def cb_fpga(self, msg: Vector3):
        self.fpga = msg

    def log_row(self):
        if self.system is None or self.fpga is None:
            return

        now = self.now()

        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        if dt <= 1e-4:
            return

        # ---- Power ----
        P_system = self.system.z
        P_fpga = self.fpga.z
        P_total = P_system + P_fpga

        # ---- Energy integration ----
        self.E_system += (P_system * dt) / 3600.0
        self.E_fpga += (P_fpga * dt) / 3600.0
        self.E_total += (P_total * dt) / 3600.0

        # ---- Relative time ----
        t_rel = now - self.t0

        # ---- Write row ----
        self.writer.writerow([
            t_rel,
            self.phase,
            P_total,
            P_system,
            P_fpga,
            self.E_total,
            self.E_system,
            self.E_fpga
        ])

        self.file.flush()

    def destroy_node(self):
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
            self.get_logger().info("CSV logging stopped")
        super().destroy_node()


def main():
    rclpy.init()
    node = SnnLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()