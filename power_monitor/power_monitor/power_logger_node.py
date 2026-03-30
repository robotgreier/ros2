import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import csv
import time
from datetime import datetime
from collections import deque
import statistics

BATTERY_WH = 49.02  # LiPo 11.4V * 4300mAh = 49.02Wh


class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = "/opt/robot_ws/log/power_monitor"
        self.filename = f"{log_dir}/power_log_{timestamp}.csv"

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "topic", "voltage", "current", "power"])

        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # latest sensor readings
        self.sys = None
        self.fpga = None

        # publisher for smoothed battery info
        self.battery_pub = self.create_publisher(Vector3, "/battery/status", 10)

        # rolling windows for median smoothing
        self.win_size = 20
        self.V_window = deque(maxlen=self.win_size)
        self.I_window = deque(maxlen=self.win_size)
        self.P_window = deque(maxlen=self.win_size)

        # exponentially smoothed values
        self.V_s = None
        self.I_s = None
        self.P_s = None
        self.alpha = 0.2  # smoothing factor

        # subscriptions
        self.create_subscription(Vector3, "/system/power", self.cb_system, 10)
        self.create_subscription(Vector3, "/fpga/power", self.cb_fpga, 10)

    # ------------------- Utilities -------------------

    def smooth(self, window, new_value, prev_smooth):
        """
        Median filter + exponential moving average.
        """
        window.append(new_value)
        median_val = statistics.median(window)

        if prev_smooth is None:
            return median_val

        return self.alpha * median_val + (1.0 - self.alpha) * prev_smooth

    # ------------------- Battery math -------------------

    def voltage_to_percentage(self, voltage):
        cell_v = voltage / 3.0

        if cell_v >= 4.20: return 100.0
        if cell_v <= 3.40: return 0.0

        table = [
            (4.20, 100),
            (3.95, 75),
            (3.80, 50),
            (3.68, 25),
            (3.55, 10),
            (3.40, 0),
        ]

        for i in range(len(table) - 1):
            v1, p1 = table[i]
            v2, p2 = table[i + 1]
            if v2 <= cell_v <= v1:
                return p2 + (p1 - p2) * (cell_v - v2) / (v1 - v2)

        return 0.0

    def estimate_runtime_hours(self, voltage, current_total):
        percent = self.voltage_to_percentage(voltage) / 100.0
        remaining_wh = percent * BATTERY_WH

        if current_total < 0.01:
            return float('inf')

        power_total = voltage * current_total
        if power_total <= 0.5:
            return float('inf')

        return remaining_wh / power_total

    # ------------------- Callbacks -------------------

    def cb_system(self, msg):
        self.sys = msg
        self.write_csv("system", msg)
        self.try_publish_battery()

    def cb_fpga(self, msg):
        self.fpga = msg
        self.write_csv("fpga", msg)
        self.try_publish_battery()

    # ------------------- Logging -------------------

    def write_csv(self, topic, msg):
        t = time.time()
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([t, topic, msg.x, msg.y, msg.z])

    # ------------------ Combined battery status ------------------

    def try_publish_battery(self):
        if self.sys is None or self.fpga is None:
            return

        # raw signals
        V_raw = self.sys.x
        I_raw = self.sys.y + self.fpga.y
        P_raw = self.sys.z + self.fpga.z

        # smoothed values
        self.V_s = self.smooth(self.V_window, V_raw, self.V_s)
        self.I_s = self.smooth(self.I_window, I_raw, self.I_s)
        self.P_s = self.smooth(self.P_window, P_raw, self.P_s)

        percent = self.voltage_to_percentage(self.V_s)
        runtime_h = self.estimate_runtime_hours(self.V_s, self.I_s)
        runtime_min = runtime_h * 60.0

        # log to console
        self.get_logger().info(
            f"[battery] V={self.V_s:.2f}V  I={self.I_s:.2f}A  "
            f"P={self.P_s:.1f}W  {percent:.0f}%  "
            f"runtime={runtime_min:.0f} min"
        )

        # publish smoothed battery status
        msg = Vector3()
        msg.x = self.P_s        # watts
        msg.y = percent         # %
        msg.z = runtime_min     # minutes
        self.battery_pub.publish(msg)


def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()