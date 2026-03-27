import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import csv
import time
from datetime import datetime

BATTERY_WH = 55.5  # Example: 3S 5000 mAh LiPo

class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"/opt/robot_ws/src/ros2/power_monitor/power_log/power_log_{timestamp}.csv"

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","topic","voltage","current","power"])

        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # latest readings
        self.sys = None
        self.fpga = None

        self.create_subscription(Vector3, "/system/power", self.cb_system, 10)
        self.create_subscription(Vector3, "/fpga/power", self.cb_fpga, 10)

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
            v2, p2 = table[i+1]
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
        self.try_print_battery()

    def cb_fpga(self, msg):
        self.fpga = msg
        self.write_csv("fpga", msg)
        self.try_print_battery()

    # ------------------- Logging -------------------

    def write_csv(self, topic, msg):
        t = time.time()
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([t, topic, msg.x, msg.y, msg.z])

    # ------------------ Combined battery status ------------------

    def try_print_battery(self):
        if self.sys is None or self.fpga is None:
            return

        V = self.sys.x
        I_total = self.sys.y + self.fpga.y
        P_total = self.sys.z + self.fpga.z

        percent = self.voltage_to_percentage(V)
        runtime_h = self.estimate_runtime_hours(V, I_total)

        self.get_logger().info(
            f"[battery] V={V:.2f}V  I={I_total:.2f}A  "
            f"P={P_total:.1f}W  {percent:.0f}%  "
            f"runtime={runtime_h*60:.0f} min"
        )

def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()