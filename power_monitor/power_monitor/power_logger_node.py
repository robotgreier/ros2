import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import csv
import time
from datetime import datetime
import math

BATTERY_WH = 49.02  # LiHV 3S: 11.4 V * 4.3 Ah = 49.02 Wh


class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/opt/robot_ws/log/power_monitor"
        self.filename = f"{log_dir}/power_log_{timestamp}.csv"

        # CSV header
        with open(self.filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                "timestamp",
                "source",
                "voltage_V",
                "current_A",
                "power_W"
            ])

        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # latest filtered inputs
        self.system = None
        self.fpga = None

        # battery status publisher
        self.battery_pub = self.create_publisher(
            Vector3, "/battery/status", 10
        )

        # subscriptions (already filtered upstream)
        self.create_subscription(
            Vector3, "/system/power", self.cb_system, 10
        )
        self.create_subscription(
            Vector3, "/fpga/power", self.cb_fpga, 10
        )

    # ---------------- CSV logging ----------------

    def write_csv(self, source, msg):
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                time.time(),
                source,
                msg.x,
                msg.y,
                msg.z
            ])

    # ---------------- Battery math ----------------

    def voltage_to_percentage(self, voltage):
        """
        LiHV 3S open-circuit approximation.
        """
        cell_v = voltage / 3.0

        table = [
            (4.35, 100),
            (4.00, 80),
            (3.87, 60),
            (3.77, 40),
            (3.65, 20),
            (3.40, 0),
        ]

        if cell_v >= table[0][0]:
            return 100.0
        if cell_v <= table[-1][0]:
            return 0.0

        for (v1, p1), (v2, p2) in zip(table, table[1:]):
            if v2 <= cell_v <= v1:
                return p2 + (p1 - p2) * (cell_v - v2) / (v1 - v2)

        return 0.0

    def estimate_runtime_minutes(self, voltage, current):
        if current < 0.05:
            return math.inf

        percent = self.voltage_to_percentage(voltage) / 100.0
        remaining_wh = percent * BATTERY_WH
        power = voltage * current

        if power <= 0.5:
            return math.inf

        return (remaining_wh / power) * 60.0

    # ---------------- Callbacks ----------------

    def cb_system(self, msg):
        """
        msg.x = V (filtered)
        msg.y = I (filtered)
        msg.z = P (filtered)
        """
        self.system = msg
        self.write_csv("system", msg)
        self.try_publish_battery()

    def cb_fpga(self, msg):
        self.fpga = msg
        self.write_csv("fpga", msg)
        self.try_publish_battery()

    # ---------------- Battery aggregation ----------------

    def try_publish_battery(self):
        if self.system is None or self.fpga is None:
            return

        # Voltage assumed identical source rail
        V = self.system.x

        # Total load
        I = self.system.y + self.fpga.y
        P = self.system.z + self.fpga.z

        percent = self.voltage_to_percentage(V)
        runtime_min = self.estimate_runtime_minutes(V, I)

        self.get_logger().info(
            f"[battery] V={V:.2f}V  I={I:.2f}A  "
            f"P={P:.1f}W  {percent:.0f}%  "
            f"runtime={runtime_min:.0f} min"
        )

        msg = Vector3()
        msg.x = P              # total power [W]
        msg.y = percent        # SOC [%]
        msg.z = runtime_min    # remaining time [min]

        self.battery_pub.publish(msg)


def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
