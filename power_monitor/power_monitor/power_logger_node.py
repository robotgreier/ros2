import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import csv
import os
import time
from datetime import datetime
import math
import uuid

BATTERY_WH = 49.02  # LiHV 3S: 11.4 V * 4.3 Ah = 49.02 Wh

STATE_NAMES = {
    0: "SEARCH_ITEM",
    1: "APPROACH_ITEM",
    2: "SEARCH_DROPOFF",
    3: "APPROACH_DROPOFF",
}

class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        # Run metadata for logging
        self.run_id = str(uuid.uuid4())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/opt/robot_ws/src/ros2/power_monitor/csv_logs"
        os.makedirs(log_dir, exist_ok=True)
        self.filename = f"{log_dir}/power_log_{timestamp}.csv"

        self.get_logger().info(f"Run ID: {self.run_id}")
        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # CSV header
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "ros_time_s",
            "run_id", 
            "state_id",
            "state_name",
            "source",
            "voltage_V",
            "current_A",
            "power_W",
            "energy_inc_Wh",
            "energy_total_Wh",
        ])

        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # latest filtered inputs
        self.system = None
        self.fpga = None
        self.system_time = None
        self.fpga_time = None
        self.current_state = None
        self.last_time = None
        self.energy_total_Wh = 0.0
        self.energy_per_state = {
            0: 0.0, # SEARCH_ITEM
            1: 0.0, # APPROACH_ITEM
            2: 0.0, # SEARCH_DROPOFF
            3: 0.0, # APPROACH_DROPOFF
        }

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

        self.create_subscription(
            Int32, "/state/current", self.cb_state, 10
        )

        # Utility function to get ROS time in seconds
        def now_ros_seconds(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- CSV logging ----------------

    def write_csv(self, source, msg, energy_inc):
        state_name = STATE_NAMES.get(self.current_state, "UNKNOWN")
        
        self.writer.writerow([
            self.now_ros_seconds(),
            self.run_id,
            self.current_state,
            state_name,
            source,
            msg.x,
            msg.y,
            msg.z,
            energy_inc,
            self.energy_total_Wh,
        ])

        self.file.flush()

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

    def cb_state(self, msg: Int32):
        self.current_state = msg.data

    def cb_system(self, msg):
        """
        msg.x = V (filtered)
        msg.y = I (filtered)
        msg.z = P (filtered)
        """
        self.system = msg
        self.system_time = self.now_ros_seconds()
        self.try_publish_battery("system", msg)

    def cb_fpga(self, msg: Vector3):
        self.fpga = msg
        self.fpga_time = self.now_ros_seconds()
        self.write_csv("fpga", msg)
        self.try_publish_battery("fpga", msg)

    # ---------------- Battery aggregation ----------------

    def try_publish_battery(self, source, msg):
        if self.system is None or self.fpga is None:
            return
    
        # Test if data is close in time
        if abs(self.system_time - self.fpga_time) > 0.1:
            return

        # Voltage assumed identical source rail
        V = self.system.x

        # Total load
        I = self.system.y + self.fpga.y
        P = self.system.z + self.fpga.z

        # Energy integration
        now = self.now_ros_seconds()
        energy_inc = 0.0
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.0:
                energy_inc = (P * dt) / 3600.0
                self.energy_total_Wh += energy_inc
                if self.current_state in self.energy_per_state:
                    self.energy_per_state[self.current_state] += energy_inc

        self.last_time = now

        # Log CSV
        self.write_csv(source, msg, energy_inc)

        # Battery status estimation for dashboard
        percent = self.voltage_to_percentage(V)
        runtime_min = self.estimate_runtime_minutes(V, I)

        self.get_logger().info(
            f"[battery] V={V:.2f}V  I={I:.2f}A  "
            f"P={P:.1f}W  {percent:.0f}%  "
            f"E={self.energy_total_Wh: 2f}Wh  "
            f"state={STATE_NAMES.get(self.current_state, "UNKNOWN")}  "
            f"runtime={runtime_min:.0f} min"
        )

        # Publish battery status
        out = Vector3()
        out.x = P              # total power [W]
        out.y = percent        # SOC [%]
        out.z = runtime_min    # remaining time [min]
        self.battery_pub.publish(out)

    # Cleanup on shutdown
    def destroy_node(self):
        self.get_logger().info("Final energy per state [Wh]:")
        for state, energy in self.energy_per_state.items():
            self.get_logger().info(f" {STATE_NAMES[state]:<18}: {energy:.3f} Wh")
        self.file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
