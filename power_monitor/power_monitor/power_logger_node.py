import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import csv
import time
from datetime import datetime

class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"/home/ubuntu/power_log_{timestamp}.csv"

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","topic","voltage","current","power"])

        self.get_logger().info(f"Logging to CSV: {self.filename}")

        self.create_subscription(Vector3, "/system/power", self.cb_system, 10)
        self.create_subscription(Vector3, "/fpga/power", self.cb_fpga, 10)

    def write(self, topic, msg):
        t = time.time()
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([t, topic, msg.x, msg.y, msg.z])

    def cb_system(self, msg):
        self.write("system", msg)

    def cb_fpga(self, msg):
        self.write("fpga", msg)

def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()