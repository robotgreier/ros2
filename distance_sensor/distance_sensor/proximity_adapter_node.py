#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range, LaserScan
import math

class ProximityAdapter(Node):
    def __init__(self):
        super().__init__("proximity_adapter")

        self.declare_parameter("gazebo_input_topic", "/ultrasonic/front/raw")
        self.declare_parameter("robot_input_topic", "/ultrasonic/front/raw_range")
        self.declare_parameter("output_topic", "/ultrasonic/front/scan")

        gazebo_input_topic = self.get_parameter("gazebo_input_topic").value
        robot_input_topic = self.get_parameter("robot_input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.get_logger().info(
            f"ProximityAdapter listening on:\n"
            f"  Range:     {robot_input_topic}\n"
            f"  LaserScan: {gazebo_input_topic}\n"
            f"Publishing unified Range on: {output_topic}"
        )

        # Publiser alltid Range ut
        self.pub = self.create_publisher(Range, output_topic, 10)

        # Abonner på begge, men på *forskjellige* topics
        self.create_subscription(Range, robot_input_topic, self.cb_range, 10)
        self.create_subscription(LaserScan, gazebo_input_topic, self.cb_laserscan, 10)

    # --- Callbacks ---

    # robot
    def cb_range(self, msg: Range):
        self.pub.publish(msg)

    # gazebo
    def cb_laserscan(self, msg: LaserScan):
        # Ta minste gyldige stråle
        vals = [r for r in msg.ranges if math.isfinite(r) and r > 0.0]
        d = min(vals) if vals else float("inf")

        out = Range()
        out.header = msg.header
        out.range = d
        out.min_range = msg.range_min
        out.max_range = msg.range_max
        self.pub.publish(out)

def main():
    rclpy.init()
    node = ProximityAdapter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
