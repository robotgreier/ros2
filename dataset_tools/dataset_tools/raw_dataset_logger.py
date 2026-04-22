"""
raw_dataset_logger.py

ROS 2 dataset logger for collecting raw training data from simulation or teleop runs.

Purpose
-------
This node creates a synchronized raw dataset for later analysis and encoder design.
For each incoming camera frame, it:

1. Saves the image to a PNG file
2. Logs the latest available sensor/control values to a CSV file

Logged data
-----------
- Camera image (/camera/image_raw)
- Front ultrasonic-style LaserScan
    * center distance only
- ArUco target data (/vision/aruco/target)
    * x_norm only
- Teleop command (/cmd_vel/teleop)
    * linear.x
    * angular.z

Design choice
-------------
The image callback is used as the main logging trigger.
This means each CSV row corresponds to exactly one saved image frame.

Why this is useful
------------------
This creates a raw dataset that can later be used to:
- tune input encoding parameters
- inspect distance and visual alignment behavior
- build supervised or event-based training datasets
- cross-reference images with sensor values by timestamp
"""

import os
import csv
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class RawDatasetLogger(Node):
    def __init__(self):
        super().__init__("raw_dataset_logger")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("scan_topic", "/ultrasonic/front/scan")
        self.declare_parameter("aruco_topic", "/vision/aruco/target")
        self.declare_parameter("teleop_topic", "/cmd_vel/teleop")
        self.declare_parameter("dataset_root", os.path.expanduser("ros2/dataset_tools/trening_1"))
        # self.declare_parameter("dataset_root", os.path.expanduser("~/.ros/datasets/trening_1"))

        camera_topic = self.get_parameter("camera_topic").value
        scan_topic = self.get_parameter("scan_topic").value
        aruco_topic = self.get_parameter("aruco_topic").value
        teleop_topic = self.get_parameter("teleop_topic").value
        dataset_root = os.path.expanduser(self.get_parameter("dataset_root").value)

        # -----------------------------
        # Create unique run folder
        # -----------------------------
        run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(dataset_root, run_stamp)
        self.image_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "data.csv")

        # -----------------------------
        # CSV setup
        # -----------------------------
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ns",
            "image_filename",
            "range_center_m",
            "aruco_x_norm",
            "cmd_linear_x",
            "cmd_angular_z",
        ])
        self.csv_file.flush()

        # -----------------------------
        # Latest cached values
        # -----------------------------
        self.latest_range_center = float("nan")
        self.latest_aruco_x = float("nan")

        self.latest_cmd_linear_x = 0.0
        self.latest_cmd_angular_z = 0.0

        self.bridge = CvBridge()

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Image, camera_topic, self.cb_image, 10)
        self.create_subscription(LaserScan, scan_topic, self.cb_scan, 10)
        self.create_subscription(Float32MultiArray, aruco_topic, self.cb_aruco, 10)
        self.create_subscription(Twist, teleop_topic, self.cb_teleop, 10)

        self.get_logger().info("Dataset logger started")
        self.get_logger().info(f"Run directory: {self.run_dir}")
        self.get_logger().info(f"CSV file: {self.csv_path}")

    def cb_scan(self, msg: LaserScan):
        """Cache the single forward distance from the front ultrasonic-style LaserScan."""
        ranges = list(msg.ranges)
        if not ranges:
            return

        center_idx = len(ranges) // 2
        center_val = ranges[center_idx]

        if msg.range_min <= center_val <= msg.range_max:
            self.latest_range_center = float(center_val)
        else:
            self.latest_range_center = float("nan")

    def cb_aruco(self, msg: Float32MultiArray):
        """
        Cache only x_norm from /vision/aruco/target.

        Layout:
          0: visible
          1: id
          2: x_norm
          ...
        """
        data = list(msg.data)
        if len(data) < 3:
            self.get_logger().warn(f"Aruco message too short: len={len(data)}")
            return

        self.latest_aruco_x = float(data[2])

    def cb_teleop(self, msg: Twist):
        """Cache latest teleop command."""
        self.latest_cmd_linear_x = float(msg.linear.x)
        self.latest_cmd_angular_z = float(msg.angular.z)

    def cb_image(self, msg: Image):
        """
        Main logging trigger:
        save the image and write one CSV row using latest cached values.
        """
        timestamp_ns = int(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        image_filename = f"{timestamp_ns}.png"
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(image_path, cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            return

        row = [
            timestamp_ns,
            image_filename,
            self.latest_range_center,
            self.latest_aruco_x,
            self.latest_cmd_linear_x,
            self.latest_cmd_angular_z,
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RawDatasetLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
