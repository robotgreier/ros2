import math
from typing import Dict, List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8MultiArray, Int32MultiArray

from .proximity_bracket_encoder import ProximityBracketEncoder
from .keypoints_grid_encoder import KeypointsGridEncoder


class EncodingNode(Node):
    """
    Subscribes to sensor topics, encodes them into named channels (lists of 0/1),
    packs channels into one combined vector, and publishes it as UInt8MultiArray.

    Packing order is configurable by the 'pack_order' parameter.
    """

    def __init__(self):
        super().__init__("encoding_node")

        # ---- Parameters ----
        self.declare_parameter("output_topic", "/snn/input")

        self.declare_parameter("pack_order", ["proximity", "keypoints_grid"])

        self.declare_parameter("proximity_topic", "/ultrasonic/front/scan")
        self.declare_parameter("proximity_bin_edges", [1.0, 2.0, 3.0])
        self.declare_parameter("proximity_inf_as_far", True)

        self.declare_parameter("keypoints_topic", "/features/keypoints_grid")
        self.declare_parameter("keypoints_threshold", 5)

        # ---- Read parameters ----
        output_topic = self.get_parameter("output_topic").value
        self.pack_order = list(self.get_parameter("pack_order").value)

        proximity_topic = self.get_parameter("proximity_topic").value
        proximity_bin_edges = list(self.get_parameter("proximity_bin_edges").value)
        proximity_inf_as_far = bool(self.get_parameter("proximity_inf_as_far").value)

        keypoints_topic = self.get_parameter("keypoints_topic").value
        keypoints_threshold = int(self.get_parameter("keypoints_threshold").value)

        # ---- Encoders ----
        self.prox_encoder = ProximityBracketEncoder(
            bin_edges=proximity_bin_edges,
            inf_as_far=proximity_inf_as_far,
        )

        self.kp_encoder = KeypointsGridEncoder(threshold=keypoints_threshold)

        # ---- Channels (named chunks) ----
        # Each channel value is a list[int] of 0/1 spikes.
        self.channels: Dict[str, List[int]] = {
            "proximity": [0],     # single-bit channel
            "keypoints_grid": [], # variable length, filled on first message
        }

        # ---- Publisher ----
        self.pub = self.create_publisher(UInt8MultiArray, output_topic, 10)

        # ---- Subscribers ----
        self.create_subscription(LaserScan, proximity_topic, self.on_proximity_scan, 10)
        self.create_subscription(Int32MultiArray, keypoints_topic, self.on_keypoints_grid, 10)

        self.get_logger().info(
            f"EncodingNode publishing {output_topic}. pack_order={self.pack_order}. "
            f"proximity_topic={proximity_topic}, keypoints_topic={keypoints_topic}"
        )

    def pack_vector(self) -> List[int]:
        """
        Concatenate channels in pack_order. Missing channels contribute nothing.
        """
        out: List[int] = []
        for name in self.pack_order:
            chunk = self.channels.get(name, [])
            out.extend(chunk)
        return out

    def publish_vector(self) -> None:
        msg = UInt8MultiArray()
        msg.data = self.pack_vector()
        self.pub.publish(msg)

    def on_proximity_scan(self, msg: LaserScan) -> None:
        # Robust extraction: choose minimum finite range; treat all-invalid as +inf
        vals = [r for r in msg.ranges if math.isfinite(r) and r > 0.0]
        d = min(vals) if vals else float("inf")

        spike = self.prox_encoder.update(d)
        self.channels["proximity"] = [spike]

        self.publish_vector()

    def on_keypoints_grid(self, msg: Int32MultiArray) -> None:
        spikes, shape = self.kp_encoder.update_from_msg(msg)
        self.channels["keypoints_grid"] = spikes

        # Optional: log when grid shape is known/changes (helps during experiments)
        if shape is not None:
            self.get_logger().debug(f"Keypoints grid: {shape.rows}x{shape.cols}, len={len(spikes)}")

        self.publish_vector()


def main():
    rclpy.init()
    node = EncodingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
