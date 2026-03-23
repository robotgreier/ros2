import math
from typing import Dict, List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8MultiArray, Int32MultiArray, Float32MultiArray, UInt8

from .proximity_bracket_encoder import ProximityBracketEncoder
from .keypoints_grid_encoder import KeypointsGridEncoder
from .aruco_direction_encoder import ArucoDirectionEncoder

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

        self.declare_parameter("pack_order", ["keypoints_grid", "proximity", "aruco_dir"])

        self.declare_parameter("proximity_topic", "/ultrasonic/front/scan")
        self.declare_parameter("proximity_n_dist_bits", 4)
        self.declare_parameter("proximity_dist_max_m", 1.5)

        self.declare_parameter("keypoints_topic", "/features/keypoints_grid")
        self.declare_parameter("keypoints_n_threshold_levels", 5)
        self.declare_parameter("keypoints_threshold_max", 20)

        # ---- Read parameters ----
        output_topic = self.get_parameter("output_topic").value
        self.pack_order = list(self.get_parameter("pack_order").value)

        proximity_topic = self.get_parameter("proximity_topic").value
        proximity_n_dist_bits = int(self.get_parameter("proximity_n_dist_bits").value)
        proximity_dist_max_m = float(self.get_parameter("proximity_dist_max_m").value)

        keypoints_topic = self.get_parameter("keypoints_topic").value
        keypoints_n_threshold_levels = int(self.get_parameter("keypoints_n_threshold_levels").value)
        keypoints_threshold_max = int(self.get_parameter("keypoints_threshold_max").value)

        # ---- Encoders ----
        self.prox_encoder = ProximityBracketEncoder(
            n_dist_bits=proximity_n_dist_bits,
            dist_max_m=proximity_dist_max_m,
        )

        self.kp_encoder = KeypointsGridEncoder(
            n_threshold_levels=keypoints_n_threshold_levels,
            threshold_max=keypoints_threshold_max,
        )

        # ---- Channels (named chunks) ----
        # Each channel value is a list[int] of 0/1 spikes.
        self.channels: Dict[str, List[int]] = {
            "proximity": [0] * proximity_n_dist_bits,
            "keypoints_grid": [], # variable length, filled on first message
        }

        # Last valid (finite) proximity reading; used to detect sensor saturation.
        # The ultrasonic driver returns inf for both "out of range" and "below
        # range_min" — track the last valid reading so we can tell them apart.
        self._last_proximity_d: float = float("inf")

        # ---- Publisher ----
        self.pub = self.create_publisher(UInt8MultiArray, output_topic, 10)

        # ---- Subscribers ---- accept both Range and LaserScan
        self.create_subscription(Range, proximity_topic, self.on_proximity_scan, 10)
        self.create_subscription(LaserScan, proximity_topic, self.on_proximity_scan, 10)
        self.create_subscription(Int32MultiArray, keypoints_topic, self.on_keypoints_grid, 10)

        self.get_logger().info(
            f"EncodingNode publishing {output_topic}. pack_order={self.pack_order}. "
            f"proximity_topic={proximity_topic}, keypoints_topic={keypoints_topic}"
        )

        # --- ArUco / state parameters ---
        self.declare_parameter("aruco_topic", "/vision/aruco/target")
        self.declare_parameter("task_state_topic", "/task/state")
        self.declare_parameter("aruco_n_bins", 5)

        aruco_topic = self.get_parameter("aruco_topic").value
        task_state_topic = self.get_parameter("task_state_topic").value
        aruco_n_bins = int(self.get_parameter("aruco_n_bins").value)

        self.aruco_encoder = ArucoDirectionEncoder(n_aruco_bins=aruco_n_bins)

        # Keep latest state locally (default SEARCH_ITEM)
        self.current_state = 0

        # Add channel
        self.channels["aruco_dir"] = [0] * aruco_n_bins

        # Subscribe to task state and aruco target
        self.create_subscription(UInt8, task_state_topic, self.on_task_state, 10)
        self.create_subscription(Float32MultiArray, aruco_topic, self.on_aruco_target, 10)

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

    def on_proximity_scan(self, msg) -> None:
        # Accept both Range and LaserScan
        # Robust extraction: choose minimum finite range; treat all-invalid as +inf.
        # The ultrasonic driver returns inf for BOTH "out of range" and "below
        # range_min" — they are indistinguishable from the range data alone.
        # Heuristic: if the last valid reading was inside the lowest thermometric
        # bracket (dist_max_m / n_dist_bits), the robot was already "very close"
        # and the sensor has saturated rather than lost the target — emit all-ones.
        
        # --- Normalize message into a list of ranges ---
        if isinstance(msg, Range):
        # Real robot - HC-SR04
            vals = [msg.range] if math.isfinite(msg.range) and msg.range > 0.0 else []
        
        # Gazebo simulation - LaserSCan with 1 beam
        elif isinstance(msg, LaserScan):
        vals = [r for r in msg.range if math.isfinite(r) and r > 0.0]    
        
        else:
            self.get_logger().warn(f"Unknown proximity msg type: {type(msg)}")
            return

        lowest_bracket = self.prox_encoder.thresholds[-1]  # smallest threshold
        
        if vals:
            d = min(vals)
            self._last_proximity_d = d
        elif self._last_proximity_d < lowest_bracket:
            d = self._last_proximity_d  # sensor saturated — stay at last close reading
        else:
            d = float("inf")

        self.channels["proximity"] = self.prox_encoder.update(d)

        self.publish_vector()

    def on_keypoints_grid(self, msg: Int32MultiArray) -> None:
        spikes, shape = self.kp_encoder.update_from_msg(msg)
        self.channels["keypoints_grid"] = spikes

        # Optional: log when grid shape is known/changes (helps during experiments)
        if shape is not None:
            self.get_logger().debug(f"Keypoints grid: {shape.rows}x{shape.cols}, len={len(spikes)}")

        self.publish_vector()
    
    def on_task_state(self, msg: UInt8) -> None:
        self.current_state = int(msg.data)

        # If we enter a SEARCH state, force the aruco channel to 000 immediately.
        if self.current_state in (0, 2):
            self.channels["aruco_dir"] = [0] * self.aruco_encoder.n_aruco_bins
            self.publish_vector()


    def on_aruco_target(self, msg: Float32MultiArray) -> None:
        # Expected layout from img_recog:
        # [0]=detect_flag, [2]=x_norm
        data = msg.data
        if data is None or len(data) < 3:
            # Not enough data to read detect_flag and x_norm
            self.channels["aruco_dir"] = [0] * self.aruco_encoder.n_aruco_bins
            self.publish_vector()
            return

        detect_flag = float(data[0])
        x_norm = float(data[2])

        code = self.aruco_encoder.encode(
            state=self.current_state,
            detect_flag=detect_flag,
            x_norm=x_norm,
        )

        self.channels["aruco_dir"] = code
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
