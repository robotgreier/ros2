#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, UInt8


# Task states
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


class ProximityStopNode(Node):
    """
    Publishes /proximity_stop (Bool) based on minimum finite range in LaserScan,
    gated by task state (disabled during APPROACH_* states by default).
    """

    def __init__(self):
        super().__init__("proximity_stop_node")

        # Parameters
        self.declare_parameter("scan_topic", "/ultrasonic/front/scan")
        self.declare_parameter("state_topic", "/task/state")
        self.declare_parameter("stop_topic", "/proximity_stop")

        self.declare_parameter("distance_threshold", 0.05)  # meters
        self.declare_parameter("disabled_states", [APPROACH_ITEM, APPROACH_DROPOFF])

        self.scan_topic = self.get_parameter("scan_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.stop_topic = self.get_parameter("stop_topic").value

        self.distance_threshold = float(self.get_parameter("distance_threshold").value)
        self.disabled_states = set(int(x) for x in self.get_parameter("disabled_states").value)

        # Internal state
        self.current_state = SEARCH_ITEM
        self.last_stop = None  # publish only on changes

        # QoS: sensors often use BEST_EFFORT
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.stop_pub = self.create_publisher(Bool, self.stop_topic, 10)

        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, sensor_qos
        )
        self.state_sub = self.create_subscription(
            UInt8, self.state_topic, self.on_state, 10
        )

        self.get_logger().info(
            f"Listening scan: {self.scan_topic} | state: {self.state_topic} | publishing: {self.stop_topic}\n"
            f"distance_threshold={self.distance_threshold} m | disabled_states={sorted(self.disabled_states)}"
        )

    def on_state(self, msg: UInt8) -> None:
        self.current_state = int(msg.data)

    def on_scan(self, msg: LaserScan) -> None:
        # Gate emergency stop by state
        stop_enabled = (self.current_state not in self.disabled_states)

        # Compute min finite positive range
        vals = [r for r in msg.ranges if math.isfinite(r) and r > 0.0]
        dmin = min(vals) if vals else float("inf")

        # Stop condition
        stop = bool(stop_enabled and (dmin < self.distance_threshold))

        # Publish only when changed (reduces spam)
        if self.last_stop is None or stop != self.last_stop:
            out = Bool()
            out.data = stop
            self.stop_pub.publish(out)
            self.last_stop = stop

            self.get_logger().info(
                f"state={self.current_state} stop_enabled={stop_enabled} dmin={dmin:.3f} -> proximity_stop={int(stop)}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ProximityStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
