#!/usr/bin/env python3
from bisect import bisect_right
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8


class ProximityBracketEvent(Node):
    """
    Subscribes to a LaserScan that effectively contains 1 range sample (ranges[0]).
    Converts the range into a discrete "bracket/bin" and publishes an event bit:
      - 0 normally
      - 1 only when the bracket changes vs last bracket
    """

    def __init__(self):
        super().__init__('proximity_bracket_event')

        # Parameters
        self.declare_parameter('input_topic', '/ultrasonic/front/scan')
        self.declare_parameter('output_topic', '/proximity/event')

        # Bin edges in meters, ascending. Example: [0.2, 0.5, 1.0]
        # This creates bins:
        #   bin 0: (-inf, 0.2]
        #   bin 1: (0.2, 0.5]
        #   bin 2: (0.5, 1.0]
        #   bin 3: (1.0, +inf)
        self.declare_parameter('bin_edges', [0.2, 0.5, 1.0])

        # If True: publish only when event==1
        # If False: publish 0/1 every callback (your “publish zero unless change” wording)
        self.declare_parameter('publish_on_change_only', False)

        # If True: ignore invalid/inf/nan readings (do not update bracket)
        self.declare_parameter('ignore_invalid', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.bin_edges = list(self.get_parameter('bin_edges').value)
        self.publish_on_change_only = bool(self.get_parameter('publish_on_change_only').value)
        self.ignore_invalid = bool(self.get_parameter('ignore_invalid').value)

        self._validate_bins(self.bin_edges)

        self._prev_bin = None

        self.sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self._on_scan,
            10
        )
        self.pub = self.create_publisher(UInt8, self.output_topic, 10)

        self.get_logger().info(
            f"Listening on {self.input_topic}, publishing events on {self.output_topic}, "
            f"bin_edges={self.bin_edges}, publish_on_change_only={self.publish_on_change_only}"
        )

    def _validate_bins(self, edges):
        if not edges:
            raise ValueError("bin_edges must be a non-empty list of ascending floats")
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                raise ValueError("bin_edges must be strictly ascending")

    def _range_to_bin(self, r: float) -> int:
        # bisect_right returns an index 0..len(edges)
        # This matches the bin definition above.
        return bisect_right(self.bin_edges, r)

    def _on_scan(self, msg: LaserScan):
        # Robust: works for 1-ray or multi-ray
        vals = [r for r in msg.ranges if math.isfinite(r) and r > 0.0]
        if not vals:
            if self.ignore_invalid:
                return
            r = float('inf')
        else:
            r = min(vals)

        current_bin = self._range_to_bin(r)

        event = 0
        if self._prev_bin is None:
            self._prev_bin = current_bin
        elif current_bin != self._prev_bin:
            event = 1
            self._prev_bin = current_bin

        if self.publish_on_change_only:
            if event == 1:
                out = UInt8()
                out.data = 1
                self.pub.publish(out)
        else:
            out = UInt8()
            out.data = event
            self.pub.publish(out)



def main():
    rclpy.init()
    node = ProximityBracketEvent()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
