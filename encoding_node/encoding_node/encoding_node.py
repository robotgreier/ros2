# encoding_node/encoding_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8MultiArray


from .proximity_bracket_encoder import ProximityBracketEncoder


class EncodingNode(Node):
    def __init__(self):
        super().__init__("encoding_node")

        self.declare_parameter("proximity_topic", "/ultrasonic/front/scan")
        self.declare_parameter("proximity_bin_edges", [1.0, 2.0, 3.0])
        #!temp
        self.declare_parameter("output_topic", "/snn/input")
        #/temp

        prox_topic = self.get_parameter("proximity_topic").value
        bin_edges = list(self.get_parameter("proximity_bin_edges").value)
        #!temp
        output_topic = self.get_parameter("output_topic").value
        #/temp

        self.prox_encoder = ProximityBracketEncoder(
            bin_edges=bin_edges,
            inf_as_far=True,   # important in enclosed worlds / no-hit cases
        )

        self.create_subscription(LaserScan, prox_topic, self.on_proximity_scan, 10)

        # TODO: your encoding array publisher for the SNN goes here

        # !Temporary publisher for testing!

        # Publisher (1 element array for now)
        
        
        self.pub = self.create_publisher(UInt8MultiArray, output_topic, 10)

        # Internal encoding vector (expand later)
        self.encoding = [0]   # index 0 = proximity spike

        # Subscriber
        self.create_subscription(
            LaserScan,
            prox_topic,
            self.on_proximity_scan,
            10
        )

        #/Temporary

    def on_proximity_scan(self, msg: LaserScan):
        # Robust: use min finite range; treat all-inf as inf
        vals = [r for r in msg.ranges if r > 0.0 and r != float("inf")]
        d = min(vals) if vals else float("inf")

        spike = self.prox_encoder.update(d)

        # TODO: write spike into your encoding array
        # e.g. self.encoding[IDX_PROX] = spike
        # and publish the array message

        #!Temporary!
        # Update encoding vector
        self.encoding[0] = spike

        # Publish
        out = UInt8MultiArray()
        out.data = self.encoding
        self.pub.publish(out)
        #/Temporary

def main():
    rclpy.init()
    node = EncodingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
