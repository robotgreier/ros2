from typing import List

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8MultiArray, UInt8, Int32
from geometry_msgs.msg import Twist

from .action_decoder import ActionDecoder


class FpgaActionDecoderNode(Node):
    def __init__(self) -> None:
        super().__init__("fpga_action_decoder_node")

        # Parameters (easy to tune later)
        self.declare_parameter("linear_speed", 0.2)
        self.declare_parameter("angular_speed", 1.0)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)

        # Decoder
        self.decoder = ActionDecoder()

        # Publishers
        self.winner_pub = self.create_publisher(UInt8, "/snn/winner", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel/snn", 10)

        # Subscriber (FPGA output)
        self.create_subscription(
            UInt8MultiArray,
            "/fpga/action_spikes",
            self.spikes_cb,
            10,
        )

        self.get_logger().info("fpga_action_decoder_node started")

    def spikes_cb(self, msg: UInt8MultiArray) -> None:
        spikes: List[int] = list(msg.data)

        action = self.decoder.decode_one_hot(spikes)

        if action is None:
            self.get_logger().warn(f"Invalid spike vector: {spikes}")
            return

        # Publish winner
        winner_msg = UInt8()
        winner_msg.data = action
        self.winner_pub.publish(winner_msg)

        # Convert to Twist
        twist = self.action_to_twist(action)
        self.cmd_pub.publish(twist)

        self.get_logger().info(f"spikes={spikes} → action={action}")

    def action_to_twist(self, action: int) -> Twist:
        twist = Twist()

        if action == 0:  # LEFT
            twist.angular.z = self.angular_speed

        elif action == 1:  # FORWARD
            twist.linear.x = self.linear_speed

        elif action == 2:  # RIGHT
            twist.angular.z = -self.angular_speed

        elif action == 3:  # BACKWARD
            twist.linear.x = -self.linear_speed

        return twist


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FpgaActionDecoderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down fpga_action_decoder_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()