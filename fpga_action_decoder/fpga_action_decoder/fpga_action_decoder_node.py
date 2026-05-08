from typing import List

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8MultiArray, Bool, Int32, String
from geometry_msgs.msg import Twist

from .action_decoder import ActionDecoder


ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT", "BACKWARD"]  # index 0=LEFT, 1=FORWARD, 2=RIGHT, 3=BACKWARD


class FpgaActionDecoderNode(Node):
    def __init__(self) -> None:
        super().__init__("fpga_action_decoder_node")

        # Robot speed parameters
        self.declare_parameter("forward_speed", 0.125)
        self.declare_parameter("turn_speed", 0.3)

        # Topic parameters
        self.declare_parameter("cmd_vel_topic", "/cmd_vel/snn")
        self.declare_parameter("proximity_stop_topic", "/proximity_stop")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)

        self.proximity_stop: bool = False

        # Decoder
        self.decoder = ActionDecoder()

        # Publishers
        self.winner_pub = self.create_publisher(Int32, "/snn/winner", 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter("cmd_vel_topic").value, 10
        )
        self.pub_decision = self.create_publisher(String, "/snn/decision", 10)

        # Subscribers
        self.create_subscription(
            UInt8MultiArray,
            "/fpga/action_spikes",
            self.spikes_cb,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("proximity_stop_topic").value,
            self._on_proximity_stop,
            10,
        )

        self.get_logger().info("fpga_action_decoder_node started")

    def _on_proximity_stop(self, msg: Bool) -> None:
        self.proximity_stop = bool(msg.data)

    def spikes_cb(self, msg: UInt8MultiArray) -> None:
        spikes: List[int] = list(msg.data)

        action = self.decoder.decode_one_hot(spikes)

        if action is None:
            self.get_logger().warn(f"Invalid spike vector: {spikes}")
            return

        # Publish winner
        winner_msg = Int32()
        winner_msg.data = action
        self.winner_pub.publish(winner_msg)

        decision = self.publish_cmd_from_winner(action)

        self.get_logger().info(f"spikes={spikes} → action={action} → {decision}")

    def publish_cmd_from_winner(self, winner_idx: int) -> str:
        cmd = Twist()
        decision = "IDLE"

        if self.proximity_stop:
            if winner_idx == 0:      # LEFT
                cmd.linear.x = 0.0
                cmd.angular.z = +self.turn_speed
                decision = ACTION_NAMES[0]

            elif winner_idx == 1:    # BACKWARDS
                cmd.linear.x = -self.forward_speed
                cmd.angular.z = 0.0
                decision = ACTION_NAMES[1]

            elif winner_idx == 2:    # RIGHT
                cmd.linear.x = 0.0
                cmd.angular.z = -self.turn_speed
                decision = ACTION_NAMES[2]

            elif winner_idx < 0:
                decision = "IDLE"
                cmd.linear.x = -self.forward_speed
                cmd.angular.z = 0.0

            else:
                decision = "STOP_PROXIMITY"

        elif winner_idx < 0:
            decision = "IDLE"
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0

        else:
            if winner_idx == 0:      # LEFT
                cmd.linear.x = 0.0
                cmd.angular.z = +self.turn_speed
                decision = ACTION_NAMES[0]

            elif winner_idx == 1:    # BACKWARDS
                cmd.linear.x = -self.forward_speed
                cmd.angular.z = 0.0
                decision = ACTION_NAMES[1]

            elif winner_idx == 2:    # RIGHT
                cmd.linear.x = 0.0
                cmd.angular.z = -self.turn_speed
                decision = ACTION_NAMES[2]

            elif winner_idx == 3:    # BACKWARD
                cmd.linear.x = -self.forward_speed
                cmd.angular.z = 0.0
                decision = ACTION_NAMES[3]

            else:
                decision = "UNKNOWN"

        self.cmd_vel_pub.publish(cmd)
        self.pub_decision.publish(String(data=decision))

        return decision


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
