#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8
from task_manager.srv import SetTaskState



class TaskState:
    SEARCH_ITEM = 0
    APPROACH_ITEM = 1
    SEARCH_DROPOFF = 2
    APPROACH_DROPOFF = 3


VALID_TRANSITIONS = {
    TaskState.SEARCH_ITEM: [TaskState.APPROACH_ITEM],
    TaskState.APPROACH_ITEM: [TaskState.SEARCH_DROPOFF],
    TaskState.SEARCH_DROPOFF: [TaskState.APPROACH_DROPOFF],
    TaskState.APPROACH_DROPOFF: [TaskState.SEARCH_ITEM],
}


STATE_NAMES = {
    TaskState.SEARCH_ITEM: "SEARCH_ITEM",
    TaskState.APPROACH_ITEM: "APPROACH_ITEM",
    TaskState.SEARCH_DROPOFF: "SEARCH_DROPOFF",
    TaskState.APPROACH_DROPOFF: "APPROACH_DROPOFF",
}


class StateManager(Node):

    def __init__(self):
        super().__init__('state_manager')

        # Initial state
        self.current_state = TaskState.SEARCH_ITEM

        # Publisher
        self.state_pub = self.create_publisher(UInt8, '/task/state', 10)

        # Service
        self.state_srv = self.create_service(
            SetTaskState,
            '/task/set_state',
            self.handle_set_state
        )

        # Timer to periodically publish state (5 Hz)
        self.timer = self.create_timer(0.2, self.publish_state)

        self.get_logger().info(
            f"State Manager started. Initial state: {STATE_NAMES[self.current_state]}"
        )

    def publish_state(self):
        msg = UInt8()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    def handle_set_state(self, request, response):
        new_state = request.new_state

        if new_state not in STATE_NAMES:
            response.success = False
            response.message = "Invalid state value"
            self.get_logger().warn("Rejected invalid state request")
            return response

        if new_state in VALID_TRANSITIONS[self.current_state]:
            old_state = self.current_state
            self.current_state = new_state

            response.success = True
            response.message = "State updated successfully"

            self.get_logger().info(
                f"State transition: {STATE_NAMES[old_state]} -> {STATE_NAMES[new_state]}"
            )
        else:
            response.success = False
            response.message = "Invalid state transition"

            self.get_logger().warn(
                f"Rejected transition: {STATE_NAMES[self.current_state]} -> {STATE_NAMES[new_state]}"
            )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
