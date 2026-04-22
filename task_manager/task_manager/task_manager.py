#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8

from task_manager_interfaces.srv import SetTaskState


class TaskState:
    SEARCH_ITEM = 0
    APPROACH_ITEM = 1
    SEARCH_DROPOFF = 2
    APPROACH_DROPOFF = 3


VALID_TRANSITIONS = {
    TaskState.SEARCH_ITEM: [
        TaskState.SEARCH_ITEM,
        TaskState.APPROACH_ITEM,      # found item tag
    ],
    TaskState.APPROACH_ITEM: [
        TaskState.APPROACH_ITEM,
        TaskState.SEARCH_ITEM,        # lost item tag → search again
        TaskState.SEARCH_DROPOFF,     # pick completed → start dropoff search
    ],
    TaskState.SEARCH_DROPOFF: [
        TaskState.SEARCH_DROPOFF,
        TaskState.APPROACH_DROPOFF,   # found dropoff tag
    ],
    TaskState.APPROACH_DROPOFF: [
        TaskState.APPROACH_DROPOFF,
        TaskState.SEARCH_DROPOFF,     # lost dropoff tag → search again
        TaskState.SEARCH_ITEM,        # drop completed → start new cycle
    ],
    }

STATE_NAMES = {
    TaskState.SEARCH_ITEM: "SEARCH_ITEM",
    TaskState.APPROACH_ITEM: "APPROACH_ITEM",
    TaskState.SEARCH_DROPOFF: "SEARCH_DROPOFF",
    TaskState.APPROACH_DROPOFF: "APPROACH_DROPOFF",
    }


class TaskManager(Node):
    def __init__(self):
        super().__init__("task_manager")

        self.current_state = TaskState.SEARCH_ITEM

        self.state_pub = self.create_publisher(UInt8, "/task/state", 10)
        self.set_state_srv = self.create_service(SetTaskState, "/task/set_state", self.handle_set_state)

        # publish at 5 Hz so late subscribers always learn the state quickly
        self.timer = self.create_timer(0.2, self.publish_state)

        self.get_logger().info(f"TaskManager started. Initial state: {STATE_NAMES[self.current_state]}")

    def publish_state(self):
        msg = UInt8()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    def handle_set_state(self, request, response):
        new_state = int(request.new_state)
        requester = str(request.requester)

        if new_state not in STATE_NAMES:
            response.success = False
            response.message = f"Invalid state value: {new_state}"
            self.get_logger().warn(response.message)
            return response

        allowed = list(VALID_TRANSITIONS.get(self.current_state, []))

        # Special-case transitions allowed only for grab_node
        if requester == "grab_node":
            if self.current_state == TaskState.SEARCH_ITEM:
                allowed.append(TaskState.SEARCH_DROPOFF)
            elif self.current_state == TaskState.SEARCH_DROPOFF:
                allowed.append(TaskState.SEARCH_ITEM)

        if new_state in allowed:
            old_state = self.current_state
            self.current_state = new_state

            response.success = True
            response.message = (
                f"Transition OK: "
                f"{STATE_NAMES[old_state]} -> {STATE_NAMES[new_state]}"
            )

            self.get_logger().info(
                f"Client: {requester}, {response.message}"
            )

        else:
            allowed_names = [STATE_NAMES[s] for s in allowed]

            response.success = False
            response.message = (
                f"Rejected transition: "
                f"{STATE_NAMES[self.current_state]} -> {STATE_NAMES[new_state]} | "
                f"Allowed: {allowed_names}"
            )

            self.get_logger().warn(
                f"Client: {requester}, {response.message}"
            )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
