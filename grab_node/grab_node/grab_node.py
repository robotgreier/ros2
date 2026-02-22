#!/usr/bin/env python3

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from task_manager_interfaces.srv import SetTaskState

from std_srvs.srv import Trigger

# ---- Task States ----
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


# ---- Event Codes ----
EVENT_IDLE = 0
EVENT_GRABBED = 1
EVENT_DROPPED = 2
EVENT_BUSY = 3


# ---- Gripper Commands ----
GRIPPER_IDLE = 0
GRIPPER_GRIP = 1
GRIPPER_DROP = 2


class GrabState(Enum):
    IDLE = 0
    WAITING_ALIGNMENT = 1
    EXECUTING_FORWARD = 2
    ACTUATING = 3
    EXECUTING_BACKUP = 4
    WAITING_SERVICE = 5


class GrabNode(Node):

    def __init__(self):
        super().__init__("grab_node")

        # -------- Parameters --------
        self.declare_parameter("center_threshold", 0.1)
        self.declare_parameter("item_distance_threshold", 0.10)
        self.declare_parameter("dropoff_distance_threshold", 0.10)

        self.declare_parameter("approach_speed", 0.05)
        self.declare_parameter("approach_distance_item", 0.10)
        self.declare_parameter("approach_distance_dropoff", 0.10)

        self.declare_parameter("backup_speed", 0.05)
        self.declare_parameter("backup_distance", 0.10)

        self.declare_parameter("use_sim_gripper", True)
        self.use_sim_gripper = self.get_parameter("use_sim_gripper").value 

        self.center_threshold = self.get_parameter("center_threshold").value
        self.item_dist_thresh = self.get_parameter("item_distance_threshold").value
        self.drop_dist_thresh = self.get_parameter("dropoff_distance_threshold").value

        self.approach_speed = self.get_parameter("approach_speed").value
        self.approach_item_dist = self.get_parameter("approach_distance_item").value
        self.approach_drop_dist = self.get_parameter("approach_distance_dropoff").value

        self.backup_speed = self.get_parameter("backup_speed").value
        self.backup_distance = self.get_parameter("backup_distance").value

        # -------- Publishers --------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel/grab", 10)
        self.event_pub = self.create_publisher(UInt8, "/grab_node/event", 10)
        self.gripper_pub = self.create_publisher(UInt8, "/gripper/command", 10)

        # -------- Subscribers --------
        self.create_subscription(Float32MultiArray,
                                 "/vision/aruco/target",
                                 self.cb_aruco,
                                 10)

        self.create_subscription(UInt8,
                                 "/task/state",
                                 self.cb_task_state,
                                 10)

        # -------- Service Client --------
        self.cli = self.create_client(SetTaskState, "/task/set_state")

        # -------- Internal State --------
        self.state = GrabState.IDLE
        self.current_task_state = SEARCH_ITEM

        self.x_norm = None
        self.distance = None

        self.motion_timer = None
        self.service_future = None

        self.get_logger().info("Grab node initialized.")

        # For simulation
        if self.use_sim_gripper:
            self.grab_cli = self.create_client(Trigger, "/gripper/grab")
            self.drop_cli = self.create_client(Trigger, "/gripper/drop")

    # =====================================================
    # ------------------ Callbacks ------------------------
    # =====================================================

    def cb_task_state(self, msg: UInt8):
        self.current_task_state = msg.data

        if self.current_task_state in [APPROACH_ITEM, APPROACH_DROPOFF]:
            if self.state == GrabState.IDLE:
                self.state = GrabState.WAITING_ALIGNMENT
                self.get_logger().info("Entering WAITING_ALIGNMENT")
        else:
            self.state = GrabState.IDLE

    def cb_aruco(self, msg: Float32MultiArray):
        data = msg.data

        if len(data) < 14:
            return

        self.x_norm = data[2]
        self.distance = data[6]

        if self.state == GrabState.WAITING_ALIGNMENT:
            self.check_alignment_and_distance()

    # =====================================================
    # ------------------ Logic ----------------------------
    # =====================================================

    def check_alignment_and_distance(self):

        if self.x_norm is None or self.distance is None:
            return

        centered = abs(self.x_norm) < self.center_threshold

        if self.current_task_state == APPROACH_ITEM:
            close_enough = self.distance < self.item_dist_thresh
        elif self.current_task_state == APPROACH_DROPOFF:
            close_enough = self.distance < self.drop_dist_thresh
        else:
            return

        if centered and close_enough:
            self.get_logger().info("Alignment and distance OK. Starting approach.")
            self.start_forward_motion()

    def start_forward_motion(self):

        self.publish_event(EVENT_BUSY)

        if self.current_task_state == APPROACH_ITEM:
            dist = self.approach_item_dist
        else:
            dist = self.approach_drop_dist

        duration = dist / self.approach_speed

        self.publish_velocity(self.approach_speed)

        self.state = GrabState.EXECUTING_FORWARD
        self.motion_timer = self.create_timer(duration, self.finish_forward_motion)

    def finish_forward_motion(self):
        self.stop_motion()
        self.motion_timer.cancel()

        self.state = GrabState.ACTUATING
        self.perform_actuation()

    def perform_actuation(self):

        if self.current_task_state == APPROACH_ITEM:

            self.publish_event(EVENT_GRABBED)

            if self.use_sim_gripper:
                self.call_gripper_service(self.grab_cli)
            else:
                self.gripper_pub.publish(UInt8(data=GRIPPER_GRIP))

            next_state = SEARCH_DROPOFF
            self.call_set_state(next_state)

        elif self.current_task_state == APPROACH_DROPOFF:

            self.publish_event(EVENT_DROPPED)

            if self.use_sim_gripper:
                self.call_gripper_service(self.drop_cli)
            else:
                self.gripper_pub.publish(UInt8(data=GRIPPER_DROP))

            self.start_backup_motion()

    def call_gripper_service(self, client):

        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for gripper service...")

        req = Trigger.Request()
        future = client.call_async(req)

        def callback(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info("Gripper action successful.")
                else:
                    self.get_logger().error("Gripper action failed.")
            except Exception as e:
                self.get_logger().error(f"Gripper service error: {e}")

        future.add_done_callback(callback)

    def start_backup_motion(self):

        duration = self.backup_distance / self.backup_speed
        self.publish_velocity(-self.backup_speed)

        self.state = GrabState.EXECUTING_BACKUP
        self.motion_timer = self.create_timer(duration, self.finish_backup_motion)

    def finish_backup_motion(self):
        self.stop_motion()
        self.motion_timer.cancel()

        next_state = SEARCH_ITEM
        self.call_set_state(next_state)

    # =====================================================
    # ------------------ Service --------------------------
    # =====================================================

    def call_set_state(self, new_state):

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /task/set_state service...")

        req = SetTaskState.Request()
        req.new_state = new_state
        req.requester = "grab_node"

        self.service_future = self.cli.call_async(req)
        self.service_future.add_done_callback(self.service_response_callback)

        self.state = GrabState.WAITING_SERVICE

    def service_response_callback(self, future):

        try:
            response = future.result()
            if response.success:
                self.get_logger().info("State change successful.")
            else:
                self.get_logger().error(f"State change failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

        self.state = GrabState.IDLE

    # =====================================================
    # ------------------ Utilities ------------------------
    # =====================================================

    def publish_velocity(self, linear_x):
        cmd = Twist()
        cmd.linear.x = linear_x
        self.cmd_pub.publish(cmd)

    def stop_motion(self):
        self.cmd_pub.publish(Twist())

    def publish_event(self, code):
        self.event_pub.publish(UInt8(data=code))


def main(args=None):
    rclpy.init(args=args)
    node = GrabNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
