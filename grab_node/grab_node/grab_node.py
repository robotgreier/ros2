#!/usr/bin/env python3

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import UInt8, Bool
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

# Gripper commands
GRIPPER_DROP = 0
GRIPPER_GRIP = 1

# Gripper states
GRIPPER_STATE_OPEN = 0
GRIPPER_STATE_MOVING = 1
GRIPPER_STATE_CLOSED = 2
GRIPPER_STATE_ERROR = 255

class GrabState(Enum):
    IDLE = 0
    WAITING_ALIGNMENT = 1
    EXECUTING_FORWARD = 2
    ACTUATING = 3
    EXECUTING_BACKUP = 4
    WAITING_SERVICE = 5
    CREEPING_TO_ITEM = 6


class GrabNode(Node):

    def __init__(self):
        super().__init__("grab_node")

        # -------- Parameters --------
        self.declare_parameter("center_threshold", 0.15)
        self.declare_parameter("item_distance_threshold", 0.3)
        self.declare_parameter("dropoff_distance_threshold", 0.3)

        self.declare_parameter("approach_speed", 0.125)
        self.declare_parameter("approach_distance_item", 0.4)
        self.declare_parameter("approach_distance_dropoff", 0.3)

        self.declare_parameter("backup_speed", 0.125)
        self.declare_parameter("backup_distance", 0.6)

        self.declare_parameter("motion_publish_rate_hz", 20.0)

        self.declare_parameter("item_final_distance", -1.0)
        self.declare_parameter("dropoff_final_distance", 0.02)

        self.declare_parameter("min_forward_distance", 0.2)
        self.declare_parameter("max_forward_distance", 0.45)

        self.declare_parameter("creep_speed", 0.1)
        self.declare_parameter("grip_timeout_sec", 4.0)
        self.declare_parameter("failed_grab_backup_distance", 0.3)

        # For simulation
        self.declare_parameter("use_sim_gripper", False)
        self.use_sim_gripper = self.get_parameter("use_sim_gripper").value 
        #/ For simulation

        self.center_threshold = self.get_parameter("center_threshold").value
        self.item_dist_thresh = self.get_parameter("item_distance_threshold").value
        self.drop_dist_thresh = self.get_parameter("dropoff_distance_threshold").value

        self.approach_speed = self.get_parameter("approach_speed").value
        self.approach_item_dist = self.get_parameter("approach_distance_item").value
        self.approach_drop_dist = self.get_parameter("approach_distance_dropoff").value

        self.backup_speed = self.get_parameter("backup_speed").value
        self.backup_distance = self.get_parameter("backup_distance").value

        self.motion_publish_rate_hz = self.get_parameter("motion_publish_rate_hz").value

        self.item_final_distance = self.get_parameter("item_final_distance").value
        self.dropoff_final_distance = self.get_parameter("dropoff_final_distance").value

        self.min_forward_distance = self.get_parameter("min_forward_distance").value
        self.max_forward_distance = self.get_parameter("max_forward_distance").value

        self.creep_speed = self.get_parameter("creep_speed").value
        self.grip_timeout_sec = self.get_parameter("grip_timeout_sec").value
        self.failed_grab_backup_distance = self.get_parameter("failed_grab_backup_distance").value

        self.gripper_state = None
        self.waiting_for_gripper = False
        self.expected_gripper_state = None

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
        
        self.gripper_state_sub = self.create_subscription(UInt8,
                                                        '/gripper/state',
                                                        self.gripper_state_callback,
                                                        10)
        
        self.create_subscription(Bool,
                                "/gripper/proximity_trigger",
                                self.cb_proximity_trigger,
                                10)

        # -------- Service Client --------
        self.cli = self.create_client(SetTaskState, "/task/set_state")

        # Motion 
        self.motion_end_time = None # when the current motion should end
        self.active_cmd = Twist() # the command to keep publishing
        self.motion_timer = None # optional, if you still want to keep the name
        self.cmd_stream_timer = None # a repeating timer that publishes /cmd_vel/grab

        # This starts the repeating publisher timer once,
        #  and it will only actively publish during movement states.
        self.start_cmd_stream()

        # -------- Internal State --------
        self.state = GrabState.IDLE
        self.current_task_state = SEARCH_ITEM

        self.x_norm = None
        self.distance = None

        self.motion_timer = None
        self.service_future = None

        self.get_logger().info("Grab node initialized.")

        # To avoid state clash
        self.sequence_active = False
        
        # Latest proximity trigger from the gripper sensor
        self.proximity_triggered = False

        # Deadline used while creeping toward the item
        self.grip_timeout_deadline = None

        # Used to distinguish normal dropoff backup from failed-grab backup
        self.backup_after_failed_grab = False

        # For simulation
        if self.use_sim_gripper:
            self.grab_cli = self.create_client(Trigger, "/gripper/grab")
            self.drop_cli = self.create_client(Trigger, "/gripper/drop")

    # =====================================================
    # ------------------ Callbacks ------------------------
    # =====================================================

    def cb_task_state(self, msg: UInt8):
        new_task_state = msg.data

        if self.sequence_active:
            return

        self.current_task_state = new_task_state

        if self.current_task_state in [APPROACH_ITEM, APPROACH_DROPOFF]:
            if self.state == GrabState.IDLE:
                self.reset_creep_state()
                self.state = GrabState.WAITING_ALIGNMENT
                self.get_logger().info("Entering WAITING_ALIGNMENT")
            return

        if self.state == GrabState.WAITING_ALIGNMENT:
            self.reset_creep_state()
            self.state = GrabState.IDLE

    def cb_aruco(self, msg: Float32MultiArray):
        data = msg.data

        if len(data) < 14:
            return

        self.x_norm = data[2]
        self.distance = data[6]

        if self.state == GrabState.WAITING_ALIGNMENT:
            self.check_alignment_and_distance()

        self.get_logger().info(
            f"Aruco update: x_norm={self.x_norm:.3f}, "
            f"distance={self.distance:.3f}, "
            f"tvec=({data[7]:.3f}, {data[8]:.3f}, {data[9]:.3f})"
        )

    def gripper_state_callback(self, msg):
        self.gripper_state = int(msg.data)
        self.get_logger().info(f"Received gripper state: {self.gripper_state}")

        if not self.waiting_for_gripper:
            return

        if self.gripper_state == GRIPPER_STATE_ERROR:
            self.get_logger().error("Gripper reported ERROR state.")
            self.waiting_for_gripper = False
            self.sequence_active = False
            self.state = GrabState.IDLE
            return

        if self.gripper_state == self.expected_gripper_state:
            self.get_logger().info(
                f"Expected gripper state {self.expected_gripper_state} confirmed."
            )

            self.waiting_for_gripper = False

            if self.current_task_state == APPROACH_ITEM:
                self.reset_creep_state()
                self.publish_event(EVENT_GRABBED)

                next_state = SEARCH_DROPOFF
                self.call_set_state(next_state)

                self.sequence_active = False
                self.state = GrabState.IDLE

            elif self.current_task_state == APPROACH_DROPOFF:
                self.publish_event(EVENT_DROPPED)
                self.start_backup_motion()
                self.state = GrabState.EXECUTING_BACKUP

    def cb_proximity_trigger(self, msg: Bool):
        # Keep the latest trigger state from the gripper sensor.
        self.proximity_triggered = bool(msg.data)
    

    # =====================================================
    # ------------------ Logic ----------------------------
    # =====================================================

    def check_alignment_and_distance(self):

        if self.x_norm is None or self.distance is None:
            return

        # Reject invalid or zero distance
        if self.distance <= 0.0:
            self.get_logger().warn(f"Ignoring invalid distance: {self.distance:.3f}")
            return

        centered = abs(self.x_norm) < self.center_threshold

        # Optional: enforce a minimum valid distance (more robust)
        distance_valid = self.distance > 0.2

        if self.current_task_state == APPROACH_ITEM:
            close_enough = self.distance < self.item_dist_thresh
        elif self.current_task_state == APPROACH_DROPOFF:
            close_enough = self.distance < self.drop_dist_thresh
        else:
            return

        # Only start approach if EVERYTHING is valid
        if centered and close_enough and distance_valid:
            self.get_logger().info("Alignment and distance OK. Starting approach.")
            self.start_forward_motion()

    def start_forward_motion(self):
        # Tell the rest of the system that grab_node has taken control.
        self.publish_event(EVENT_BUSY)

        if self.current_task_state == APPROACH_ITEM:
            # For item pickup, switch to a slow creep and wait for the
            # proximity sensor to tell us when the object is inside the gripper zone.
            self.sequence_active = True
            self.proximity_triggered = False
            self.grip_timeout_deadline = (
                self.get_clock().now() + Duration(seconds=float(self.grip_timeout_sec))
            )

            self.active_cmd = Twist()
            self.active_cmd.linear.x = float(self.creep_speed)
            self.active_cmd.angular.z = 0.0

            self.state = GrabState.CREEPING_TO_ITEM
            self.cmd_pub.publish(self.active_cmd)

            self.get_logger().info(
                f"Starting creep toward item: speed={self.creep_speed:.3f}, "
                f"timeout={self.grip_timeout_sec:.2f}s"
            )
            return

        # Dropoff still uses the old distance-based timed forward motion.
        if self.distance is None or self.distance <= 0.0:
            self.get_logger().warn(
                f"Invalid distance in start_forward_motion: {self.distance}"
            )
            self.state = GrabState.WAITING_ALIGNMENT
            return

        desired_final_distance = float(self.dropoff_final_distance)

        self.sequence_active = True

        remaining_distance = float(self.distance) - desired_final_distance
        remaining_distance = max(float(self.min_forward_distance), remaining_distance)
        remaining_distance = min(float(self.max_forward_distance), remaining_distance)

        self.get_logger().info(
            f"Forward check: task_state={self.current_task_state}, "
            f"measured_distance={self.distance:.3f}, "
            f"desired_final_distance={desired_final_distance:.3f}, "
            f"remaining_distance={remaining_distance:.3f}"
        )

        self.remaining_distance = remaining_distance
        duration = remaining_distance / float(self.approach_speed)

        self.get_logger().info(
            f"Starting forward motion: measured_distance={self.distance:.3f}, "
            f"target_final_distance={desired_final_distance:.3f}, "
            f"remaining_distance={remaining_distance:.3f}, duration={duration:.3f}s"
        )

        self.start_timed_motion(
            linear_x=float(self.approach_speed),
            duration=duration,
            new_state=GrabState.EXECUTING_FORWARD,
        )

    def finish_forward_motion(self):
        self.stop_motion()
        self.motion_end_time = None

        self.state = GrabState.ACTUATING
        self.perform_actuation()

    def perform_actuation(self):

        if self.waiting_for_gripper:
            self.get_logger().warn("Already waiting for gripper confirmation. Ignoring actuation request.")
            return

        if self.current_task_state == APPROACH_ITEM:

            if self.use_sim_gripper:
                self.get_logger().info(
                    f"Attempting grip now. Estimated distance to object: {self.remaining_distance}"
                )
                self.call_gripper_service(self.grab_cli)

                self.publish_event(EVENT_GRABBED)
                next_state = SEARCH_DROPOFF
                self.call_set_state(next_state)
                self.sequence_active = False
                self.state = GrabState.IDLE

            else:
                self.get_logger().info("Publishing physical gripper GRIP command.")
                self.proximity_triggered = False
                self.waiting_for_gripper = True
                self.expected_gripper_state = GRIPPER_STATE_CLOSED
                self.gripper_pub.publish(UInt8(data=GRIPPER_GRIP))

        elif self.current_task_state == APPROACH_DROPOFF:

            if self.use_sim_gripper:
                self.call_gripper_service(self.drop_cli)
                self.publish_event(EVENT_DROPPED)
                self.start_backup_motion()
                self.state = GrabState.EXECUTING_BACKUP

            else:
                self.get_logger().info("Publishing physical gripper DROP command.")
                self.waiting_for_gripper = True
                self.expected_gripper_state = GRIPPER_STATE_OPEN
                self.gripper_pub.publish(UInt8(data=GRIPPER_DROP))

    # ###Temporary
    # def perform_actuation(self):
    #     if self.current_task_state == APPROACH_ITEM:
    #         self.get_logger().info("TEST MODE: would grab now")
    #         self.state = GrabState.IDLE

    #     elif self.current_task_state == APPROACH_DROPOFF:
    #         self.get_logger().info("TEST MODE: would drop now, starting backup")
    #         self.start_backup_motion()
    # ###/Temporary

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
        duration = float(self.backup_distance) / float(self.backup_speed)

        self.get_logger().info(
            f"Starting backup motion: distance={self.backup_distance:.3f}, "
            f"speed={self.backup_speed:.3f}, duration={duration:.3f}s"
        )

        self.start_timed_motion(
            linear_x=-float(self.backup_speed),
            duration=duration,
            new_state=GrabState.EXECUTING_BACKUP,
        )

    def finish_backup_motion(self):
        self.stop_motion()
        self.motion_end_time = None

        # Recovery path after a failed grab attempt.
        if self.backup_after_failed_grab:
            self.get_logger().warn("Failed grab recovery complete. Returning to SEARCH_ITEM.")

            self.reset_creep_state()
            self.sequence_active = False
            self.state = GrabState.IDLE
            self.publish_event(EVENT_IDLE)

            next_state = SEARCH_ITEM
            self.call_set_state(next_state)
            return

        # Normal dropoff backup path.
        self.sequence_active = False
        self.state = GrabState.IDLE
        self.publish_event(EVENT_IDLE)

        next_state = SEARCH_ITEM
        self.call_set_state(next_state)

    def start_timed_motion(self, linear_x: float, duration: float, new_state: GrabState):
        if duration <= 0.0:
            self.get_logger().warn("Requested motion duration <= 0. Skipping motion.")

            if new_state == GrabState.EXECUTING_FORWARD:
                self.finish_forward_motion()
            elif new_state == GrabState.EXECUTING_BACKUP:
                self.finish_backup_motion()
            return

        self.active_cmd = Twist()
        self.active_cmd.linear.x = float(linear_x)
        self.active_cmd.angular.z = 0.0

        self.state = new_state
        now = self.get_clock().now()
        self.motion_end_time = now + Duration(seconds=float(duration))

        self.cmd_pub.publish(self.active_cmd)

    def reset_creep_state(self):
        # Clear state used only during sensor-guided item pickup.
        self.proximity_triggered = False
        self.grip_timeout_deadline = None
        self.backup_after_failed_grab = False

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

        self.sequence_active = False
        self.state = GrabState.IDLE

    # =====================================================
    # ------------------ Utilities ------------------------
    # =====================================================

    def publish_velocity(self, linear_x):
        cmd = Twist()
        cmd.linear.x = linear_x
        self.cmd_pub.publish(cmd)

    def stop_motion(self):
        self.active_cmd = Twist()
        self.cmd_pub.publish(self.active_cmd)

    def publish_event(self, code):
        self.event_pub.publish(UInt8(data=code))

    def publish_velocity(self, linear_x: float, angular_z: float = 0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)
    
    def start_cmd_stream(self):
        if self.cmd_stream_timer is None:
            period = 1.0 / max(1e-6, float(self.motion_publish_rate_hz))
            self.cmd_stream_timer = self.create_timer(period, self.cmd_stream_callback)

    def cmd_stream_callback(self):
        # Creep mode for item pickup:
        # keep moving slowly until the proximity sensor triggers
        # or until the timeout expires.
        if self.state == GrabState.CREEPING_TO_ITEM:
            self.cmd_pub.publish(self.active_cmd)

            # Object is inside the gripper zone: stop and grip.
            if self.proximity_triggered:
                self.get_logger().info("Proximity trigger active. Stopping and gripping item.")
                self.stop_motion()
                self.grip_timeout_deadline = None
                self.state = GrabState.ACTUATING
                self.perform_actuation()
                return

            # Timed out before seeing the object.
            now = self.get_clock().now()
            if self.grip_timeout_deadline is not None and now >= self.grip_timeout_deadline:
                self.get_logger().warn("Grip timeout expired. Stopping and starting recovery backup.")
                self.stop_motion()
                self.grip_timeout_deadline = None
                self.proximity_triggered = False
                self.backup_after_failed_grab = True

                duration = float(self.failed_grab_backup_distance) / float(self.backup_speed)

                self.start_timed_motion(
                    linear_x=-float(self.backup_speed),
                    duration=duration,
                    new_state=GrabState.EXECUTING_BACKUP,
                )
                return

        # Existing timed motion handling for forward/dropoff backup.
        elif self.state in [GrabState.EXECUTING_FORWARD, GrabState.EXECUTING_BACKUP]:
            self.cmd_pub.publish(self.active_cmd)

            now = self.get_clock().now()
            if self.motion_end_time is not None and now >= self.motion_end_time:
                if self.state == GrabState.EXECUTING_FORWARD:
                    self.finish_forward_motion()
                elif self.state == GrabState.EXECUTING_BACKUP:
                    self.finish_backup_motion()


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
