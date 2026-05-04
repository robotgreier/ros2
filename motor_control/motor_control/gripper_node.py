"""
gripper_node.py

ROS 2 hardware gripper node for the physical taskbot gripper.

Purpose
-------
This node provides the interface between the robot control logic and the
physical servo-driven gripper. It listens for simple binary commands from
other ROS 2 nodes and converts them into safe servo movements.

Topics
------
Subscribed:
    /gripper/command    std_msgs/msg/UInt8
        0 -> GRIPPER_DROP  -> open gripper
        1 -> GRIPPER_GRIP  -> close gripper

Published:
    /gripper/state      std_msgs/msg/UInt8
        0   -> STATE_OPEN
        1   -> STATE_MOVING
        2   -> STATE_CLOSED
        255 -> STATE_ERROR

Functionality
-------------
- Initializes the Emakefun MotorHAT and servo.
- Starts in the OPEN position as a safe default.
- Receives grip/drop commands from grab_node.
- Uses a simple internal state machine to track whether the gripper is:
    OPEN, MOVING, CLOSED, or ERROR.
- Moves the servo gradually ("soft movement") instead of jumping directly
  to the target angle.
- Waits for a short settle time after reaching the target angle before
  publishing the final confirmed state.
- Ignores repeated commands if the gripper is already in the requested state.
- Falls back to opening the gripper if:
    - an invalid command is received
    - servo control fails
    - hardware initialization fails

Current Command Encoding
------------------------
The current design uses a binary command interface:
    0 = open
    1 = close

This keeps the interface simple and matches the current hardcoded
grab/drop procedure used by grab_node.

Typical Use
-----------
1. grab_node publishes a command to /gripper/command
2. gripper_node moves the physical servo
3. gripper_node publishes state updates on /gripper/state
4. grab_node can wait for the expected final state before continuing

Safety Behavior
---------------
If anything unexpected happens, the node attempts to move the gripper
to the OPEN position and logs an error or warning message.
"""

import time
import atexit

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8

from .Emakefun_MotorHAT import Emakefun_MotorHAT


class GripperNode(Node):
    # Incoming commands on /gripper/command
    GRIPPER_DROP = 0   # open gripper
    GRIPPER_GRIP = 1   # close gripper

    # Published states on /gripper/state
    STATE_OPEN = 0
    STATE_MOVING = 1
    STATE_CLOSED = 2
    STATE_ERROR = 255

    def __init__(self):
        super().__init__('gripper_node')

        # ---- Parameters you can tune later ----
        self.servo_channel = 2
        self.open_angle = 40.0
        self.closed_angle = 100.0

        self.step_size_deg = 2.0          # degrees per step
        self.step_period_sec = 0.05       # time between steps
        self.settle_time_sec = 0.5        # wait after movement before confirming state

        # ---- Internal state ----
        self.mh = None
        self.servo = None
        self.current_angle = self.open_angle
        self.target_angle = self.open_angle
        self.target_state_after_move = self.STATE_OPEN
        self.current_state = self.STATE_OPEN
        self.waiting_for_settle = False
        self.settle_deadline_ns = 0

        # ---- ROS interfaces ----
        self.command_sub = self.create_subscription(
            UInt8,
            '/gripper/command',
            self.command_callback,
            10
        )

        self.state_pub = self.create_publisher(
            UInt8,
            '/gripper/state',
            10
        )

        # Timer used for soft movement
        self.motion_timer = self.create_timer(
            self.step_period_sec,
            self.motion_timer_callback
        )
        self.motion_timer.cancel()

        self.get_logger().info('Initializing Emakefun MotorHAT...')
        try:
            self.mh = Emakefun_MotorHAT(addr=0x60)
            self.servo = self.mh.getServo(self.servo_channel)

            atexit.register(self.cleanup)

            # Start with gripper open
            self.write_servo_angle(self.open_angle)
            self.current_angle = self.open_angle
            self.current_state = self.STATE_OPEN
            self.publish_state(self.STATE_OPEN)

            self.get_logger().info('Gripper node started successfully.')
            self.get_logger().info(
                f'Open angle: {self.open_angle} deg, '
                f'Closed angle: {self.closed_angle} deg'
            )

        except Exception as e:
            self.get_logger().error(f'Failed to initialize gripper hardware: {e}')
            self.current_state = self.STATE_ERROR
            self.publish_state(self.STATE_ERROR)

    def command_callback(self, msg: UInt8):
        command = int(msg.data)
        self.get_logger().info(f'Received gripper command: {command}')

        if self.servo is None:
            self.get_logger().error('Servo not available. Opening gripper as safe fallback.')
            self.safe_open_with_error()
            return

        if command == self.GRIPPER_GRIP:
            # Ignore if already closed or already moving to closed
            if self.current_state == self.STATE_CLOSED:
                self.get_logger().info('Gripper already closed. Ignoring command.')
                return

            if self.current_state == self.STATE_MOVING and self.target_state_after_move == self.STATE_CLOSED:
                self.get_logger().info('Gripper already moving to closed position. Ignoring command.')
                return

            self.start_motion(
                target_angle=self.closed_angle,
                final_state=self.STATE_CLOSED
            )

        elif command == self.GRIPPER_DROP:
            # Ignore if already open or already moving to open
            if self.current_state == self.STATE_OPEN:
                self.get_logger().info('Gripper already open. Ignoring command.')
                return

            if self.current_state == self.STATE_MOVING and self.target_state_after_move == self.STATE_OPEN:
                self.get_logger().info('Gripper already moving to open position. Ignoring command.')
                return

            self.start_motion(
                target_angle=self.open_angle,
                final_state=self.STATE_OPEN
            )

        else:
            self.get_logger().error(f'Invalid gripper command: {command}. Opening gripper for safety.')
            self.safe_open_with_error()

    def start_motion(self, target_angle: float, final_state: int):
        self.target_angle = float(target_angle)
        self.target_state_after_move = final_state
        self.current_state = self.STATE_MOVING
        self.waiting_for_settle = False

        self.publish_state(self.STATE_MOVING)
        self.motion_timer.reset()

        self.get_logger().info(
            f'Starting gripper motion: current={self.current_angle:.1f} deg, '
            f'target={self.target_angle:.1f} deg'
        )

    def motion_timer_callback(self):
        if self.servo is None:
            self.get_logger().error('Servo missing during motion. Entering safe mode.')
            self.safe_open_with_error()
            return

        try:
            # If movement finished, wait for settle time before publishing final state
            if self.waiting_for_settle:
                now_ns = self.get_clock().now().nanoseconds
                if now_ns >= self.settle_deadline_ns:
                    self.waiting_for_settle = False
                    self.current_state = self.target_state_after_move
                    self.publish_state(self.current_state)
                    self.motion_timer.cancel()

                    if self.current_state == self.STATE_OPEN:
                        self.get_logger().info('Gripper action complete: OPEN confirmed.')
                    elif self.current_state == self.STATE_CLOSED:
                        self.get_logger().info('Gripper action complete: CLOSED confirmed.')
                return

            difference = self.target_angle - self.current_angle

            # Reached target
            if abs(difference) <= self.step_size_deg:
                self.current_angle = self.target_angle
                self.write_servo_angle(self.current_angle)

                self.waiting_for_settle = True
                self.settle_deadline_ns = (
                    self.get_clock().now().nanoseconds
                    + int(self.settle_time_sec * 1e9)
                )

                self.get_logger().info(
                    f'Gripper reached target angle {self.current_angle:.1f} deg. '
                    f'Waiting {self.settle_time_sec:.2f}s before confirming state.'
                )
                return

            # Move one step toward target
            step = self.step_size_deg if difference > 0 else -self.step_size_deg
            self.current_angle += step
            self.write_servo_angle(self.current_angle)

        except Exception as e:
            self.get_logger().error(f'Servo movement failed: {e}')
            self.safe_open_with_error()

    def write_servo_angle(self, angle: float):
        # Clamp angle just in case
        clamped_angle = max(0.0, min(180.0, angle))
        self.servo.writeServo(clamped_angle)
        self.get_logger().debug(f'Servo set to {clamped_angle:.1f} deg')

    def publish_state(self, state_value: int):
        msg = UInt8()
        msg.data = int(state_value)
        self.state_pub.publish(msg)

    def safe_open_with_error(self):
        self.current_state = self.STATE_ERROR
        self.publish_state(self.STATE_ERROR)

        # Try to open the gripper as a safe fallback
        if self.servo is not None:
            try:
                self.motion_timer.cancel()
                self.write_servo_angle(self.open_angle)
                self.current_angle = self.open_angle
                self.target_angle = self.open_angle
                self.waiting_for_settle = False

                self.get_logger().warn('Safe fallback executed: gripper opened.')
            except Exception as e:
                self.get_logger().error(f'Failed to open gripper during safe fallback: {e}')

    def cleanup(self):
        self.get_logger().info('Cleaning up gripper node...')
        try:
            if self.mh is not None:
                self.mh.close()
                time.sleep(0.1)
        except Exception as e:
            self.get_logger().warn(f'Cleanup warning: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()