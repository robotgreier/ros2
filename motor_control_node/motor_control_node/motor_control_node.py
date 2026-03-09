#!/usr/bin/env python3
import math
import time
from typing import Optional 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# Try Adafruit MotorKit first (Motor-HAT compatible at I2C 0x60)
_ADAFRUIT_OK = True
try:
    from adafruit_motorkit import MotorKit
    MOTORHAT_AVAILABLE = True
except Exception:
    MOTORHAT_AVAILABLE = False


class MotorHAL:
    """
    Hardware HAL (for RPi5 + DRI0054).
    Only used when MotorKit is available.
    """
    def __init__(self, node, i2c_address, left_id, right_id):
        if not MOTORHAT_AVAILABLE:
            raise RuntimeError("MotorKit not available")
        self.node = node
        self.kit = MotorKit(address=i2c_address)

        self.left = getattr(self.kit, f"motor{left_id}")
        self.right = getattr(self.kit, f"motor{right_id}")

        node.get_logger().info(
            f"MotorKit active at I2C 0x{i2c_address:02X} (left=M{left_id}, right=M{right_id})"
        )

    def set(self, left, right):
        self.left.throttle = left
        self.right.throttle = right

    def stop_brake(self):
        self.left.throttle = 0.0
        self.right.throttle = 0.0

    def stop_coast(self):
        self.left.throttle = None
        self.right.throttle = None


class DummyMotorHAL:
    """
    Simulation HAL for laptops/WSL2/no hardware.
    Prints commands instead of using I2C.
    """
    def __init__(self, node, i2c_address, left_id, right_id):
        node.get_logger().warn(
            "MotorKit NOT available — running in SIMULATION MODE (no I2C)."
        )
        self.node = node

    def set(self, left, right):
        self.node.get_logger().info(
            f"[SIM] Motors => L:{left:.2f}  R:{right:.2f}"
        )

    def stop_brake(self):
        self.node.get_logger().info("[SIM] STOP brake")

    def stop_coast(self):
        self.node.get_logger().info("[SIM] STOP coast")


class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control_node')

        # Parameters (declare with defaults; override in launch)
        self.declare_parameter('wheel_separation', 0.30)       # meters
        self.declare_parameter('wheel_radius', 0.05)           # meters
        self.declare_parameter('max_wheel_linear_speed', 0.8)  # m/s per wheel
        self.declare_parameter('i2c_address', 0x60)            # DRI0054 default
        self.declare_parameter('left_motor_id', 1)             # 1..4
        self.declare_parameter('right_motor_id', 2)            # 1..4
        self.declare_parameter('invert_left', False)
        self.declare_parameter('invert_right', False)
        self.declare_parameter('cmd_vel_timeout', 0.5)         # seconds
        self.declare_parameter('slew_rate', 5.0)               # throttle units/sec
        self.declare_parameter('stop_mode', 'brake')           # 'brake' or 'coast'

        self.L = float(self.get_parameter('wheel_separation').value)
        self.r = float(self.get_parameter('wheel_radius').value)
        self.vmax = float(self.get_parameter('max_wheel_linear_speed').value)
        self.i2c_addr = int(self.get_parameter('i2c_address').value)
        self.left_id = int(self.get_parameter('left_motor_id').value)
        self.right_id = int(self.get_parameter('right_motor_id').value)
        self.invL = bool(self.get_parameter('invert_left').value)
        self.invR = bool(self.get_parameter('invert_right').value)
        self.timeout = float(self.get_parameter('cmd_vel_timeout').value)
        self.slew_rate = float(self.get_parameter('slew_rate').value)
        self.stop_mode = str(self.get_parameter('stop_mode').value).lower()

        # Motors
        if MOTORHAT_AVAILABLE:
            self.hal = MotorHAL(self, self.i2c_addr, self.left_id, self.right_id)
        else:
            self.hal = DummyMotorHAL(self, self.i2c_addr, self.left_id, self.right_id)


        # Subscribers
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.sub_estop = self.create_subscription(Bool, '/e_stop', self.on_e_stop, 10)

        # Watchdog
        self.last_cmd_time = time.monotonic()
        self.estop = False
        self.curr_left = 0.0
        self.curr_right = 0.0
        self.timer = self.create_timer(0.02, self.watchdog_loop)   # 50 Hz update

        self.get_logger().info("motor_control_node up — waiting for /cmd_vel")

    # ---- Helpers -------------------------------------------------------------

    def clip(self, x, x_min, x_max):
        return min(max(x, x_min), x_max)

    def sign_invert(self, x, invert):
        return -x if invert else x

    def vel_to_throttle(self, v_lin: float) -> float:
        """
        Convert wheel linear speed [m/s] to MotorKit throttle [-1..1].
        """
        if self.vmax <= 0.0:
            return 0.0
        return self.clip(v_lin / self.vmax, -1.0, 1.0)

    def slew(self, current: float, target: float, dt: float) -> float:
        """
        Limit rate of change of throttle for smoother driving and lower spikes.
        """
        if self.slew_rate <= 0:
            return target
        max_step = self.slew_rate * dt
        if target > current:
            return min(target, current + max_step)
        else:
            return max(target, current - max_step)

    # ---- Callbacks -----------------------------------------------------------

    def on_e_stop(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self.stop_motors()

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd_time = time.monotonic()
        v = float(msg.linear.x)
        w = float(msg.angular.z)

        # Differential drive mapping to wheel linear speeds
        v_left  = v - w * (self.L / 2.0)
        v_right = v + w * (self.L / 2.0)

        # Convert to throttle [-1..1]
        t_left  = self.vel_to_throttle(v_left)
        t_right = self.vel_to_throttle(v_right)

        # Apply per-side inversion
        t_left  = self.sign_invert(t_left,  self.invL)
        t_right = self.sign_invert(t_right, self.invR)

        # Save targets; actual application is in watchdog_loop() to apply slew & safety
        self.target_left = t_left
        self.target_right = t_right

    # ---- Control loop & safety ----------------------------------------------

    target_left: float = 0.0
    target_right: float = 0.0

    def stop_motors(self):
        if self.stop_mode == 'coast':
            self.hal.stop_coast()
        else:
            self.hal.stop_brake()
        self.curr_left = 0.0
        self.curr_right = 0.0

    def watchdog_loop(self):
        now = time.monotonic()
        dt = 0.02  # timer period

        # Dead-man timeout or e-stop
        if self.estop or (now - self.last_cmd_time) > self.timeout:
            self.stop_motors()
            return

        # Slew-limit towards targets
        self.curr_left = self.slew(self.curr_left, self.target_left, dt)
        self.curr_right = self.slew(self.curr_right, self.target_right, dt)
        self.hal.set(self.curr_left, self.curr_right)


def main():
    rclpy.init()
    node = MotorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()
