#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import atexit

# Emakefun MotorHAT code (same folder)
from .Emakefun_MotorHAT import Emakefun_MotorHAT, Emakefun_DCMotor, Emakefun_Servo
from .Emakefun_MotorDriver import PWM
from .Emakefun_I2C import Emakefun_I2C

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control_node')

        # --- ROS-parameteroppsett ---
        self.declare_parameter('wheel_base', 0.15)      # meter
        self.declare_parameter('max_lin_vel', 0.1)     # m/s
        self.declare_parameter('max_ang_vel', 0.1)      # rad/s
        self.declare_parameter('max_pwm', 100)          # pwm upper limit
        self.declare_parameter('min_pwm', 30)           # pwm lower limit (smooth start)
        self.declare_parameter('cmd_vel_timeout', 0.5)  # seconds

        self.declare_parameter('vel_smooth_alpha', 0.80) # pwm smoothing
        self.declare_parameter('idle_decay', 0.95)      # pwm smoothing

        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)
        self.max_pwm = int(self.get_parameter('max_pwm').value)
        self.min_pwm = int(self.get_parameter('min_pwm').value)
        self.timeout = float(self.get_parameter('cmd_vel_timeout').value)
        
        self.alpha = float(self.get_parameter('vel_smooth_alpha').value)    # pwm smoothing
        self.idle_decay = float(self.get_parameter('idle_decay').value)      # pwm smoothing
        
        self.pwm_l_prev = 0.0   # pwm smoothing
        self.pwm_r_prev = 0.0   # pwm smoothing

        # Emakefun Motorhat (I2C-address 0x60 - DRI0054-documentation)
        self.get_logger().info("Initialiserer Emakefun_MotorHAT på I2C 0x60...")
        self.mh = Emakefun_MotorHAT(addr=0x60)

        # DC Motor connection M1 og M2
        self.m_left  = self.mh.getMotor(1)
        self.m_right = self.mh.getMotor(2)
        
        self.left_dir = 1
        self.right_dir = -1   # invert right motor


        # Turn off motors on shutdown
        atexit.register(self.turn_off_motors)

        # ROS2 subscription and timer
        self.last_cmd_time = time.time()
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_timer(0.02, self.timer_callback)  # 50 Hz watchdog

        self.get_logger().info("motor_control_node er klar.")

    # -----------------------
    # Motor shutdown
    # -----------------------
    def turn_off_motors(self):
        try:
            self.m_left.run(Emakefun_MotorHAT.RELEASE)
            self.m_right.run(Emakefun_MotorHAT.RELEASE)
        except Exception:
            pass

    # -----------------------
    # cmd_vel callback
    # -----------------------
    def cmd_vel_callback(self, msg):
        self.last_cmd_time = time.time()

        v = msg.linear.x        # m/s, cmd_vel message: 0.1
        w = msg.angular.z       # rad/s, cmd_vel message: 0.1

        # Limit velocities to max values
        v = max(-self.max_lin_vel, min(self.max_lin_vel, v))
        w = max(-self.max_ang_vel, min(self.max_ang_vel, w))

        # Normalize velocities to [-1, 1]
        v_norm = v / self.max_lin_vel if self.max_lin_vel > 0 else 0.0
        w_norm = w / self.max_ang_vel if self.max_ang_vel > 0 else 0.0

        # Differential drive mixing for left and right motors
        pwm_l_norm = v_norm - w_norm
        pwm_r_norm = v_norm + w_norm

        # Scale PWM values to fit within [-255, 255] while preserving the ratio
        scale = max(1.0, abs(pwm_l_norm), abs(pwm_r_norm))
        pwm_l_target = self.max_pwm * pwm_l_norm / scale
        pwm_r_target = self.max_pwm * pwm_r_norm / scale

        
        # # pwm smoothing
        idle = pwm_l_target == 0 and pwm_r_target == 0.0)

        if idle:
            pwm_l = self.pwm_l_prev * self.idle_decay
            pwm_r = self.pwm_r_prev * self.idle_decay
        else:
            pwm_l = self.alpha * self.pwm_l_prev + (1.0 - self.alpha) * pwm_l_target
            pwm_r = self.alpha * self.pwm_r_prev + (1.0 - self.alpha) * pwm_r_target

        self.pwm_l_prev = pwm_l
        self.pwm_r_prev = pwm_r

        pwm_l_i = int(pwm_l)
        pwm_r_i = int(pwm_r)


        # Apply PWM to motors
        self.apply_pwm(self.m_left,  self.left_dir  * pwm_l_i)
        self.apply_pwm(self.m_right, self.right_dir * pwm_r_i)

    # -----------------------
    # Failsafe timeout
    # -----------------------
    def timer_callback(self):
        if time.time() - self.last_cmd_time > self.timeout:
            self.apply_pwm(self.m_left, 0)
            self.apply_pwm(self.m_right, 0)

    # -----------------------
    # Apply PWM to motor with direction and speed control
    # -----------------------
    def apply_pwm(self, motor, pwm_value):
        if pwm_value == 0:
            motor.run(Emakefun_MotorHAT.RELEASE)
            motor.setSpeed(0)
            return

        speed = abs(pwm_value)

        if speed < self.min_pwm:
            speed = self.min_pwm

        if speed > self.max_pwm:
            speed = self.max_pwm

        if pwm_value > 0:
            motor.run(Emakefun_MotorHAT.FORWARD)
        else:
            motor.run(Emakefun_MotorHAT.BACKWARD)

        motor.setSpeed(speed)


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.turn_off_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
