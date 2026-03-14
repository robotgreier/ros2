#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import atexit

# Lokale Emakefun-filer (samme mappe)
from .Emakefun_MotorHAT import Emakefun_MotorHAT, Emakefun_DCMotor, Emakefun_Servo
from .Emakefun_MotorDriver import PWM
from .Emakefun_I2C import Emakefun_I2C

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control_node')

        # --- ROS-parameteroppsett ---
        self.declare_parameter('wheel_base', 0.20)       # meter
        self.declare_parameter('max_lin_vel', 0.6)       # m/s → PWM 255
        self.declare_parameter('max_ang_vel', 2.0)       # rad/s
        self.declare_parameter('cmd_vel_timeout', 0.5)   # sekunder

        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)
        self.timeout = float(self.get_parameter('cmd_vel_timeout').value)

        # --- Emakefun Motorhat (I2C-adresse 0x60 iht. DRI0054-dokumentasjon) ---
        self.get_logger().info("Initialiserer Emakefun_MotorHAT på I2C 0x60...")
        self.mh = Emakefun_MotorHAT(addr=0x60)

        # Vi bruker M1 og M2
        self.m_left  = self.mh.getMotor(1)
        self.m_right = self.mh.getMotor(2)

        # Slå av motorer ved avslutning
        atexit.register(self.turn_off_motors)

        # ROS2-kommunikasjon
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
    # cmd_vel mottatt
    # -----------------------
    def cmd_vel_callback(self, msg):
        self.last_cmd_time = time.time()

        v = msg.linear.x
        w = msg.angular.z

        # Begrens til maksverdier
        v = max(-self.max_lin_vel, min(self.max_lin_vel, v))
        w = max(-self.max_ang_vel, min(self.max_ang_vel, w))

        # Differensialdrift
        v_l = v - (w * self.wheel_base / 2.0)
        v_r = v + (w * self.wheel_base / 2.0)

        # Konverter til PWM
        pwm_l = self.velocity_to_pwm(v_l)
        pwm_r = self.velocity_to_pwm(v_r)

        # Påfør
        self.apply_pwm(self.m_left, pwm_l)
        self.apply_pwm(self.m_right, pwm_r)

    # -----------------------
    # Failsafe timeout
    # -----------------------
    def timer_callback(self):
        if time.time() - self.last_cmd_time > self.timeout:
            self.apply_pwm(self.m_left, 0)
            self.apply_pwm(self.m_right, 0)

    # -----------------------
    # Konverter hjulhastighet til PWM [-255, 255]
    # -----------------------
    def velocity_to_pwm(self, v_wheel):
        if self.max_lin_vel <= 0:
            return 0

        pwm = int(255 * (v_wheel / self.max_lin_vel))
        pwm = max(-255, min(255, pwm))
        return pwm

    # -----------------------
    # Påfør PWM til motor
    # -----------------------
    def apply_pwm(self, motor, pwm_value):
        if pwm_value == 0:
            motor.run(Emakefun_MotorHAT.RELEASE)
            motor.setSpeed(0)
            return

        speed = abs(pwm_value)
        if speed > 255:
            speed = 255

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
