#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import atexit

from .Emakefun_MotorHAT import Emakefun_MotorHAT

class ServoTestNode(Node):
    def __init__(self):
        super().__init__('servo_test')

        self.get_logger().info("Initialiserer Emakefun MotorHAT...")
        self.mh = Emakefun_MotorHAT(addr=0x60)
        self.servo = self.mh.getServo(1)

        atexit.register(self.cleanup)

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.position = 0

        self.get_logger().info("ServoTestNode startet.")

    def timer_callback(self):
        """Kjøres hver 1. sekund"""
        if self.position == 0:
            angle = 0
            self.position = 1
        else:
            angle = 45
            self.position = 0

        self.get_logger().info(f"Setter servo til {angle} grader...")
        self.servo.writeServo(angle)

    def cleanup(self):
        self.get_logger().info("Stopper MotorHAT og skrur av motorer...")
        self.mh.close()
        time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = ServoTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
