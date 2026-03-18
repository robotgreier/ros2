#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import RPi.GPIO as GPIO
import time


class HCSR04Node(Node):
    def __init__(self):
        super().__init__('distance_HCSR04_node')

        # Sett GPIO pins HER
        self.TRIGGER_PIN = 23   # BCM numbering
        self.ECHO_PIN = 24

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)

        self.publisher = self.create_publisher(Int32, '/ultrasonic/front/scan', 10)

        # Publiseringsfrekvens
        self.timer = self.create_timer(0.1, self.read_distance)  # 10 Hz

        self.get_logger().info("HC-SR04 sensor node er startet.")

    def read_distance(self):
        # Send 10 µs trigger pulse
        GPIO.output(self.TRIGGER_PIN, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIGGER_PIN, False)

        # Vent på echo start (max timeout)
        start_time = time.time()
        timeout = start_time + 0.02  # 20 ms

        while GPIO.input(self.ECHO_PIN) == 0:
            start_time = time.time()
            if start_time > timeout:
                self.get_logger().warn("Echo start timeout")
                return

        # Vent på echo slutt
        stop_time = time.time()
        timeout = stop_time + 0.02  # 20 ms

        while GPIO.input(self.ECHO_PIN) == 1:
            stop_time = time.time()
            if stop_time > timeout:
                self.get_logger().warn("Echo end timeout")
                return

        # Beregn avstand i mm
        # Lydhastighet ≈ 34300 cm/s → 343 m/s → 34300 mm/s
        elapsed = stop_time - start_time
        distance_mm = int((elapsed * 343000) / 2)

        msg = Int32()
        msg.data = distance_mm
        self.publisher.publish(msg)

        # Debug (kan slås av)
        self.get_logger().debug(f"Avstand: {distance_mm} mm")

    def destroy_node(self):
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
        rclpy.init(args=args)
        node = HCSR04Node()

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
