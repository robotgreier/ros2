#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import gpiod
import time



class DistanceSensorNode(Node):

    def __init__(self):
        super().__init__("distance_sensor_node")

        # GPIO via libgpiod
        chip = gpiod.Chip("gpiochip0")

        # Set GPIO pins på Raspberry Pi 
        self.TRIG = 23
        self.ECHO = 24

        self.trig_line = chip.get_line(self.TRIG)
        self.echo_line = chip.get_line(self.ECHO)

        self.trig_line.request(consumer="trig", type=gpiod.LINE_REQ_DIR_OUT)
        self.echo_line.request(consumer="echo", type=gpiod.LINE_REQ_DIR_IN)

        self.publisher = self.create_publisher(Range, "/ultrasonic/front/raw", 10)
        self.timer = self.create_timer(0.20, self.measure)   # 5 Hz

        self.get_logger().info("HC-SR04 libgpiod-node startet.")

    def measure(self):
        # Send 10 us puls
        self.trig_line.set_value(1)
        time.sleep(10e-6)
        self.trig_line.set_value(0)

        # Vent på ECHO = HIGH
        start = time.time()
        timeout = start + 0.03  # 30 ms

        while self.echo_line.get_value() == 0:
            if time.time() > timeout:
                return

        echo_start = time.time()

        # Vent på ECHO = LOW
        while self.echo_line.get_value() == 1:
            if time.time() > timeout:
                return

        echo_end = time.time()

        # Beregn avstand (mm)
        duration = echo_end - echo_start  # sekunder
        distance = (duration * 343000) / 2  # mm

        # Publiser Range
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ultrasonic_front"
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.4
        msg.min_range = 0.02
        msg.max_range = 4.0
        msg.range = distance / 1000.0   # meter

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DistanceSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()