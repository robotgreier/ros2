#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import gpiod
import time
from collections import deque
import statistics



class DistanceSensorNode(Node):

    def __init__(self):
        super().__init__("distance_sensor_node")

        # GPIO via libgpiod
        chip = gpiod.Chip("gpiochip0")

        # Set GPIO pins på Raspberry Pi 
        self.TRIG = 24
        self.ECHO = 23

        self.trig_line = chip.get_line(self.TRIG)
        self.echo_line = chip.get_line(self.ECHO)

        self.trig_line.request(consumer="trig", type=gpiod.LINE_REQ_DIR_OUT)
        self.echo_line.request(consumer="echo", type=gpiod.LINE_REQ_DIR_IN)

        # Filters
        self.median_window = deque(maxlen=3) # Used in median filter
        self.filtered_value = None

        self.publisher = self.create_publisher(Range, "/ultrasonic/front/raw_range", 10)
        self.timer = self.create_timer(0.20, self.measure)   # 5 Hz

        self.get_logger().info("HC-SR04 libgpiod-node startet.")

    def measure(self):
        # Send 10 us puls
        self.trig_line.set_value(0)
        time.sleep(0.000002) # 2 µs for stabilizing
        self.trig_line.set_value(1)
        time.sleep(0.000015) # 15 µs for triggering
        self.trig_line.set_value(0)

        # Vent på ECHO = HIGH
        start_wait = time.perf_counter_ns()
  
        while self.echo_line.get_value() == 0:
            if time.perf_counter_ns() - start_wait > 60000000:
                return

        echo_start = time.perf_counter_ns()

        # Vent på ECHO = LOW
        while self.echo_line.get_value() == 1:
            if time.perf_counter_ns() - echo_start > 60000000:
                return

        echo_end = time.perf_counter_ns()

        # Beregn avstand (mm)
        duration_ns = echo_end - echo_start  # ns
        duration_s = duration_ns / 1e9 # s
        distance_mm = (duration_s * 343000) / 2 # 343m/s: speed of sound in dry air at 20 degrees Celsius, divided by 2 for back/forth
        distance_m = distance_mm / 1000.0 

        # Validation
        if distance_m <= 0.0 or distance_m > 4.0:
            return
        
        # Median filter
        self.median_window.append(distance_m)
        median_val = statistics.median(self.median_window)

        # Exponential smooth (IIR filter)
        alpha = 0.4
        if self.filtered_value is None:
            self.filtered_value = median_val
        else: 
            self.filtered_value = alpha * median_val + (1.0 -alpha) * self.filtered_value

        # Publiser Range
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ultrasonic_front"
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.5 # 0.5 radians = 28.6 degrees
        msg.min_range = 0.02
        msg.max_range = 4.0
        msg.range = float(self.filtered_value)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DistanceSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()