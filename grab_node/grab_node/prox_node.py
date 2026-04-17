#!/usr/bin/env python3

import time
import statistics
from collections import deque
from smbus2 import SMBus
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool


# APDS-9930 Registers

APDS_ADDR   = 0x39
ENABLE      = 0x80
ENABLE_PON  = 0x01
ENABLE_PEN  = 0x04
PPCOUNT     = 0x8E
PPULSE_8US  = 0x20
PDATA       = 0x9C  # Proximity data (2 bytes)
CONTROL     = 0x8F


# PPCOUNT configuration:
# Upper 2 bits: pulse length
# Lower 6 bits: pulse count
PPULSE_LEN_8US = 0x00          # 8 µs pulse
PPLUSES_32     = 32            # 32 pulses
PPULSE_8US_32  = PPULSE_LEN_8US | PPLUSES_32

# CONTROL register bits:
# Bits [7:6]: LED drive current
# Bits [5:4]: Proximity gain
CONTROL_LED_50MA_PGAIN_1X = 0x20   # 



# Sensor Driver 

class APDS9930:
    def __init__(self, bus_id=1):
        self.bus = SMBus(bus_id)

        # Power on
        self.bus.write_byte_data(APDS_ADDR, ENABLE, ENABLE_PON)
        time.sleep(0.01)

        # Enable proximity
        self.bus.write_byte_data(APDS_ADDR, ENABLE, ENABLE_PON | ENABLE_PEN)

        # 8us proximity pulse count (tune if needed, lower value gives faster readings but less sensitivity)
        self.bus.write_byte_data(APDS_ADDR, PPCOUNT, PPULSE_8US_32)

        # Set control register (LED drive current and proximity gain)
        self.bus.write_byte_data(APDS_ADDR, CONTROL, CONTROL_LED_50MA_PGAIN_1X)

    def read_proximity(self) -> int:
        low = self.bus.read_byte_data(APDS_ADDR, PDATA)
        high = self.bus.read_byte_data(APDS_ADDR, PDATA + 1)
        return (high << 8) | low


# Node implementation

class ProxNode(Node):

    def __init__(self):
        super().__init__("prox_node")

        # Parameters 
        self.declare_parameter("sample_rate_hz", 100.0)
        self.declare_parameter("median_window", 5)
        self.declare_parameter("trigger_threshold", 300.0)  # RAW proximity value
        self.declare_parameter("trigger_hold_ms", 40.0)

        self.sample_rate = float(self.get_parameter("sample_rate_hz").value)
        self.window_size = int(self.get_parameter("median_window").value)
        self.threshold = float(self.get_parameter("trigger_threshold").value)
        self.hold_time = float(self.get_parameter("trigger_hold_ms").value) / 1000.0

        # Sensor 
        self.sensor = APDS9930()

        # Filtering to stabilize readings
        self.samples = deque(maxlen=self.window_size)

        self.trigger_start_time = None
        self.trigger_active = False

        # Publish stabilized value
        self.raw_pub = self.create_publisher(
            Float32, "/gripper/proximity_raw", 10
        )

        self.trigger_pub = self.create_publisher(
            Bool, "/gripper/proximity_trigger", 10
        )

        # Timer to poll sensor at specified rate
        period = 1.0 / max(1e-6, self.sample_rate)
        self.timer = self.create_timer(period, self.poll_sensor)

        self.get_logger().info(
            f"prox_node started | "
            f"median={self.window_size}, "
            f"threshold={self.threshold}, "
            f"hold={self.hold_time*1000:.0f} ms"
        )

    # Read sensor, apply median filter, handle trigger logic with delay and publish results

    def poll_sensor(self):
        raw = self.sensor.read_proximity()
        self.samples.append(raw)

        if len(self.samples) < self.window_size:
            return  # wait until buffer full

        median_value = statistics.median(self.samples)

        # Publish median raw value
        self.raw_pub.publish(
            Float32(data=float(median_value))
        )

        now = time.monotonic()

        # Trigger logic with delay time

        if median_value >= self.threshold:
            if self.trigger_start_time is None:
                self.trigger_start_time = now
            elif (now - self.trigger_start_time) >= self.hold_time:
                if not self.trigger_active:
                    self.trigger_active = True
                    self.trigger_pub.publish(Bool(data=True))
        else:
            self.trigger_start_time = None
            if self.trigger_active:
                self.trigger_active = False
                self.trigger_pub.publish(Bool(data=False))


# main

def main(args=None):
    rclpy.init(args=args)
    node = ProxNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()