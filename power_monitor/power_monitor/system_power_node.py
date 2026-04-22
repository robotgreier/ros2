import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from collections import deque
import statistics

from .ina219 import INA219


class SystemPowerNode(Node):
    def __init__(self):
        super().__init__('system_power_node')

        self.ina = INA219(address=0x45)
        self.pub = self.create_publisher(Vector3, 'system/power', 10)
        self.timer = self.create_timer(0.5, self.read_power)

        # --- filtering parameters ---
        self.win_size = 20
        self.alpha = 0.2

        self.Vw = deque(maxlen=self.win_size)
        self.Iw = deque(maxlen=self.win_size)
        self.Pw = deque(maxlen=self.win_size)

        self.Vs = None
        self.Is = None
        self.Ps = None

        self.get_logger().info("System Power Node started on I2C address 0x45")

    # ------------- filtering -------------

    def smooth(self, window, new_value, prev_value):
        window.append(new_value)
        med = statistics.median(window)

        if prev_value is None:
            return med

        return self.alpha * med + (1.0 - self.alpha) * prev_value

    # ------------- main loop -------------

    def read_power(self):
        # raw INA readings
        v_raw = float(self.ina.read_voltage())
        i_raw = float(self.ina.read_current())
        p_raw = v_raw * i_raw   # recompute to keep consistency

        # filtered values
        self.Vs = self.smooth(self.Vw, v_raw, self.Vs)
        self.Is = self.smooth(self.Iw, i_raw, self.Is)
        self.Ps = self.smooth(self.Pw, p_raw, self.Ps)

        msg = Vector3()
        msg.x = self.Vs   # volts
        msg.y = self.Is   # amps
        msg.z = self.Ps   # watts

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SystemPowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
