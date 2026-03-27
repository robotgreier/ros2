import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from .ina219 import INA219

class SystemPowerNode(Node):
    def __init__(self):
        super().__init__('system_power_node')

        self.ina = INA219(address=0x44)
        self.pub = self.create_publisher(Vector3, 'system/power', 10)
        self.timer = self.create_timer(0.5, self.read_power)

        self.get_logger().info("System Power Node started on I2C address 0x44")

    def read_power(self):
        msg = Vector3()
        v = float(self.ina.read_voltage())
        c = float(self.ina.read_current())
        p = float(self.ina.read_power())
        msg.x, msg.y, msg.z = v, c, p
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = SystemPowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()