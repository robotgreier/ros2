import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from .ina219 import INA219

class FPGAPowerNode(Node):
    def __init__(self):
        super().__init__('fpga_power_node')

        self.ina = INA219(address=0x44)
        self.pub = self.create_publisher(Vector3, 'fpga/power', 10)
        self.timer = self.create_timer(0.5, self.read_power)

        self.get_logger().info("FPGA Power Node started on I2C address 0x45")

    def read_power(self):
        msg = Vector3()
        v = float(self.ina.read_voltage())
        c = float(self.ina.read_current())
        p = float(self.ina.read_power())
        msg.x, msg.y, msg.z = v, c, p
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = FPGAPowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()