import rclpy
from rclpy.node import Node
from power_monitor.msg import PowerReading
from .ina219 import INA219

class FPGAPowerNode(Node):
    def __init__(self):
        super().__init__('fpga_power_node')

        self.ina = INA219(address=0x45)
        self.pub = self.create_publisher(PowerReading, 'fpga/power', 10)
        self.timer = self.create_timer(0.5, self.read_power)

        self.get_logger().info("FPGA Power Node started on I2C address 0x45")

    def read_power(self):
        msg = PowerReading()
        msg.voltage = float(self.ina.read_voltage())
        msg.current = float(self.ina.read_current())
        msg.power   = float(self.ina.read_power())
        self.pub.publish(msg)

def main():
        rclpy.init()
        node = FPGAPowerNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
