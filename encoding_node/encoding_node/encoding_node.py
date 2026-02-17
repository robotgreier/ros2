import rclpy
from rclpy.node import Node
from fpga_msgs.msg import SpikingGrid   # custom message type for spiking grid data

class EncodingNode(Node):
    def __init__(self):
        super().__init__('encoding_node')

        self.subscription = self.create_subscription(
            SpikingGrid,
            '/keypoint_grid',   # ropic to subscribe to for keypoint grid data
            self.grid_callback, 
            10
        )
        self.subscription   # prevents unused variable warning

        self.get_logger().info("Encoding node subscribed to /keypoint_grid")

    def grid_callback(self, msg: SpikingGrid):
        self.get_logger().info(
            f"Got grid {msg.grid_width}x{msg.grid_height} with {len(msg.spikes)} spike entries"
        ) 
        
        '''
        TODO: add encoding logic to convert the keypoint_grid data into a suitable format for the topic /input_snn.
        '''

def main(args=None):
    rclpy.init(args=args)
    node = EncodingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
