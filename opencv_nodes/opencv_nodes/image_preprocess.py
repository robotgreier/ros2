import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePreprocess(Node):
    def __init__(self):
        super().__init__('image_preprocess_node')
        self.declare_parameter('in_topic', '/camera/image_raw')
        self.declare_parameter('out_topic', '/camera/image_preprocessed')

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, in_topic, self.cb, 10)
        self.pub = self.create_publisher(Image, out_topic, 10)

        self.get_logger().info(f"Subscribing: {in_topic}")
        self.get_logger().info(f"Publishing:  {out_topic}")

    def cb(self, msg: Image):
        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Example OpenCV pipeline (SLAM-friendly)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Convert back to ROS Image (mono8 is often nice for feature debug)
        out = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
        out.header = msg.header
        self.pub.publish(out)

def main():
    rclpy.init()
    node = ImagePreprocess()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
