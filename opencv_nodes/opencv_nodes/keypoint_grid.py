import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class KeypointGrid(Node):
    def __init__(self):
        super().__init__('keypoint_grid_node')

        # Adjustable parameters (grid size can change later)
        self.declare_parameter('in_topic', '/camera/image_preprocessed')
        self.declare_parameter('out_topic', '/features/keypoints_grid')
        self.declare_parameter('rows', 7)
        self.declare_parameter('cols', 7)

        # Keypoint detector settings
        self.declare_parameter('detector', 'ORB')   # 'ORB' or 'FAST'
        self.declare_parameter('max_features', 1500) # ORB only
        self.declare_parameter('fast_threshold', 5) # FAST only
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_topic', '/debug/keypoint_grid')
        self.declare_parameter('debug_width', 320)
        self.declare_parameter('debug_height', 240)

        self.bridge = CvBridge()

        self.in_topic = self.get_parameter('in_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.rows = int(self.get_parameter('rows').value)
        self.cols = int(self.get_parameter('cols').value)

        detector = self.get_parameter('detector').value.upper()
        self.detector_name = detector

        self.declare_parameter('publish_debug_hz', 2.0)
        self.declare_parameter('debug_resize_width', 640)

        self.debug_period = 1.0 / float(self.get_parameter('publish_debug_hz').value)
        self.debug_resize_width = int(self.get_parameter('debug_resize_width').value)
        self._last_debug_time = 0.0

        self.declare_parameter('response_topic', '/features/keypoints_response')
        self.response_topic = self.get_parameter('response_topic').value
        self.response_pub = self.create_publisher(Float32MultiArray, self.response_topic, 10)


        if detector == 'FAST':
            thr = int(self.get_parameter('fast_threshold').value)
            self.detector = cv2.FastFeatureDetector_create(thr, nonmaxSuppression=True)
        else:
            n = int(self.get_parameter('max_features').value)
            self.detector = cv2.ORB_create(nfeatures=n)

        self.pub = self.create_publisher(Int32MultiArray, self.out_topic, 10)
        self.sub = self.create_subscription(Image, self.in_topic, self.cb, 10)

        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)
        self.debug_width = int(self.get_parameter('debug_width').value)
        self.debug_height = int(self.get_parameter('debug_height').value)

        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image,
                self.get_parameter('debug_topic').value,
                10
            )
        else:
            self.debug_pub = None


        self.get_logger().info(
            f"KeypointGrid: {self.detector_name} | in={self.in_topic} out={self.out_topic} grid={self.rows}x{self.cols}"
        )

    def cb(self, msg: Image):
        # Convert ROS -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = frame.shape[:2]

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = clahe.apply(frame)


        # Detect keypoints
        if self.detector_name == 'FAST':
            kps = self.detector.detect(frame, None)
        else:
            kps = self.detector.detect(frame, None)
        
        # Count keypoints/bin and collect response/bin
        counts = np.zeros((self.rows, self.cols), dtype=np.int32)
        resp_sum = np.zeros((self.rows, self.cols), dtype=np.float32)

        cell_w = w / self.cols
        cell_h = h / self.rows

        for kp in kps:
            x, y = kp.pt
            c = int(x / cell_w)
            r = int(y / cell_h)

            if c < 0: c = 0
            if c >= self.cols: c = self.cols - 1
            if r < 0: r = 0
            if r >= self.rows: r = self.rows - 1

            counts[r, c] += 1
            resp_sum[r, c] += float(kp.response)

        # Calculate the mean response
        mean_resp = np.zeros((self.rows, self.cols), dtype=np.float32)
        mask = counts > 0
        mean_resp[mask] = resp_sum[mask] / counts[mask]


        # Publish Int32MultiArray row-major
        out = Int32MultiArray()

        # Add layout metadata (optional but nice)
        dim_row = MultiArrayDimension(label='rows', size=self.rows, stride=self.rows * self.cols)
        dim_col = MultiArrayDimension(label='cols', size=self.cols, stride=self.cols)
        out.layout.dim = [dim_row, dim_col]
        out.layout.data_offset = 0

        out.data = counts.flatten().tolist()
        self.pub.publish(out)

        # Publish mean response (same layout, same indexing)
        out_resp = Float32MultiArray()
        out_resp.layout.dim = [dim_row, dim_col]
        out_resp.layout.data_offset = 0
        out_resp.data = mean_resp.flatten().tolist()
        self.response_pub.publish(out_resp)

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.publish_debug and self.debug_pub is not None and (now - self._last_debug_time) >= self.debug_period:
            self._last_debug_time = now

            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            ...
            # Optional resize for debug only
            if self.debug_resize_width > 0 and vis.shape[1] != self.debug_resize_width:
                scale = self.debug_resize_width / vis.shape[1]
                vis = cv2.resize(vis, (self.debug_resize_width, int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            # Ensure 3-channel uint8 for debug
            if vis.dtype != np.uint8:
                vis = vis.astype(np.uint8)
            if len(vis.shape) != 3 or vis.shape[2] != 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # Force constant debug size (prevents viewer resizing/blinking)
            vis = cv2.resize(vis, (self.debug_width, self.debug_height), interpolation=cv2.INTER_AREA)
            vis = np.ascontiguousarray(vis)  # stable memory layout -> stable 'step'


            dbg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            dbg.header = msg.header
            self.debug_pub.publish(dbg)



        # Optional debug image (grid + keypoints)
        if self.publish_debug and self.debug_pub is not None:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # draw grid
            for r in range(1, self.rows):
                y = int(r * cell_h)
                cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
            for c in range(1, self.cols):
                x = int(c * cell_w)
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)

            # draw keypoints
            for kp in kps:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

            dbg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            dbg.header = msg.header
            self.debug_pub.publish(dbg)

def main():
    rclpy.init()
    node = KeypointGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
