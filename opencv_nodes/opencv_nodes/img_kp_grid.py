import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge

import cv2
import numpy as np


class ImgKpGrid(Node):
    def __init__(self):
        super().__init__('img_kp_grid')

        # Topics / grid
        self.declare_parameter('in_topic', '/camera/image_raw')
        self.declare_parameter('out_topic', '/features/keypoints_grid')
        self.declare_parameter('rows', 3)
        self.declare_parameter('cols', 4)

        # Detector
        self.declare_parameter('detector', 'ORB')          # 'ORB' or 'FAST'
        self.declare_parameter('max_features', 800)        # ORB only
        self.declare_parameter('fast_threshold', 20)       # FAST only

        # Keypoint filtering
        self.declare_parameter('response_threshold', 0.0)  # keep kp.response >= this

        # Optional contrast boost
        self.declare_parameter('use_clahe', False)
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid', 8)

        # Optional debug image
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('debug_topic', '/debug/img_kp_grid')
        self.declare_parameter('debug_width', 320)
        self.declare_parameter('debug_height', 240)

        self.bridge = CvBridge()

        self.in_topic = self.get_parameter('in_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.rows = int(self.get_parameter('rows').value)
        self.cols = int(self.get_parameter('cols').value)

        self.detector_name = str(self.get_parameter('detector').value).upper()
        if self.detector_name == 'FAST':
            thr = int(self.get_parameter('fast_threshold').value)
            self.detector = cv2.FastFeatureDetector_create(thr, nonmaxSuppression=True)
        else:
            n = int(self.get_parameter('max_features').value)
            self.detector = cv2.ORB_create(nfeatures=n)

        self.response_threshold = float(self.get_parameter('response_threshold').value)

        self.use_clahe = bool(self.get_parameter('use_clahe').value)
        self.clahe = None
        if self.use_clahe:
            clip = float(self.get_parameter('clahe_clip_limit').value)
            tile = int(self.get_parameter('clahe_tile_grid').value)
            self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

        self.pub = self.create_publisher(Int32MultiArray, self.out_topic, 10)
        self.sub = self.create_subscription(Image, self.in_topic, self.cb, 10)

        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)
        self.debug_topic = self.get_parameter('debug_topic').value
        self.debug_width = int(self.get_parameter('debug_width').value)
        self.debug_height = int(self.get_parameter('debug_height').value)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10) if self.publish_debug else None

        self.get_logger().info(
            f"img_kp_grid running | in={self.in_topic} out={self.out_topic} "
            f"grid={self.rows}x{self.cols} detector={self.detector_name} "
            f"resp_thresh={self.response_threshold} clahe={self.use_clahe}"
        )

    def cb(self, msg: Image):
        # Convert ROS -> OpenCV grayscale (mono8)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = frame.shape[:2]

        # Optional CLAHE
        if self.clahe is not None:
            frame = self.clahe.apply(frame)

        # Detect keypoints
        kps = self.detector.detect(frame, None)

        # Filter by response threshold
        if self.response_threshold > 0.0:
            kps = [kp for kp in kps if float(kp.response) >= self.response_threshold]

        # Bin counts
        counts = np.zeros((self.rows, self.cols), dtype=np.int32)
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

        # Publish Int32MultiArray (row-major)
        out = Int32MultiArray()
        dim_row = MultiArrayDimension(label='rows', size=self.rows, stride=self.rows * self.cols)
        dim_col = MultiArrayDimension(label='cols', size=self.cols, stride=self.cols)
        out.layout.dim = [dim_row, dim_col]
        out.layout.data_offset = 0
        out.data = counts.flatten().tolist()
        self.pub.publish(out)

        # Optional debug image (grid + keypoints)
        if self.publish_debug and self.debug_pub is not None:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Draw grid
            for rr in range(1, self.rows):
                y = int(rr * cell_h)
                cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
            for cc in range(1, self.cols):
                x = int(cc * cell_w)
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)

            # Draw keypoints
            for kp in kps:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)

            # Force constant debug size + stable layout
            vis = cv2.resize(vis, (self.debug_width, self.debug_height), interpolation=cv2.INTER_AREA)
            vis = np.ascontiguousarray(vis)

            dbg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            dbg.header = msg.header
            self.debug_pub.publish(dbg)


def main():
    rclpy.init()
    node = ImgKpGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
