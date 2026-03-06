#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt8, Float32MultiArray
from cv_bridge import CvBridge

import cv2
import numpy as np

from task_manager_interfaces.srv import SetTaskState

# Task states (match your task_manager)
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


def _parse_dict(name: str):
    """Map string -> OpenCV aruco dictionary constant."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class ImgRecog(Node):
    """
    Publishes /vision/aruco/target as Float32MultiArray with layout:

    idx: field
      0: visible (1.0 or 0.0)
      1: id (float, but represents int)
      2: x_norm ([-1..1], left=-1, center=0, right=+1)
      3: y_norm ([-1..1], top=-1, center=0, bottom=+1)
      4: area_px (marker quad area in pixels)
      5: pixel_width (avg of top/bottom edge lengths in pixels)
      6: distance_m (meters if pose available else -1)
      7-9: tvec (x,y,z) meters (if pose else 0,0,0)
      10-12: rvec (x,y,z) radians axis-angle (if pose else 0,0,0)
      13: state (current task state)
    """

    def __init__(self):
        super().__init__("img_recog")

        # Params
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("state_topic", "/task/state")
        self.declare_parameter("out_topic", "/vision/aruco/target")

        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("item_ids", [0, 1, 2, 3])      # change via params/launch
        self.declare_parameter("dropoff_ids", [11])   # change via params/launch

        # Pose estimation needs marker_length (meters)
        self.declare_parameter("marker_length_m", 0.08)  # 8cm default; adjust to your printed marker size
        self.declare_parameter("target_policy", "largest")  # "largest" or "closest_to_center"

        # Parameters for state transitions
        self.declare_parameter("set_state_service", "/task/set_state")
        self.declare_parameter("found_frames", 2)   # how many consecutive frames required to switch SEARCH->APPROACH
        self.declare_parameter("lost_frames", 5)    # how many consecutive missing frames to switch APPROACH->SEARCH
        self.declare_parameter("enable_state_auto", True)

        self.set_state_service = self.get_parameter("set_state_service").value
        self.found_frames = int(self.get_parameter("found_frames").value)
        self.lost_frames = int(self.get_parameter("lost_frames").value)
        self.enable_state_auto = bool(self.get_parameter("enable_state_auto").value)

        # Create client for task_master
        self.set_state_client = self.create_client(SetTaskState, self.set_state_service)
        self._consec_found = 0
        self._consec_lost = 0
        self._pending_request = False

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.out_topic = self.get_parameter("out_topic").value

        dict_name = self.get_parameter("aruco_dictionary").value
        self.dictionary = _parse_dict(dict_name)

        self.item_ids = set(int(x) for x in self.get_parameter("item_ids").value)
        self.dropoff_ids = set(int(x) for x in self.get_parameter("dropoff_ids").value)

        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.target_policy = str(self.get_parameter("target_policy").value).lower().strip()

        # Detector parameters (OpenCV version-safe)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_params = cv2.aruco.DetectorParameters_create()
        else:
            self.detector_params = cv2.aruco.DetectorParameters()

        # ROS I/O
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, self.image_topic, self.cb_img, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_info, 10)
        self.sub_state = self.create_subscription(UInt8, self.state_topic, self.cb_state, 10)
        self.pub = self.create_publisher(Float32MultiArray, self.out_topic, 10)

        # State
        self.current_state: int = SEARCH_ITEM

        # Camera model (filled from CameraInfo)
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None

        self.get_logger().info(f"Subscribing image: {self.image_topic}")
        self.get_logger().info(f"Subscribing camera_info: {self.camera_info_topic}")
        self.get_logger().info(f"Subscribing state: {self.state_topic}")
        self.get_logger().info(f"Publishing: {self.out_topic}")
        self.get_logger().info(f"Dictionary: {dict_name}")
        self.get_logger().info(f"item_ids={sorted(self.item_ids)} dropoff_ids={sorted(self.dropoff_ids)}")
        self.get_logger().info(f"marker_length_m={self.marker_length_m} target_policy={self.target_policy}")

    def cb_state(self, msg: UInt8):
        self.current_state = int(msg.data)

    def cb_info(self, msg: CameraInfo):
        # Only need to parse once; but safe to update if it changes
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # distortion can be length 0, 4, 5, 8, etc.
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) > 0 else np.zeros((5,), dtype=np.float64)

    def _valid_ids_for_state(self) -> set:
        if self.current_state in (SEARCH_ITEM, APPROACH_ITEM):
            return self.item_ids
        if self.current_state in (SEARCH_DROPOFF, APPROACH_DROPOFF):
            return self.dropoff_ids
        return set()

    @staticmethod
    def _quad_area(corners: np.ndarray) -> float:
        # corners shape (4,2)
        return float(cv2.contourArea(corners.astype(np.float32)))

    @staticmethod
    def _pixel_width(corners: np.ndarray) -> float:
        # average of top and bottom edge lengths
        p0, p1, p2, p3 = corners
        top = float(np.linalg.norm(p1 - p0))
        bottom = float(np.linalg.norm(p2 - p3))
        return 0.5 * (top + bottom)

    @staticmethod
    def _norm_xy(center: Tuple[float, float], w: int, h: int) -> Tuple[float, float]:
        cx, cy = center
        # map to [-1,1], (0,0) at center
        x = (cx - (w / 2.0)) / (w / 2.0)
        y = (cy - (h / 2.0)) / (h / 2.0)
        return float(np.clip(x, -1.0, 1.0)), float(np.clip(y, -1.0, 1.0))

    def _choose_best(self, candidates: List[dict], w: int, h: int) -> dict:
        if self.target_policy == "closest_to_center":
            # minimize |x_norm| + |y_norm|
            def score(c):
                return abs(c["x_norm"]) + abs(c["y_norm"])
            return min(candidates, key=score)
        # default: largest area
        return max(candidates, key=lambda c: c["area_px"])

    def cb_img(self, msg: Image):
        # Convert ROS image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners_list, ids, _rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )

        valid_ids = self._valid_ids_for_state()
        candidates: List[dict] = []

        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten().tolist()
            for i, mid in enumerate(ids_flat):
                mid = int(mid)
                if mid not in valid_ids:
                    continue

                corners = corners_list[i].reshape(4, 2).astype(np.float64)
                center = (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))
                x_norm, y_norm = self._norm_xy(center, w, h)

                area_px = self._quad_area(corners)
                px_w = self._pixel_width(corners)

                candidates.append({
                    "id": mid,
                    "corners": corners,
                    "center": center,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "area_px": area_px,
                    "pixel_width": px_w,
                })

        # Default “no detection” publish
        out = Float32MultiArray()
        out.data = [
            0.0,   # visible
            -1.0,  # id
            0.0,   # x_norm
            0.0,   # y_norm
            0.0,   # area_px
            0.0,   # pixel_width
            -1.0,  # distance_m
            0.0, 0.0, 0.0,  # tvec
            0.0, 0.0, 0.0,  # rvec
            float(self.current_state),
        ]

        if len(candidates) == 0:
            self.pub.publish(out)
            
            if self.current_state == APPROACH_ITEM:
                self._consec_lost += 1
                self._consec_found = 0
                if self._consec_lost >= self.lost_frames:
                    self._request_state(SEARCH_ITEM)
            elif self.current_state == APPROACH_DROPOFF:
                self._consec_lost += 1
                self._consec_found = 0
                if self._consec_lost >= self.lost_frames:
                    self._request_state(SEARCH_DROPOFF)
            else:
                # in SEARCH states, no tag visible is normal
                self._consec_lost = 0
                self._consec_found = 0

            return

        best = self._choose_best(candidates, w, h)

        if self.current_state == SEARCH_ITEM:
            self._consec_found += 1
            self._consec_lost = 0
            if self._consec_found >= self.found_frames:
                self._request_state(APPROACH_ITEM)

        elif self.current_state == SEARCH_DROPOFF:
            self._consec_found += 1
            self._consec_lost = 0
            if self._consec_found >= self.found_frames:
                self._request_state(APPROACH_DROPOFF)

        else:
            # in APPROACH states, seeing the tag is normal; reset lost counter
            self._consec_lost = 0
            self._consec_found = 0

        # Pose estimation if camera intrinsics available
        distance_m = -1.0
        tvec = (0.0, 0.0, 0.0)
        rvec = (0.0, 0.0, 0.0)

        if self.K is not None and self.D is not None and self.marker_length_m > 0.0:
            # estimatePoseSingleMarkers expects corners as list/array shaped (N,1,4,2) or (N,4,2)
            c = best["corners"].astype(np.float32).reshape(1, 1, 4, 2)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(c, self.marker_length_m, self.K, self.D)
            rv = rvecs[0][0]
            tv = tvecs[0][0]
            rvec = (float(rv[0]), float(rv[1]), float(rv[2]))
            tvec = (float(tv[0]), float(tv[1]), float(tv[2]))
            distance_m = float(math.sqrt(tv[0] ** 2 + tv[1] ** 2 + tv[2] ** 2))

        out.data = [
            1.0,
            float(best["id"]),
            float(best["x_norm"]),
            float(best["y_norm"]),
            float(best["area_px"]),
            float(best["pixel_width"]),
            float(distance_m),
            float(tvec[0]), float(tvec[1]), float(tvec[2]),
            float(rvec[0]), float(rvec[1]), float(rvec[2]),
            float(self.current_state),
        ]
        self.pub.publish(out)

    # Helper funcion to request state change
    def _request_state(self, new_state: int):
        if not self.enable_state_auto:
            return
        if self._pending_request:
            return
        if not self.set_state_client.service_is_ready():
            # Don’t block in the image callback
            self.get_logger().warn("set_state service not ready yet")
            return

        req = SetTaskState.Request()
        req.new_state = int(new_state)

        self._pending_request = True
        future = self.set_state_client.call_async(req)
        future.add_done_callback(self._on_set_state_done)

    def _on_set_state_done(self, future):
        self._pending_request = False
        try:
            resp = future.result()
            if not resp.success:
                self.get_logger().warn(f"State change rejected: {resp.message}")
        except Exception as e:
            self.get_logger().warn(f"State change call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImgRecog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()