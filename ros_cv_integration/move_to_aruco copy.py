#!/usr/bin/env python3
"""
ROS2 node that detects ArUco markers, stores their 3D positions,
and moves a MoveIt2-controlled arm sequentially to each corner
safely in the main executor thread.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class ArucoSequentialMover(Node):
    def __init__(self):
        super().__init__("aruco_sequential_mover")
        self.bridge = CvBridge()

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None

        # Camera mount on link6
        self.T_cam_link6_trans = np.array([0.0, 0.0, 0.1])
        self.T_cam_link6_rot = R.from_euler('xyz', [0,0,0]).as_matrix()

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Subscribers
        self.create_subscription(Image, "/camera/image_raw", self.rgb_callback, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.info_callback, 10)

        # Latest frames
        self.latest_rgb = None
        self.latest_depth = None

        # Detected points
        self.detected_points = []  # array of dicts with x, y, z, marker_id
        self.current_index = 0  # index for sequential motion
        self.motion_in_progress = False

        # MoveIt2 setup
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint1","joint2","joint3","joint4","joint5","joint6"],
            base_link_name="base_link",
            end_effector_name="link6",
            group_name="arm",
            callback_group=callback_group
        )
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5

        # Post-defined pose at the end
        self.post_pose = [0.3, 0.0, 0.5]
        self.post_quat = [0.0, 0.0, 0.0, 1.0]

        # Timer for detection and sequential motion
        self.create_timer(0.2, self.timer_callback)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3,3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received")

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def timer_callback(self):
        self.detect_aruco()
        self.sequential_motion()

    def detect_aruco(self):
        if self.latest_rgb is None or self.latest_depth is None or self.camera_matrix is None:
            return

        frame = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.detected_points = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_corners in enumerate(corners):
                marker_corners = marker_corners[0]
                for j, (u, v) in enumerate(marker_corners):
                    if 0 <= int(v) < self.latest_depth.shape[0] and 0 <= int(u) < self.latest_depth.shape[1]:
                        z = float(self.latest_depth[int(v), int(u)])
                        if self.latest_depth.dtype == np.uint16:
                            z *= 0.001
                    else:
                        z = 0.0
                    fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
                    cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    self.detected_points.append({"marker_id": int(ids[i][0]), "x": x, "y": y, "z": z})

        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    def sequential_motion(self):
        if self.motion_in_progress or not self.detected_points:
            return

        if self.current_index >= len(self.detected_points):
            # All points done, move to post-defined pose
            self.get_logger().info(f"Moving to post-defined pose {self.post_pose}")
            self.motion_in_progress = True
            self.moveit2.move_to_pose(position=self.post_pose, quat_xyzw=self.post_quat, cartesian=False)
            self.moveit2.wait_until_executed()
            self.get_logger().info("All motions complete.")
            self.motion_in_progress = False
            self.current_index = 0
            return

        pt = self.detected_points[self.current_index]
        try:
            t_link6_base = self.tf_buffer.lookup_transform("base_link", "link6", rclpy.time.Time())
            t_lb_trans = np.array([t_link6_base.transform.translation.x - 0.3,
                                   t_link6_base.transform.translation.y,
                                   t_link6_base.transform.translation.z])
            t_lb_rot = R.from_quat([t_link6_base.transform.rotation.x,
                                    t_link6_base.transform.rotation.y,
                                    t_link6_base.transform.rotation.z,
                                    t_link6_base.transform.rotation.w]).as_matrix()
        except:
            self.get_logger().warn("Failed to get link6->base TF. Skipping point.")
            self.current_index += 1
            return

        P_cam = np.array([pt["x"], pt["y"], pt["z"]])
        P_base = t_lb_rot @ self.T_cam_link6_rot @ P_cam + t_lb_rot @ self.T_cam_link6_trans + t_lb_trans
        target_pos = [float(P_base[0]), float(P_base[1]), float(P_base[2])+0.05]

        self.get_logger().info(f"Moving to ArUco {pt['marker_id']} at {target_pos}")
        self.motion_in_progress = True
        self.moveit2.move_to_pose(position=target_pos, quat_xyzw=[0,0,0,1], cartesian=False)
        self.moveit2.wait_until_executed()
        self.motion_in_progress = False
        self.current_index += 1


def main():
    rclpy.init()
    node = ArucoSequentialMover()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
