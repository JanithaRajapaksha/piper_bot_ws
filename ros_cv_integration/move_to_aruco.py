#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArucoDepthTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_depth_tf_publisher')
        self.bridge = CvBridge()

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Camera parameters ---
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Define fixed camera mount on link6
        self.T_cam_link6_trans = np.array([0.0, 0.0, 0.1])  # example offset in meters
        self.T_cam_link6_rot = R.from_euler('xyz', [0, 0, 0]).as_matrix()  # replace with your camera orientation

        # --- ArUco setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # --- Subscribers ---
        self.rgb_sub = self.create_subscription(Image, '/camera/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Latest frames
        self.latest_rgb = None
        self.latest_depth = None

        # Store 3D points in camera frame
        self.points_3d_cam = []

        # Store 3D points transformed once into base frame
        self.points_3d_base = []

        self.frame_processed = False
        self.current_point_idx = 0

        # Timer to publish transforms
        self.tf_timer = self.create_timer(5, self.publish_latest_transforms)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3,3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received")

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process_frame()

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_process_frame()

    def try_process_frame(self):
        if self.frame_processed or self.latest_rgb is None or self.latest_depth is None or self.camera_matrix is None:
            return

        frame = self.latest_rgb.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is not None:
            self.points_3d_cam = []
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_corners in enumerate(corners):
                marker_corners = marker_corners[0]  # shape (4,2)
                for j, (u, v) in enumerate(marker_corners):
                    if 0 <= int(v) < self.latest_depth.shape[0] and 0 <= int(u) < self.latest_depth.shape[1]:
                        z = float(self.latest_depth[int(v), int(u)])
                        if self.latest_depth.dtype == np.uint16:
                            z = z * 0.001  # mm -> meters
                    else:
                        z = 0.0

                    fx = self.camera_matrix[0,0]
                    fy = self.camera_matrix[1,1]
                    cx = self.camera_matrix[0,2]
                    cy = self.camera_matrix[1,2]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    self.points_3d_cam.append({
                        "marker_id": int(ids[i][0]),
                        "corner_id": j,
                        "x": x,
                        "y": y,
                        "z": z
                    })

                    # Draw for visualization
                    cv2.circle(frame, (int(u), int(v)), 5, (0,0,255), -1)
                    cv2.putText(frame, f"{z:.2f}m", (int(u)+5,int(v)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            # ✅ Convert all corners to base frame ONCE
            try:
                t_link6_base = self.tf_buffer.lookup_transform(
                    "base_link",
                    "link6",
                    rclpy.time.Time()
                )
                t_lb_trans = np.array([
                    t_link6_base.transform.translation.x,
                    t_link6_base.transform.translation.y,
                    t_link6_base.transform.translation.z
                ])
                t_lb_rot = R.from_quat([
                    t_link6_base.transform.rotation.x,
                    t_link6_base.transform.rotation.y,
                    t_link6_base.transform.rotation.z,
                    t_link6_base.transform.rotation.w
                ]).as_matrix()

                self.points_3d_base = []
                for pt in self.points_3d_cam:
                    P_cam = np.array([pt['x'], pt['y'], pt['z']])
                    P_base_fixed = t_lb_rot @ self.T_cam_link6_rot @ P_cam + t_lb_rot @ self.T_cam_link6_trans + t_lb_trans
                    self.points_3d_base.append({
                        "marker_id": pt['marker_id'],
                        "corner_id": pt['corner_id'],
                        "x": P_base_fixed[0],
                        "y": P_base_fixed[1],
                        "z": P_base_fixed[2]
                    })

                self.get_logger().info(f"Transformed {len(self.points_3d_base)} corners to base_link frame (fixed).")
                self.frame_processed = True  # ✅ Stop further processing

            except Exception as e:
                self.get_logger().warn(f"Failed to get link6->base TF: {e}")

        # cv2.imshow("Aruco Depth TF Capture", frame)
        # cv2.waitKey(1)

    def publish_latest_transforms(self):
        if not self.points_3d_base:
            return

        # Publish one corner at a time (loop)
        pt = self.points_3d_base[self.current_point_idx]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "aruco_active_corner"
        t.transform.translation.x = float(pt['x']) - 0.2
        t.transform.translation.y = float(pt['y'])
        t.transform.translation.z = float(pt['z'])
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f"Published fixed transform for corner {pt['corner_id']} of marker {pt['marker_id']} (index {self.current_point_idx})"
        )

        # Move to next corner
        self.current_point_idx = (self.current_point_idx + 1) % len(self.points_3d_base)


def main():
    rclpy.init()
    node = ArucoDepthTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
