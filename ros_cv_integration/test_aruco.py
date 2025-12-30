import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class ArucoSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # ArUco Dictionary (Common defaults: DICT_5X5_250, DICT_4X4_50, DICT_ARUCO_ORIGINAL)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Marker length in meters
        self.marker_length = 1.0 

        self.camera_matrix = None
        self.dist_coeffs = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            self.get_logger().info('Waiting for camera info...')
            return

        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        # Draw rejected markers for debugging (in Red)
        if len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Define marker object points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
            # This replaces estimatePoseSingleMarkers which is deprecated/removed in newer OpenCV
            marker_points = np.array([
                [-self.marker_length / 2.0, self.marker_length / 2.0, 0],
                [self.marker_length / 2.0, self.marker_length / 2.0, 0],
                [self.marker_length / 2.0, -self.marker_length / 2.0, 0],
                [-self.marker_length / 2.0, -self.marker_length / 2.0, 0]
            ], dtype=np.float32)

            for i in range(len(ids)):
                # Estimate pose for each marker
                _, rvec, tvec = cv2.solvePnP(
                    marker_points, corners[i], self.camera_matrix, self.dist_coeffs
                )

                # Draw axis
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs, 
                    rvec, tvec, 0.5
                )

                # Draw pose text on frame
                x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                text_pose = f"x={x:.2f} y={y:.2f} z={z:.2f}"
                corner_top_left = corners[i][0][0].astype(int)
                cv2.putText(frame, text_pose, (corner_top_left[0], corner_top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Log pose
                self.get_logger().info(f"ID: {ids[i][0]} | x:{x:.2f} y:{y:.2f} z:{z:.2f}")

                # Publish TF
                # Convert rvec (Rodrigues) to Rotation Matrix, then to Quaternion
                rot_matrix, _ = cv2.Rodrigues(rvec)
                quat = R.from_matrix(rot_matrix).as_quat() # [x, y, z, w]

                t = TransformStamped()
                t.header.stamp = msg.header.stamp
                t.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera_optical_frame"
                t.child_frame_id = f"aruco_marker_{ids[i][0]}"

                t.transform.translation.x = float(tvec[0][0])
                t.transform.translation.y = float(tvec[1][0])
                t.transform.translation.z = float(tvec[2][0])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])

                self.tf_broadcaster.sendTransform(t)

        cv2.imshow("ArUco Pose", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArucoSubscriber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
