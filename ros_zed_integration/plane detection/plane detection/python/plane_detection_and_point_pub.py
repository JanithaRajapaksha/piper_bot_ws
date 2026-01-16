#!/usr/bin/env python3
import cv2
import pyzed.sl as sl
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, Transform
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
import sys
import select
import threading
import os
import json
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types as genai_types

# Load environment variables from .env file
load_dotenv()

# -- Helper Functions for Transform Math --
def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Convert a geometry_msgs/Transform to a 4x4 numpy matrix."""
    trans = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    quat = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    rot_mat = R.from_quat(quat).as_matrix()
    
    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = trans
    return mat

def matrix_to_transform(mat: np.ndarray) -> Transform:
    """Convert a 4x4 numpy matrix to a geometry_msgs/Transform."""
    transform = Transform()
    transform.translation.x = mat[0, 3]
    transform.translation.y = mat[1, 3]
    transform.translation.z = mat[2, 3]
    
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    transform.rotation.x = quat[0]
    transform.rotation.y = quat[1]
    transform.rotation.z = quat[2]
    transform.rotation.w = quat[3]
    return transform
# -----------------------------------------


class PlaneDetectionNode(Node):
    def __init__(self):
        super().__init__('plane_detection_node')
        self.event_publisher = self.create_publisher(String, 'detection_events', 10)
        self.clicked_point = None
        self.clicked = False
        
        self.static_transform_stamped = None
        
        self.plane_normal_zed = None
        self.last_clicked_point_2d = None

        # For thread-safe data passing
        self.new_coords = None
        self.latest_frame_for_gemini = None
        self.frame_lock = threading.Lock()

        # --- Gemini AI Initialization (Older API Pattern) ---
        self.gemini_client = None
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                self.get_logger().warn("GOOGLE_API_KEY not found in .env file. Gemini functionality will be disabled.")
            else:
                self.gemini_client = genai.Client(api_key=api_key)
                self.get_logger().info("Gemini client initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Gemini client: {e}")
        # -----------------------------

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('use_click', True, ParameterDescriptor(description="True to use mouse click, False to use startup x/y coordinates."))
        self.declare_parameter('x_coord', -1, ParameterDescriptor(description="Startup X coordinate for non-click mode (normalized 0-1000)."))
        self.declare_parameter('y_coord', -1, ParameterDescriptor(description="Startup Y coordinate for non-click mode (normalized 0-1000)."))

        self.use_click = self.get_parameter('use_click').get_parameter_value().bool_value
        x_coord = self.get_parameter('x_coord').get_parameter_value().integer_value
        y_coord = self.get_parameter('y_coord').get_parameter_value().integer_value

        init = sl.InitParameters()
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.zed = sl.Camera()
        try:
            if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
                self.get_logger().error("Failed to open ZED camera.")
                rclpy.shutdown()
                return
        except Exception as e:
            self.get_logger().error(f"Failed to open ZED camera: {e}")
            rclpy.shutdown()
            return

        if not self.use_click and x_coord != -1 and y_coord != -1:
            self.get_logger().info(f"Using startup coordinates ({x_coord}, {y_coord}).")
            self.new_coords = (x_coord, y_coord)

        self.R_zed_to_ros = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])

        self.zed.enable_positional_tracking()
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.plane = sl.Plane()
        self.pose = sl.Pose()
        self.plane_params = sl.PlaneDetectionParameters()
        
        self.window_name = "ZED Plane Detection"
        cv2.namedWindow(self.window_name)
        if self.use_click:
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.get_logger().info("Plane detection node started. Click on the image to detect a plane.")
        else:
            self.get_logger().info("Plane detection node started in coordinate-input mode. Click is disabled.")

        self.timer = self.create_timer(1/30.0, self.timer_callback)

        self.input_thread = threading.Thread(target=self.runtime_input_handler)
        self.input_thread.daemon = True
        self.input_thread.start()

    def get_point_from_gemini(self, frame, object_name="book"):
        if not self.gemini_client:
            self.get_logger().warn("Gemini client not initialized. Skipping API call.")
            return None

        prompt = f"""
        Give the middle point of the {object_name}.
        The point is in [y, x] format normalized to 0-1000.
        Return only the JSON object.
        Example:
        [
            {{"label": "middle of the {object_name}", "point": [500, 500]}}
        ]
        """
        
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            self.get_logger().error("Failed to encode frame to JPEG.")
            return None
        image_bytes = encoded_image.tobytes()

        try:
            image_response = self.gemini_client.models.generate_content(
              model="gemini-robotics-er-1.5-preview",
              contents=[
                genai_types.Part.from_bytes(
                  data=image_bytes,
                  mime_type='image/jpeg',
                ),
                prompt
              ]
            )
            
            text = image_response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            points_data = json.loads(text)

            if points_data and isinstance(points_data, list) and "point" in points_data[0]:
                y_norm, x_norm = points_data[0]["point"]
                return (x_norm, y_norm) # Return as (x, y)
            else:
                self.get_logger().warn(f"Gemini response did not contain a valid point: {image_response.text}")
                return None

        except Exception as e:
            self.get_logger().error(f"Error calling Gemini API or parsing response: {e}")
            return None

    def runtime_input_handler(self):
        self.get_logger().info("\n\n"
            "---------------------------------------------------"
            " RUNTIME INPUT"
            "---------------------------------------------------"
            " - Type 'x y' (e.g., '500 500') and press Enter."
            " - Type 'gemini <object>' (e.g., 'gemini cup')."
            "---------------------------------------------------")
        while rclpy.ok():
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
                if rlist:
                    user_input = sys.stdin.readline().strip()
                    if not user_input: continue

                    parts = user_input.split()
                    if parts and parts[0].lower() == 'gemini':
                        object_name = " ".join(parts[1:]) if len(parts) > 1 else "book"
                        self.get_logger().info(f"'gemini' command received for object: '{object_name}'. Analyzing current frame...")
                        
                        with self.frame_lock:
                            frame_to_process = self.latest_frame_for_gemini.copy() if self.latest_frame_for_gemini is not None else None
                        
                        if frame_to_process is not None:
                            point = self.get_point_from_gemini(frame_to_process, object_name)
                            if point:
                                x, y = point
                                self.get_logger().info(f"Gemini identified point: x={x}, y={y}")
                                self.new_coords = (x, y)
                            else:
                                self.get_logger().warn("Gemini did not return a valid point.")
                        else:
                            self.get_logger().warn("No camera frame available to send to Gemini.")

                    elif len(parts) == 2:
                        try:
                            x = int(parts[0])
                            y = int(parts[1])
                            if 0 <= x <= 1000 and 0 <= y <= 1000:
                                self.get_logger().info(f"Received runtime coordinates: x={x}, y={y}")
                                self.new_coords = (x, y)
                            else:
                                self.get_logger().warn("Coordinates must be between 0 and 1000.")
                        except ValueError:
                            self.get_logger().warn("Invalid coordinate format. Use integer values for 'x y'.")
                    else:
                        self.get_logger().warn("Invalid input. Enter 'x y' or 'gemini <object>'.")

            except Exception:
                if rclpy.ok():
                    self.get_logger().error("Error in runtime input handler.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = [x, y]
            self.clicked = True
            self.get_logger().info(f"Clicked at ({x}, {y})")

    def zed_mat_to_cv(self, mat):
        return mat.get_data()

    def relay_link6_to_camera(self):
        try:
            tf_link6 = self.tf_buffer.lookup_transform('base_link', 'link6', rclpy.time.Time())
            t_camera = TransformStamped()
            t_camera.header.stamp = self.get_clock().now().to_msg()
            t_camera.header.frame_id = 'base_link'
            t_camera.child_frame_id = 'camera_link'
            t_camera.transform = tf_link6.transform
            self.tf_broadcaster.sendTransform(t_camera)
        except Exception as e:
            self.get_logger().warn(f"Transform lookup 'base_link' to 'link6' failed: {e}", throttle_duration_sec=5)

    def timer_callback(self):
        # Check for new coordinates from the input thread
        if self.new_coords:
            x_coord, y_coord = self.new_coords
            self.new_coords = None # Consume the coordinates
            
            resolution = self.zed.get_camera_information().camera_configuration.resolution
            denormalized_x = int((x_coord / 1000.0) * resolution.width)
            denormalized_y = int((y_coord / 1000.0) * resolution.height)

            self.clicked_point = [denormalized_x, denormalized_y]
            self.clicked = True

        self.relay_link6_to_camera()

        if self.static_transform_stamped:
            self.static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.static_transform_stamped)

        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.point_cloud.get_resolution())
            tracking_state = self.zed.get_position(self.pose)
            
            frame_rgba = self.zed_mat_to_cv(self.image)
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)
            
            with self.frame_lock:
                self.latest_frame_for_gemini = frame_bgr.copy()

            if self.clicked and tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                if self.tf_buffer.can_transform('base_link', 'camera_link', rclpy.time.Time()):
                    self.calculate_and_store_static_transform()
                    self.clicked = False
                else:
                    self.get_logger().warn("Waiting for 'base_link' to 'camera_link' transform to become available...", throttle_duration_sec=2.0)
            
            if self.plane_normal_zed is not None and self.last_clicked_point_2d is not None:
                cv2.circle(frame_bgr, (self.last_clicked_point_2d[0], self.last_clicked_point_2d[1]), 6, (0, 0, 255), -1)
                start_point = (self.last_clicked_point_2d[0], self.last_clicked_point_2d[1])
                end_point = (start_point[0] + int(self.plane_normal_zed[0] * 100), start_point[1] - int(self.plane_normal_zed[1] * 100))
                cv2.line(frame_bgr, start_point, end_point, (0, 255, 0), 2)

            cv2.imshow(self.window_name, frame_bgr)
            key = cv2.waitKey(1)
            if key == 27:
                self.get_logger().info("ESC pressed, shutting down.")
                self.destroy_node()
                rclpy.shutdown()
                return

    def calculate_and_store_static_transform(self):
        """On click, calculate the full transform from base_link to the point and store it."""
        err, point3d_camera = self.point_cloud.get_value(self.clicked_point[0], self.clicked_point[1])
        if not (err == sl.ERROR_CODE.SUCCESS and math.isfinite(point3d_camera[0])):
            self.get_logger().warn("Could not get 3D point from the click.")
            return

        status = self.zed.find_plane_at_hit(self.clicked_point, self.plane, self.plane_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().warn("Could not find a plane at the clicked point.")
            return

        point_zed = point3d_camera[:3]
        
        world_normal = self.plane.get_normal()
        rotation = sl.Rotation()
        self.pose.get_rotation_matrix(rotation)
        cam_rot_mat = rotation.r
        normal_zed = cam_rot_mat.T @ world_normal

        point_ros = self.R_zed_to_ros @ point_zed
        normal_ros = self.R_zed_to_ros @ normal_zed

        self.plane_normal_zed = normal_zed
        self.last_clicked_point_2d = self.clicked_point

        z_axis = normal_ros / np.linalg.norm(normal_ros)
        up_vector = np.array([0.0, 0.0, 1.0])
        ref_vector = up_vector if not (np.allclose(z_axis, up_vector) or np.allclose(z_axis, -up_vector)) else np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(ref_vector, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        
        rot_matrix_cam_to_point = np.column_stack((x_axis, y_axis, z_axis))
        
        mat_cam_to_point = np.eye(4)
        mat_cam_to_point[:3, :3] = rot_matrix_cam_to_point
        mat_cam_to_point[:3, 3] = point_ros

        try:
            tf_base_to_cam = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            mat_base_to_cam = transform_to_matrix(tf_base_to_cam.transform)
        except Exception as e:
            self.get_logger().error(f"Could not get transform from 'base_link' to 'camera_link': {e}. Cannot compute static transform.")
            return

        mat_base_to_point = mat_base_to_cam @ mat_cam_to_point

        self.static_transform_stamped = TransformStamped()
        self.static_transform_stamped.header.frame_id = 'base_link'
        self.static_transform_stamped.child_frame_id = 'clicked_point_frame'
        self.static_transform_stamped.transform = matrix_to_transform(mat_base_to_point)
        
        self.get_logger().info(f"Set new static transform for 'clicked_point_frame' relative to 'base_link'.")

        # Publish event message
        event_msg = String()
        event_msg.data = "point"
        self.event_publisher.publish(event_msg)
        self.get_logger().info("Published 'point' event to /detection_events topic.")

    def on_shutdown(self):
        self.get_logger().info("Shutting down node.")
        cv2.destroyAllWindows()
        if self.zed.is_opened():
            self.zed.disable_positional_tracking()
            self.zed.close()

def main(args=None):
    rclpy.init(args=args)
    plane_detection_node = PlaneDetectionNode()
    
    try:
        rclpy.spin(plane_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        plane_detection_node.on_shutdown()
        plane_detection_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()