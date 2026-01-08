"""
    This sample shows how to detect planes in a 3D scene using Gemini to select a point,
    and displays it on an OpenGL window.
"""
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import argparse
import cv2
import numpy as np
import os
import json
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import datetime

# ROS 2 imports
import rclpy
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

def main(opt):
    load_dotenv()
    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        print("Please set the GOOGLE_API_KEY environment variable.")
        exit()

    rclpy.init(args=None)
    node = rclpy.create_node('gemini_plane_detector')
    tf_broadcaster = tf2_ros.TransformBroadcaster(node)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    init = sl.InitParameters()
    parse_args(init, opt)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}. Exit program.")
        exit()

    camera_infos = zed.get_camera_information()
    has_imu = camera_infos.sensors_configuration.gyroscope_parameters.is_available

    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, has_imu)

    image = sl.Mat()
    pose = sl.Pose()
    plane = sl.Plane()
    mesh = sl.Mesh()

    find_plane_status = sl.ERROR_CODE.PLANE_NOT_FOUND
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    
    user_action = gl.UserAction()
    user_action.clear()

    zed.enable_positional_tracking()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    plane_parameters = sl.PlaneDetectionParameters()
    
    gemini_triggered = False

    while viewer.is_available():
        rclpy.spin_once(node, timeout_sec=0.0)
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            image_ocv = image.get_data()
            tracking_state = zed.get_position(pose)

            # Click in the 3D view to trigger Gemini-based plane detection
            if user_action.hit:
                print("Click detected. Processing frame with Gemini...")

                # --- Gemini Logic ---
                image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                import io
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                image_bytes = img_byte_arr.getvalue()

                prompt = """
                Point me the phone and give me the mid point.
                """

                print("Sending image to Gemini API...")
                image_response = client.models.generate_content(
                    model="gemini-robotics-er-1.5-preview",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                        prompt
                    ],
                    config=types.GenerateContentConfig(temperature=0.5)
                )
                print("Gemini response:")
                print(image_response.text)

                try:
                    # Create a timestamped directory for the output
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_dir = os.path.join("gemini_output", timestamp)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the original image
                    original_image_path = os.path.join(output_dir, "original.png")
                    pil_image.save(original_image_path)

                    # Save the JSON response
                    json_response_path = os.path.join(output_dir, "response.json")
                    with open(json_response_path, 'w') as f:
                        json.dump(image_response.text, f)

                    text = image_response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    elif text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    
                    parsed_json = json.loads(text)
                    point_data = None
                    if isinstance(parsed_json, list) and parsed_json:
                        point_data = parsed_json[0]
                    elif isinstance(parsed_json, dict):
                        point_data = parsed_json
                    
                    if point_data and "point" in point_data:
                        output_image = pil_image.copy()
                        draw = ImageDraw.Draw(output_image)
                        width, height = output_image.size

                        y_norm, x_norm = point_data["point"]
                        x = (x_norm / 1000) * width
                        y = (y_norm / 1000) * height
                        r = 5
                        draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white")
                        
                        image_click = [x, y]
                        print(f"Gemini selected point (denormalized): {image_click}")

                        # --- Plane Detection at Gemini's point ---
                        find_plane_status = zed.find_plane_at_hit(image_click, plane, plane_parameters)
                        
                        if find_plane_status == sl.ERROR_CODE.SUCCESS:
                            mesh = plane.extract_mesh()
                            viewer.update_mesh(mesh, plane.type)
                            print("✅ Plane detected with Gemini point.")

                            # --- Visualize Surface Normal ---
                            center = plane.get_center()
                            normal = plane.get_normal()
                            
                            # Get camera parameters
                            calib = camera_infos.camera_configuration.calibration_parameters.left_cam
                            fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
                            world_to_cam = sl.Transform()
                            world_to_cam.init_matrix(pose.pose_data())
                            world_to_cam.inverse()
                            view_matrix = world_to_cam.m

                            def project(point3d):
                                p_cam = np.dot(view_matrix, np.append(point3d, 1))
                                # ROS (X-fwd, Y-left, Z-up) to CV (Z-fwd, X-right, Y-down)
                                x_cv, y_cv, z_cv = -p_cam[1], -p_cam[2], p_cam[0]
                                if z_cv > 0.1:
                                    return (int(cx + fx * x_cv / z_cv), int(cy + fy * y_cv / z_cv))
                                return None

                            uv_center = project(center)
                            if uv_center:
                                uv_normal_end = project(center + normal * 0.2) # 20cm long normal
                                if uv_normal_end:
                                    draw.line([uv_center, uv_normal_end], fill="blue", width=3)

                        else:
                            print(f"⚠️ Could not find a plane at the point specified by Gemini. Status: {find_plane_status}")

                        # Save the final image with point and normal
                        output_image_path = os.path.join(output_dir, "trajectory.png")
                        output_image.save(output_image_path)
                        print(f"Saved visualization to {output_image_path}")

                    else:
                        print("⚠️ Gemini did not return a valid point.")

                except Exception as e:
                    print(f"Error processing Gemini response or detecting plane: {e}")

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                if find_plane_status == sl.ERROR_CODE.SUCCESS:
                    plane_pose = plane.get_pose()
                    plane_tf = TransformStamped()
                    plane_tf.header.stamp = node.get_clock().now().to_msg()
                    plane_tf.header.frame_id = "base_link"
                    plane_tf.child_frame_id = "detected_plane"

                    translation = plane_pose.get_translation().get()
                    orientation = plane_pose.get_orientation().get()

                    plane_tf.transform.translation.x = float(translation[0])
                    plane_tf.transform.translation.y = float(translation[1])
                    plane_tf.transform.translation.z = float(translation[2])
                    plane_tf.transform.rotation.x = float(orientation[0])
                    plane_tf.transform.rotation.y = float(orientation[1])
                    plane_tf.transform.rotation.z = float(orientation[2])
                    plane_tf.transform.rotation.w = float(orientation[3])

                    # Using a static transform broadcaster as the plane should be static relative to the world once detected.
                    static_broadcaster = tf2_ros.StaticTransformBroadcaster(node)
                    static_broadcaster.sendTransform(plane_tf)
                    
                    # Reset status to avoid re-publishing
                    find_plane_status = sl.ERROR_CODE.PLANE_NOT_FOUND 
                    print("✅ Plane transform published.")


            user_action = viewer.update_view(image, pose.pose_data(), tracking_state)

            frame_vis = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            cv2.imshow("Gemini Plane Detection", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    viewer.exit()
    zed.disable_positional_tracking()
    zed.close()
    node.destroy_node()
    rclpy.shutdown()

def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("resolution" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d.', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
        print("Specify only input_svo_file or ip_address, not both. Exit program")
        exit()
    main(opt)
