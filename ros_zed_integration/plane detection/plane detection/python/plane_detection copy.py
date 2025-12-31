########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample shows how to detect planes in a 3D scene and
    displays it on an OpenGL window
"""
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import argparse
import cv2
import numpy as np

# ROS 2 imports
import rclpy
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.time import Time



def main(opt):
    rclpy.init(args=None)
    node = rclpy.create_node('plane_detector_tf_publisher')
    tf_broadcaster = tf2_ros.TransformBroadcaster(node)

    init = sl.InitParameters()
    parse_args(init, opt)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD  # Use ROS coordinate system to publish TF
    # Create a camera object
    zed = sl.Camera()
    # Open the camera
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    # Get camera info and check if IMU data is available
    camera_infos = zed.get_camera_information()
    has_imu =  camera_infos.sensors_configuration.gyroscope_parameters.is_available

    # Initialize OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, has_imu)

    image = sl.Mat()    # current left image
    pose = sl.Pose()    # positional tracking data
    plane = sl.Plane()  # detected plane 
    mesh = sl.Mesh()    # plane mesh

    find_plane_status = sl.ERROR_CODE.PLANE_NOT_FOUND
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF

    # Timestamp of the last mesh request
    last_call = time.time()

    user_action = gl.UserAction()
    user_action.clear()

    # Enable positional tracking before starting spatial mapping
    zed.enable_positional_tracking()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    # the plane detection parameters can be change
    plane_parameters = sl.PlaneDetectionParameters()

    while viewer.is_available():
        if zed.grab(runtime_parameters) <= sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Convert sl.Mat to OpenCV (Numpy array)
            # The image is in BGRA format (4 channels)
            image_ocv = image.get_data()

            # Update pose data
            tracking_state = zed.get_position(pose)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Publish Camera Pose (map -> base_link)
                t_cam = TransformStamped()
                zed_ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                t_cam.header.stamp = Time(seconds=zed_ts.get_seconds(), nanoseconds=zed_ts.get_nanoseconds() % 1000000000).to_msg()
                t_cam.header.frame_id = 'map'
                t_cam.child_frame_id = 'base_link'

                translation = pose.get_translation().get()
                orientation = pose.get_orientation().get()

                t_cam.transform.translation.x = float(translation[0])
                t_cam.transform.translation.y = float(translation[1])
                t_cam.transform.translation.z = float(translation[2])

                t_cam.transform.rotation.x = float(orientation[0])
                t_cam.transform.rotation.y = float(orientation[1])
                t_cam.transform.rotation.z = float(orientation[2])
                t_cam.transform.rotation.w = float(orientation[3])

                tf_broadcaster.sendTransform(t_cam)

                duration = time.time() - last_call  

                # Plane detection on mouse click
                if user_action.hit:
                    image_click = [
                        user_action.hit_coord[0] * camera_infos.camera_configuration.resolution.width,
                        user_action.hit_coord[1] * camera_infos.camera_configuration.resolution.height
                    ]
                    find_plane_status = zed.find_plane_at_hit(
                        image_click, plane, plane_parameters
                    )

                # Floor plane detection
                if duration > 0.5 and user_action.press_space:
                    reset_tracking_floor_frame = sl.Transform()
                    find_plane_status = zed.find_floor_plane(
                        plane, reset_tracking_floor_frame
                    )
                    last_call = time.time()

                if find_plane_status == sl.ERROR_CODE.SUCCESS:
                    mesh = plane.extract_mesh()

                    center = plane.get_center()      # np.array([x, y, z])
                    normal = plane.get_normal()      # np.array([nx, ny, nz])

                    scale = 1.0
                    p0 = center
                    p1 = center + normal * scale

                    viewer.update_mesh(mesh, plane.type)

                    # # 🔹 PASS NUMPY ARRAYS
                    # viewer.update_normal(p0, p1)
                    print("Center:", center, "Normal:", normal)

                    # Publish the plane's pose to TF
                    t = TransformStamped()

                    # Get the timestamp from the ZED SDK
                    zed_ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                    t.header.stamp = Time(seconds=zed_ts.get_seconds(), nanoseconds=zed_ts.get_nanoseconds() % 1000000000).to_msg()
                    
                    # The 'map' frame is the world frame from the ZED ROS2 wrapper.
                    # This publishes the plane's pose in the fixed world frame.
                    t.header.frame_id = 'map'
                    t.child_frame_id = 'detected_plane'

                    # The pose of the plane is a transform that aligns the plane's local coordinate system
                    # with the world frame. By convention, the local Z-axis of the plane is its normal.
                    plane_pose = plane.get_pose()

                    # Create a rotation of 180 degrees around the Z axis to invert the X axis
                    rotation_180_z = sl.Transform()
                    # A 180-degree rotation around Z is equivalent to a yaw of pi radians.
                    rotation_180_z.set_euler_angles(0, 0, np.pi, radian=True)

                    # Apply the rotation to the plane's pose.
                    # The multiplication order T_new = T_old * T_applied applies the rotation in the local frame of T_old.
                    # The result is a Matrix4f, so we must re-initialize an sl.Transform object from it.
                    rotated_pose_matrix = plane_pose * rotation_180_z
                    final_plane_pose = sl.Transform()
                    final_plane_pose.init_matrix(rotated_pose_matrix)
                    translation = final_plane_pose.get_translation().get()
                    orientation = final_plane_pose.get_orientation().get()

                    t.transform.translation.x = float(translation[0])
                    t.transform.translation.y = float(translation[1])
                    t.transform.translation.z = float(translation[2])

                    t.transform.rotation.x = float(orientation[0])
                    t.transform.rotation.y = float(orientation[1])
                    t.transform.rotation.z = float(orientation[2])
                    t.transform.rotation.w = float(orientation[3])

                    tf_broadcaster.sendTransform(t)


            user_action = viewer.update_view(
                image, pose.pose_data(), tracking_state
            )

            # Visualize in another CV2 frame
            frame_vis = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK and find_plane_status == sl.ERROR_CODE.SUCCESS:
                center = plane.get_center()
                normal = plane.get_normal()
                cv2.putText(frame_vis, "Plane Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_vis, f"Center: {center}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame_vis, f"Normal: {normal}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Project 3D points to 2D image
                calib = camera_infos.camera_configuration.calibration_parameters.left_cam
                fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

                world_to_cam = sl.Transform()
                world_to_cam.init_matrix(pose.pose_data())
                world_to_cam.inverse()
                view_matrix = world_to_cam.m

                # Calculate distance to the plane center
                center_cam = np.dot(view_matrix, np.append(center, 1))
                distance = np.linalg.norm(center_cam[:3])
                cv2.putText(frame_vis, f"Distance: {distance:.2f}m", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                def project(point3d):
                    p_cam = np.dot(view_matrix, np.append(point3d, 1))
                    # ROS (X-fwd, Y-left, Z-up) to CV (Z-fwd, X-right, Y-down)
                    x, y, z = -p_cam[1], -p_cam[2], p_cam[0]
                    if z > 0.1:
                        return (int(cx + fx * x / z), int(cy + fy * y / z))
                    return None

                uv_center = project(center)
                if uv_center:
                    cv2.circle(frame_vis, uv_center, 5, (0, 255, 0), -1)
                    uv_normal = project(center + normal * 0.5)
                    if uv_normal:
                        cv2.arrowedLine(frame_vis, uv_center, uv_normal, (0, 0, 255), 2)
            else:
                if tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
                    cv2.putText(frame_vis, "Tracking not ready", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_vis, "Click on a plane in the 3D view to detect it", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("ZED Plane Detection", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    viewer.exit()
    image.free(sl.MEM.CPU)
    mesh.clear()

    # Disable modules and close camera
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
