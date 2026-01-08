#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.engine.results import Results

from threading import Lock, Thread
from time import sleep
from typing import List

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

if gl.GPU_ACCELERATION_AVAILABLE:
    import cupy as cp

# Globals
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None  # We need it to keep the ownership of the sl.Mat


def draw_3d_bbox(image, bbox_3d, cam_params, color=(0, 255, 0)):
    pts_2d = project_3d_to_2d(bbox_3d, cam_params)

    if any(p is None for p in pts_2d):
        return

    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # vertical
    ]

    for i, j in edges:
        cv2.line(image, pts_2d[i], pts_2d[j], color, 2)


def xywh2abcd_(xywh: np.ndarray) -> np.ndarray:
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = max(0, xywh[0] - 0.5 * xywh[2])
    x_max = (xywh[0] + 0.5 * xywh[2])
    y_min = max(0, xywh[1] - 0.5 * xywh[3])
    y_max = (xywh[1] + 0.5 * xywh[3])

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_masks_(dets: Results) -> List[sl.CustomMaskObjectData]:
    global sl_mats
    output = []
    sl_mats = []
    for det in dets.cpu().numpy():
        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomMaskObjectData()

        # Bounding box
        box = det.boxes
        xywh = box.xywh[0]
        abcd = xywh2abcd_(xywh)

        obj.bounding_box_2d = abcd
        obj.label = box.cls
        obj.probability = box.conf
        obj.is_grounded = False

        # Mask
        if det.masks is not None:
            mask_bin = (det.masks.data[0] * 255).astype(np.uint8)
            mask_bin = mask_bin[int(abcd[0][1]): int(abcd[2][1]),
                                int(abcd[0][0]): int(abcd[2][0])]
            if not mask_bin.flags.c_contiguous:
                mask_bin = np.ascontiguousarray(mask_bin)

            # Mask as a sl mat
            sl_mat = sl.Mat(width=mask_bin.shape[1],
                            height=mask_bin.shape[0],
                            mat_type=sl.MAT_TYPE.U8_C1,
                            memory_type=sl.MEM.CPU)
            p_sl_as_cv = sl_mat.get_data()
            np.copyto(p_sl_as_cv, mask_bin)
            sl_mats.append(sl_mat)

            obj.box_mask = sl_mat
        else:
            print("[Warning] No mask found in the prediction. Did you use a seg model?")

        output.append(obj)

    return output


def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.2, iou_thres: float = 0.45) -> None:
    global image_net, exit_signal, run_signal, detections

    print("Initializing Network...")
    model = YOLO(weights)
    print("Network Initialized...")

    while not exit_signal:
        if run_signal:
            lock.acquire()

            if gl.GPU_ACCELERATION_AVAILABLE:
                img_cupy = cp.asarray(image_net)[:, :, :3]  # Remove alpha channel on GPU
                # YOLO internally does some preprocessing, so we convert to numpy to let it handle the image
                # This is not the most efficient way, but it works for demonstration purposes.
                # It would be better to keep everything on GPU, with a zero-copy transfer like
                # `tensor = torch.from_dlpack(img_cupy.toDlpack())`
                img = cp.asnumpy(img_cupy)
            else:
                img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)

            # Run inference
            det = model.predict(img, save=False, retina_masks=True, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]

            # ZED CustomMasks format
            detections = detections_to_custom_masks_(det)
            lock.release()
            run_signal = False

        sleep(0.01)

def project_3d_to_2d(points_3d, cam_params):
    fx = cam_params.fx
    fy = cam_params.fy
    cx = cam_params.cx
    cy = cam_params.cy

    pts_2d = []

    for p in points_3d:
        X = p[0]
        Y = p[1]
        Z = p[2]

        # ZED: forward is NEGATIVE Z
        if Z >= 0:
            pts_2d.append(None)
            continue

        Z = -Z  # convert to OpenCV convention

        u = int((X * fx / Z) + cx)
        v = int((-Y * fy / Z) + cy)

        pts_2d.append((u, v))

    return pts_2d




def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections

    # Determine memory type based on CuPy availability and user preference
    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not args.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU

    # Display memory type being used
    if use_gpu:
        print("🚀 Using GPU data transfer with CuPy")

    capture_thread = Thread(target=torch_thread_,
                            kwargs={'weights': args.weights, 'img_size': args.img_size, "conf_thres": args.conf_thres})
    capture_thread.start()

    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if args.svo is not None:
        input_type.set_from_svo_file(args.svo)
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10

    # Initialize the camera
    print("Initializing Camera...")
    zed = sl.Camera()
    status = zed.open(init_params)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    print("Camera Initialized")

    # Enable Positional Tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable Object Detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True
    zed.enable_object_detection(obj_param)

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, mem_type)
    image_left = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type)

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance*1000, 1)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    # Prepare runtime retrieval
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    image_left_tmp = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type)
    objects = sl.Objects()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) <= sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, mem_type)
            image_net = image_left_tmp.get_data(memory_type=mem_type, deep_copy=False)
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_mask_objects(detections)
            lock.release()
            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, point_cloud_res)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, mem_type, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud, objects)
            # 2D rendering
            if gl.GPU_ACCELERATION_AVAILABLE:
                image_left_ocv = cp.asnumpy(cp.asarray(image_left.get_data(memory_type=mem_type, deep_copy=False)))
            else:
                np.copyto(image_left_ocv, image_left.get_data(memory_type=mem_type, deep_copy=False))
            # Draw 3D boxes on the 2D image
            cam_params = camera_config.calibration_parameters.left_cam

            # Convert once (always valid)
            image_bgr = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)

            cam_params = camera_config.calibration_parameters.left_cam

            for obj in objects.object_list:
                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    draw_3d_bbox(image_bgr, obj.bounding_box, cam_params)

                    # Compute center of the 3D bounding box
                    center = np.mean(obj.bounding_box, axis=0)  # [x, y, z]

                    # Optionally, orientation can be identity (no rotation) or use obj.rotation if available
                    quat = [0, 0, 0, 1]  # identity quaternion

                    t = TransformStamped()
                    t.header.stamp = ros_node.get_clock().now().to_msg()
                    t.header.frame_id = "base_link"
                    t.child_frame_id = f"object_{obj.id}" if hasattr(obj, 'id') else f"object_unknown"

                    t.transform.translation.x = float(center[0])
                    t.transform.translation.y = float(center[1])
                    t.transform.translation.z = float(center[2])
                    t.transform.rotation.x = float(quat[0])
                    t.transform.rotation.y = float(quat[1])
                    t.transform.rotation.z = float(quat[2])
                    t.transform.rotation.w = float(quat[3])


                    tf_broadcaster.sendTransform(t)


            cv2.imshow("ZED | 3D BBoxes on Image", image_bgr)


            key = cv2.waitKey(10)
            if key in (27, ord('q'), ord('Q')):
                exit_signal = True
            if key == 105: #for 'i' key 
                track_view_generator.zoomIn()
            if key == 111 : #for 'o' key
                track_view_generator.zoomOut() 
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':

    rclpy.init()
    ros_node = Node("zed_object_tf_broadcaster")
    tf_broadcaster = TransformBroadcaster(ros_node)


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11m-seg.pt', help='model.pt path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--disable-gpu-data-transfer', action='store_true', help='Disable GPU data transfer acceleration with CuPy even if CuPy is available')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)

    ros_node.destroy_node()
    rclpy.shutdown()

