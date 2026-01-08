#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
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


# -------------------------------
# Globals for YOLO thread
# -------------------------------
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None
selected_object_id = None


# -------------------------------
# YOLO helper functions
# -------------------------------
def xywh2abcd_(xywh: np.ndarray) -> np.ndarray:
    output = np.zeros((4, 2))
    x_min = max(0, xywh[0] - 0.5 * xywh[2])
    x_max = xywh[0] + 0.5 * xywh[2]
    y_min = max(0, xywh[1] - 0.5 * xywh[3])
    y_max = xywh[1] + 0.5 * xywh[3]

    output[0] = [x_min, y_min]
    output[1] = [x_max, y_min]
    output[2] = [x_max, y_max]
    output[3] = [x_min, y_max]

    return output


def detections_to_custom_masks_(dets: Results) -> List[sl.CustomMaskObjectData]:
    global sl_mats
    output = []
    sl_mats = []

    for det in dets.cpu().numpy():
        obj = sl.CustomMaskObjectData()
        box = det.boxes
        xywh = box.xywh[0]
        abcd = xywh2abcd_(xywh)

        obj.bounding_box_2d = abcd
        obj.label = box.cls
        obj.probability = box.conf
        obj.is_grounded = False

        # Mask handling
        if det.masks is not None:
            mask_bin = (det.masks.data[0] * 255).astype(np.uint8)
            mask_bin = mask_bin[int(abcd[0][1]): int(abcd[2][1]),
                                int(abcd[0][0]): int(abcd[2][0])]

            if not mask_bin.flags.c_contiguous:
                mask_bin = np.ascontiguousarray(mask_bin)

            sl_mat = sl.Mat(
                width=mask_bin.shape[1],
                height=mask_bin.shape[0],
                mat_type=sl.MAT_TYPE.U8_C1,
                memory_type=sl.MEM.CPU
            )
            np.copyto(sl_mat.get_data(), mask_bin)
            sl_mats.append(sl_mat)
            obj.box_mask = sl_mat

        output.append(obj)

    return output


def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.4, iou_thres: float = 0.45) -> None:
    global image_net, exit_signal, run_signal, detections

    model = YOLO(weights)
    print("[YOLO] Model loaded")

    while not exit_signal:
        if run_signal:
            try:
                lock.acquire()
                if gl.GPU_ACCELERATION_AVAILABLE:
                    img = cp.asnumpy(cp.asarray(image_net)[:, :, :3])
                else:
                    img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)

                det = model.predict(
                    img,
                    save=False,
                    retina_masks=True,
                    imgsz=img_size,
                    conf=conf_thres,
                    iou=iou_thres,
                    verbose=False
                )[0]

                detections = detections_to_custom_masks_(det)

            finally:
                lock.release()
                run_signal = False

        sleep(0.01)


# -------------------------------
# 3D projection & drawing helpers
# -------------------------------
def project_3d_to_2d(points_3d, cam_params):
    fx = cam_params.fx
    fy = cam_params.fy
    cx = cam_params.cx
    cy = cam_params.cy

    pts_2d = []
    for p in points_3d:
        X, Y, Z = p
        if Z >= 0:
            pts_2d.append(None)
            continue
        Z = -Z
        u = int((X * fx / Z) + cx)
        v = int((-Y * fy / Z) + cy)
        pts_2d.append((u, v))

    return pts_2d


def mouse_click_callback(event, x, y, flags, param):
    global selected_object_id
    node = param  # pass the node when setting the callback

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which object was clicked
        if not hasattr(node, 'objects') or node.objects is None:
            return
            
        for obj in node.objects.object_list:
            if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                continue

            # Project 3D bbox to 2D
            pts_2d = project_3d_to_2d(obj.bounding_box, node.cam_params)
            if pts_2d is None or any(p is None for p in pts_2d):
                continue

            x_coords = [p[0] for p in pts_2d]
            y_coords = [p[1] for p in pts_2d]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected_object_id = obj.id
                print(f"[INFO] Selected object ID: {selected_object_id}")
                break


def draw_3d_bbox(image, bbox_3d, cam_params, color=(0, 255, 0)):
    if bbox_3d is None or len(bbox_3d) != 8:
        return

    pts_2d = project_3d_to_2d(bbox_3d, cam_params)
    if len(pts_2d) != 8 or any(p is None for p in pts_2d):
        return

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, j in edges:
        cv2.line(image, pts_2d[i], pts_2d[j], color, 2)


def draw_axes_3d(image, bbox_3d, cam_params, axis_length=0.1):
    if bbox_3d is None or len(bbox_3d) != 8:
        return

    center = np.mean(bbox_3d, axis=0)
    axes = {
        'x': center + np.array([axis_length, 0, 0]),
        'y': center + np.array([0, axis_length, 0]),
        'z': center + np.array([0, 0, axis_length])
    }

    center_2d = project_3d_to_2d([center], cam_params)[0]
    x_axis_2d = project_3d_to_2d([axes['x']], cam_params)[0]
    y_axis_2d = project_3d_to_2d([axes['y']], cam_params)[0]
    z_axis_2d = project_3d_to_2d([axes['z']], cam_params)[0]

    if None in (center_2d, x_axis_2d, y_axis_2d, z_axis_2d):
        return

    cv2.arrowedLine(image, center_2d, x_axis_2d, (0, 0, 255), 2)  # X red
    cv2.arrowedLine(image, center_2d, y_axis_2d, (0, 255, 0), 2)  # Y green
    cv2.arrowedLine(image, center_2d, z_axis_2d, (255, 0, 0), 2)  # Z blue


def bbox_to_quaternion(bbox_3d):
    if bbox_3d is None or len(bbox_3d) != 8:
        return 0, 0, 0, 1

    x_axis = ((bbox_3d[1] + bbox_3d[6]) / 2) - ((bbox_3d[0] + bbox_3d[7]) / 2)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = ((bbox_3d[3] + bbox_3d[2]) / 2) - ((bbox_3d[7] + bbox_3d[6]) / 2)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = ((bbox_3d[4] + bbox_3d[5]) / 2) - ((bbox_3d[0] + bbox_3d[1]) / 2)
    z_axis /= np.linalg.norm(z_axis)

    # Orthogonalize
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    rot_mat = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(rot_mat)
    return r.as_quat()  # x, y, z, w


# -------------------------------
# ROS2 Node
# -------------------------------
class ZEDObjectTFNode(Node):
    def __init__(self, weights, img_size, conf_thres, svo_file=None):
        super().__init__('zed_object_tf_broadcaster')
        global exit_signal

        # TF tools
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.svo_file = svo_file

        # Start YOLO thread
        self.yolo_thread = Thread(target=torch_thread_, args=(self.weights, self.img_size, self.conf_thres))
        self.yolo_thread.start()

        # Initialize ZED
        self.init_zed()

        # Timer to publish TFs
        self.timer = self.create_timer(1/30, self.publish_object_tfs)

    def init_zed(self):
        input_type = sl.InputType()
        if self.svo_file:
            input_type.set_from_svo_file(self.svo_file)

        self.init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.depth_maximum_distance = 10

        self.zed = sl.Camera()
        status = self.zed.open(self.init_params)
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(status)}")

        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        self.zed.enable_object_detection(obj_param)

        self.runtime_params = sl.RuntimeParameters()
        self.obj_runtime_params = sl.CustomObjectDetectionRuntimeParameters()
        self.objects = sl.Objects()
        self.image_left_tmp = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)

        self.camera_info = self.zed.get_camera_information()
        self.cam_params = self.camera_info.camera_configuration.calibration_parameters.left_cam

    def relay_link6_to_camera(self):
        """
        Republish the transform from link6->base_link as camera_link->base_link
        """
        try:
            # Lookup transform from base_link to link6
            tf_link6 = self.tf_buffer.lookup_transform('base_link', 'link6', rclpy.time.Time())
            
            # Create a new TransformStamped for camera_link
            t_camera = TransformStamped()
            t_camera.header.stamp = self.get_clock().now().to_msg()
            t_camera.header.frame_id = 'base_link'
            t_camera.child_frame_id = 'camera_link'

            # Copy translation and rotation from link6
            t_camera.transform.translation = tf_link6.transform.translation
            t_camera.transform.rotation = tf_link6.transform.rotation

            # Broadcast it
            self.tf_broadcaster.sendTransform(t_camera)

        except Exception as e:
            # Ignore errors if transform is not ready
            self.get_logger().warn(f"Transform lookup failed: {e}")

    def show_cv_image(self):
        if image_net is None:
            return

        frame_bgr = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)
        for obj in self.objects.object_list:
            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                color = (0, 255, 0)
                if obj.id == selected_object_id:
                    color = (0, 0, 255)
                draw_3d_bbox(frame_bgr, obj.bounding_box, self.cam_params, color)
                draw_axes_3d(frame_bgr, obj.bounding_box, self.cam_params)

        cv2.imshow("ZED Camera (ROS2)", frame_bgr)
        cv2.setMouseCallback("ZED Camera (ROS2)", mouse_click_callback, self)
        cv2.waitKey(1)

    def publish_object_tfs(self):
        self.relay_link6_to_camera()
        global image_net, run_signal, detections, lock, selected_object_id

        if self.zed.grab(self.runtime_params) <= sl.ERROR_CODE.SUCCESS:
            try:
                lock.acquire()
                self.zed.retrieve_image(self.image_left_tmp, sl.VIEW.LEFT, sl.MEM.CPU)
                image_net = self.image_left_tmp.get_data(memory_type=sl.MEM.CPU, deep_copy=False)
                run_signal = True
            finally:
                lock.release()

            self.show_cv_image()

            while run_signal:
                sleep(0.001)

            try:
                lock.acquire()
                self.zed.ingest_custom_mask_objects(detections)
            finally:
                lock.release()

            self.zed.retrieve_custom_objects(self.objects, self.obj_runtime_params)

            for obj in self.objects.object_list:
                if obj.id != selected_object_id:
                    continue

                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK and obj.bounding_box is not None:
                    if len(obj.bounding_box) != 8:
                        continue

                    # Compute center
                    center = np.mean(obj.bounding_box, axis=0)

                    # Transform from ZED camera to ROS camera_link
                    R_zed_to_ros = np.array([
                        [0, -1, 0],   # ROS X = ZED Z
                        [-1, 0, 0],   # ROS Y = ZED X
                        [0, 0, -1]    # ROS Z = ZED Y
                    ])
                    bbox_ros = (R_zed_to_ros @ obj.bounding_box.T).T

                    # Compute center
                    center = np.mean(bbox_ros, axis=0)

                    # Publish TF
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = "camera_link"
                    t.child_frame_id = f"object_{obj.id}"
                    t.transform.translation.x = float(center[0])
                    t.transform.translation.y = float(center[1])
                    t.transform.translation.z = float(center[2])
                    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = bbox_to_quaternion(bbox_ros)

                    self.tf_broadcaster.sendTransform(t)


# -------------------------------
# Main function
# -------------------------------
def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11m-seg.pt')
    parser.add_argument('--svo', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.4)
    args = parser.parse_args()

    node = ZEDObjectTFNode(weights=args.weights, img_size=args.img_size,
                            conf_thres=args.conf_thres, svo_file=args.svo)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        global exit_signal
        exit_signal = True
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
