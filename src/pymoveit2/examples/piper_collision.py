#!/usr/bin/env python3
"""
Robust example for adding/removing collision objects in MoveIt2 with mesh preloading.
Ensures faces are correctly converted to uint32 for ROS2 shape_msgs.
"""

from os import path
from threading import Thread
import rclpy
import trimesh
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
import numpy as np
from pymoveit2 import MoveIt2

# Default mesh (Suzanne) shipped with pymoveit2
DEFAULT_EXAMPLE_MESH = path.join(
    path.dirname(path.realpath(__file__)), "assets", "suzanne.stl"
)

def load_mesh(filepath: str) -> trimesh.Trimesh:
    """
    Load a mesh using trimesh and fix vertex & face types for MoveIt2.
    """
    mesh = trimesh.load_mesh(filepath, force='mesh')

    # Convert vertices to float32
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)

    # Convert faces to plain uint32 array
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # Handle structured arrays if trimesh returned them
    if faces.dtype.names is not None:
        faces = np.stack([faces[name] for name in faces.dtype.names], axis=1)

    # Ensure shape is (-1,3) and dtype uint32
    mesh.faces = faces.reshape(-1, 3).astype(np.uint32)

    return mesh

def main():
    rclpy.init()
    node = Node("ex_collision_mesh")

    # Declare ROS2 parameters
    node.declare_parameter("filepath", "")
    node.declare_parameter("action", "add")  # add, remove, move
    node.declare_parameter("position", [0.5, 0.0, 0.5])
    node.declare_parameter("quat_xyzw", [0.0, 0.0, -0.7071, 0.7071])
    node.declare_parameter("scale", [1.0, 1.0, 1.0])
    node.declare_parameter("preload_mesh", False)

    # Callback group for MoveIt2
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=["joint1","joint2","joint3","joint4","joint5","joint6"],
        base_link_name="base_link",
        end_effector_name="link6",
        group_name="arm",
        callback_group=callback_group,
    )

    # Spin node in background thread
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.create_rate(1.0).sleep()  # short wait

    # Read parameters
    filepath = node.get_parameter("filepath").get_parameter_value().string_value
    action = node.get_parameter("action").get_parameter_value().string_value
    position = list(node.get_parameter("position").get_parameter_value().double_array_value)
    quat_xyzw = list(node.get_parameter("quat_xyzw").get_parameter_value().double_array_value)
    scale = list(node.get_parameter("scale").get_parameter_value().double_array_value)
    preload_mesh = node.get_parameter("preload_mesh").get_parameter_value().bool_value

    # Use default mesh if not provided
    if not filepath:
        node.get_logger().info(f"Using default mesh: {DEFAULT_EXAMPLE_MESH}")
        filepath = DEFAULT_EXAMPLE_MESH

    # Check mesh file exists
    if not path.exists(filepath):
        node.get_logger().error(f"Mesh file '{filepath}' does not exist")
        rclpy.shutdown()
        exit(1)

    object_id = path.basename(filepath).split(".")[0]

    # Preload mesh if requested
    mesh = None
    if preload_mesh:
        mesh = load_mesh(filepath)
        filepath = None  # ensure MoveIt2 uses preloaded mesh

    # Execute action
    try:
        if action == "add":
            node.get_logger().info(f"Adding collision mesh '{object_id}' at {position}")
            moveit2.add_collision_mesh(
                filepath=filepath,
                id=object_id,
                position=position,
                quat_xyzw=quat_xyzw,
                scale=scale,
                mesh=mesh
            )
        elif action == "remove":
            node.get_logger().info(f"Removing collision mesh '{object_id}'")
            moveit2.remove_collision_object(id=object_id)
        elif action == "move":
            node.get_logger().info(f"Moving collision mesh '{object_id}' to {position}")
            moveit2.move_collision(id=object_id, position=position, quat_xyzw=quat_xyzw)
        else:
            raise ValueError(f"Unknown action '{action}'")
    except Exception as e:
        node.get_logger().error(f"Error handling collision mesh: {e}")

    # Shutdown
    rclpy.shutdown()
    executor_thread.join()

if __name__ == "__main__":
    main()
