#!/usr/bin/env python3
"""
Move to a TF frame position with fixed orientation using pymoveit2
"""

from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from pymoveit2 import MoveIt2


def main():
    rclpy.init()

    # ----------------------------
    # Node
    # ----------------------------
    node = Node("pose_goal_from_tf")

    # ----------------------------
    # Parameters
    # ----------------------------
    node.declare_parameter("target_frame", "clicked_point_frame")
    node.declare_parameter("base_frame", "base_link")

    # Fixed orientation (kept constant)
    # node.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])
    node.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])


    node.declare_parameter("cartesian", False)
    node.declare_parameter("cartesian_max_step", 0.1)
    node.declare_parameter("cartesian_fraction_threshold", 0.0)

    # ----------------------------
    # TF
    # ----------------------------
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # ----------------------------
    # MoveIt2
    # ----------------------------
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        base_link_name="base_link",
        end_effector_name="link6",
        group_name="arm",
        callback_group=callback_group,
    )

    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # ----------------------------
    # Executor (REQUIRED for TF)
    # ----------------------------
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    node.get_logger().info("Waiting for TF...")

    # ----------------------------
    # Get parameters
    # ----------------------------
    target_frame = node.get_parameter("target_frame").value
    base_frame = node.get_parameter("base_frame").value
    quat_xyzw = node.get_parameter("quat_xyzw").value

    cartesian = node.get_parameter("cartesian").value
    cartesian_max_step = node.get_parameter("cartesian_max_step").value
    cartesian_fraction_threshold = node.get_parameter(
        "cartesian_fraction_threshold"
    ).value

    # ----------------------------
    # Lookup TF (position only)
    # ----------------------------
    transform: TransformStamped = None
    rate = node.create_rate(10)

    while rclpy.ok():
        try:
            transform = tf_buffer.lookup_transform(
                base_frame,
                target_frame,
                rclpy.time.Time()
            )
            break
        except Exception:
            continue

    if transform is None:
        node.get_logger().error("TF lookup failed")
        rclpy.shutdown()
        return

    position = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ]

    # Optional approach offset (recommended for grasping)
    position[2] += 0.10  # 10 cm above object

    node.get_logger().info(
        f"Moving to TF position {position} with fixed quat {quat_xyzw}"
    )

    # ----------------------------
    # Move
    # ----------------------------
    moveit2.move_to_pose(
        position=position,
        quat_xyzw=quat_xyzw,
        cartesian=cartesian,
        cartesian_max_step=cartesian_max_step,
        cartesian_fraction_threshold=cartesian_fraction_threshold,
    )

    moveit2.wait_until_executed()

    # ----------------------------
    # Shutdown
    # ----------------------------
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
