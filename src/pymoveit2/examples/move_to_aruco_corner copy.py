#!/usr/bin/env python3
"""
Continuously move to a single TF frame using pymoveit2
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

    node = Node("tf_pose_goal_loop")

    # ----------------------------
    # Parameters
    # ----------------------------
    target_frame = node.declare_parameter("target_frame", "clicked_point_frame").value
    base_frame = node.declare_parameter("base_frame", "base_link").value
    quat_xyzw = node.declare_parameter("quat_xyzw", [0.0, 0.676, 0.0, 0.737]).value
    cartesian = node.declare_parameter("cartesian", False).value
    cartesian_max_step = node.declare_parameter("cartesian_max_step", 0.0025).value
    cartesian_fraction_threshold = node.declare_parameter("cartesian_fraction_threshold", 0.0).value
    approach_offset = node.declare_parameter("approach_offset", 0.10).value  # 10 cm above object

    # ----------------------------
    # TF setup
    # ----------------------------
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # ----------------------------
    # MoveIt2 setup
    # ----------------------------
    callback_group = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=["joint1","joint2","joint3","joint4","joint5","joint6"],
        base_link_name="base_link",
        end_effector_name="link6",
        group_name="arm",
        callback_group=callback_group,
    )
    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # ----------------------------
    # Executor (needed for TF)
    # ----------------------------
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = node.create_rate(10)

    node.get_logger().info(f"Listening to TF: {target_frame}")

    # ----------------------------
    # Main loop
    # ----------------------------
    while rclpy.ok():
        transform: TransformStamped = None
        # Wait for TF
        while rclpy.ok():
            try:
                transform = tf_buffer.lookup_transform(
                    base_frame,
                    target_frame,
                    rclpy.time.Time()
                )
                break
            except Exception:
                rate.sleep()

        if transform is None:
            node.get_logger().error(f"TF lookup failed for {target_frame}")
            continue

        position = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z + approach_offset
        ]

        node.get_logger().info(f"Moving to {target_frame}: pos={position}, quat={quat_xyzw}")

        # Move robot
        moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
            cartesian_max_step=cartesian_max_step,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )
        moveit2.wait_until_executed()

        node.get_logger().info(f"Reached {target_frame}, waiting for next loop...")

    # ----------------------------
    # Shutdown
    # ----------------------------
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
