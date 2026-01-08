#!/usr/bin/env python3
"""
Move to a TF frame position, then move to a predefined pose using pymoveit2.
Second motion executes ONLY if first motion succeeds.
"""

from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from pymoveit2 import MoveIt2


def main():
    rclpy.init()

    node = Node("pose_goal_from_tf")

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    node.declare_parameter("target_frame", "object_2")
    node.declare_parameter("base_frame", "base_link")

    node.declare_parameter("quat_xyzw", [0.0, 0.7, 0.0, 0.714])

    node.declare_parameter("post_position", [0.3, 0.0, 0.5])
    node.declare_parameter("post_quat_xyzw", [0.0, 0.7, 0.0, 0.714])

    node.declare_parameter("cartesian", False)
    node.declare_parameter("cartesian_max_step", 0.0025)
    node.declare_parameter("cartesian_fraction_threshold", 0.0)

    # --------------------------------------------------
    # TF
    # --------------------------------------------------
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # --------------------------------------------------
    # MoveIt2
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Executor
    # --------------------------------------------------
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # --------------------------------------------------
    # Read parameters
    # --------------------------------------------------
    target_frame = node.get_parameter("target_frame").value
    base_frame = node.get_parameter("base_frame").value
    quat_xyzw = node.get_parameter("quat_xyzw").value

    post_position = node.get_parameter("post_position").value
    post_quat_xyzw = node.get_parameter("post_quat_xyzw").value

    cartesian = node.get_parameter("cartesian").value
    cartesian_max_step = node.get_parameter("cartesian_max_step").value
    cartesian_fraction_threshold = node.get_parameter(
        "cartesian_fraction_threshold"
    ).value

    # --------------------------------------------------
    # TF lookup
    # --------------------------------------------------
    node.get_logger().info("⏳ Waiting for TF...")

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
            rate.sleep()

    if transform is None:
        node.get_logger().error("❌ TF lookup failed")
        rclpy.shutdown()
        executor_thread.join()
        return

    position = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ]

    position[2] += 0.10  # approach offset

    # --------------------------------------------------
    # First move (TF pose)
    # --------------------------------------------------
    node.get_logger().info(f"➡️ Moving to TF pose: {position}")

    moveit2.move_to_pose(
        position=position,
        quat_xyzw=quat_xyzw,
        cartesian=cartesian,
        cartesian_max_step=cartesian_max_step,
        cartesian_fraction_threshold=cartesian_fraction_threshold,
    )

    success = moveit2.wait_until_executed()

    if not success:
        node.get_logger().error("❌ Failed to reach TF pose. Aborting.")
        rclpy.shutdown()
        executor_thread.join()
        return

    node.get_logger().info("✅ Reached TF pose successfully")

    # --------------------------------------------------
    # Second move (predefined pose)
    # --------------------------------------------------
    node.get_logger().info(f"➡️ Moving to predefined pose: {post_position}")

    moveit2.move_to_pose(
        position=post_position,
        quat_xyzw=post_quat_xyzw,
        cartesian=False,
    )

    success = moveit2.wait_until_executed()

    if not success:
        node.get_logger().error("❌ Failed to reach predefined pose")
    else:
        node.get_logger().info("✅ Predefined pose reached successfully")

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
