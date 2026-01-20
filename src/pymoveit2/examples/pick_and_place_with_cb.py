#!/usr/bin/env python3
"""
TF-based motion followed by joint-space sequence.
Terminates automatically after completing the motion.

Trigger:
  Publish "point" to /detection_events
"""

from threading import Thread
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from pymoveit2 import MoveIt2


class TfThenJointSequence(Node):
    def __init__(self):
        super().__init__("tf_then_joint_sequence")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("target_frame", "clicked_point_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])

        self.target_frame = self.get_parameter("target_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.quat_xyzw = self.get_parameter("quat_xyzw").value

        # ----------------------------
        # Flags
        # ----------------------------
        self.already_triggered = False

        # ----------------------------
        # Callback group
        # ----------------------------
        self.callback_group = ReentrantCallbackGroup()

        # ----------------------------
        # TF
        # ----------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------------
        # MoveIt2
        # ----------------------------
        self.moveit2 = MoveIt2(
            node=self,
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
            callback_group=self.callback_group,
        )

        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        # ----------------------------
        # Joint targets
        # ----------------------------
        self.predefined_joints = [1.573, 2.003, -0.754, 0.0, 0.0, 0.0]
        self.zero_joints = [0.0] * 6

        # ----------------------------
        # Subscriber
        # ----------------------------
        self.sub = self.create_subscription(
            String,
            "detection_events",
            self.detection_callback,
            10,
        )

        self.get_logger().info("Waiting for 'point' on /detection_events...")

    # ==========================================================
    # Detection trigger
    # ==========================================================
    def detection_callback(self, msg: String):
        if self.already_triggered:
            return

        if msg.data != "point":
            return

        self.already_triggered = True
        self.get_logger().info("Detection received → starting sequence")

        Thread(target=self.execute_sequence).start()

    # ==========================================================
    # Motion sequence
    # ==========================================================
    def execute_sequence(self):
        try:
            self.get_logger().info("Waiting for TF...")

            transform: TransformStamped = None
            while rclpy.ok():
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        self.target_frame,
                        rclpy.time.Time(),
                    )
                    break
                except Exception:
                    time.sleep(0.1)

            if transform is None:
                self.get_logger().error("TF lookup failed")
                rclpy.shutdown()
                return

            # ----------------------------
            # 1. Move to TF pose
            # ----------------------------
            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z + 0.10,
            ]

            self.get_logger().info(f"Moving to TF pose: {position}")
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=self.quat_xyzw,
                cartesian=False,
            )
            self.moveit2.wait_until_executed()

            # ----------------------------
            # 2. Wait
            # ----------------------------
            self.get_logger().info("Waiting at TF pose...")
            time.sleep(2.0)

            # ----------------------------
            # 3. Predefined joint pose
            # ----------------------------
            self.get_logger().info("Moving to predefined joint pose")
            self.moveit2.move_to_configuration(self.predefined_joints)
            self.moveit2.wait_until_executed()

            # ----------------------------
            # 4. Zero position
            # ----------------------------
            self.get_logger().info("Moving to zero position")
            self.moveit2.move_to_configuration(self.zero_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Sequence complete")

        finally:
            # Clean shutdown (IMPORTANT)
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TfThenJointSequence()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)

    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
