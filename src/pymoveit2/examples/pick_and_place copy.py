#!/usr/bin/env python3
"""
TF-based motion followed by joint-space sequence + gripper.
Runs once on startup and exits.
"""

from threading import Thread
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

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
        # self.declare_parameter("quat_xyzw", [0.000, 0.786, -0.000, 0.619])
        self.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])

        self.target_frame = self.get_parameter("target_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.quat_xyzw = self.get_parameter("quat_xyzw").value

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
        # MoveIt2 – ARM
        # ----------------------------
        self.arm = MoveIt2(
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

        self.arm.max_velocity = 0.3
        self.arm.max_acceleration = 0.3

        # ----------------------------
        # MoveIt2 – GRIPPER
        # ----------------------------
        self.gripper = MoveIt2(
            node=self,
            joint_names=["joint7"],
            base_link_name="link6",
            end_effector_name="link7",
            group_name="gripper",
            callback_group=self.callback_group,
        )

        self.gripper.max_velocity = 0.5
        self.gripper.max_acceleration = 0.5

        # ----------------------------
        # Joint targets
        # ----------------------------
        self.predefined_joints = [1.573, 2.003, -0.754, 0.0, 0.0, 0.0]
        self.zero_joints = [0.0, 1.545, 0.0, 0.0, -1.220, 0.0]
        self.move_up_joints = [1.470, 2.054, -1.475, 0.0, 0.0, 0.0]

        self.gripper_open = [0.035]
        self.gripper_closed = [0.0]

        # ----------------------------
        # Start sequence immediately
        # ----------------------------
        Thread(target=self.execute_sequence, daemon=True).start()

    # ==========================================================
    # Motion sequence
    # ==========================================================
    def execute_sequence(self):
        try:
            self.get_logger().info("Waiting for TF...")

            transform = None
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
                return

            # ----------------------------
            # 0. Open gripper
            # ----------------------------
            self.get_logger().info("Opening gripper")
            self.gripper.move_to_configuration(self.gripper_open)
            self.gripper.wait_until_executed()

            # ----------------------------
            # 1. Move to pose
            # ----------------------------
            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z + 0.0,
            ]

            # Optional override
            # position = [0.355, 0.000, 0.171]

            self.get_logger().info(f"Moving to pose: {position}")
            self.arm.move_to_pose(
                position=position,
                quat_xyzw=self.quat_xyzw,
                cartesian=False,
            )
            self.arm.wait_until_executed()

            time.sleep(2.0)

            # ----------------------------
            # 2. Pick
            # ----------------------------
            self.get_logger().info("Closing gripper")
            self.gripper.move_to_configuration(self.gripper_closed)
            self.gripper.wait_until_executed()

            self.get_logger().info("Moving up")
            self.arm.move_to_configuration(self.move_up_joints)
            self.arm.wait_until_executed()

            # ----------------------------
            # 3. Place
            # ----------------------------
            self.get_logger().info("Moving to predefined pose")
            self.arm.move_to_configuration(self.predefined_joints)
            self.arm.wait_until_executed()

            self.get_logger().info("Opening gripper")
            self.gripper.move_to_configuration(self.gripper_open)
            self.gripper.wait_until_executed()

            # ----------------------------
            # 4. Home
            # ----------------------------
            self.get_logger().info("Moving to zero")
            self.arm.move_to_configuration(self.zero_joints)
            self.arm.wait_until_executed()

            self.get_logger().info("Sequence complete — shutting down")

        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")

        finally:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TfThenJointSequence()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.spin()


if __name__ == "__main__":
    main()
