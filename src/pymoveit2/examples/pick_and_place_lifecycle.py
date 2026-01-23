#!/usr/bin/env python3
"""
Lifecycle TF-based motion followed by joint-space sequence.
Starts when lifecycle is ACTIVATED.
Terminates automatically after completing the motion.
"""

from threading import Thread
import time
import rclpy
from lifecycle_msgs.msg import Transition
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from pymoveit2 import MoveIt2


class TfThenJointSequence(LifecycleNode):
    def __init__(self):
        super().__init__("tf_then_joint_sequence")

        # Parameters (declared early)
        self.declare_parameter("target_frame", "clicked_point_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])

        self.callback_group = ReentrantCallbackGroup()

        self.tf_buffer = None
        self.tf_listener = None
        self.moveit2 = None

        self.predefined_joints = [1.573, 2.003, -0.754, 0.0, 0.0, 0.0]
        self.zero_joints = [0.0] * 6

        self.target_frame = self.get_parameter("target_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.quat_xyzw = self.get_parameter("quat_xyzw").value
        # MoveIt2
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

    # ==========================================================
    # Lifecycle callbacks
    # ==========================================================
    def on_configure(self, state):
        self.get_logger().info("Configuring node...")

        

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        

        self.get_logger().info("Node configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info("Node activated → starting motion")
        Thread(target=self.execute_sequence, daemon=True).start()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info("Cleaning up node resources")
        self.moveit2 = None
        self.tf_listener = None
        self.tf_buffer = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info("Shutting down lifecycle node")
        return TransitionCallbackReturn.SUCCESS

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
                return

            # 1. Move to TF pose
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

            # 2. Wait
            time.sleep(2.0)

            # 3. Predefined joint pose
            self.moveit2.move_to_configuration(self.predefined_joints)
            self.moveit2.wait_until_executed()

            # 4. Zero position
            self.moveit2.move_to_configuration(self.zero_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Sequence complete → shutting down lifecycle")

            self.trigger_deactivate()




        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")
            self.trigger_shutdown()


def main():
    rclpy.init()
    node = TfThenJointSequence()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # rclpy.shutdown()


if __name__ == "__main__":
    main()
