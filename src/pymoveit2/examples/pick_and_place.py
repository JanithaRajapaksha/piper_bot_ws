#!/usr/bin/env python3

from threading import Thread, Lock
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
        # Internal state
        # ----------------------------
        self.motion_running = False
        self.lock = Lock()

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("target_frame", "clicked_point_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])

        # ----------------------------
        # TF
        # ----------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------------
        # MoveIt2
        # ----------------------------
        self.callback_group = ReentrantCallbackGroup()

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
        self.predefined_joints = [
            1.573,
            2.003,
            -0.754,
            0.0,
            0.0,
            0.0,
        ]

        self.zero_joints = [0.0] * 6

        # ----------------------------
        # Detection trigger
        # ----------------------------
        self.create_subscription(
            String,
            "detection_events",
            self.detection_callback,
            10,
        )

        self.get_logger().info("Waiting for 'point' on detection_events")

    # ==========================================================
    # Detection callback
    # ==========================================================
    def detection_callback(self, msg: String):
        if msg.data != "point":
            return

        with self.lock:
            if self.motion_running:
                self.get_logger().warn("Motion already running. Ignoring trigger.")
                return
            self.motion_running = True

        self.get_logger().info("Detection received → starting motion")
        Thread(target=self.execute_sequence, daemon=True).start()

    # ==========================================================
    # Motion sequence
    # ==========================================================
    def execute_sequence(self):
        try:
            target_frame = self.get_parameter("target_frame").value
            base_frame = self.get_parameter("base_frame").value
            quat_xyzw = self.get_parameter("quat_xyzw").value

            self.get_logger().info("Waiting for TF...")

            transform: TransformStamped = None
            rate = self.create_rate(10)

            while rclpy.ok():
                try:
                    transform = self.tf_buffer.lookup_transform(
                        base_frame,
                        target_frame,
                        rclpy.time.Time(),
                    )
                    break
                except Exception:
                    rate.sleep()

            if transform is None:
                self.get_logger().error("TF lookup failed")
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
                quat_xyzw=quat_xyzw,
                cartesian=False,
            )
            self.moveit2.wait_until_executed()

            # ----------------------------
            # 2. WAIT
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

            rclpy.shutdown()   # <-- ADD THIS

        finally:
            with self.lock:
                self.motion_running = False


def main():
    rclpy.init()
    node = TfThenJointSequence()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
