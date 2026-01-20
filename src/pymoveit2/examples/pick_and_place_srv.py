#!/usr/bin/env python3

from threading import Thread, Lock
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from pymoveit2 import MoveIt2


class TfThenJointService(Node):
    def __init__(self):
        super().__init__("tf_then_joint_service")

        self.callback_group = ReentrantCallbackGroup()
        self.busy_lock = Lock()

        # Frames & pose
        self.target_frame = "clicked_point_frame"
        self.base_frame = "base_link"
        self.quat_xyzw = [0.000, 0.676, -0.000, 0.737]

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6"
            ],
            base_link_name="base_link",
            end_effector_name="link6",
            group_name="arm",
            callback_group=self.callback_group,
        )

        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self.predefined_joints = [1.573, 2.003, -0.754, 0.0, 0.0, 0.0]
        self.zero_joints = [0.0] * 6

        # Service
        self.srv = self.create_service(
            Trigger,
            "run_tf_sequence",
            self.handle_service,
            callback_group=self.callback_group,
        )

        self.get_logger().info("TF motion service ready")

    # ================================
    # Service callback
    # ================================
    def handle_service(self, request, response):
        if not self.busy_lock.acquire(blocking=False):
            response.success = False
            response.message = "Robot is busy"
            return response

        Thread(target=self.execute_sequence, daemon=True).start()

        response.success = True
        response.message = "Sequence started"
        return response

    # ==========================================================
    # Motion sequence with try/except per step
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

            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z + 0.10,
            ]

            # ----------------------------
            # 1. Move to TF pose
            # ----------------------------
            try:
                self.get_logger().info(f"Moving to TF pose: {position}")
                self.moveit2.move_to_pose(
                    position=position,
                    quat_xyzw=self.quat_xyzw,
                    cartesian=False,
                )
                self.moveit2.wait_until_executed()
            except Exception as e:
                self.get_logger().error(f"Failed to move to TF pose: {e}")

            # ----------------------------
            # 2. Wait
            # ----------------------------
            try:
                self.get_logger().info("Waiting at TF pose...")
                time.sleep(2.0)
            except Exception as e:
                self.get_logger().warn(f"Wait interrupted: {e}")

            # ----------------------------
            # 3. Predefined joint pose
            # ----------------------------
            try:
                self.get_logger().info("Moving to predefined joint pose")
                self.moveit2.move_to_configuration(self.predefined_joints)
                self.moveit2.wait_until_executed()
            except Exception as e:
                self.get_logger().error(f"Failed to move to predefined joints: {e}")

            # ----------------------------
            # 4. Zero position
            # ----------------------------
            try:
                self.get_logger().info("Moving to zero position")
                self.moveit2.move_to_configuration(self.zero_joints)
                self.moveit2.wait_until_executed()
            except Exception as e:
                self.get_logger().error(f"Failed to move to zero joints: {e}")

            self.get_logger().info("Sequence complete")

        finally:
            # Clean shutdown
            rclpy.shutdown()



def main():
    rclpy.init()

    node = TfThenJointService()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
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
