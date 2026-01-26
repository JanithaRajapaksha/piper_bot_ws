#!/usr/bin/env python3
"""
TF-based motion sequence for multiple cleaning points + gripper.
Processes all frames cleaning_sequence_point0, cleaning_sequence_point1, ...
Then returns to zero joints at the very end.
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
        self.declare_parameter("target_frame_prefix", "cleaning_sequence_point")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("quat_xyzw", [0.000, 0.786, -0.000, 0.619])
        # self.declare_parameter("quat_xyzw", [0.000, 0.676, -0.000, 0.737])

        self.target_frame_prefix = self.get_parameter("target_frame_prefix").value
        self.base_frame = self.get_parameter("base_frame").value
        self.quat_xyzw = self.get_parameter("quat_xyzw").value

        # ----------------------------
        # Motion tuning parameters (pose)
        # ----------------------------
        self.declare_parameter("tolerance_position", 0.001)
        self.declare_parameter("tolerance_orientation", 0.01)
        self.declare_parameter("weight_position", 0.5)
        self.declare_parameter("weight_orientation", 0.0)

        self.tolerance_position = self.get_parameter("tolerance_position").value
        self.tolerance_orientation = self.get_parameter("tolerance_orientation").value
        self.weight_position = self.get_parameter("weight_position").value
        self.weight_orientation = self.get_parameter("weight_orientation").value

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
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
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
        # Joint targets & gripper values
        # ----------------------------
        self.predefined_joints = [1.573, 2.003, -0.754, 0.0, 0.0, 0.0]   # place/intermediate pose
        self.zero_joints        = [0.0, 1.545, 0.0, 0.0, -1.220, 0.0]    # home
        self.move_up_joints     = [1.470, 2.054, -1.475, 0.0, 0.0, 0.0]  # lift after pick
        self.gripper_open       = [0.035]
        self.gripper_closed     = [0.0]

        # ----------------------------
        # Start sequence in background thread
        # ----------------------------
        Thread(target=self.execute_sequence, daemon=True).start()

    # ==========================================================
    # Motion sequence – handles multiple points
    # ==========================================================
    def execute_sequence(self):
        try:
            self.get_logger().info("Waiting for at least one cleaning point TF...")

            # Collect all available cleaning point transforms
            transform_list = []
            index = 0
            timeout_start = time.time()
            max_wait_sec = 1.0  # adjust if needed

            while rclpy.ok() and (time.time() - timeout_start < max_wait_sec):
                target_frame = f"{self.target_frame_prefix}{index}"
                try:
                    tf_msg = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        target_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.3)
                    )
                    transform_list.append((target_frame, tf_msg))
                    self.get_logger().info(f"Found point: {target_frame}")
                    index += 1
                    timeout_start = time.time()  # reset timeout on success
                except Exception:
                    # No more points? Wait a bit longer in case more are coming
                    time.sleep(0.2)

            if not transform_list:
                self.get_logger().error("No cleaning points found (timeout). Shutting down.")
                return

            self.get_logger().info(f"Processing {len(transform_list)} cleaning points...")



            # Process each point in sequence
            for i, (frame_name, tf_msg) in enumerate(transform_list):
                self.get_logger().info(f"--- Point {i+1}/{len(transform_list)}: {frame_name} ---")

                position = [
                    tf_msg.transform.translation.x,
                    tf_msg.transform.translation.y,
                    tf_msg.transform.translation.z + 0.01,  # approach from 1 cm above
                ]

                # 1. Move above the point
                self.get_logger().info(f"Moving above: {position}")
                self.arm.move_to_pose(
                    position=position,
                    quat_xyzw=self.quat_xyzw,
                    tolerance_position=self.tolerance_position,
                    tolerance_orientation=self.tolerance_orientation,
                    weight_position=self.weight_position,
                    weight_orientation=self.weight_orientation,
                    cartesian=False,
                )
                self.arm.wait_until_executed()
                time.sleep(0.5)  # small settle time

              



                time.sleep(1.0)  # pause between points if desired

            # After ALL points are processed → final home
            self.get_logger().info("All points processed → returning to zero/home")
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