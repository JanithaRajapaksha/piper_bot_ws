#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
import time

class TrajectorySniffer(Node):
    def __init__(self):
        super().__init__('trajectory_sniffer')

        # Action server to sniff trajectories
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_cb
        )

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, '/joint_ctrl_cmd', 10)

        # Current joint state
        self.joint_state = JointState()
        self.joint_state.name = []  # Will be filled from trajectory
        self.joint_state.position = []

        self.get_logger().info("TrajectorySniffer initialized")

    def execute_cb(self, goal_handle):
        traj = goal_handle.request.trajectory

        # Initialize joint state names if empty
        if not self.joint_state.name:
            self.joint_state.name = traj.joint_names
            self.joint_state.position = [0.0] * len(traj.joint_names)

        self.get_logger().info(f"Received trajectory with {len(traj.points)} points")

        # Publish each trajectory point
        for i, point in enumerate(traj.points):
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state.position = point.positions
            self.joint_pub.publish(self.joint_state)

            self.get_logger().info(f"Point {i} published: {point.positions}")

            # Wait for the point's time_from_start
            # t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            # time.sleep(t)  # simple blocking delay

        # Abort the goal (we're just publishing, not controlling hardware)
        goal_handle.abort()
        return FollowJointTrajectory.Result()

def main():
    rclpy.init()
    node = TrajectorySniffer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
