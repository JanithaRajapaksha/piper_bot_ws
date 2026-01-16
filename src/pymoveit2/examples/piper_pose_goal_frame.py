#!/usr/bin/env python3
"""
ROS2 node that listens to detection_events and launches
`ros2 run pymoveit2 piper_pose_goal_frame.py` while streaming logs.
"""

import subprocess
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ShellTriggerNode(Node):
    def __init__(self):
        super().__init__("shell_trigger_node")

        self.process = None
        self.lock = threading.Lock()  # prevent concurrent launches

        self.create_subscription(
            String,
            "detection_events",
            self.detection_callback,
            10,
        )

        self.get_logger().info(
            "ShellTriggerNode listening on 'detection_events'"
        )

    def detection_callback(self, msg: String):
        if msg.data != "point":
            return

        with self.lock:
            # If process is still running, ignore trigger
            if self.process and self.process.poll() is None:
                self.get_logger().warn("Motion script already running. Ignoring trigger.")
                return

            self.get_logger().info("Trigger received. Launching MoveIt script.")

            try:
                # Start subprocess with stdout/stderr piped
                self.process = subprocess.Popen(
                    ["ros2", "run", "pymoveit2", "pick_and_place.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,  # output as text
                )

                # Start a thread to read output
                threading.Thread(target=self.stream_output, daemon=True).start()

            except Exception as e:
                self.get_logger().error(f"Failed to launch script: {e}")

    def stream_output(self):
        """Reads the subprocess output line by line and prints to ROS log."""
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            line = line.strip()
            if line:
                self.get_logger().info(f"[MoveIt] {line}")

        self.process.stdout.close()
        self.process.wait()
        self.get_logger().info("MoveIt script finished.")


def main():
    rclpy.init()
    node = ShellTriggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down shell trigger node.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
