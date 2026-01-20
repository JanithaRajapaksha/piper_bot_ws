#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time


class MotorSerialNode(Node):
    def __init__(self):
        super().__init__('motor_serial_node')

        # ---- Parameters ----
        self.declare_parameter('port', '/dev/ttyUSB0')   # Linux default
        self.declare_parameter('baud', 9600)

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value

        # ---- Serial ----
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # ESP32 reset delay
            self.get_logger().info(f"Connected to ESP32 on {port}")
        except Exception as e:
            self.get_logger().error(f"Serial connection failed: {e}")
            raise e

        # ---- Subscriber ----
        self.sub = self.create_subscription(
            String,
            'motor_cmd',
            self.cmd_callback,
            10
        )

        self.get_logger().info("Motor serial node ready (send '1' or '0')")

    def cmd_callback(self, msg: String):
        cmd = msg.data.strip()

        if cmd not in ['1', '0']:
            self.get_logger().warn(f"Ignoring invalid command: {cmd}")
            return

        try:
            # Send command + newline (safe for Windows/Linux)
            self.ser.write((cmd + '\n').encode())
            self.get_logger().info(f"Sent command to motor: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MotorSerialNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
