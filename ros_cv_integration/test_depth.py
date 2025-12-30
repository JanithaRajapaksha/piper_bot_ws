import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthSubscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

    def depth_callback(self, msg):
        # Convert ROS Image → OpenCV (keep depth)
        depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

        # Convert to meters if needed
        if depth_image.dtype == np.uint16:
            depth_m = depth_image.astype(np.float32) * 0.001  # mm → m
        else:
            depth_m = depth_image.astype(np.float32)

        # ---- Visualization parameters ----
        MIN_DEPTH = 0.3   # meters
        MAX_DEPTH = 3.0   # meters

        # Clip depth range
        depth_clipped = np.clip(depth_m, MIN_DEPTH, MAX_DEPTH)

        # Normalize to 0–255
        depth_norm = ((depth_clipped - MIN_DEPTH) /
                    (MAX_DEPTH - MIN_DEPTH) * 255.0)

        depth_norm = depth_norm.astype(np.uint8)

        # Apply color map
        depth_colormap = cv2.applyColorMap(
            depth_norm,
            cv2.COLORMAP_JET
        )

        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = DepthSubscriber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
