import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RGBDSubscriber(Node):
    def __init__(self):
        super().__init__('rgbd_subscriber')
        self.bridge = CvBridge()

        # --- Mouse Click Handling ---
        self.clicked_point = None
        self.depth_text = ""
        self.window_name = "RGB and Depth Side-by-Side"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        # --- End Mouse Click Handling ---

        # Keep track of latest frames
        self.latest_rgb_frame = None
        self.latest_depth_frame = None

        # Create subscribers
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Create a timer to process and display images
        self.timer = self.create_timer(0.05, self.display_images) # 20 Hz

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events to display depth at a point."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is on the RGB frame (the left side)
            if self.latest_rgb_frame is not None and x < self.latest_rgb_frame.shape[1]:
                self.clicked_point = (x, y)

                # Ensure depth data is available
                if self.latest_depth_frame is not None:
                    # Get depth value at the clicked point
                    depth_raw = self.latest_depth_frame[y, x]

                    # Convert to meters if needed (assuming uint16 is in mm)
                    if self.latest_depth_frame.dtype == np.uint16:
                        depth_m = float(depth_raw) * 0.001
                    else:
                        depth_m = float(depth_raw) # Assuming it's already in meters

                    self.depth_text = f"Depth: {depth_m:.3f}m"
                    self.get_logger().info(
                        f"Clicked at ({x}, {y}) on RGB image. {self.depth_text}"
                    )
            else:
                # Click was on the depth side or images not ready
                self.clicked_point = None
                self.depth_text = ""


    def rgb_callback(self, msg):
        """Callback for RGB image topic."""
        try:
            # Convert ROS Image -> OpenCV
            self.latest_rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        """Callback for depth image topic."""
        try:
            # Convert ROS Image -> OpenCV (passthrough to keep original depth info)
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def display_images(self):
        """Process and display RGB and depth images side-by-side."""
        if self.latest_rgb_frame is None or self.latest_depth_frame is None:
            # Wait until both frames are available
            return

        # --- Process Depth Image for Visualization ---
        depth_image = self.latest_depth_frame

        if depth_image.dtype == np.uint16:
            depth_m = depth_image.astype(np.float32) * 0.001
        else:
            depth_m = depth_image.astype(np.float32)

        MIN_DEPTH, MAX_DEPTH = 0.3, 3.0
        depth_clipped = np.clip(depth_m, MIN_DEPTH, MAX_DEPTH)
        depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # --- Combine Images ---
        rgb_frame = self.latest_rgb_frame
        h_rgb, w_rgb, _ = rgb_frame.shape
        h_depth, w_depth, _ = depth_colormap.shape

        if h_rgb != h_depth or w_rgb != w_depth:
            depth_colormap = cv2.resize(depth_colormap, (w_rgb, h_rgb))

        combined_image = np.hstack((rgb_frame, depth_colormap))

        # --- Draw Click Information ---
        # Make a copy to draw on, so we don't alter the original frames
        display_frame = combined_image.copy()
        if self.clicked_point:
            # Draw a circle at the clicked point
            cv2.circle(display_frame, self.clicked_point, 5, (0, 255, 0), -1)
            # Position the text near the circle
            text_pos = (self.clicked_point[0] + 15, self.clicked_point[1])
            # Add the depth text
            cv2.putText(display_frame, self.depth_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, self.depth_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Display ---
        cv2.imshow(self.window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = RGBDSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()