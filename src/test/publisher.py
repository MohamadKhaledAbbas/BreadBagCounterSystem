# File: src/image_publisher.py

import rclpy
from rclpy.node import Node
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image
import time


class FramePublisherNode(Node):
    def __init__(self):
        super().__init__('frame_publisher_node')

        # ROS 2 Publisher: Publishes sensor_msgs/Image on the 'camera/image_raw' topic
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)

        # Timer callback to publish frames at 10 Hz (10 FPS)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.bridge = cv_bridge.CvBridge()
        self.frame_id = 0
        self.get_logger().info('ROS 2 Image Publisher Node Initialized.')

    def timer_callback(self):
        self.frame_id += 1

        # 1. Create Dummy NumPy Frame (100x100 RGB image)
        h, w, c = 100, 100, 3
        # Fill value changes with time (0-255)
        np_frame = np.full((h, w, c), fill_value=self.frame_id % 256, dtype=np.uint8)

        # 2. Convert NumPy array (cv2 image) to ROS Image message
        # Use 'bgr8' or 'rgb8' depending on your CvBridge settings; 'bgr8' is common for OpenCV.
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(np_frame, encoding="bgr8")
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Optional: Set timestamp for better ROS tracking
        ros_image_msg.header.stamp = self.get_clock().now().to_msg()

        # 3. Publish the message
        self.publisher_.publish(ros_image_msg)
        self.get_logger().info(f'Publishing Frame {self.frame_id}. Pixel Value: {np_frame[0, 0, 0]}')


def main(args=None):
    rclpy.init(args=args)
    publisher_node = FramePublisherNode()
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()