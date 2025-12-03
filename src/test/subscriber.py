# File: src/image_subscriber.py

import rclpy
from rclpy.node import Node
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image
import time


class FrameSubscriberNode(Node):
    def __init__(self):
        super().__init__('frame_subscriber_node')

        # ROS 2 Subscriber: Subscribes to sensor_msgs/Image from 'camera/image_raw'
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)

        self.bridge = cv_bridge.CvBridge()
        self.get_logger().info('ROS 2 Image Subscriber Node Initialized.')

    def listener_callback(self, msg):
        # 1. Convert ROS Image message to NumPy array
        try:
            # Use the encoding from the message or 'bgr8'
            np_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # 2. Process the received frame (e.g., displaying, counting)
        h, w, c = np_frame.shape
        self.get_logger().info(f'Received Frame. Shape: ({h}, {w}, {c}). Pixel Value: {np_frame[0, 0, 0]}')

        # NOTE: Your actual CV2 window logic would go here:
        # cv2.imshow("Live Monitor", np_frame)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    subscriber_node = FrameSubscriberNode()
    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()