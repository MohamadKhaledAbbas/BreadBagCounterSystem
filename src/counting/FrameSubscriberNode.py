import numpy as np
import time  # Needed for timer/rate control
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK
from src.counting.IPC import FRAME_TOPIC

# Conditional ROS2 imports
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
else:
    # Stub base class for non-RDK platforms
    class Node:
        """Stub Node class for platforms without ROS2."""
        def __init__(self, node_name):
            pass
        
        def get_logger(self):
            return logger
        
        def create_subscription(self, *args, **kwargs):
            return None
        
        def destroy_node(self):
            pass
# File: src/counting/IPC.py (FrameSubscriber Class)

if IS_RDK:
    class FrameSubscriber(Node):
        def __init__(self, topic_name=FRAME_TOPIC):
            # 1. Initialize the ROS 2 Node
            super().__init__('frame_subscriber_node')

            # 2. Create the Subscriber
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.listener_callback,  # Function to call on message receipt
                10)

            # 3. Create the CvBridge instance
            self.bridge = CvBridge()
            self.latest_frame = None
            self.frame_counter = 0

            logger.info(f"[SUB] Initialized ROS 2 Subscriber on topic '{topic_name}'.")

        def listener_callback(self, msg):
            """Callback function executed every time a new Image message arrives."""
            self.frame_counter += 1

            # 1. Convert ROS Image message back to NumPy array
            try:
                # The desired_encoding must match the encoding used in the publisher
                np_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return

            # 2. Store the latest frame for application logic
            self.latest_frame = np_frame

            self.get_logger().debug(f'Received Frame {self.frame_counter}. Shape: {np_frame.shape}')

        def get_latest_frame(self):
            """Public method for the main application loop to retrieve the frame."""
            # Returns the NumPy array and consumes the stored frame (optional, for safety)
            if self.latest_frame is None:
                return None

            # Optional: return a copy to ensure the frame isn't modified by both the callback and the app loop
            return self.latest_frame.copy()

            # No need for connect/close_shm; connection is managed by ROS 2/DDS

        def close_node(self):
            self.destroy_node()
else:
    # Stub FrameSubscriber for non-RDK platforms
    class FrameSubscriber(Node):
        """Stub FrameSubscriber for platforms without ROS2."""
        
        def __init__(self, topic_name=FRAME_TOPIC):
            super().__init__('frame_subscriber_node')
            self.latest_frame = None
            logger.info(f"[STUB] FrameSubscriber initialized (non-RDK platform, no subscription)")
        
        def get_latest_frame(self):
            """Stub method that returns None."""
            return None
        
        def close_node(self):
            """Stub close method."""
            pass