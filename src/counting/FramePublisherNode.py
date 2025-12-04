import numpy as np
import threading
from src.counting.IPC import FRAME_TOPIC
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

# Conditional ROS2 imports
if IS_RDK:
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
        
        def get_clock(self):
            class StubClock:
                def now(self):
                    class StubTime:
                        def to_msg(self):
                            return None
                    return StubTime()
            return StubClock()
        
        def create_publisher(self, *args, **kwargs):
            return None
        
        def create_timer(self, *args, **kwargs):
            return None
        
        def destroy_node(self):
            pass

if IS_RDK:
    class FramePublisher(Node):
        def __init__(self, topic_name=FRAME_TOPIC, publish_rate_hz=30.0):
            # rclpy.init() must be called externally once per process
            super().__init__('ipc_frame_publisher_node')

            self.publisher_ = self.create_publisher(Image, topic_name, 1)  # Queue size of 1 for latest frame
            self.bridge = CvBridge()
            self.timer = self.create_timer(1.0 / publish_rate_hz, self.timer_callback)

            # Buffer for the latest frame written by the logic thread
            self._frame_buffer = None
            self._frame_lock = threading.Lock()

            logger.info(f"[PUB] Initialized ROS 2 Publisher on topic '{topic_name}' at {publish_rate_hz} Hz.")

        def publish(self, frame):
            """Called by the main logic thread to place a frame in the buffer."""
            if frame is None:
                return
            # Use a lock to ensure thread-safe writing to the buffer
            with self._frame_lock:
                self._frame_buffer = frame

        def timer_callback(self):
            """Executed by the ROS 2 executor thread to publish the latest frame."""
            # 1. Read the frame from the buffer
            with self._frame_lock:
                frame_to_publish = self._frame_buffer
                self._frame_buffer = None  # Consume the frame after reading

            # --- CRITICAL FIX: EXPLICIT TYPE AND DATA CHECK ---
            if not isinstance(frame_to_publish, np.ndarray):
                # If we reach here, 'frame_to_publish is None' failed. We check again
                # to properly log the empty buffer case (DEBUG) vs. an actual wrong object (ERROR).
                if frame_to_publish is None:
                    self.get_logger().debug(
                        'Timer fired, but frame buffer was empty (Secondary Check). Waiting for valid frame.')
                else:
                    # This is a genuinely unexpected object type, which is an ERROR.
                    self.get_logger().error(
                        f'Critical: Frame buffer contained non-numpy object type: {type(frame_to_publish)}. Skipping publish.')

                # Do not clear the buffer; wait for a valid frame to overwrite it.
                return

            # 2. Convert NumPy array to ROS Image message
            try:
                # Use 'bgr8' encoding, standard for OpenCV
                ros_image_msg = self.bridge.cv2_to_imgmsg(frame_to_publish, encoding="bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return

            # 3. Publish the message
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(ros_image_msg)
            self.get_logger().debug(f'Published frame. Shape: {frame_to_publish.shape}')

        # No more close_unlink is needed! ROS 2 cleans up on shutdown.
        def close_node(self):
            self.destroy_node()
else:
    # Stub FramePublisher for non-RDK platforms
    class FramePublisher(Node):
        """Stub FramePublisher for platforms without ROS2."""
        
        def __init__(self, topic_name=FRAME_TOPIC, publish_rate_hz=30.0):
            super().__init__('ipc_frame_publisher_node')
            logger.info(f"[STUB] FramePublisher initialized (non-RDK platform, no publishing)")
        
        def publish(self, frame):
            """Stub publish method that does nothing."""
            pass
        
        def close_node(self):
            """Stub close method."""
            pass