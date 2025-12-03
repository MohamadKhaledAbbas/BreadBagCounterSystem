import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from hbm_img_msgs.msg import HbmMsg1080P
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from src.frame_source.FrameSource import FrameSource

from src.utils.AppLogging import logger


class FrameServer(Node, FrameSource):
    """
    ROS 2 Subscriber that listens for incoming frames and buffers the latest
    frame for consumption by the main logic thread. This node is designed
    to be added to an external SingleThreadedExecutor.
    """

    def __init__(self, topic='/nv12_images'):
        # IMPORTANT: rclpy.init() must be called externally before this class is instantiated.
        super().__init__('frame_server')
        self.subscription = self.create_subscription(
            HbmMsg1080P,
            topic,
            self.listener_callback,
            qos_profile_sensor_data)

        # Only keep the latest frame
        self.frame_queue = queue.Queue(maxsize=1)
        self.last_frame_time = time.time()

        # --- REMOVED ---
        # Removed the internal _ros_spin_thread logic.
        # The execution (spinning) is now handled by the external ExecutorThread.
        # ---------------

    def listener_callback(self, msg):
        now = time.time()
        img = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        try:
            # NV12 conversion logic
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return

        latency_ms = (now - self.last_frame_time) * 1000
        self.last_frame_time = now

        # Leaky queue: drop oldest if full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((bgr, latency_ms))

    def frames(self):
        # We check rclpy.ok() to ensure we stop if the ROS context shuts down
        while rclpy.ok():
            try:
                frame, latency_ms = self.frame_queue.get(timeout=1)
                yield frame, latency_ms
            except queue.Empty:
                continue

    def cleanup(self):
        """Destroys the node, relying on the main app to shutdown the ROS context."""
        logger.info("[Ros2FrameServer] cleanup called. Destroying node.")

        # Destroy the node itself
        try:
            self.destroy_node()
        except Exception as e:
            logger.debug(f"[Ros2FrameServer] destroy_node() raised (ignored): {e}")

        logger.info("[Ros2FrameServer] cleanup finished")