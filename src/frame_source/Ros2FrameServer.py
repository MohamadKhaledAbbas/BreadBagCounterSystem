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

class FrameServer(Node, FrameSource):
    def __init__(self, topic='/nv12_images'):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('frame_server')
        self.subscription = self.create_subscription(
            HbmMsg1080P,
            topic,
            self.listener_callback,
            qos_profile_sensor_data)
        self.frame_queue = queue.Queue(maxsize=10)
        self.last_frame_time = time.time()

        self._ros_spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._ros_spin_thread.start()

    def listener_callback(self, msg):
        now = time.time()
        img = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        try:
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return
        latency_ms = (now - self.last_frame_time) * 1000
        self.last_frame_time = now
        self.frame_queue.put((bgr, latency_ms))

    def frames(self):
        while rclpy.ok():
            try:
                frame, latency_ms = self.frame_queue.get(timeout=1)
                yield frame, latency_ms
            except queue.Empty:
                continue

    def cleanup(self):
        if self._ros_spin_thread.is_alive():
            self._ros_spin_thread.join()
        self.frame_queue.join()
        rclpy.shutdown()
