import queue
import threading
import time

import cv2

from src.frame_source.FrameSource import FrameSource

class OpenCVFrameSource(FrameSource):
    def __init__(self, source, queue_size=1):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        self.queue = queue.Queue(maxsize=queue_size)
        self.last_frame_time = time.time()
        self.running = True
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()

    def _read_frames(self):
        while self.running:
            ret, bgr = self.cap.read()
            now = time.time()
            latency_ms = (now - self.last_frame_time) * 1000
            self.last_frame_time = now
            if not ret:
                self.running = False
                break
            # Always keep only the latest frame: leaky queue
            if self.queue.full():
                try:
                    self.queue.get(block=False)
                except queue.Empty:
                    pass
            try:
                self.queue.put((bgr, latency_ms), timeout=1)
            except queue.Full:
                pass  # Should not happen now

            time.sleep(0.01)  # Faster polling for UI

    def frames(self):
        while self.running or not self.queue.empty():
            try:
                frame, latency_ms = self.queue.get(timeout=1)
                yield frame, latency_ms
            except queue.Empty:
                continue

    def cleanup(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join()
        self.cap.release()