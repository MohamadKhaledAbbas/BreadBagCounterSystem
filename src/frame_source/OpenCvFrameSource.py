import queue
import threading
import time

import cv2

from src.frame_source.FrameSource import FrameSource


class OpenCVFrameSource(FrameSource):
    def __init__(self, source, queue_size=10):
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
            try:
                self.queue.put((bgr, latency_ms), timeout=1)
            except queue.Full:
                # Drop oldest frame, or skip, depending on your needs
                try:
                    self.queue.get(block=False)  # Remove one
                    self.queue.put((bgr, latency_ms), timeout=1)
                except queue.Empty:
                    pass  # Ignore if queue suddenly emptied
            time.sleep(0.05)

        self.cleanup()

    def frames(self):
        while self.running or not self.queue.empty():
            try:
                frame, latency_ms = self.queue.get(timeout=1)
                yield frame, latency_ms
            except queue.Empty:
                continue

    def cleanup(self):
        # Call this to gracefully stop the reading thread (if needed)
        self.running = False
        self.queue.join()
        if self.read_thread.is_alive():
            self.read_thread.join()
        self.cap.release()
