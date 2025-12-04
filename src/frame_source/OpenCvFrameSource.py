import queue
import threading
import time
import cv2
from src.frame_source.FrameSource import FrameSource

class OpenCVFrameSource(FrameSource):
    def __init__(self, source, queue_size=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # 0 = infinite queue = no frame dropping
        self.queue = queue.Queue(maxsize=queue_size)

        self.running = True
        self.last_frame_time = None

        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()

    def _read_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            now = time.time()

            if not ret:
                self.running = False
                break

            if self.last_frame_time is None:
                latency_ms = 0.0
            else:
                latency_ms = (now - self.last_frame_time) * 1000.0

            self.last_frame_time = now

            # Block if consumer is slower (no frame skipping)
            self.queue.put((frame, latency_ms))
            time.sleep(0.1)

        self.cap.release()

    def frames(self):
        while self.running or not self.queue.empty():
            yield self.queue.get()

    def cleanup(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join()
        self.cap.release()
