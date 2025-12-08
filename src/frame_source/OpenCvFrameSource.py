import queue
import threading
import time
import cv2
from src.frame_source.FrameSource import FrameSource
from src.utils.AppLogging import logger

class OpenCVFrameSource(FrameSource):
    def __init__(self, source, queue_size=0, target_fps=None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # Get source FPS for logging
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[OpenCVFrameSource] Source FPS: {self.source_fps}")

        # 0 = infinite queue = no frame dropping
        self.queue = queue.Queue(maxsize=queue_size)

        self.running = True
        self.last_frame_time = None
        
        # Target FPS for frame pacing (None = no rate limiting)
        self.target_fps = target_fps
        if self.target_fps is not None:
            if self.target_fps <= 0:
                raise ValueError(f"target_fps must be positive, got {self.target_fps}")
            self.frame_interval = 1.0 / self.target_fps
            logger.info(f"[OpenCVFrameSource] Target FPS: {self.target_fps}, frame interval: {self.frame_interval:.4f}s")
        else:
            self.frame_interval = None
            logger.info(f"[OpenCVFrameSource] No target FPS specified, reading frames as fast as possible")

        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()

    def _read_frames(self):
        while self.running:
            cycle_start = time.perf_counter()
            
            ret, frame = self.cap.read()
            
            if not ret:
                self.running = False
                break

            # Calculate inter-frame interval for latency reporting
            if self.last_frame_time is None:
                inter_frame_interval_ms = 0.0
            else:
                inter_frame_interval_ms = (cycle_start - self.last_frame_time) * 1000.0

            self.last_frame_time = cycle_start

            # Block if consumer is slower (no frame skipping)
            self.queue.put((frame, inter_frame_interval_ms))
            
            # Frame pacing: wait to achieve target FPS
            if self.frame_interval is not None:
                # Measure total time for this cycle (read + queue operations)
                cycle_end = time.perf_counter()
                elapsed = cycle_end - cycle_start
                sleep_time = self.frame_interval - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.cap.release()

    def frames(self):
        while self.running or not self.queue.empty():
            yield self.queue.get()

    def cleanup(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join()
        self.cap.release()
