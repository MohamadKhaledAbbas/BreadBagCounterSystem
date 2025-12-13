import queue
import threading
import time
import cv2
from src.frame_source.FrameSource import FrameSource
from src.utils.AppLogging import logger


class OpenCVFrameSource(FrameSource):
    """
    OpenCV-based frame source for video files, webcams, or RTSP streams.
    
    Supports two modes:
    1. **Production mode** (testing_mode=False): Uses a background thread to read frames
       into a queue. Suitable for real-time processing on optimized hardware like RDK X5.
       
    2. **Testing mode** (testing_mode=True): Reads frames synchronously on-demand.
       No background thread, no frame buffering, no frame dropping.
       Ideal for testing on slower machines (e.g., Windows PCs) where processing
       cannot keep up with real-time frame rates. Ensures every frame is processed
       even if playback takes 4-5x longer than real-time.
    """
    
    def __init__(self, source, queue_size=0, target_fps=None, testing_mode=False):
        """
        Initialize the OpenCV frame source.
        
        Args:
            source: Video source (file path, camera index, or RTSP URL)
            queue_size: Queue size for production mode (0 = unlimited)
            target_fps: Target FPS for frame pacing in production mode (None = no limit)
            testing_mode: If True, enables synchronous on-demand frame reading
                         for testing on slower machines without frame drops
        """
        self.source = source
        self.testing_mode = testing_mode
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # Get source video properties
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"[OpenCVFrameSource] Source FPS: {self.source_fps}, Total frames: {self.total_frames}")
        
        self.running = True
        self.last_frame_time = None
        self._frame_count = 0
        
        if testing_mode:
            # Testing mode: synchronous, on-demand frame reading
            logger.info("[OpenCVFrameSource] Testing mode ENABLED - synchronous frame reading")
            logger.info("[OpenCVFrameSource] All frames will be processed without dropping")
            logger.info("[OpenCVFrameSource] Playback may be slower than real-time")
            self.queue = None
            self.read_thread = None
            self.target_fps = None
            self.frame_interval = None
        else:
            # Production mode: background thread with queue
            logger.info("[OpenCVFrameSource] Production mode - background thread frame reading")
            
            # 0 = infinite queue = no frame dropping
            self.queue = queue.Queue(maxsize=queue_size)
            
            # Target FPS for frame pacing (None = no rate limiting)
            self.target_fps = target_fps
            if self.target_fps is not None:
                if self.target_fps <= 0:
                    raise ValueError(f"target_fps must be positive, got {self.target_fps}")
                self.frame_interval = 1.0 / self.target_fps
                logger.info(f"[OpenCVFrameSource] Target FPS: {self.target_fps}, frame interval: {self.frame_interval:.4f}s")
            else:
                self.frame_interval = None
                logger.info("[OpenCVFrameSource] No target FPS specified, reading frames at source FPS")

            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()

    def _read_frames(self):
        """Background thread for reading frames in production mode."""
        while self.running:
            cycle_start = time.perf_counter()
            
            ret, frame = self.cap.read()
            
            if not ret:
                self.running = False
                break

            self._frame_count += 1

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

    def _read_frame_sync(self):
        """
        Synchronously read the next frame (testing mode).
        
        Returns:
            tuple: (frame, latency_ms) or None if no more frames
        """
        if not self.running:
            return None
            
        cycle_start = time.perf_counter()
        
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            self.running = False
            return None
        
        self._frame_count += 1
        
        # Calculate inter-frame interval for latency reporting
        if self.last_frame_time is None:
            inter_frame_interval_ms = 0.0
        else:
            inter_frame_interval_ms = (cycle_start - self.last_frame_time) * 1000.0
        
        self.last_frame_time = cycle_start
        
        # Log progress periodically in testing mode
        if self._frame_count % 100 == 0:
            if self.total_frames > 0:
                progress = (self._frame_count / self.total_frames * 100)
                logger.info(
                    f"[OpenCVFrameSource] Testing mode progress: "
                    f"frame {self._frame_count}/{self.total_frames} ({progress:.1f}%)"
                )
            else:
                # Live stream (webcam/RTSP) - no total frames available
                logger.info(
                    f"[OpenCVFrameSource] Testing mode: processed {self._frame_count} frames"
                )
        
        return (frame, inter_frame_interval_ms)

    def frames(self):
        """
        Yield frames from the video source.
        
        In production mode: yields from the internal queue (non-blocking background read).
        In testing mode: reads frames synchronously on-demand (blocking, no drops).
        """
        if self.testing_mode:
            # Testing mode: synchronous on-demand reading
            while self.running:
                result = self._read_frame_sync()
                if result is None:
                    break
                yield result
            
            logger.info(f"[OpenCVFrameSource] Testing mode completed: processed {self._frame_count} frames")
        else:
            # Production mode: read from queue
            while self.running or not self.queue.empty():
                try:
                    # Use short timeout to allow checking self.running flag
                    # This enables graceful shutdown while still being responsive
                    yield self.queue.get(timeout=0.1)
                except queue.Empty:
                    # Queue is temporarily empty, check if we should continue
                    continue

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
            if self.read_thread.is_alive():
                logger.warning(
                    "[OpenCVFrameSource] Read thread did not stop within 2s - "
                    "this may indicate a blocking read operation. "
                    "Thread will be terminated when process exits."
                )
        
        if self.cap.isOpened():
            self.cap.release()
        
        logger.info(f"[OpenCVFrameSource] Cleanup complete, processed {self._frame_count} frames")
