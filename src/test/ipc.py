# File: queue_ipc.py
import time
import numpy as np
import multiprocessing as mp


# --- Dummy Logger ---
class DummyLogger:
    def info(self, msg):
        print(f"[LOG] INFO: {msg}")

    def error(self, msg):
        print(f"[LOG] ERROR: {msg}")


logger = DummyLogger()


def publisher_process(frame_queue: mp.Queue):
    """Generates frames and puts them onto the queue."""
    logger.info("[PUB] Starting publisher process.")
    h, w, c = 100, 100, 3

    try:
        frame_id = 0
        while True:
            frame_id += 1
            # 1. Create a frame where the pixel value changes with frame_id
            # NOTE: Frame data is COPIED/PICKLED when put into the Queue.
            dummy_frame = np.full((h, w, c), fill_value=frame_id % 256, dtype=np.uint8)

            # 2. Put frame onto the queue (blocking call)
            frame_queue.put(dummy_frame)
            logger.info(f"[PUB] Frame {frame_id} sent. Value: {dummy_frame[0, 0, 0]}")

            time.sleep(0.1)  # Simulate 10 FPS

    except KeyboardInterrupt:
        logger.info("[PUB] Publisher shutting down.")
    except Exception as e:
        logger.error(f"[PUB] Error: {e}")
    finally:
        # Signal the subscriber that no more data is coming
        frame_queue.put(None)
        logger.info("[PUB] Sent termination signal to queue.")


def subscriber_process(frame_queue: mp.Queue):
    """Reads frames from the queue."""
    logger.info("[SUB] Starting subscriber process.")

    try:
        while True:
            # 1. Get frame from the queue (blocking call)
            frame = frame_queue.get()

            # 2. Check for termination signal (sent by publisher)
            if frame is None:
                logger.info("[SUB] Received termination signal. Shutting down.")
                break

            # 3. Process the frame (It's already a copy in SUB's memory)
            logger.info(f"[SUB] Received frame. Value: {frame[0, 0, 0]}, Shape: {frame.shape}")

    except KeyboardInterrupt:
        logger.info("[SUB] Subscriber shutting down.")
    except Exception as e:
        logger.error(f"[SUB] Error: {e}")