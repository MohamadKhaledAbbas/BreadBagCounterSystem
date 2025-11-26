import cv2
import threading
import queue
import time


class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=1)
        self.running = True

        # NEW: Event to signal "New Frame Ready"
        self.frame_ready_event = threading.Event()

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                # If stream ends or fails, ensure we don't block forever
                self.running = False
                self.frame_ready_event.set()
                break

            # Drop old frame if queue is full (Keep latest)
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put((ret, frame))

            # SIGNAL: Wake up the main thread!
            self.frame_ready_event.set()

            # Small sleep to ensure other threads can grab the GIL
            # This is the "Automatic 50ms" logic handled at the thread level
            time.sleep(0.005)

    def read(self):
        # WAIT: Block here until the event is set (Automatic Sync)
        # timeout=1.0 prevents freezing if camera dies
        got_frame = self.frame_ready_event.wait(timeout=1.0)

        if not got_frame or not self.running:
            return False, None

        # Reset event so we wait again next time
        self.frame_ready_event.clear()

        return self.q.get()

    def release(self):
        self.running = False
        self.frame_ready_event.set()  # Unblock anyone waiting
        self.capture.release()
        self.thread.join()

    def isOpened(self):
        return self.capture.isOpened()