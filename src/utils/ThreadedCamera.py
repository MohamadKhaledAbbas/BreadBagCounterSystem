import cv2
import threading
import queue


class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=1)  # Keep only latest frame
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard old frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return True, self.q.get()

    def release(self):
        self.running = False
        self.capture.release()
        self.thread.join()

    def isOpened(self):
        return self.capture.isOpened()