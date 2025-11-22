import queue
import threading
from collections import defaultdict

from src.classifier.BaseClassifier import BaseClassifier


class AsyncClassificationService:
    """Handles the second model in a separate thread."""
    def __init__(self, classifier: BaseClassifier):
        self.classifier = classifier
        self.queue = queue.Queue()
        self.counts = defaultdict(int)
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def process(self, roi_image):
        if self.running:
            self.queue.put(roi_image)

    def get_counts(self):
        with self.lock:
            return dict(self.counts)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None: break

            label = self.classifier.predict(item)

            with self.lock:
                self.counts[label] += 1

            self.queue.task_done()