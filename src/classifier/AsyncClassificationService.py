import queue
import threading
from collections import defaultdict
from typing import Callable, List

from src.classifier.BaseClassifier import BaseClassifier
from src.logging.Database import DatabaseManager

ResultCallback = Callable[[int, str], None]

class AsyncClassificationService:
    """
    Handles the second model in a separate thread AND logs to Database.
    """

    def __init__(self, classifier: BaseClassifier, db_manager: DatabaseManager):
        self.classifier = classifier
        self.db = db_manager  # Injected Dependency
        self.queue = queue.Queue()
        self.callbacks: List[ResultCallback] = []
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def register_callback(self, callback: ResultCallback):
        """Add a function to be called when a classification is complete."""
        self.callbacks.append(callback)

    def process(self, track_id: int, roi_image):
        if self.running:
            self.queue.put((track_id, roi_image))

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None: break

            track_id, image = item

            # 1. Predict
            label = self.classifier.predict(image)

            # 2. Notify all listeners (Decoupled)
            for callback in self.callbacks:
                try:
                    callback(track_id, label)
                except Exception as e:
                    print(f"Error in callback: {e}")


            self.queue.task_done()
