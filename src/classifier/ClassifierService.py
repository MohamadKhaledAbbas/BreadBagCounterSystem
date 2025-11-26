# src/classifier/AsyncClassificationService.py
import os
import queue
import threading
import time
from typing import Callable, List, Dict, Any
from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash

# Callback type
ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False):

        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois  # True = Data Collection Mode

        self.callbacks: List[ResultCallback] = []
        self.running = True


    def register_callback(self, callback: ResultCallback):
        self.callbacks.append(callback)

    def process(self, track_id: int, roi_image):
        try:
            # 1. Predict
            label, conf = self.classifier.predict(roi_image)

            # 2. Compute Hash
            phash_obj = compute_phash(roi_image)
            phash_str = str(phash_obj)

            # 3. Smart Saving Logic
            # Determine Target Folder: data/classes/RedBag OR data/unknown/a3f1b2...
            if label == "Unknown":
                # Group unknowns by their Hash
                target_dir = os.path.join(self.data_root, "unknown", phash_str)
            else:
                # Group knowns by their Class Name
                target_dir = os.path.join(self.data_root, "classes", label)

            os.makedirs(target_dir, exist_ok=True)

            # Check if we need to save this specific image
            should_save = False
            existing_files = os.listdir(target_dir)

            if self.save_all_rois:
                should_save = True  # Data Collection Mode: Save everything
            elif not existing_files:
                should_save = True  # Production Mode: Save only if folder is empty (First Thumbnail)

            image_path = None
            if should_save:
                timestamp = int(time.time())
                filename = f"{timestamp}_{track_id}.jpg"
                save_path = os.path.join(target_dir, filename)

                # Use utils or cv2 directly
                import cv2
                cv2.imwrite(save_path, roi_image)
                image_path = save_path
            elif existing_files:
                # If we didn't save new one, point to the existing one for the DB
                image_path = os.path.join(target_dir, existing_files[0])

            # 4. Bundle Result
            result_data = {
                "label": label,
                "phash": phash_str,
                "image_path": image_path,  # Could be new or existing
                "confidence": conf
            }

            for cb in self.callbacks:
                cb(track_id, result_data)

        except Exception as e:
            print(f"Worker Error: {e}")
