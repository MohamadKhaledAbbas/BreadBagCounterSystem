# src/utils.py
import os
import cv2
import imagehash
from PIL import Image


def compute_phash(roi_array):
    # pHash requires PIL Image
    image = Image.fromarray(cv2.cvtColor(roi_array, cv2.COLOR_BGR2RGB))
    return imagehash.phash(image)


def save_roi_image(roi_array, folder_type: str, filename: str) -> str:
    # Define base paths
    BASE_DIR = "data"
    TARGET_DIR = os.path.join(BASE_DIR, folder_type)

    os.makedirs(TARGET_DIR, exist_ok=True)

    full_path = os.path.join(TARGET_DIR, filename)
    cv2.imwrite(full_path, roi_array)

    return full_path
