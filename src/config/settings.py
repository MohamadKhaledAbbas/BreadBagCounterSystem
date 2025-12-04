import os
from dataclasses import dataclass
from src.utils.platform import IS_RDK

@dataclass
class AppConfig:
    video_path: str = os.getenv("VIDEO_PATH", "D:\\Recordings\\New_Recordings\\RED.mp4")

    # Platform-specific model paths
    # RDK uses .bin models optimized for BPU, Windows/other platforms use .pt or .onnx models
    detection_model: str = os.getenv(
        "DETECTION_MODEL",
        "data/model/detect_yolo_small_v2_bayese_640x640_nv12.bin" if IS_RDK
        else "data/model/detect_yolo_small_v3.pt"
    )
    classification_model: str = os.getenv(
        "CLASS_MODEL",
        "data/model/classify_yolo_small_v4_bayese_224x224_nv12.bin" if IS_RDK
        else "data/model/classify_yolo_small_v4.pt"
    )

    db_path: str = os.getenv("DB_PATH", "data/db/bag_events.db")
    
    # Classifier class names
    classifier_classes: dict = None
    
    # Detector class names  
    detector_classes: dict = None
    
    def __post_init__(self):
        if self.classifier_classes is None:
            self.classifier_classes = {
                0: 'Blue_Yellow', 
                1: 'Bran', 
                2: 'Brown_Orange_Overlay', 
                3: 'Brown_Orange_Small', 
                4: 'Green_Yellow', 
                5: 'Red_Yellow', 
                6: 'Wheatberry'
            }
        if self.detector_classes is None:
            self.detector_classes = {
                0: 'bread-bag-closed', 
                1: 'bread-bag-opened'
            }

config = AppConfig()
