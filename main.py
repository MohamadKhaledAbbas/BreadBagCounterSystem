import os

from src import constants
from src.classifier.BpuClassifyer import BpuClassifier
from src.counting.BagCounterApp import BagCounterApp
from src.detection.BpuDetector import BpuDetector
from src.logging.Database import DatabaseManager
from src.tracking.Tracker import ObjectTracker
from src.utils.AppLogging import logger

if __name__ == "__main__":

    # Configuration
    VIDEO_PATH = "/media/72E436DAE436A071/Wheatberry_Green20251128080005_20251128090005.mp4"
    DETECTION_MODEL = "data/model/detect_yolo_small_v3_bayese_640x640_nv12.bin"
    CLASS_MODEL = "data/model/classify_yolo_small_v4_bayese_224x224_nv12.bin"
    db_manager = DatabaseManager("data/db/bag_events.db")
    detector = BpuDetector(DETECTION_MODEL, {0: 'bread-bag-closed', 1: 'bread-bag-opened'})
    tracker = ObjectTracker()
    classifier = BpuClassifier(CLASS_MODEL,{ 0: 'Blue_Yellow', 1: 'Bran', 2: 'Brown_Orange_Overlay', 3: 'Brown_Orange_Small', 4: 'Green_Yellow', 5: 'Red_Yellow', 6: 'Wheatberry'})

    is_development = db_manager.get_config_value(constants.is_development_key) == "1"

    try:
        print(f"os.environ['HOME'] = {os.environ['HOME']}")
    except KeyError:
        os.environ["HOME"] = "/home/sunrise"

    app = BagCounterApp(
        video_path = VIDEO_PATH,
        detector_engine=detector,
        tracker=tracker,
        classifier_engine=classifier,
        db=db_manager,
        is_development = is_development,
    )

    logger.info("[INFO] Detection Model: {}".format(DETECTION_MODEL))
    logger.info("[INFO] Classification Model: {}".format(CLASS_MODEL))
    logger.info("[INFO] DB: {}".format(db_manager.db_path))

    app.run()