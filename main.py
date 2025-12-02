import argparse

from src import constants
from src.classifier.BpuClassifyer import BpuClassifier
from src.constants import is_production_key
from src.counting.BagCounterApp import BagCounterApp
from src.detection.BpuDetector import BpuDetector
from src.logging.Database import DatabaseManager
from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.tracking.Tracker import ObjectTracker

if __name__ == "__main__":

    # Configuration
    VIDEO_PATH = "output.mp4"
    DETECTION_MODEL = "data/model/detect_yolo_small_v3_bayese_640x640_nv12.bin"
    CLASS_MODEL = "data/model/classify_yolo_nano_v2_bayese_224x224_nv12.bin"
    db_manager = DatabaseManager("data/db/bag_events.db")
    detector = BpuDetector(DETECTION_MODEL, {0: 'bread-bag-closed', 1: 'bread-bag-opened'})
    tracker = ObjectTracker()
    classifier = BpuClassifier(CLASS_MODEL,{ 0: 'Blue_Yellow', 1: 'Bran', 2: 'Brown_Orange_Overlay', 3: 'Brown_Orange_Small', 4: 'Green_Yellow', 5: 'Red_Yellow', 6: 'Wheatberry'})

    is_production = db_manager.get_config_value(constants.is_production_key) == "1"
    show_ui_screen = db_manager.get_config_value(constants.show_ui_screen_key) == "1"
    print(f"[DEBUG] is_production: {is_production}, show_ui_screen: {show_ui_screen}")

    if is_production:
        frame_source = FrameSourceFactory.create("ros2")
    else:
        frame_source = FrameSourceFactory.create("opencv", source=VIDEO_PATH)

    app = BagCounterApp(
        video_path=VIDEO_PATH,
        detector_engine=detector,
        tracker=tracker,
        classifier_engine=classifier,
        db=db_manager,
        frame_source=frame_source,
        show_ui_screen=show_ui_screen
    )

    print("[INFO] loading video from {}".format(VIDEO_PATH))
    if is_production:
        print("[INFO] running in production mode, reading from RTSP stream...")
    else:
        print("[INFO] running in development mode, reading from Video file...")
    if show_ui_screen:
        print("[INFO] started with ui screen enabled...")
    else:
        print("[INFO] started with ui screen enabled...")
    print("[INFO] Detection Model: {}".format(DETECTION_MODEL))
    print("[INFO] Classification Model: {}".format(CLASS_MODEL))
    print("[INFO] DB: {}".format(db_manager.db_path))

    app.run()