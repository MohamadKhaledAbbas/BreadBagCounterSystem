import argparse
from src.classifier.BpuClassifyer import BpuClassifier
from src.counting.BagCounterApp import BagCounterApp
from src.detection.BpuDetector import BpuDetector
from src.logging.Database import DatabaseManager
from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.tracking.Tracker import ObjectTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_production', type=str, required=True, help='true or false')
    parser.add_argument('--show_ui_screen', type=str, required=True, help='true or false')
    args = parser.parse_args()

    # Make them booleans:
    IS_PRODUCTION_MODE = args.is_production.lower() == "true"
    SHOW_UI_SCREEN = args.show_ui_screen.lower() == "true"

    # Configuration
    VIDEO_PATH = "output2.mp4"
    DETECTION_MODEL = "data/model/best_detect_bayese_640x640_nv12.bin"
    CLASS_MODEL = "data/model/best_classify_bayese_224x224_nv12.bin"
    db_manager = DatabaseManager("data/db/bag_events.db")
    detector = BpuDetector(DETECTION_MODEL, {0: 'bread-bag-closed', 1: 'bread-bag-opened'})
    tracker = ObjectTracker()
    classifier = BpuClassifier(CLASS_MODEL,{ 0: 'Blue-Bag', 1: 'Brown-Bag', 2: 'Dark-Brown-Bag', 3: 'Green-Bag', 4: 'Red-Bag', 5: 'Yellow-Bag'})

    if IS_PRODUCTION_MODE:
        frame_source = FrameSourceFactory.create("ros2")
    else:
        frame_source = FrameSourceFactory.create("opencv", source=VIDEO_PATH)

    app = BagCounterApp(
        video_path=VIDEO_PATH,
        detector_engine=detector,
        tracker=tracker,
        classifier_engine=classifier,
        db_manager=db_manager,
        frame_source=frame_source,
        show_ui_screen=SHOW_UI_SCREEN
    )

    print("[INFO] loading video from {}".format(VIDEO_PATH))
    if args.is_production:
        print("[INFO] running in production mode, reading from RTSP stream...")
    else:
        print("[INFO] running in development mode, reading from Video file...")
    print("[INFO] Detection Model: {}".format(DETECTION_MODEL))
    print("[INFO] Classification Model: {}".format(CLASS_MODEL))
    print("[INFO] DB: {}".format(db_manager.db_path))

    app.run()