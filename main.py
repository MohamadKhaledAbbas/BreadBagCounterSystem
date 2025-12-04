import os

from src import constants
from src.classifier.BpuClassifyer import BpuClassifier
from src.counting.BagCounterApp import BagCounterApp
from src.detection.BpuDetector import BpuDetector
from src.logging.Database import DatabaseManager
from src.config.settings import config
from src.utils.AppLogging import logger

if __name__ == "__main__":

    # Configuration
    VIDEO_PATH = config.video_path or "/media/72E436DAE436A071/Wheatberry_Green20251128080005_20251128090005.mp4"
    DETECTION_MODEL = config.detection_model
    CLASS_MODEL = config.classification_model
    db_manager = DatabaseManager(config.db_path)
    detector = BpuDetector(DETECTION_MODEL, config.detector_classes)
    classifier = BpuClassifier(CLASS_MODEL, config.classifier_classes)

    is_development = db_manager.get_config_value(constants.is_development_key) == "1"

    try:
        logger.info(f"os.environ['HOME'] = {os.environ['HOME']}")
    except KeyError:
        os.environ["HOME"] = "/home/sunrise"
        logger.info("HOME environment variable not set, using default: /home/sunrise")

    app = BagCounterApp(
        video_path = VIDEO_PATH,
        detector_engine=detector,
        classifier_engine=classifier,
        db=db_manager,
        is_development = is_development,
    )

    logger.info("[INFO] Detection Model: {}".format(DETECTION_MODEL))
    logger.info("[INFO] Classification Model: {}".format(CLASS_MODEL))
    logger.info("[INFO] DB: {}".format(db_manager.db_path))

    app.run()