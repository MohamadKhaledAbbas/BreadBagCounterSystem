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
    db_manager = DatabaseManager(config.db_path)
    detector = BpuDetector(config.detection_model, config.detector_classes)
    classifier = BpuClassifier(config.classification_model, config.classifier_classes)

    is_development = db_manager.get_config_value(constants.is_development_key) == "1"

    try:
        logger.info(f"os.environ['HOME'] = {os.environ['HOME']}")
    except KeyError:
        os.environ["HOME"] = "/home/sunrise"
        logger.info("HOME environment variable not set, using default: /home/sunrise")

    app = BagCounterApp(
        video_path = config.video_path or "/media/72E436DAE436A071/Wheatberry_Green20251128080005_20251128090005.mp4",
        detector_engine=detector,
        classifier_engine=classifier,
        db=db_manager,
        is_development = is_development,
    )

    logger.info("[INFO] Detection Model: {}".format(config.detection_model))
    logger.info("[INFO] Classification Model: {}".format(config.classification_model))
    logger.info("[INFO] DB: {}".format(db_manager.db_path))

    app.run()