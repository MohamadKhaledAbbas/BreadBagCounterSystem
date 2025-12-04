import os

from src import constants
from src.counting.BagCounterApp import BagCounterApp
from src.logging.Database import DatabaseManager
from src.config.settings import config
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

# Platform-aware detector and classifier imports
if IS_RDK:
    from src.detection.BpuDetector import BpuDetector as Detector
    from src.classifier.BpuClassifyer import BpuClassifier as Classifier
    logger.info("[Platform] Running on RDK - using BPU models")
else:
    from src.detection.UltralyticsDetector import UltralyticsDetector as Detector
    from src.classifier.UltralyticsClassifier import UltralyticsClassifier as Classifier
    logger.info("[Platform] Running on non-RDK platform - using Ultralytics models")

if __name__ == "__main__":

    # Configuration
    db_manager = DatabaseManager(config.db_path)
    detector = Detector(config.detection_model, config.detector_classes)
    classifier = Classifier(config.classification_model, config.classifier_classes)

    is_development = db_manager.get_config_value(constants.is_development_key) == "1"

    try:
        logger.info(f"os.environ['HOME'] = {os.environ['HOME']}")
    except KeyError:
        os.environ["HOME"] = "/home/sunrise"
        logger.info("HOME environment variable not set, using default: /home/sunrise")

    app = BagCounterApp(
        video_path = config.video_path,
        detector_engine=detector,
        classifier_engine=classifier,
        db=db_manager,
        is_development = is_development,
    )

    logger.info("[INFO] Detection Model: {}".format(config.detection_model))
    logger.info("[INFO] Classification Model: {}".format(config.classification_model))
    logger.info("[INFO] DB: {}".format(db_manager.db_path))

    app.run()