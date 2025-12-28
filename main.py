"""
BreadBag Counter System - Main Entry Point

V2: Enhanced with model version tracking, structured logging, and health checks.
"""

import os

from src import constants
from src.counting.BagCounterApp import BagCounterApp
from src.logging.Database import DatabaseManager
from src.config.settings import config
from src.utils.AppLogging import logger, get_log_file_paths
from src.utils.platform import IS_RDK, IS_WINDOWS

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
    
    # V2: Log startup banner
    logger.info("=" * 60)
    logger.info("  BreadBag Counter System V2")
    logger.info("  Production-Grade Accuracy Pipeline")
    logger.info("=" * 60)
    
    # V2: Log file paths for debugging
    log_paths = get_log_file_paths()
    logger.info(f"[Startup] Log files: {log_paths}")
    
    # V2: Log configuration with model versions
    config.log_configuration()

    # Configuration
    db_manager = DatabaseManager(config.db_path)
    detector = Detector(config.detection_model, config.detector_classes)
    classifier = Classifier(config.classification_model,  config.classifier_classes)

    is_development = db_manager.get_config_value(constants.is_development_key) == "1"
    logger.info(f"[Startup] Development mode: {is_development}")

    try:
        logger.info(f"os.environ['HOME'] = {os.environ['HOME']}")
    except KeyError:
        os.environ["HOME"] = "/home/sunrise"
        logger.info("HOME environment variable not set, using default: /home/sunrise")

    app = BagCounterApp(
        video_path=config.video_path,
        detector_engine=detector,
        classifier_engine=classifier,
        db=db_manager,
        is_development=is_development,
    )

    logger.info("[Startup] Detection Model: {}".format(config.detection_model))
    logger.info("[Startup] Classification Model: {}".format(config.classification_model))
    logger.info("[Startup] DB: {}".format(db_manager.db_path))

    profiler_enabled = db_manager.get_config_value(constants.is_profiler_enabled)
    if profiler_enabled:
        logger.info("[Startup] profiler is enabled... running profiler...")
        import cProfile
        import pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            logger.info("[Startup] Starting main application loop...")
            app.run()
        finally:
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(25)
            print(s.getvalue())
            profiler.dump_stats("data/logs/cprofile.prof")
    else:
        logger.info("[Startup] Starting main application loop...")
        app.run()