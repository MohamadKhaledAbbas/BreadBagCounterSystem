import rclpy
from rclpy.executors import SingleThreadedExecutor
import threading
from src.utils.AppLogging import logger
import time

# Topic name for publishing frames
FRAME_TOPIC = "breadcount/image_raw"


class ExecutorThread(threading.Thread):
    """A dedicated thread to run the ROS 2 Executor's spin loop."""

    def __init__(self, executor: SingleThreadedExecutor):
        super().__init__(daemon=True)
        self.executor = executor
        self.should_run = True

    def run(self):
        logger.debug("[ROS2-THREAD] Executor Thread Started. Spinning...")
        try:
            # The executor.spin() handles the execution of all added nodes (like FramePublisher)
            self.executor.spin()
        except Exception as e:
            # Catch exceptions to prevent silent thread death
            logger.error(f"[ROS2-THREAD ERROR] Executor spin failed: {e}")
        finally:
            logger.debug("[ROS2-THREAD] Executor Thread Finished.")


# Global state to manage ROS 2 context
_ROS_EXECUTOR = None


def init_ros2_context():
    """Initializes the ROS 2 context and the SingleThreadedExecutor."""
    global _ROS_EXECUTOR
    if not rclpy.utilities.ok():
        rclpy.init()
        logger.info("[ROS2] rclpy context initialized.")

    if _ROS_EXECUTOR is None:
        _ROS_EXECUTOR = SingleThreadedExecutor()
        logger.info("[ROS2] SingleThreadedExecutor initialized.")

    return _ROS_EXECUTOR


def shutdown_ros2_context():
    """Shuts down the ROS 2 context and executor."""
    global _ROS_EXECUTOR
    if _ROS_EXECUTOR:
        # Stop the executor gently
        _ROS_EXECUTOR.shutdown()
        _ROS_EXECUTOR = None
        logger.info("[ROS2] Executor shut down.")

    if rclpy.utilities.ok():
        rclpy.shutdown()
        logger.info("[ROS2] rclpy context shutdown.")
