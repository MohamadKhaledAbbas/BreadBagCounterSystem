# File: main_ui.py (Refactored for ROS 2)

import cv2
import time
import rclpy
from rclpy.executors import SingleThreadedExecutor
import threading

# --- NEW IMPORTS ---
# Assuming you update src/counting/IPC.py to include the needed imports
# (rclpy, Node, Image, CvBridge, etc.)
from src.counting.FrameSubscriberNode import FrameSubscriber
from src.counting.IPC import init_ros2_context, shutdown_ros2_context
# -------------------

def main():
    print("Starting UI Monitor (ROS 2 Subscriber)...")

    # 1. Initialize ROS 2 Context
    init_ros2_context()

    # 2. Create the Subscriber Node
    subscriber = FrameSubscriber()

    # 3. Create Executor and run it in a separate thread
    # The executor handles incoming messages and calls the listener_callback
    executor = SingleThreadedExecutor()
    executor.add_node(subscriber)

    # Run the executor in a thread so it doesn't block the cv2 loop
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    window_name = "BreadCounting Live Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- UI loop ---
    while True:
        # Get the latest frame from the thread-safe buffer
        frame = subscriber.get_latest_frame()

        if frame is not None:
            # Display the frame using OpenCV
            cv2.imshow(window_name, frame)

        # q = close window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # If no frame, small sleep is necessary to yield CPU time
        if frame is None:
            time.sleep(0.01)

    # --- Cleanup ---
    cv2.destroyAllWindows()

    # Stop the executor and the ROS 2 node gracefully
    executor.remove_node(subscriber)
    executor.shutdown()
    subscriber.close_node()

    # Shutdown the ROS 2 context
    shutdown_ros2_context()

    if executor_thread.is_alive():
        executor_thread.join()


if __name__ == "__main__":
    main()