#!/usr/bin/env python3
"""
Homography Calibration Tool for Bread Bag Counter System.

This script provides an interactive calibration tool for setting up the homography
transformation used for accurate size-based bread bag classification.

Features:
- Interactive corner selection from video frame, image, live camera, or ROS2 NV12 topic
- Table dimension validation
- Interactive bread bag reference drawing for threshold calibration
- Calibration testing with sample bboxes
- Environment variable generation for . env file
- Save calibration data (image with corners + JSON)

Usage:
    # From video frame
    python scripts/calibrate_homography.py --video path/to/video.mp4 --frame 100

    # From image
    python scripts/calibrate_homography.py --image path/to/frame.jpg

    # From live camera
    python scripts/calibrate_homography. py --camera 0

    # From ROS2 NV12 topic
    python scripts/calibrate_homography.py --ros2-topic /nv12_images

    # Test existing calibration
    python scripts/calibrate_homography.py --dry-run
"""

import argparse
import cv2
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.homography import HomographyTransform
from src.utils.AppLogging import logger

# ROS2 imports (optional - only if ROS2 is available)
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from hbm_img_msgs.msg import HbmMsg1080P

    ROS2_AVAILABLE = True


    class ROS2FrameCapture(Node):
        """ROS2 node for capturing a single frame from NV12 topic."""

        def __init__(self, topic_name="/nv12_images"):
            super().__init__('calibration_frame_capture')
            self.subscription = self.create_subscription(
                HbmMsg1080P,
                topic_name,
                self.listener_callback,
                qos_profile_sensor_data
            )
            self.captured_frame = None
            self.frame_count = 0
            self.get_logger().info(f"Subscribed to topic '{topic_name}' for frame capture.")

        def listener_callback(self, msg):
            """Convert NV12 message to BGR frame."""
            h = msg.height
            w = msg.width

            # Convert NV12 data to BGR
            img_data = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
            nv12_img = img_data.reshape((h * 3 // 2, w))
            bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)

            self.captured_frame = bgr
            self.frame_count += 1

        def get_frame(self):
            """Return the latest captured frame."""
            return self.captured_frame
except ImportError:
    print("⚠️  ROS2 libraries not available.  --ros2-topic option will be disabled.")


class CalibrationTool:
    """Interactive homography calibration tool."""

    def __init__(self):
        self.corners: List[List[float]] = []
        self.image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.window_name = "Homography Calibration - Click 4 Corners (TL, TR, BR, BL)"
        self.bag_references: Dict[str, List[List[float]]] = {}  # Store bag reference corners

    def load_image_from_video(self, video_path: str, frame_number: int) -> bool:
        """Load a frame from a video file."""
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error:  Could not open video:  {video_path}")
            return False

        # Seek to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"❌ Error: Could not read frame {frame_number} from video")
            return False

        self.image = frame
        self.display_image = frame.copy()
        print(f"✅ Loaded frame {frame_number} from video: {frame.shape}")
        return True

    def load_image_from_file(self, image_path: str) -> bool:
        """Load an image from a file."""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found:  {image_path}")
            return False

        image = cv2.imread(image_path)
        resized = cv2.resize(image, (1280, 720))
        if resized is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return False

        self.image = resized
        self.display_image = resized.copy()
        print(f"✅ Loaded image: {resized.shape}")
        return True

    def load_image_from_camera(self, camera_id: int) -> bool:
        """Capture a frame from a live camera."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return False

        print("📷 Camera opened.  Press SPACE to capture, ESC to cancel...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read from camera")
                cap.release()
                return False

            # Display live preview
            preview = frame.copy()
            cv2.putText(preview, "Press SPACE to capture, ESC to cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Preview", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                print("❌ Capture cancelled")
                return False
            elif key == 32:  # SPACE
                self.image = frame
                self.display_image = frame.copy()
                cap.release()
                cv2.destroyAllWindows()
                print(f"✅ Captured frame:  {frame.shape}")
                return True

    def load_image_from_ros2_topic(self, topic_name: str, timeout_sec: float = 10.0) -> bool:
        """Capture a frame from a ROS2 NV12 topic."""
        if not ROS2_AVAILABLE:
            print("❌ Error: ROS2 libraries not available")
            return False

        print(f"📡 Connecting to ROS2 topic: {topic_name}")
        print(f"   Waiting for frame (timeout: {timeout_sec}s)...")
        print("   Press SPACE to capture current frame, ESC to cancel...")

        # Initialize ROS2
        rclpy.init()

        try:
            # Create capture node
            capture_node = ROS2FrameCapture(topic_name)

            # Create preview window
            preview_window = "ROS2 Frame Preview"
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

            # Spin until we get a frame or timeout
            import time
            start_time = time.time()
            captured = False

            while rclpy.ok():
                # Process ROS2 callbacks
                rclpy.spin_once(capture_node, timeout_sec=0.01)

                # Get current frame
                current_frame = capture_node.get_frame()

                if current_frame is not None:
                    # Display live preview
                    preview = current_frame.copy()
                    print(f"frame size = {preview.shape}")
                    cv2.putText(preview, f"Frame {capture_node.frame_count} - Press SPACE to capture, ESC to cancel",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Resize for display if too large
                    display_height = 720
                    if preview.shape[0] > display_height:
                        scale = display_height / preview.shape[0]
                        display_width = int(preview.shape[1] * scale)
                        preview_resized = cv2.resize(preview, (display_width, display_height))
                    else:
                        preview_resized = preview

                    cv2.imshow(preview_window, preview_resized)

                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("❌ Capture cancelled")
                        break
                    elif key == 32:  # SPACE
                        # Capture the current frame (use original, not resized)
                        self.image = current_frame.copy()
                        self.display_image = current_frame.copy()
                        captured = True
                        print(f"✅ Captured frame from ROS2 topic: {current_frame.shape}")
                        break
                else:
                    # No frame yet, show waiting message
                    if time.time() - start_time > timeout_sec:
                        print(f"❌ Error:  Timeout waiting for frame from topic '{topic_name}'")
                        break

                    cv2.waitKey(10)

            # Cleanup
            cv2.destroyWindow(preview_window)
            capture_node.destroy_node()
            rclpy.shutdown()

            return captured

        except Exception as e:
            print(f"❌ Error capturing from ROS2 topic: {e}")
            try:
                rclpy.shutdown()
            except:
                pass
            return False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append([float(x), float(y)])
                print(f"📍 Corner {len(self.corners)}: ({x}, {y})")

                # Draw the corner on the display image
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, f"{len(self.corners)}",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                # Draw lines between corners
                if len(self.corners) > 1:
                    for i in range(len(self.corners) - 1):
                        pt1 = tuple(map(int, self.corners[i]))
                        pt2 = tuple(map(int, self.corners[i + 1]))
                        cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)

                # Close the polygon if we have all 4 corners
                if len(self.corners) == 4:
                    pt1 = tuple(map(int, self.corners[3]))
                    pt2 = tuple(map(int, self.corners[0]))
                    cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
                    print("✅ All 4 corners selected!  Press any key to continue...")

                cv2.imshow(self.window_name, self.display_image)

    def select_corners_interactive(self) -> bool:
        """Interactive corner selection using OpenCV."""
        if self.image is None:
            print("❌ Error: No image loaded")
            return False

        print("\n" + "=" * 70)
        print("📐 CORNER SELECTION")
        print("=" * 70)
        print("Instructions:")
        print("  1. Click on the 4 corners of the work table")
        print("  2. Order:  Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("  3. Press 'r' to reset if you make a mistake")
        print("  4. Press any other key to finish after selecting 4 corners")
        print("=" * 70 + "\n")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Reset corners
            if key == ord('r'):
                self.corners = []
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
                print("🔄 Corners reset.  Click again...")

            # Finish selection
            elif key != 255 and len(self.corners) == 4:
                break

            # Allow ESC to cancel
            elif key == 27:
                cv2.destroyAllWindows()
                print("❌ Corner selection cancelled")
                return False

        cv2.destroyAllWindows()

        if len(self.corners) != 4:
            print(f"❌ Error: Expected 4 corners, got {len(self.corners)}")
            return False

        return True

    def bag_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for bag reference selection."""
        bag_type, temp_corners = param

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(temp_corners) < 4:
                temp_corners.append([float(x), float(y)])
                print(f"📍 {bag_type} bag corner {len(temp_corners)}: ({x}, {y})")

                # Draw the corner on the display image
                color = (255, 0, 255) if bag_type == "small" else (0, 165, 255)  # Magenta for small, Orange for large
                cv2.circle(self.display_image, (x, y), 5, color, -1)
                cv2.putText(self.display_image, f"{len(temp_corners)}",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

                # Draw lines between corners
                if len(temp_corners) > 1:
                    for i in range(len(temp_corners) - 1):
                        pt1 = tuple(map(int, temp_corners[i]))
                        pt2 = tuple(map(int, temp_corners[i + 1]))
                        cv2.line(self.display_image, pt1, pt2, color, 2)

                # Close the polygon if we have all 4 corners
                if len(temp_corners) == 4:
                    pt1 = tuple(map(int, temp_corners[3]))
                    pt2 = tuple(map(int, temp_corners[0]))
                    cv2.line(self.display_image, pt1, pt2, color, 2)
                    print(f"✅ All 4 corners selected for {bag_type} bag! Press any key to continue...")

                cv2.imshow(self.window_name, self.display_image)

    def select_bag_reference_interactive(self, bag_type: str) -> Optional[List[List[float]]]:
        """Interactive selection of a reference bag for size calibration."""
        if self.image is None:
            print("❌ Error: No image loaded")
            return None

        temp_corners = []
        color_name = "MAGENTA" if bag_type == "small" else "ORANGE"

        print("\n" + "=" * 70)
        print(f"🍞 {bag_type.upper()} BAG REFERENCE SELECTION ({color_name})")
        print("=" * 70)
        print("Instructions:")
        print(f"  1. Click on the 4 corners of a {bag_type} bread bag")
        print("  2. Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("  3. Press 'r' to reset if you make a mistake")
        print("  4. Press 's' to skip if no reference bag is available")
        print("  5. Press any other key to finish after selecting 4 corners")
        print("=" * 70 + "\n")

        self.window_name = f"{bag_type.capitalize()} Bag Reference - Click 4 Corners"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.bag_mouse_callback, (bag_type, temp_corners))
        cv2.imshow(self.window_name, self.display_image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Reset corners
            if key == ord('r'):
                # Remove drawn elements for this bag
                temp_corners.clear()
                # Redraw from scratch
                self.display_image = self.image.copy()
                # Redraw table corners
                for i, corner in enumerate(self.corners):
                    pt = tuple(map(int, corner))
                    cv2.circle(self.display_image, pt, 5, (0, 255, 0), -1)
                    cv2.putText(self.display_image, f"{i + 1}",
                                (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                for i in range(4):
                    pt1 = tuple(map(int, self.corners[i]))
                    pt2 = tuple(map(int, self.corners[(i + 1) % 4]))
                    cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
                # Redraw other bag references
                for saved_type, saved_corners in self.bag_references.items():
                    saved_color = (255, 0, 255) if saved_type == "small" else (0, 165, 255)
                    for i, corner in enumerate(saved_corners):
                        pt = tuple(map(int, corner))
                        cv2.circle(self.display_image, pt, 5, saved_color, -1)
                    for i in range(4):
                        pt1 = tuple(map(int, saved_corners[i]))
                        pt2 = tuple(map(int, saved_corners[(i + 1) % 4]))
                        cv2.line(self.display_image, pt1, pt2, saved_color, 2)
                cv2.imshow(self.window_name, self.display_image)
                print(f"🔄 {bag_type.capitalize()} bag corners reset. Click again...")

            # Skip this bag type
            elif key == ord('s'):
                print(f"⏭️  Skipped {bag_type} bag reference")
                cv2.destroyAllWindows()
                return None

            # Finish selection
            elif key != 255 and len(temp_corners) == 4:
                cv2.destroyAllWindows()
                return temp_corners

            # Allow ESC to cancel
            elif key == 27:
                cv2.destroyAllWindows()
                print("❌ Bag reference selection cancelled")
                return None

    def calculate_bag_area(self, corners: List[List[float]], transform: HomographyTransform) -> float:
        """Calculate the area of a bag given its corners."""
        # Convert corners to a bounding box (x1, y1, x2, y2)
        corners_array = np.array(corners, dtype=np.float32)
        x_coords = corners_array[:, 0]
        y_coords = corners_array[:, 1]

        bbox = (
            float(np.min(x_coords)),
            float(np.min(y_coords)),
            float(np.max(x_coords)),
            float(np.max(y_coords))
        )

        # Get size in cm
        size_cm = transform.get_bbox_size_cm(bbox)
        area_cm2 = size_cm[0] * size_cm[1]

        return area_cm2

    def validate_corners(self) -> bool:
        """Validate that corners form a reasonable quadrilateral."""
        if len(self.corners) != 4:
            return False

        # Check if corners form a convex quadrilateral
        corners_array = np.array(self.corners, dtype=np.float32)

        # Compute area using shoelace formula
        x = corners_array[:, 0]
        y = corners_array[:, 1]
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))

        if area < 10000:  # Minimum 100x100 pixels
            print(f"❌ Error: Table area too small ({area:.0f} px²). "
                  "Please select a larger region.")
            return False

        # Check for reasonable aspect ratio
        width = max(
            np.linalg.norm(corners_array[1] - corners_array[0]),
            np.linalg.norm(corners_array[2] - corners_array[3])
        )
        height = max(
            np.linalg.norm(corners_array[3] - corners_array[0]),
            np.linalg.norm(corners_array[2] - corners_array[1])
        )

        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            print(f"⚠️  Warning: Unusual aspect ratio ({aspect_ratio:.2f}). "
                  "This might indicate incorrect corner selection.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False

        print(f"✅ Corners validated:  area={area:.0f}px², aspect_ratio={aspect_ratio:.2f}")
        return True

    def get_table_dimensions(self) -> Optional[Tuple[float, float]]:
        """Prompt user for actual table dimensions."""
        print("\n" + "=" * 70)
        print("📏 TABLE DIMENSIONS")
        print("=" * 70)
        print("Enter the actual physical dimensions of your work table:")
        print("  (Measure the table where the bread bags are placed)")
        print("=" * 70 + "\n")

        while True:
            try:
                width_str = input("Table width (cm): ")
                width = float(width_str)

                if width < 40 or width > 200:
                    print("⚠️  Warning: Width should typically be between 40-200 cm")
                    response = input("Continue with this value?  (y/n): ")
                    if response.lower() != 'y':
                        continue

                height_str = input("Table height (cm): ")
                height = float(height_str)

                if height < 30 or height > 150:
                    print("⚠️  Warning: Height should typically be between 30-150 cm")
                    response = input("Continue with this value? (y/n): ")
                    if response.lower() != 'y':
                        continue

                print(f"✅ Table dimensions:  {width} cm × {height} cm")
                return (width, height)

            except ValueError:
                print("❌ Error: Please enter valid numbers")
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None

    def get_bag_thresholds(self, transform: HomographyTransform) -> Optional[Tuple[float, float]]:
        """Get bread bag size thresholds by drawing reference bags or manual input."""
        print("\n" + "=" * 70)
        print("🍞 BREAD BAG SIZE THRESHOLDS")
        print("=" * 70)
        print("Choose calibration method:")
        print("  1. Draw reference bags on the image (recommended)")
        print("  2. Enter threshold values manually")
        print("=" * 70 + "\n")

        while True:
            choice = input("Select method (1 or 2): ").strip()

            if choice == "1":
                # Interactive drawing mode
                return self._get_thresholds_by_drawing(transform)
            elif choice == "2":
                # Manual input mode
                return self._get_thresholds_manual()
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")

    def _get_thresholds_by_drawing(self, transform: HomographyTransform) -> Optional[Tuple[float, float]]:
        """Get thresholds by drawing reference bags."""
        print("\n📐 Drawing reference bags for automatic threshold calculation...")
        print("   You will mark the boundaries of small and large bags.")
        print()

        # Select small bag reference
        small_corners = self.select_bag_reference_interactive("small")
        if small_corners:
            self.bag_references["small"] = small_corners
            small_area = self.calculate_bag_area(small_corners, transform)
            print(f"✅ Small bag area: {small_area:.1f} cm²")
        else:
            small_area = None

        # Select large bag reference
        large_corners = self.select_bag_reference_interactive("large")
        if large_corners:
            self.bag_references["large"] = large_corners
            large_area = self.calculate_bag_area(large_corners, transform)
            print(f"✅ Large bag area: {large_area:.1f} cm²")
        else:
            large_area = None

        # Calculate thresholds based on drawn bags
        if small_area and large_area:
            # Use the average of small and large as the threshold
            small_threshold = small_area * 1.2  # 20% margin above small
            large_threshold = large_area * 0.8  # 20% margin below large

            print(f"\n✅ Calculated thresholds from reference bags:")
            print(f"   Small threshold: {small_threshold:.1f} cm² (based on small bag)")
            print(f"   Large threshold: {large_threshold:.1f} cm² (based on large bag)")
            print(f"\n   Classification:")
            print(f"   - Small bags:   area < {small_threshold:.1f} cm²")
            print(f"   - Medium bags: {small_threshold:.1f} cm² ≤ area < {large_threshold:.1f} cm²")
            print(f"   - Large bags:  area ≥ {large_threshold:.1f} cm²")

            response = input("\nAccept these thresholds? (y/n): ")
            if response.lower() == 'y':
                return (small_threshold, large_threshold)
            else:
                print("Switching to manual input...")
                return self._get_thresholds_manual()

        elif small_area:
            # Only small bag drawn
            small_threshold = small_area * 1.2
            large_threshold = small_area * 2.0  # Estimate large as 2x small

            print(f"\n✅ Using small bag reference:")
            print(f"   Small threshold: {small_threshold:.1f} cm²")
            print(f"   Large threshold: {large_threshold:.1f} cm² (estimated)")

            response = input("\nAccept these thresholds? (y/n): ")
            if response.lower() == 'y':
                return (small_threshold, large_threshold)
            else:
                print("Switching to manual input...")
                return self._get_thresholds_manual()

        elif large_area:
            # Only large bag drawn
            small_threshold = large_area * 0.5  # Estimate small as 0.5x large
            large_threshold = large_area * 0.8

            print(f"\n✅ Using large bag reference:")
            print(f"   Small threshold: {small_threshold:.1f} cm² (estimated)")
            print(f"   Large threshold: {large_threshold:.1f} cm²")

            response = input("\nAccept these thresholds? (y/n): ")
            if response.lower() == 'y':
                return (small_threshold, large_threshold)
            else:
                print("Switching to manual input...")
                return self._get_thresholds_manual()
        else:
            # No bags drawn
            print("\n⚠️  No reference bags drawn.  Switching to manual input...")
            return self._get_thresholds_manual()

    def _get_thresholds_manual(self) -> Optional[Tuple[float, float]]:
        """Get thresholds via manual keyboard input."""
        print("\n" + "=" * 70)
        print("⌨️  MANUAL THRESHOLD INPUT")
        print("=" * 70)
        print("Enter the area thresholds for classifying bread bags:")
        print("  Small bags:   area < small_threshold")
        print("  Medium bags: small_threshold ≤ area < large_threshold")
        print("  Large bags:   area ≥ large_threshold")
        print("=" * 70 + "\n")

        while True:
            try:
                small_str = input("Small bag threshold (cm²) [default: 100. 0]: ").strip()
                small_threshold = float(small_str) if small_str else 100.0

                if small_threshold < 10 or small_threshold > 500:
                    print("⚠️  Warning: Small threshold should typically be between 10-500 cm²")
                    response = input("Continue with this value? (y/n): ")
                    if response.lower() != 'y':
                        continue

                large_str = input("Large bag threshold (cm²) [default: 150.0]: ").strip()
                large_threshold = float(large_str) if large_str else 150.0

                if large_threshold < 20 or large_threshold > 1000:
                    print("⚠️  Warning: Large threshold should typically be between 20-1000 cm²")
                    response = input("Continue with this value? (y/n): ")
                    if response.lower() != 'y':
                        continue

                if large_threshold <= small_threshold:
                    print("❌ Error: Large threshold must be greater than small threshold")
                    continue

                print(f"✅ Bag thresholds: Small < {small_threshold} cm² < Medium < {large_threshold} cm² ≤ Large")
                return (small_threshold, large_threshold)

            except ValueError:
                print("❌ Error: Please enter valid numbers")
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None

    def test_calibration(self, transform: HomographyTransform) -> bool:
        """Test the calibration with sample bboxes and drawn references."""
        if self.image is None or not transform.is_calibrated():
            return False

        print("\n" + "=" * 70)
        print("🧪 CALIBRATION TEST")
        print("=" * 70)
        print("Visualizing calibration with reference bags and test box...")
        print("=" * 70 + "\n")

        # Create visualization image
        test_image = self.display_image.copy()

        # Draw a test box in the center
        corners_array = np.array(self.corners, dtype=np.float32)
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])

        test_box_size_cm = 10.
        0
        px_per_cm = transform.px_per_cm if transform.px_per_cm else 10.0
        test_box_size_px = test_box_size_cm * px_per_cm / 2

        test_bbox = (
            center_x - test_box_size_px,
            center_y - test_box_size_px,
            center_x + test_box_size_px,
            center_y + test_box_size_px
        )

        size_cm = transform.get_bbox_size_cm(test_bbox)
        area_cm2 = size_cm[0] * size_cm[1]

        # Draw test box
        x1, y1, x2, y2 = map(int, test_bbox)
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
        cv2.putText(test_image, f"Test:  {size_cm[0]:.1f} x {size_cm[1]:.1f} cm",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(test_image, f"Area: {area_cm2:.1f} cm²",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add legend
        legend_y = 30
        cv2.putText(test_image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.rectangle(test_image, (10, legend_y - 10), (30, legend_y + 5), (0, 255, 0), -1)
        cv2.putText(test_image, "Table", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        if "small" in self.bag_references:
            cv2.rectangle(test_image, (10, legend_y - 10), (30, legend_y + 5), (255, 0, 255), -1)
            cv2.putText(test_image, "Small Bag", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        if "large" in self.bag_references:
            cv2.rectangle(test_image, (10, legend_y - 10), (30, legend_y + 5), (0, 165, 255), -1)
            cv2.putText(test_image, "Large Bag", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        cv2.rectangle(test_image, (10, legend_y - 10), (30, legend_y + 5), (0, 255, 255), -1)
        cv2.putText(test_image, "Test Box", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Calibration Test", test_image)
        print(f"Test bbox size: {size_cm[0]:.1f} cm × {size_cm[1]:.1f} cm")
        print(f"Test bbox area: {area_cm2:.1f} cm²")
        print(f"Pixel density: {px_per_cm:.2f} px/cm")

        if self.bag_references:
            print("\nReference bags:")
            for bag_type, corners in self.bag_references.items():
                area = self.calculate_bag_area(corners, transform)
                print(f"  {bag_type.capitalize()} bag: {area:.1f} cm²")

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True

    def generate_env_config(
            self,
            table_size_cm: Tuple[float, float],
            small_threshold: float = 100.0,
            large_threshold: float = 150.0
    ) -> str:
        """Generate environment variable configuration."""
        corners_json = json.dumps(self.corners)

        config = f"""
# Homography Configuration
# Add these lines to your .env file

HOMOGRAPHY_ENABLED=true
HOMOGRAPHY_TABLE_CORNERS='{corners_json}'
HOMOGRAPHY_TABLE_WIDTH_CM={table_size_cm[0]}
HOMOGRAPHY_TABLE_HEIGHT_CM={table_size_cm[1]}
HOMOGRAPHY_SMALL_THRESHOLD_CM2={small_threshold}
HOMOGRAPHY_LARGE_THRESHOLD_CM2={large_threshold}
"""
        return config

    def save_calibration(
            self,
            output_dir: str,
            table_size_cm: Tuple[float, float],
            small_threshold: float = 100.0,
            large_threshold: float = 150.0
    ) -> bool:
        """Save calibration data to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save calibration image
        image_path = os.path.join(output_dir, "calibration_image.jpg")
        cv2.imwrite(image_path, self.display_image)
        print(f"✅ Saved calibration image:  {image_path}")

        # Save calibration JSON
        calibration_data = {
            "table_corners_px": self.corners,
            "table_size_cm": {
                "width": table_size_cm[0],
                "height": table_size_cm[1]
            },
            "bag_thresholds_cm2": {
                "small": small_threshold,
                "large": large_threshold
            },
            "bag_references": self.bag_references,  # Save reference bag corners
            "image_shape": self.image.shape[: 2] if self.image is not None else None
        }

        json_path = os.path.join(output_dir, "calibration_data.json")
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"✅ Saved calibration data:  {json_path}")

        return True


def dry_run_test():
    """Test existing calibration from environment variables."""
    print("\n" + "=" * 70)
    print("🔍 DRY RUN - Testing Existing Calibration")
    print("=" * 70 + "\n")

    from src.classifier.homography import get_homography_transform

    transform = get_homography_transform()

    if not transform.enabled:
        print("❌ Homography is not enabled in environment variables")
        print("   Set HOMOGRAPHY_ENABLED=true in your .env file")
        return False

    if not transform.is_calibrated():
        print("❌ Homography is enabled but not calibrated")
        print("   Missing calibration data in environment variables")
        return False

    info = transform.get_calibration_info()
    print("✅ Homography is enabled and calibrated!")
    print(f"\nCalibration info:")
    print(f"  Table corners: {info['table_corners_px']}")
    print(f"  Table size: {info['table_size_cm']} cm")
    print(f"  Pixel density: {info['px_per_cm']:.2f} px/cm")

    # Test with a sample bbox
    print("\n📊 Testing with sample bboxes:")
    test_bboxes = [
        (100, 100, 150, 150),  # 50x50 px
        (200, 200, 280, 280),  # 80x80 px
        (300, 300, 400, 400),  # 100x100 px
    ]

    for bbox in test_bboxes:
        size_cm = transform.get_bbox_size_cm(bbox)
        area_cm2 = size_cm[0] * size_cm[1]
        print(f"  Bbox {bbox}: {size_cm[0]:.1f} × {size_cm[1]:.1f} cm = {area_cm2:.1f} cm²")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive homography calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate from video frame
  python scripts/calibrate_homography.py --video path/to/video.mp4 --frame 100

  # Calibrate from image
  python scripts/calibrate_homography.py --image path/to/frame.jpg

  # Calibrate from live camera
  python scripts/calibrate_homography.py --camera 0

  # Calibrate from ROS2 NV12 topic
  python scripts/calibrate_homography.py --ros2-topic /nv12_images

  # Test existing calibration
  python scripts/calibrate_homography.py --dry-run
        """
    )

    # Input source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--video', type=str, help='Path to video file')
    source_group.add_argument('--image', type=str, help='Path to image file')
    source_group.add_argument('--camera', type=int, help='Camera device ID (e.g., 0)')
    source_group.add_argument('--ros2-topic', type=str,
                              help='ROS2 topic name for NV12 images (e.g., /nv12_images)')
    source_group.add_argument('--dry-run', action='store_true',
                              help='Test existing calibration from environment')

    # Additional options
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame number to extract from video (default: 0)')
    parser.add_argument('--output', type=str, default='data/calibration',
                        help='Output directory for calibration files (default: data/calibration)')
    parser.add_argument('--ros2-timeout', type=float, default=10.0,
                        help='Timeout for ROS2 frame capture in seconds (default: 10.0)')

    args = parser.parse_args()

    # Dry run mode
    if args.dry_run:
        success = dry_run_test()
        sys.exit(0 if success else 1)

    # Check ROS2 availability if needed
    if args.ros2_topic and not ROS2_AVAILABLE:
        print("❌ Error: ROS2 libraries not available")
        print("   Install ROS2 and required packages:")
        print("   - rclpy")
        print("   - hbm_img_msgs")
        sys.exit(1)

    # Create calibration tool
    tool = CalibrationTool()

    # Load image based on source
    print("\n" + "=" * 70)
    print("🎯 HOMOGRAPHY CALIBRATION TOOL")
    print("=" * 70 + "\n")

    if args.video:
        if not tool.load_image_from_video(args.video, args.frame):
            sys.exit(1)
    elif args.image:
        if not tool.load_image_from_file(args.image):
            sys.exit(1)
    elif args.camera is not None:
        if not tool.load_image_from_camera(args.camera):
            sys.exit(1)
    elif args.ros2_topic:
        if not tool.load_image_from_ros2_topic(args.ros2_topic, args.ros2_timeout):
            sys.exit(1)

    # Interactive corner selection
    if not tool.select_corners_interactive():
        sys.exit(1)

    # Validate corners
    if not tool.validate_corners():
        print("❌ Corner validation failed.  Please try again.")
        sys.exit(1)

    # Get table dimensions
    table_size_cm = tool.get_table_dimensions()
    if table_size_cm is None:
        sys.exit(1)

    # Create homography transform (needed for bag threshold calculation)
    transform = HomographyTransform(
        table_corners_px=tool.corners,
        table_size_cm=table_size_cm,
        enabled=True
    )

    if not transform.is_calibrated():
        print("❌ Failed to compute homography transformation")
        sys.exit(1)

    # Get bag size thresholds (interactive drawing or manual input)
    bag_thresholds = tool.get_bag_thresholds(transform)
    if bag_thresholds is None:
        sys.exit(1)

    small_threshold, large_threshold = bag_thresholds

    # Test calibration
    tool.test_calibration(transform)

    # Generate and display configuration
    print("\n" + "=" * 70)
    print("📝 CONFIGURATION")
    print("=" * 70)

    env_config = tool.generate_env_config(
        table_size_cm,
        small_threshold,
        large_threshold
    )
    print(env_config)

    # Save calibration data
    if tool.save_calibration(args.output, table_size_cm, small_threshold, large_threshold):
        env_file_path = os.path.join(args.output, "calibration. env")
        with open(env_file_path, 'w') as f:
            f.write(env_config)
        print(f"✅ Saved environment config: {env_file_path}")

    print("\n" + "=" * 70)
    print("✅ CALIBRATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Copy the environment variables above to your .env file")
    print("  2. Restart your application")
    print("  3. Verify homography is working with:  python scripts/calibrate_homography.py --dry-run")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()