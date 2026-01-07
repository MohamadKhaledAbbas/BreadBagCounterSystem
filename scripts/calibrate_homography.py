#!/usr/bin/env python3
"""
Homography Calibration Tool for Bread Bag Counter System.

This script provides an interactive calibration tool for setting up the homography
transformation used for accurate size-based bread bag classification.

Features:
- Interactive corner selection from video frame, image, or live camera
- Table dimension validation
- Calibration testing with sample bboxes
- Environment variable generation for .env file
- Save calibration data (image with corners + JSON)

Usage:
    # From video frame
    python scripts/calibrate_homography.py --video path/to/video.mp4 --frame 100
    
    # From image
    python scripts/calibrate_homography.py --image path/to/frame.jpg
    
    # From live camera
    python scripts/calibrate_homography.py --camera 0
    
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
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.homography import HomographyTransform
from src.utils.AppLogging import logger


class CalibrationTool:
    """Interactive homography calibration tool."""
    
    def __init__(self):
        self.corners: List[List[float]] = []
        self.image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.window_name = "Homography Calibration - Click 4 Corners (TL, TR, BR, BL)"
        
    def load_image_from_video(self, video_path: str, frame_number: int) -> bool:
        """Load a frame from a video file."""
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
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
            print(f"❌ Error: Image file not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return False
        
        self.image = image
        self.display_image = image.copy()
        print(f"✅ Loaded image: {image.shape}")
        return True
    
    def load_image_from_camera(self, camera_id: int) -> bool:
        """Capture a frame from a live camera."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return False
        
        print("📷 Camera opened. Press SPACE to capture, ESC to cancel...")
        
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
                print(f"✅ Captured frame: {frame.shape}")
                return True
    
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
                    print("✅ All 4 corners selected! Press any key to continue...")
                
                cv2.imshow(self.window_name, self.display_image)
    
    def select_corners_interactive(self) -> bool:
        """Interactive corner selection using OpenCV."""
        if self.image is None:
            print("❌ Error: No image loaded")
            return False
        
        print("\n" + "="*70)
        print("📐 CORNER SELECTION")
        print("="*70)
        print("Instructions:")
        print("  1. Click on the 4 corners of the work table")
        print("  2. Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("  3. Press 'r' to reset if you make a mistake")
        print("  4. Press any other key to finish after selecting 4 corners")
        print("="*70 + "\n")
        
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
                print("🔄 Corners reset. Click again...")
            
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
    
    def validate_corners(self) -> bool:
        """Validate that corners form a reasonable quadrilateral."""
        if len(self.corners) != 4:
            return False
        
        # Check if corners form a convex quadrilateral
        corners_array = np.array(self.corners, dtype=np.float32)
        
        # Compute area using shoelace formula
        x = corners_array[:, 0]
        y = corners_array[:, 1]
        area = 0.5 * abs(sum(x[i]*y[(i+1)%4] - x[(i+1)%4]*y[i] for i in range(4)))
        
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
        
        print(f"✅ Corners validated: area={area:.0f}px², aspect_ratio={aspect_ratio:.2f}")
        return True
    
    def get_table_dimensions(self) -> Optional[Tuple[float, float]]:
        """Prompt user for actual table dimensions."""
        print("\n" + "="*70)
        print("📏 TABLE DIMENSIONS")
        print("="*70)
        print("Enter the actual physical dimensions of your work table:")
        print("  (Measure the table where the bread bags are placed)")
        print("="*70 + "\n")
        
        while True:
            try:
                width_str = input("Table width (cm): ")
                width = float(width_str)
                
                if width < 40 or width > 200:
                    print("⚠️  Warning: Width should typically be between 40-200 cm")
                    response = input("Continue with this value? (y/n): ")
                    if response.lower() != 'y':
                        continue
                
                height_str = input("Table height (cm): ")
                height = float(height_str)
                
                if height < 30 or height > 150:
                    print("⚠️  Warning: Height should typically be between 30-150 cm")
                    response = input("Continue with this value? (y/n): ")
                    if response.lower() != 'y':
                        continue
                
                print(f"✅ Table dimensions: {width} cm × {height} cm")
                return (width, height)
                
            except ValueError:
                print("❌ Error: Please enter valid numbers")
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None
    
    def test_calibration(self, transform: HomographyTransform) -> bool:
        """Test the calibration with a sample bbox."""
        if self.image is None or not transform.is_calibrated():
            return False
        
        print("\n" + "="*70)
        print("🧪 CALIBRATION TEST")
        print("="*70)
        print("Drawing a sample bounding box to test size measurement...")
        print("="*70 + "\n")
        
        # Create a sample bbox in the center of the table
        corners_array = np.array(self.corners, dtype=np.float32)
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        
        # Create a 10x10cm test box (approximately)
        test_box_size_cm = 10.0
        px_per_cm = transform.px_per_cm if transform.px_per_cm else 10.0
        test_box_size_px = test_box_size_cm * px_per_cm / 2  # Half size for radius
        
        test_bbox = (
            center_x - test_box_size_px,
            center_y - test_box_size_px,
            center_x + test_box_size_px,
            center_y + test_box_size_px
        )
        
        # Measure the test bbox
        size_cm = transform.get_bbox_size_cm(test_bbox)
        area_cm2 = size_cm[0] * size_cm[1]
        
        # Visualize
        test_image = self.display_image.copy()
        x1, y1, x2, y2 = map(int, test_bbox)
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(test_image, f"Test: {size_cm[0]:.1f} x {size_cm[1]:.1f} cm", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(test_image, f"Area: {area_cm2:.1f} cm²", 
                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow("Calibration Test", test_image)
        print(f"Test bbox size: {size_cm[0]:.1f} cm × {size_cm[1]:.1f} cm")
        print(f"Test bbox area: {area_cm2:.1f} cm²")
        print(f"Pixel density: {px_per_cm:.2f} px/cm")
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
        table_size_cm: Tuple[float, float]
    ) -> bool:
        """Save calibration data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save calibration image
        image_path = os.path.join(output_dir, "calibration_image.jpg")
        cv2.imwrite(image_path, self.display_image)
        print(f"✅ Saved calibration image: {image_path}")
        
        # Save calibration JSON
        calibration_data = {
            "table_corners_px": self.corners,
            "table_size_cm": {
                "width": table_size_cm[0],
                "height": table_size_cm[1]
            },
            "image_shape": self.image.shape[:2] if self.image is not None else None
        }
        
        json_path = os.path.join(output_dir, "calibration_data.json")
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"✅ Saved calibration data: {json_path}")
        
        return True


def dry_run_test():
    """Test existing calibration from environment variables."""
    print("\n" + "="*70)
    print("🔍 DRY RUN - Testing Existing Calibration")
    print("="*70 + "\n")
    
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
  
  # Test existing calibration
  python scripts/calibrate_homography.py --dry-run
        """
    )
    
    # Input source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--video', type=str, help='Path to video file')
    source_group.add_argument('--image', type=str, help='Path to image file')
    source_group.add_argument('--camera', type=int, help='Camera device ID (e.g., 0)')
    source_group.add_argument('--dry-run', action='store_true',
                             help='Test existing calibration from environment')
    
    # Additional options
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame number to extract from video (default: 0)')
    parser.add_argument('--output', type=str, default='data/calibration',
                       help='Output directory for calibration files (default: data/calibration)')
    parser.add_argument('--small-threshold', type=float, default=100.0,
                       help='Small bag threshold in cm² (default: 100.0)')
    parser.add_argument('--large-threshold', type=float, default=150.0,
                       help='Large bag threshold in cm² (default: 150.0)')
    
    args = parser.parse_args()
    
    # Dry run mode
    if args.dry_run:
        success = dry_run_test()
        sys.exit(0 if success else 1)
    
    # Create calibration tool
    tool = CalibrationTool()
    
    # Load image based on source
    print("\n" + "="*70)
    print("🎯 HOMOGRAPHY CALIBRATION TOOL")
    print("="*70 + "\n")
    
    if args.video:
        if not tool.load_image_from_video(args.video, args.frame):
            sys.exit(1)
    elif args.image:
        if not tool.load_image_from_file(args.image):
            sys.exit(1)
    elif args.camera is not None:
        if not tool.load_image_from_camera(args.camera):
            sys.exit(1)
    
    # Interactive corner selection
    if not tool.select_corners_interactive():
        sys.exit(1)
    
    # Validate corners
    if not tool.validate_corners():
        print("❌ Corner validation failed. Please try again.")
        sys.exit(1)
    
    # Get table dimensions
    table_size_cm = tool.get_table_dimensions()
    if table_size_cm is None:
        sys.exit(1)
    
    # Create homography transform
    transform = HomographyTransform(
        table_corners_px=tool.corners,
        table_size_cm=table_size_cm,
        enabled=True
    )
    
    if not transform.is_calibrated():
        print("❌ Failed to compute homography transformation")
        sys.exit(1)
    
    # Test calibration
    tool.test_calibration(transform)
    
    # Generate and display configuration
    print("\n" + "="*70)
    print("📝 CONFIGURATION")
    print("="*70)
    
    env_config = tool.generate_env_config(
        table_size_cm, 
        args.small_threshold, 
        args.large_threshold
    )
    print(env_config)
    
    # Save calibration data
    if tool.save_calibration(args.output, table_size_cm):
        env_file_path = os.path.join(args.output, "calibration.env")
        with open(env_file_path, 'w') as f:
            f.write(env_config)
        print(f"✅ Saved environment config: {env_file_path}")
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Copy the environment variables above to your .env file")
    print("  2. Restart your application")
    print("  3. Verify homography is working with: python scripts/calibrate_homography.py --dry-run")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
