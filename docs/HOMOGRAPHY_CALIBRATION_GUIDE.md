# Homography Calibration Guide

## Table of Contents
1. [Overview](#overview)
2. [Why Homography?](#why-homography)
3. [Calibration Process](#calibration-process)
4. [Configuration](#configuration)
5. [Testing and Verification](#testing-and-verification)
6. [Troubleshooting](#troubleshooting)
7. [Tuning Thresholds](#tuning-thresholds)

---

## Overview

Homography-based size classification transforms bounding box measurements from pixel coordinates (camera perspective) to real-world coordinates (centimeters), enabling accurate size-based bread bag classification regardless of camera angle or distance.

**Key Benefits:**
- ✅ **Physically accurate**: Measures actual bread size, not visual perception
- ✅ **Perspective-invariant**: Works at any camera angle/distance
- ✅ **Debuggable**: Real measurements (cm²) vs arbitrary pixel thresholds
- ✅ **Lightweight**: ~1ms overhead, no ML model needed
- ✅ **Production-grade**: Used in industrial inspection systems worldwide

---

## Why Homography?

### The Problem with Pixel-Based Classification

Traditional size-based classification using bounding box dimensions in pixels is unreliable:

```
┌─────────────────────────────────────┐
│  Camera View (Perspective)          │
│                                     │
│    [Small bag far away]             │  Same pixel size!
│         ┌─────┐                     │
│         │ 80px│                     │
│         └─────┘                     │
│                                     │
│    [Large bag close up]             │
│         ┌─────┐                     │
│         │ 80px│                     │
│         └─────┘                     │
└─────────────────────────────────────┘
```

**Issues:**
- Perspective distortion: Same object appears different sizes at different distances
- Camera angle variations affect perceived size
- No physical context - 10,000 px² could be any actual size

### The Solution: Homography

Homography uses the work table as a reference plane to compute a perspective transformation:

```
┌─────────────────────────────────────┐
│  Camera View                        │
│       /\                            │
│      /  \     Table corners         │
│     /____\    (trapezoid)           │
│                                     │
│           ↓  Homography             │
│                                     │
│  Bird's-Eye View                    │
│    ┌────────┐                       │
│    │        │   Table (rectangle)   │
│    │  80cm  │   Known size!         │
│    └────────┘                       │
└─────────────────────────────────────┘
```

**Benefits:**
1. Transform bounding boxes to real-world coordinates
2. Measure actual bread size in cm²
3. Classify using physical thresholds (e.g., < 100 cm² = Small)

---

## Calibration Process

### Prerequisites

1. **Video/Image Requirements:**
   - Clear view of the entire work table
   - Good lighting, minimal shadows
   - Table visible with clear corners
   - Representative of actual production scene

2. **Physical Measurements:**
   - Measure your work table dimensions (width × height in cm)
   - Use a measuring tape for accuracy
   - Typical table: 60-100 cm wide × 40-80 cm deep

### Step-by-Step Calibration

#### 1. Prepare Calibration Image

Choose one of these methods:

**Option A: From Video File**
```bash
python scripts/calibrate_homography.py --video path/to/video.mp4 --frame 100
```

**Option B: From Image File**
```bash
python scripts/calibrate_homography.py --image path/to/frame.jpg
```

**Option C: From Live Camera**
```bash
python scripts/calibrate_homography.py --camera 0
```

#### 2. Select Table Corners

The tool will open a window showing your image:

```
📐 CORNER SELECTION
======================================================================
Instructions:
  1. Click on the 4 corners of the work table
  2. Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
  3. Press 'r' to reset if you make a mistake
  4. Press any other key to finish after selecting 4 corners
======================================================================
```

**Tips for Accurate Corner Selection:**
- Click on the exact corners where the table edges meet
- Follow clockwise order starting from top-left
- Ensure corners form a quadrilateral (not crossed lines)
- If you make a mistake, press 'r' to reset and start over

**Visual Guide:**
```
    1 (TL) ●────────────● 2 (TR)
           │            │
           │   Table    │
           │            │
    4 (BL) ●────────────● 3 (BR)
```

#### 3. Enter Table Dimensions

The tool will prompt you for physical measurements:

```
📏 TABLE DIMENSIONS
======================================================================
Enter the actual physical dimensions of your work table:
  (Measure the table where the bread bags are placed)
======================================================================

Table width (cm): 80
Table height (cm): 60
✅ Table dimensions: 80.0 cm × 60.0 cm
```

**Measurement Tips:**
- Measure from edge to edge where you clicked corners
- Double-check measurements for accuracy
- Typical tables: 60-100 cm wide, 40-80 cm deep

#### 4. Test Calibration

The tool will display a test bounding box:

```
🧪 CALIBRATION TEST
======================================================================
Drawing a sample bounding box to test size measurement...
======================================================================

Test bbox size: 10.2 cm × 10.1 cm
Test bbox area: 103.0 cm²
Pixel density: 12.50 px/cm

Press any key to continue...
```

**What to Check:**
- Pixel density should be reasonable (8-20 px/cm for typical setups)
- Test bbox measurements should look plausible
- If values seem wrong, restart and check corner selection

#### 5. Save Configuration

The tool generates environment variables for your `.env` file:

```
📝 CONFIGURATION
======================================================================

# Homography Configuration
# Add these lines to your .env file

HOMOGRAPHY_ENABLED=true
HOMOGRAPHY_TABLE_CORNERS='[[150,100],[950,120],[980,650],[120,680]]'
HOMOGRAPHY_TABLE_WIDTH_CM=80.0
HOMOGRAPHY_TABLE_HEIGHT_CM=60.0
HOMOGRAPHY_SMALL_THRESHOLD_CM2=100.0
HOMOGRAPHY_LARGE_THRESHOLD_CM2=150.0

✅ Saved calibration image: data/calibration/calibration_image.jpg
✅ Saved calibration data: data/calibration/calibration_data.json
✅ Saved environment config: data/calibration/calibration.env
```

---

## Configuration

### Environment Variables

Add these variables to your `.env` file (located at the project root):

```bash
# Enable homography transformation
HOMOGRAPHY_ENABLED=true

# Table corner positions in pixel coordinates
# Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
HOMOGRAPHY_TABLE_CORNERS='[[150,100],[950,120],[980,650],[120,680]]'

# Physical table dimensions in centimeters
HOMOGRAPHY_TABLE_WIDTH_CM=80.0
HOMOGRAPHY_TABLE_HEIGHT_CM=60.0

# Classification thresholds in square centimeters
# Adjust these based on your actual bread bag sizes
HOMOGRAPHY_SMALL_THRESHOLD_CM2=100.0
HOMOGRAPHY_LARGE_THRESHOLD_CM2=150.0
```

### Applying Configuration

1. **Copy** the environment variables from calibration output
2. **Paste** into your `.env` file (create if it doesn't exist)
3. **Restart** the application to load new configuration
4. **Verify** with dry-run test (see next section)

### Configuration in Code

Alternatively, configure programmatically:

```python
from src.classifier.homography import HomographyTransform

transform = HomographyTransform(
    table_corners_px=[[150, 100], [950, 120], [980, 650], [120, 680]],
    table_size_cm=(80.0, 60.0),
    enabled=True
)

# Use transform
size_cm = transform.get_bbox_size_cm(bbox)
area_cm2 = size_cm[0] * size_cm[1]
```

---

## Testing and Verification

### Dry-Run Test

Verify your calibration is working:

```bash
python scripts/calibrate_homography.py --dry-run
```

**Expected Output:**
```
🔍 DRY RUN - Testing Existing Calibration
======================================================================

✅ Homography is enabled and calibrated!

Calibration info:
  Table corners: [[150, 100], [950, 120], [980, 650], [120, 680]]
  Table size: (80.0, 60.0) cm
  Pixel density: 12.50 px/cm

📊 Testing with sample bboxes:
  Bbox (100, 100, 150, 150): 4.0 × 4.0 cm = 16.0 cm²
  Bbox (200, 200, 280, 280): 6.4 × 6.4 cm = 40.96 cm²
  Bbox (300, 300, 400, 400): 8.0 × 8.0 cm = 64.0 cm²
```

### Production Testing

Monitor logs during operation to verify homography is working:

```python
# In src/classifier/disambiguation_v2.py, enable debug logging
disambiguation_v2_debug_logging=True
```

Look for log entries like:
```
[Disambiguation V2] Homography: area=95.5cm², size=9.5x10.1cm, bin=small
[Disambiguation V2] original=Brown_Orange_Overlay, final=Brown_Orange_Small,
                    size_bin=small, homography=True, tier=high
```

### Validation Checklist

- [ ] Calibration completes without errors
- [ ] Pixel density is reasonable (8-20 px/cm)
- [ ] Test bbox measurements look plausible
- [ ] Dry-run test shows expected results
- [ ] Application logs show `homography=True` during operation
- [ ] Size classifications match expected bread sizes

---

## Troubleshooting

### Issue: "Homography is not enabled"

**Cause:** Environment variable not set or `.env` file not loaded

**Solution:**
1. Check `.env` file exists in project root
2. Verify `HOMOGRAPHY_ENABLED=true` is present
3. Restart application to reload configuration
4. Check application startup logs for configuration errors

### Issue: "Failed to compute homography"

**Cause:** Invalid corner selection (degenerate quadrilateral)

**Solution:**
1. Re-run calibration with `--image` or `--video`
2. Ensure corners form a proper quadrilateral (not crossed)
3. Check corners are in correct order (clockwise from top-left)
4. Verify table is clearly visible in calibration image

### Issue: Unrealistic Size Measurements

**Example:** Small bags showing as 500 cm² or negative sizes

**Cause:** Incorrect table dimensions or corner positions

**Solution:**
1. Verify physical table measurements are correct
2. Double-check corner selection (use `--image` to review)
3. Ensure corners are clicked accurately on table edges
4. Re-calibrate with a clearer image of the table

### Issue: Pixel Density Too High/Low

**Example:** `px_per_cm: 50.0` (too high) or `px_per_cm: 2.0` (too low)

**Typical Range:** 8-20 px/cm for most camera setups

**Solution:**
1. Check camera resolution matches production setup
2. Verify table dimensions are measured correctly
3. Ensure calibration image is not cropped or scaled
4. Re-calibrate with full-resolution frame

### Issue: Gray Zone Classifications Still Occurring

**Cause:** Thresholds might be too narrow

**Solution:** Adjust thresholds (see next section)

---

## Tuning Thresholds

### Understanding Thresholds

Homography classification uses two thresholds:

```
      0                100                150                ∞
      ├─────────────────┼──────────────────┼─────────────────┤
      Small             Gray Zone          Large
```

- **HOMOGRAPHY_SMALL_THRESHOLD_CM2**: Maximum area for "Small" classification
- **HOMOGRAPHY_LARGE_THRESHOLD_CM2**: Minimum area for "Large" classification
- **Gray Zone**: Between thresholds - ambiguous size

### Measuring Actual Bread Sizes

To set optimal thresholds, measure your actual bread bags:

1. **Collect Samples:**
   - 10-20 bags of each class (Small and Regular/Large)
   - Place on calibrated table
   - Capture frames or use existing footage

2. **Measure in Production:**
   - Enable debug logging: `disambiguation_v2_debug_logging=True`
   - Run system with representative bags
   - Collect size measurements from logs

3. **Analyze Distribution:**
   - Find minimum "Large" bag size
   - Find maximum "Small" bag size
   - Look for clear separation between classes

### Example Threshold Tuning

**Scenario:** Small bags are 60-90 cm², Large bags are 120-180 cm²

**Optimal Thresholds:**
```bash
HOMOGRAPHY_SMALL_THRESHOLD_CM2=100.0   # Above all Small bags
HOMOGRAPHY_LARGE_THRESHOLD_CM2=110.0   # Below all Large bags
```

**Result:** Clear separation, minimal gray zone

**If bags overlap (e.g., Small up to 110 cm², Large from 100 cm²):**
```bash
HOMOGRAPHY_SMALL_THRESHOLD_CM2=95.0    # Conservative Small threshold
HOMOGRAPHY_LARGE_THRESHOLD_CM2=115.0   # Conservative Large threshold
```

**Result:** Some gray zone cases, but avoids misclassification

### Iterative Tuning Process

1. **Start with defaults** (100/150 cm²)
2. **Collect data** for 1-2 hours of production
3. **Analyze** size distributions in logs
4. **Adjust** thresholds based on actual sizes
5. **Test** for 30 minutes, verify improvements
6. **Repeat** until classification is reliable

### Monitoring Metrics

Track these metrics during tuning:

- **Gray zone rate**: % of bags in gray zone (target: <10%)
- **Misclassification rate**: Known bags classified incorrectly (target: <2%)
- **Confidence tier distribution**: % high vs low confidence (target: >80% high)

---

## Advanced Topics

### Multiple Camera Setups

If you have multiple cameras with different angles:

1. Calibrate each camera separately
2. Store calibrations with camera IDs
3. Load appropriate calibration based on camera source

### Dynamic Table Configuration

For systems with adjustable tables:

1. Store multiple calibrations (e.g., `table_config_A.json`)
2. Switch calibrations via configuration or API
3. Validate calibration before processing each session

### Integration with Existing Code

The homography system integrates seamlessly:

```python
from src.classifier.disambiguation_v2 import disambiguate_v2

# Automatically uses homography if enabled
result = disambiguate_v2(
    original_label="Brown_Orange_Family",
    confidence=0.75,
    bbox=(150, 200, 250, 300),
    is_open=False,
    config=tracking_config
)

# Check if homography was used
if result.metadata.get('homography_used'):
    print(f"Classified using real-world measurements: {result.label}")
else:
    print(f"Fallback to pixel-based classification: {result.label}")
```

---

## Quick Reference

### Calibration Command

```bash
# Interactive calibration from video
python scripts/calibrate_homography.py --video video.mp4 --frame 100

# Test existing calibration
python scripts/calibrate_homography.py --dry-run
```

### Environment Variables Template

```bash
HOMOGRAPHY_ENABLED=true
HOMOGRAPHY_TABLE_CORNERS='[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'
HOMOGRAPHY_TABLE_WIDTH_CM=80.0
HOMOGRAPHY_TABLE_HEIGHT_CM=60.0
HOMOGRAPHY_SMALL_THRESHOLD_CM2=100.0
HOMOGRAPHY_LARGE_THRESHOLD_CM2=150.0
```

### Common Issues Quick Fix

| Issue | Quick Fix |
|-------|-----------|
| "Not enabled" | Add `HOMOGRAPHY_ENABLED=true` to `.env` |
| Unrealistic sizes | Re-calibrate with correct table dimensions |
| High gray zone rate | Tune thresholds based on actual bag sizes |
| Pixel density odd | Verify table measurements and corner selection |

---

## Support

For additional help:
1. Check application logs in `data/logs/`
2. Review calibration files in `data/calibration/`
3. Run dry-run test to verify configuration
4. Re-calibrate if issues persist

**Remember:** Accurate calibration is critical for reliable size classification. Take time to select corners precisely and measure table dimensions accurately!
