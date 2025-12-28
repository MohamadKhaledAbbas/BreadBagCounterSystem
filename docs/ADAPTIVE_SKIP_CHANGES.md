# Adaptive Skipping and Queue Optimization Changes

## Summary
This document describes the changes made to improve adaptive frame skipping and queue management in the BreadBag Counter System.

## Changes Made

### 1. Configuration Constants Updated

#### Queue Size
- **INPUT_QUEUE_SIZE**: Increased from 100 to 500
  - Allows more frame buffering when memory permits
  - Reduces risk of frame drops during temporary processing spikes
  - Location: `src/counting/BagCounterApp.py`, line 67

#### Adaptive Skipping Thresholds
- **ADAPTIVE_SKIP_THRESHOLD**: Lowered from 0.8 (80%) to 0.7 (70%)
  - More aggressive adaptive skipping triggers earlier
  - Reduces queue pressure before reaching critical levels
  - Location: `src/counting/BagCounterApp.py`, line 76

- **MAX_DETECTION_TIME_MS**: Lowered from 35.0ms to 31.0ms
  - Stricter threshold for detection time before skipping
  - Helps maintain target frame rate of 25fps (40ms per frame)
  - Location: `src/counting/BagCounterApp.py`, line 75

#### New Constants
- **SKIP_RATE_CAP**: Set to 0.02 (2%)
  - Hard limit on frame skip rate
  - Ensures tracking reliability by preventing excessive skipping
  - Location: `src/counting/BagCounterApp.py`, line 79

- **SKIP_RATE_WINDOW**: Set to 500 frames
  - Window size for tracking skip rate
  - Approximately 20 seconds of video at 25fps
  - Location: `src/counting/BagCounterApp.py`, line 80

- **SYSTEM_STATUS_LOG_INTERVAL**: Set to 900.0 seconds (15 minutes)
  - Interval for system resource monitoring logs
  - Location: `src/counting/BagCounterApp.py`, line 83

### 2. Skip Rate Hard Cap Implementation

#### Skip Decision Tracking
- Added `_skip_decisions` deque to track last 500 frame skip decisions
- Added `_skip_cap_blocks` counter to track how many times cap prevented skipping
- Location: `src/counting/BagCounterApp.py`, lines 180-181

#### Skip Rate Cap Logic
The logic enforces a 2% maximum skip rate:

1. **Calculate Current Skip Rate**: 
   - Compute from last N decisions in the tracking window
   
2. **Predict Future Skip Rate**:
   - Calculate what the rate would be if current frame is skipped
   
3. **Enforce Cap**:
   - Block skip if predicted rate exceeds 2%
   - Allow skip only if it keeps rate at or below cap
   
4. **Track Decision**:
   - Record every skip/no-skip decision for future calculations

Location: `src/counting/BagCounterApp.py`, lines 758-825

### 3. Enhanced Logging

#### Adaptive Skip Logging
- Added detailed logging when frames are skipped due to backpressure
- Includes: queue utilization, average detection time, skip rate, total skipped
- Log level: WARNING (every 10th skip to avoid flooding)
- Example log format:
  ```
  [AdaptiveSkip] Frame skipped due to backpressure: queue=75.2%, 
  avg_detect=32.5ms (threshold=31.0ms), skip_rate=1.2%, total_skipped=150
  ```
- Location: `src/counting/BagCounterApp.py`, lines 795-805

#### Skip Cap Block Logging
- Logs when skip rate cap prevents a frame skip
- Includes: current skip rate, cap limit, queue state, detection time
- Log level: WARNING (every 5th block to avoid flooding)
- Example log format:
  ```
  [SkipCapBlock] Skip rate cap preventing frame skip: current_rate=1.8%, 
  cap=2.0%, queue=72.0%, avg_detect=33.1ms, blocks=25
  ```
- Location: `src/counting/BagCounterApp.py`, lines 808-823

#### Input Queue Pressure Logging
- Enhanced warning messages with root cause information
- Includes: utilization, threshold, average detection time, target time
- Example log format:
  ```
  [InputQueuePressure] High queue utilization: 75.0% (threshold=70%) - 
  Root cause: avg_detection_time=38.2ms (target=40.0ms). 
  Risk: frames may be dropped if processing doesn't improve.
  ```
- Location: `src/counting/BagCounterApp.py`, lines 1147-1155

#### Classification Queue Pressure Logging
- Enhanced with clear risk information
- Example log format:
  ```
  [ClassificationQueuePressure] High queue utilization: 80.0% (threshold=70%) - 
  classification thread is falling behind. Risk: classification tasks may be dropped.
  ```
- Location: `src/counting/BagCounterApp.py`, lines 1158-1163

#### Queue Statistics Logging
- Added skip rate and skip cap blocks to periodic queue stats
- Example log format:
  ```
  [QueueStats] Input: 350/500 (70.0% full, drops=5) | 
  Classification: 15/20 (75.0% full, drops=2) | 
  Skipped: 150 (rate=1.5%, cap=2.0%) | SkipCapBlocks: 25
  ```
- Location: `src/counting/BagCounterApp.py`, lines 1131-1145

### 4. System Monitoring

#### Periodic Resource Logging
- Added `_log_system_status()` method for CPU and RAM monitoring
- Uses psutil if available, fails gracefully if not installed
- Logs every 15 minutes at INFO level
- Location: `src/counting/BagCounterApp.py`, lines 275-312

#### Psutil Integration
- Checks availability at initialization
- Logs availability status
- Gracefully handles ImportError if psutil not installed
- Location: `src/counting/BagCounterApp.py`, lines 184-191

#### System Status Log Format
Example output:
```
[SystemStatus] CPU: 45.2%, RAM: 62.1% (3842.5MB / 6144.0MB)
```

Also logs via structured logging:
```json
{
  "event": "system_status",
  "timestamp": "2025-12-22T18:43:40.123456",
  "cpu_percent": 45.2,
  "memory_percent": 62.1,
  "memory_used_mb": 3842.5,
  "memory_total_mb": 6144.0
}
```

### 5. Updated Performance Config Logging
- Enhanced initialization log to show all key performance parameters
- Location: `src/counting/BagCounterApp.py`, lines 252-258

Example:
```
[BagCounterApp] V3 Performance Config: target_fps=25.0, target_frame_time=40.0ms, 
max_detection_time=31.0ms, adaptive_skip_threshold=70.0%, skip_rate_cap=2.0%, 
input_queue=500, classification_queue=20
```

### 6. Enhanced Final Statistics
- Added skip rate and skip cap blocks to shutdown statistics
- Location: `src/counting/BagCounterApp.py`, lines 1224-1238

Example:
```
[BagCounterApp] Final Stats: input_drops=10, classification_drops=2, 
frames_skipped=150, skip_rate=1.52%, skip_cap_blocks=25
```

## Testing

A comprehensive test suite has been added (`test_adaptive_skip_changes.py`) that verifies:

1. **Constants Verification**: All new constants have correct values
2. **Skip Rate Cap Logic**: Mathematical correctness of skip rate calculation and capping
3. **Feature Detection**: Presence of all new features in code
4. **Psutil Integration**: Graceful handling with and without psutil

Run tests with:
```bash
python3 test_adaptive_skip_changes.py
```

## Benefits

1. **Reduced Queue Pressure**: Larger buffer (500) and more aggressive skipping (0.7 threshold)
2. **Maintained Tracking Quality**: 2% skip rate cap ensures reliable object tracking
3. **Better Observability**: Enhanced logging shows root causes of performance issues
4. **System Health Monitoring**: Periodic CPU/RAM logging helps diagnose resource issues
5. **No Breaking Changes**: Core architecture remains unchanged, only tuning parameters adjusted

## Migration Notes

- No code changes required by users
- Psutil is optional - system works without it
- All changes are backward compatible
- Log format changes are additive (no removals)
