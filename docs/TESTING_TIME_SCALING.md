# Testing Mode Time Scaling Guide

## Overview

When running the BreadBag Counter System in testing/development mode (e.g., on Windows PCs without BPU acceleration), the system processes all frames but at a slower effective speed than production (RDK X5 board). This can cause time-based logic (event timeouts, state transitions, suppression windows) to behave differently than in production.

The **Testing Time Scaling** feature solves this by automatically or manually scaling all time-based thresholds to compensate for slower processing speed, ensuring event lifecycles behave identically to production.

## Problem Statement

### Production Environment (RDK X5)
- Processes frames at 25 FPS (40ms per frame) in real-time
- Time-based timeouts (e.g., 1000ms ghost timeout = ~25 frames) work as expected
- Events expire and commit based on wall-clock timing

### Testing Environment (Windows)
- May process frames at 5 FPS (200ms per frame) or slower
- Time-based timeouts become misaligned (1000ms ghost timeout = only ~5 frames)
- Events expire too quickly or remain active too long
- **Result**: False positives/negatives, inaccurate testing

## Solution: Time Scaling

The system can scale all millisecond-based timeout parameters by a configurable factor to maintain equivalent behavior regardless of processing speed.

### Affected Parameters

All time-based thresholds are scaled:
- `association_time_ms` - Max time gap to associate detection with event
- `ghost_timeout_ms` - How long events survive without detections
- `max_event_lifetime_ms` - Maximum event lifetime before forced expiration
- `suppression_duration_ms` - Anti-double-counting suppression window
- `min_event_creation_interval_ms` - Minimum time between new events at same location
- `open_to_closing_time_ms` - Min time in OPEN state before transitioning
- `closing_stability_time_ms` - How long closed detections must persist
- `closed_stability_time_ms` - Min time in CLOSED before commit eligible
- `max_prediction_time_ms` - Max time ahead for velocity prediction
- `min_gap_duration_for_logging_ms` - Minimum detection gap to log

### Frame-Based Parameters (Not Affected)

Parameters based on frame counts scale naturally with frame rate:
- `commit_idle_frames` - Frames without detection before commit
- `min_open_frames` - Minimum consecutive open frames
- `min_closed_frames` - Minimum consecutive closed frames
- `out_of_zone_grace_frames` - Frames outside work zone before expiration

## Configuration

### Method 1: Auto-Scaling (Recommended for Windows)

Auto-scaling is **enabled by default on Windows** platforms. The system measures actual processing speed during a warmup period (first 100 frames) and automatically calculates the appropriate scaling factor.

```python
# In src/config/tracking_config.py
@dataclass
class TrackingConfig:
    enable_auto_time_scaling: bool = IS_WINDOWS  # True on Windows, False on RDK
    testing_time_scale_factor: float = 1.0  # Initial value, auto-updated
```

The system will log the calculated factor:
```
[EventCentricTracker] Auto-scaling enabled: avg_frame_time=200.0ms, target=40.0ms, scale_factor=5.00
```

### Method 2: Manual Scaling

If you know your processing speed, set the scale factor manually:

```python
# In src/config/tracking_config.py
tracking_config.testing_time_scale_factor = 5.0  # For 5x slower processing
tracking_config.enable_auto_time_scaling = False  # Disable auto-calculation
```

### Method 3: Calculate Scale Factor

Calculate the scale factor based on your measured FPS:

```python
# Formula:
scale_factor = target_fps / actual_fps

# Example: Target is 25fps, but testing runs at 5fps
scale_factor = 25.0 / 5.0  # = 5.0

# Example: Target is 25fps, testing runs at 10fps  
scale_factor = 25.0 / 10.0  # = 2.5

# Example: Measured 200ms per frame, target is 40ms
scale_factor = 200.0 / 40.0  # = 5.0
```

## Usage Examples

### Example 1: Auto-Scaling on Windows (Default)

```python
# No configuration needed - auto-enabled on Windows
python main.py

# System automatically detects:
# - Processing at 8fps (125ms/frame)
# - Target is 25fps (40ms/frame)
# - Calculates scale_factor = 125/40 = 3.125
# - All timeouts scaled: ghost_timeout becomes 1000ms * 3.125 = 3125ms
```

### Example 2: Manual Scaling for Slow Hardware

```python
# Edit src/config/tracking_config.py
tracking_config.testing_time_scale_factor = 10.0  # Very slow machine
tracking_config.enable_auto_time_scaling = False

# Result: All timeouts scaled by 10x
# - ghost_timeout: 1000ms -> 10000ms
# - suppression_duration: 1500ms -> 15000ms
```

### Example 3: Disable Scaling for Real-Time Testing

```python
# Edit src/config/tracking_config.py
tracking_config.testing_time_scale_factor = 1.0
tracking_config.enable_auto_time_scaling = False

# Result: No scaling, production behavior
```

## Verification

### Check Logs

The system logs the applied scaling at startup:

```
[EventCentricTracker] Initialized with:
  T=2000.0ms (base=400.0ms),
  G=5000.0ms (base=1000.0ms),
  suppression_duration=7500.0ms (base=1500.0ms),
  time_scale_factor=5.0, auto_scaling=True
```

### Compare Event Behavior

With proper scaling, events should:
- Survive similar numbers of frames in both environments
- Commit at similar points in their lifecycle
- Have similar suppression behavior

Without scaling, you'll see:
- Events expiring too quickly (timeouts reached in fewer frames)
- Doubled events (suppression windows too short)
- Missed events (association windows too narrow)

## Technical Details

### Implementation

Time scaling is implemented in `EventCentricTracker`:

1. **Initialization**: Creates scaled copies of all time-based thresholds
2. **Auto-Scaling**: Measures frame intervals during warmup (100 frames)
3. **Application**: Uses scaled thresholds throughout tracking logic
4. **Event Creation**: Passes scaled config to `BreadBagEvent` instances

### Frame Timestamps vs Wall Clock

The system uses **frame-based timestamps** on Windows (`use_frame_timestamps=True`), which ensures:
- Deterministic timing regardless of processing speed
- Consistent frame intervals (frame_count * frame_duration_ms)
- Predictable event behavior

Production (RDK) uses wall-clock time for true real-time processing.

### Scaled Config Propagation

```python
# Tracker creates scaled config for events
def _get_scaled_config(self) -> EventConfig:
    return replace(
        self.config,
        association_time_ms=self._scaled_association_time_ms,
        ghost_timeout_ms=self._scaled_ghost_timeout_ms,
        # ... all time-based parameters
    )

# Events receive scaled config
new_event = BreadBagEvent(
    initial_detection=evidence,
    config=self._get_scaled_config(),  # Scaled, not original
    ...
)
```

## Best Practices

### Development Testing
1. **Use auto-scaling** (default on Windows) for accurate behavior
2. **Monitor logs** to verify reasonable scale factor (2x-10x typical)
3. **Compare results** with production logs to validate equivalence

### Performance Testing
1. **Disable scaling** (`testing_time_scale_factor=1.0`) to test raw speed
2. **Measure FPS** to understand processing capabilities
3. **Calculate required scaling** for equivalence testing

### Video Playback Testing
1. **Enable auto-scaling** to match video frame rate with processing speed
2. **Use frame timestamps** (`use_frame_timestamps=True`) for deterministic timing
3. **Verify all frames processed** (testing mode never drops frames)

### Calibration
1. **Run warmup**: Let system process 100+ frames
2. **Check scale factor**: Look for auto-scaling log message
3. **Adjust if needed**: Manual override if auto-calculation is off
4. **Validate behavior**: Compare event counts and lifecycles with production

## Troubleshooting

### Events Expiring Too Quickly
**Symptom**: Events counted prematurely, lower completion rate
**Cause**: Scale factor too low (timeouts not scaled enough)
**Solution**: Increase `testing_time_scale_factor` or enable auto-scaling

### Events Staying Active Too Long
**Symptom**: Events accumulating, slow counting, high active count
**Cause**: Scale factor too high (timeouts scaled too much)
**Solution**: Decrease `testing_time_scale_factor` or verify FPS measurement

### Double Counting
**Symptom**: Same bag counted multiple times
**Cause**: Suppression window not scaled properly
**Solution**: Verify `suppression_duration_ms` is being scaled (check logs)

### Auto-Scaling Not Activating
**Symptom**: No auto-scaling log message after 100 frames
**Cause**: `enable_auto_time_scaling=False` or on RDK platform
**Solution**: Set `tracking_config.enable_auto_time_scaling = True`

### Scale Factor Seems Wrong
**Symptom**: Calculated factor doesn't match expected processing speed
**Cause**: Frame timestamps may not reflect actual processing time
**Solution**: 
- Check if `use_frame_timestamps=True` (should be on Windows)
- Verify frame intervals in logs
- Consider manual scaling based on measured FPS

## Examples

### Scenario: Testing with 5fps Effective Speed

**Without Scaling:**
```
Target: 25fps (40ms/frame)
Actual: 5fps (200ms/frame)
ghost_timeout: 1000ms = only 5 frames of survival
Result: Events expire too quickly, bags missed
```

**With Scaling (5.0x):**
```
Target: 25fps (40ms/frame)
Actual: 5fps (200ms/frame)
Scale factor: 5.0
ghost_timeout: 1000ms * 5.0 = 5000ms = 25 frames at 5fps
Result: Events survive equivalent duration, accurate counting
```

### Scenario: High-Performance Windows Machine

**Auto-Scaling Result:**
```
Measured: 20fps (50ms/frame)
Target: 25fps (40ms/frame)
Calculated factor: 50/40 = 1.25
Decision: < 1.2x threshold, keeping scale_factor=1.0
Result: Processing close to real-time, no scaling needed
```

## Summary

Time scaling ensures that the BreadBag Counter System behaves identically in testing and production environments, regardless of processing speed. By automatically or manually scaling time-based thresholds, developers can:

- Test accurately on slower hardware
- Validate counting logic without false positives/negatives
- Process video files at any speed while maintaining correct timing
- Deploy with confidence knowing behavior matches testing

**Default behavior on Windows**: Auto-scaling enabled, no configuration required.
**Default behavior on RDK**: No scaling, real-time processing.
