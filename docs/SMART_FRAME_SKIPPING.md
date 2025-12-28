# Smart Frame Skipping in Degraded Mode

## Overview

The Smart Frame Skipping feature provides intelligent, pattern-based frame skipping during degraded mode to reduce system load while ensuring events receive sufficient frames for reliable detection and tracking.

## Problem Statement

In high-load scenarios, the system may enter **degraded mode** when:
- Input queue utilization exceeds 70%
- Average queue delay exceeds 100ms

Traditional binary frame skipping (skip or don't skip) can break event tracking if events don't receive enough frames. Smart frame skipping solves this by using intelligent patterns that balance throughput and tracking quality.

## Key Features

### 1. **Pattern-Based Skipping**

Instead of random or binary skipping, uses predictable patterns:

- **every_2nd**: Skip every 2nd frame (50% reduction, processes 50% of frames)
- **every_3rd**: Skip every 3rd frame (33% reduction, processes 67% of frames)
- **adaptive**: Dynamically adjusts based on queue pressure (recommended)

### 2. **Event-Aware Intelligence**

Ensures tracking reliability by:
- Guaranteeing minimum frames per event (default: 15 frames)
- Never skipping during critical states (CLOSING, early OPEN)
- Reducing skip rate when many active events exist
- Skipping more aggressively when no active events

### 3. **Adaptive Pattern** (Recommended for Production)

Automatically adjusts skip pattern based on queue utilization:

| Queue Utilization | Skip Pattern | Frames Processed | Description |
|------------------|--------------|------------------|-------------|
| < 50% | No pattern skip | 100% | Below threshold, rely on existing adaptive skip |
| 50-70% | Every 3rd frame | 67% | Mild load - gentle reduction |
| 70-85% | Every 2nd frame | 50% | Moderate load - balanced reduction |
| 85-95% | 2 out of 3 | 33% | Heavy load - aggressive reduction |
| 95%+ | 3 out of 4 | 25% | Critical load - maximum reduction |

## Configuration

All parameters are in `src/config/tracking_config.py`:

### Core Parameters

```python
# Enable/disable smart skipping
degraded_mode_smart_skip_enabled: bool = True

# Skip pattern: 'adaptive', 'every_2nd', or 'every_3rd'
degraded_mode_skip_pattern: str = 'adaptive'

# Minimum frames each event must receive
degraded_mode_min_frames_per_event: int = 15
```

### Event Protection

```python
# Preserve critical states (CLOSING, early OPEN)
degraded_mode_preserve_critical_states: bool = True

# Frames to consider OPEN event as "early" (critical)
degraded_mode_critical_state_frame_threshold: int = 5

# Maximum skip rate when active events exist
degraded_mode_max_skip_rate_with_events: float = 0.5  # 50%
```

### Behavior Control

```python
# Only skip when events are active
degraded_mode_skip_with_active_events_only: bool = False
```

## How It Works

### 1. Degraded Mode Detection

System enters degraded mode when:
```python
queue_utilization > 0.7  # 70% queue full
OR
avg_queue_delay > 100ms  # Average delay exceeds threshold
```

### 2. Smart Skip Decision Flow

```
Frame arrives
    ↓
Check if degraded mode active?
    ↓ Yes
Check if smart skip enabled?
    ↓ Yes
Are there active events?
    ↓ Yes
Do any events need more frames?
    ↓ No (all events have min_frames)
Is any event in critical state (CLOSING)?
    ↓ No
Is any event in early OPEN (< 5 frames)?
    ↓ No
Apply skip pattern based on queue utilization
    ↓
Skip or Process
```

### 3. Event Frame Tracking

System maintains a counter for each active event:
```python
_event_frame_counts = {
    event_id_1: 18,  # Has enough frames
    event_id_2: 8,   # Needs more frames - won't skip
    event_id_3: 25,  # Has enough frames
}
```

Events with fewer than `min_frames_per_event` frames prevent skipping.

## Production Tuning Guide

### Scenario 1: High-Throughput, Low Event Density

**Problem**: Many frames, few events, want maximum throughput

**Configuration**:
```python
degraded_mode_skip_pattern = 'adaptive'  # Adjusts to load
degraded_mode_min_frames_per_event = 12  # Slightly lower
degraded_mode_max_skip_rate_with_events = 0.6  # Allow more skipping
degraded_mode_skip_with_active_events_only = True  # Skip freely when no events
```

### Scenario 2: Dense Event Tracking, Quality Critical

**Problem**: Many events, tracking quality is paramount

**Configuration**:
```python
degraded_mode_skip_pattern = 'every_3rd'  # Conservative pattern
degraded_mode_min_frames_per_event = 20  # Higher minimum
degraded_mode_max_skip_rate_with_events = 0.4  # More conservative
degraded_mode_preserve_critical_states = True  # Always preserve
```

### Scenario 3: Balanced Production Default

**Problem**: Need reliable tracking with good throughput

**Configuration** (Current Defaults):
```python
degraded_mode_skip_pattern = 'adaptive'  # Best of both worlds
degraded_mode_min_frames_per_event = 15  # Adequate for tracking
degraded_mode_max_skip_rate_with_events = 0.5  # Balanced
degraded_mode_preserve_critical_states = True  # Safety first
```

## Monitoring and Metrics

### Log Messages

#### Smart Skip Activation
```
[SmartSkip] Frame skipped (adaptive_every_2nd): queue=75.0%, 
avg_detect=35.2ms (threshold=40.0ms), skip_rate=4.5%, total_skipped=450
```

#### Queue Statistics (Every 5 seconds)
```
[QueueStats] Input: 350/500 (70.0% full, drops=5) | 
Classification: 15/20 (75.0% full, drops=2) | 
Skipped: 450 (rate=4.5%, cap=7.0%) | SkipCapBlocks: 25 | 
SmartSkip: 180 frames (rate=40.0% in degraded)
```

#### Final Statistics (On Shutdown)
```
[BagCounterApp] Final Stats: input_drops=10, classification_drops=2, 
frames_skipped=450, skip_rate=4.52%, skip_cap_blocks=25, 
smart_skip_frames=180, smart_skip_rate=40.00%, frames_in_degraded=450
```

### Key Metrics

- **frames_skipped**: Total frames skipped (all methods)
- **smart_skip_frames**: Frames skipped by smart pattern
- **smart_skip_rate**: Skip rate during degraded mode
- **frames_in_degraded**: Total frames processed while in degraded mode

## Safety Mechanisms

### 1. Minimum Frames Guarantee

Each event is guaranteed at least `min_frames_per_event` (default: 15) frames:
- With ghost_timeout=40 frames
- Even at 50% skip rate → 20 frames processed
- Minimum 15 ensures reliable tracking with margin

### 2. Critical State Protection

Never skips frames when:
- Any event is in CLOSING state (bag being tied)
- Any event is in early OPEN (first 5 frames of new event)

### 3. Max Skip Rate Enforcement

When active events exist:
- Skip rate is capped at `max_skip_rate_with_events` (default: 50%)
- Prevents excessive skipping that could harm tracking
- Provides consistent frame delivery for active events

### 4. Backwards Compatibility

Can be completely disabled:
```python
degraded_mode_smart_skip_enabled = False  # Use legacy skip logic
```

## Performance Impact

### Throughput Improvement

At 70-85% queue utilization with adaptive pattern (skip every 2nd frame):
- **Frames processed**: 50% reduction
- **Detection load**: 50% reduction
- **Queue pressure**: Significantly reduced
- **Event survival**: Maintained with min_frames guarantee

### Tracking Quality

With proper configuration:
- **Event detection**: No degradation (events get enough frames)
- **State transitions**: Preserved (critical states never skipped)
- **Classification**: Unaffected (still collects ROIs from processed frames)

## Testing

Run the test suite:
```bash
python3 test_smart_skip.py
```

Tests validate:
- Configuration parameters exist and have sensible defaults
- Skip patterns work correctly at different queue utilizations
- Event awareness features are properly configured
- Production readiness (defaults won't break tracking)
- Backwards compatibility

## Troubleshooting

### Problem: Events are being lost

**Symptoms**: Events expire before classification

**Solution**: Increase minimum frames per event
```python
degraded_mode_min_frames_per_event = 20  # Was 15
```

### Problem: Queue still backing up

**Symptoms**: Queue remains at 70%+ even with smart skip

**Solutions**:
1. Use more aggressive pattern:
   ```python
   degraded_mode_skip_pattern = 'every_2nd'  # Fixed 50% reduction
   ```

2. Reduce minimum frames (carefully):
   ```python
   degraded_mode_min_frames_per_event = 12  # Was 15
   ```

3. Check if critical state protection is too conservative:
   ```python
   degraded_mode_critical_state_frame_threshold = 3  # Was 5
   ```

### Problem: Too much skipping, tracking degraded

**Symptoms**: Events getting exactly min_frames, state transitions missed

**Solution**: Increase safety margins
```python
degraded_mode_min_frames_per_event = 20  # More headroom
degraded_mode_max_skip_rate_with_events = 0.4  # More conservative
```

## Best Practices

1. **Start with defaults**: Current defaults are production-tested
2. **Monitor metrics**: Watch `smart_skip_rate` and event survival
3. **Tune gradually**: Change one parameter at a time
4. **Test under load**: Validate with representative workload
5. **Keep logging enabled**: Essential for tuning and debugging

## Future Enhancements

Potential improvements for future versions:

1. **Per-state skip rates**: Different rates for OPEN vs CLOSING vs CLOSED
2. **Event velocity tracking**: Skip less for fast-moving events
3. **Predictive skipping**: Use ML to predict which frames are safe to skip
4. **Dynamic min_frames**: Adjust based on event complexity
5. **Confidence-based skipping**: Skip more when classifier is confident

## References

- `src/config/tracking_config.py`: Configuration parameters
- `src/counting/BagCounterApp.py`: Implementation (`_should_smart_skip_frame`)
- `test_smart_skip.py`: Test suite
- `ADAPTIVE_SKIP_CHANGES.md`: Legacy adaptive skip documentation
