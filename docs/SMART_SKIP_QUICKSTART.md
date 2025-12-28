# Smart Frame Skipping - Quick Start Guide

## Overview

Smart Frame Skipping intelligently reduces frame processing during high load while ensuring events receive sufficient frames for reliable tracking.

## Quick Configuration

### Default (Recommended for Production)

Already enabled with sensible defaults:

```python
# In src/config/tracking_config.py
degraded_mode_smart_skip_enabled = True
degraded_mode_skip_pattern = 'adaptive'
degraded_mode_min_frames_per_event = 15
```

**No changes needed** - just run your application!

## How to Monitor

### 1. Watch the Logs

Look for smart skip messages in your logs:

```
[SmartSkip] Frame skipped (adaptive_every_2nd): queue=75.0%, 
avg_detect=35.2ms, skip_rate=4.5%, total_skipped=450
```

### 2. Check Queue Statistics (Every 5 seconds)

```
[QueueStats] Input: 350/500 (70.0% full, drops=5) | 
SmartSkip: 180 frames (rate=40.0% in degraded)
```

### 3. Review Final Statistics (On Shutdown)

```
[BagCounterApp] Final Stats: frames_skipped=450, 
smart_skip_frames=180, smart_skip_rate=40.00%
```

## Common Use Cases

### Use Case 1: I Want More Aggressive Skipping

**Problem**: Queue is still backing up at 70-80%

**Solution**: Use fixed pattern for consistent reduction

```python
# In src/config/tracking_config.py
degraded_mode_skip_pattern = 'every_2nd'  # Fixed 50% reduction
```

### Use Case 2: I Need Higher Quality Tracking

**Problem**: Worried about event quality with skipping

**Solution**: Increase minimum frames and use conservative pattern

```python
degraded_mode_skip_pattern = 'every_3rd'  # Only 33% reduction
degraded_mode_min_frames_per_event = 20   # More frames per event
```

### Use Case 3: Disable Smart Skip (Testing)

**Problem**: Want to test without smart skip

**Solution**: Disable the feature

```python
degraded_mode_smart_skip_enabled = False  # Back to legacy skip
```

## Troubleshooting

### Events Being Lost?

**Increase minimum frames:**
```python
degraded_mode_min_frames_per_event = 20  # Was 15
```

### Queue Still Full?

**Use more aggressive pattern:**
```python
degraded_mode_skip_pattern = 'every_2nd'  # Fixed 50% skip
```

**Or reduce minimum frames (carefully):**
```python
degraded_mode_min_frames_per_event = 12  # Was 15
```

### Too Much Skipping?

**Be more conservative:**
```python
degraded_mode_max_skip_rate_with_events = 0.4  # Was 0.5 (50%)
```

## Understanding the Adaptive Pattern

The adaptive pattern adjusts automatically:

| Queue Level | Action | Frames Processed |
|------------|--------|------------------|
| < 50% | No skip | 100% |
| 50-70% | Skip every 3rd | 67% |
| 70-85% | Skip every 2nd | 50% |
| 85-95% | Skip 2 of 3 | 33% |
| 95%+ | Skip 3 of 4 | 25% |

## Validation

Run the test suites to validate configuration:

```bash
# Basic tests
python3 test_smart_skip.py

# Integration tests (scenarios)
python3 test_smart_skip_integration.py
```

## Example Output

When working correctly, you'll see:

```
[BagCounterApp] ENTERING DEGRADED MODE: queue_util=72.0%
[SmartSkip] Frame skipped (adaptive_every_2nd): queue=75.0%
[SmartSkip] Frame skipped (adaptive_every_2nd): queue=73.0%
[QueueStats] SmartSkip: 45 frames (rate=48.0% in degraded)
[BagCounterApp] EXITING DEGRADED MODE: queue_util=65.0%, system recovered
```

## Best Practices

1. **Start with defaults** - They're production-tested
2. **Monitor logs** - Watch smart skip activation and rates
3. **Tune gradually** - One parameter at a time
4. **Test under load** - Use representative workload
5. **Check event survival** - Ensure events aren't being lost

## More Information

See `SMART_FRAME_SKIPPING.md` for:
- Complete configuration reference
- Detailed tuning guide
- Advanced scenarios
- Architecture details
