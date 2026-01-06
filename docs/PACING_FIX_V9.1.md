# SpoolProcessor V9.1 - FPS Pacing Fix

## Issue

User reported that despite `target_fps=35.0`, the system was only achieving `effective_fps=15.6-16.5`. Performance metrics showed:
- `avg_total_loop_ms=60-75ms` per frame
- `avg_get_next_frame_ms=20-28ms`
- `avg_publish_frame_ms=15-21ms`
- Processing time: ~35-49ms
- But total loop: 60-75ms

This indicated excessive sleep time was being added beyond what was needed.

## Root Cause

The sleep calculation logic had a bug:

```python
# OLD (BROKEN) CODE
time_until_deadline = next_deadline - publish_end
target_sleep = max(0.0, max(min_interval_sec, time_until_deadline))
```

This logic always enforced a minimum sleep of `min_interval_sec` (25ms), even when:
1. We were already behind schedule (negative `time_until_deadline`)
2. We needed shorter sleeps to achieve target FPS

### Why This Broke FPS

At 35 FPS, each frame should take 28.6ms total.

**Scenario with 40ms processing:**
- `time_until_deadline = 28.6ms - 40ms = -11.4ms` (behind schedule)
- Old logic: `max(0, max(25ms, -11.4ms)) = max(0, 25ms) = 25ms`
- **Result:** Process 40ms + sleep 25ms = 65ms total = **15.4 FPS** ✗

**Scenario with 20ms processing:**
- `time_until_deadline = 28.6ms - 20ms = 8.6ms` (ahead of schedule)
- Old logic: `max(0, max(25ms, 8.6ms)) = max(0, 25ms) = 25ms`
- **Result:** Process 20ms + sleep 25ms = 45ms total = **22.2 FPS** ✗

In both cases, the 25ms minimum sleep floor prevented achieving 35 FPS.

## Solution

Simplified the logic to always sleep exactly `time_until_deadline` (or 0 if behind schedule):

```python
# NEW (FIXED) CODE
time_until_deadline = next_deadline - publish_end

# Determine target sleep time
if time_until_deadline <= 0:
    # We're behind schedule - don't sleep at all to catch up
    target_sleep = 0
else:
    # We're ahead of schedule - sleep to hit deadline
    target_sleep = time_until_deadline

if target_sleep > 0:
    time.sleep(target_sleep)
```

### Why This Works

**Scenario with 40ms processing:**
- `time_until_deadline = 28.6ms - 40ms = -11.4ms` (behind schedule)
- New logic: `target_sleep = 0` (don't sleep when behind)
- **Result:** Process 40ms + sleep 0ms = 40ms total = **25 FPS** ✓

**Scenario with 20ms processing:**
- `time_until_deadline = 28.6ms - 20ms = 8.6ms` (ahead of schedule)
- New logic: `target_sleep = 8.6ms` (sleep exactly until deadline)
- **Result:** Process 20ms + sleep 8.6ms = 28.6ms total = **35 FPS** ✓

## Impact

### Before Fix
- Effective FPS: ~15-16 (limited by artificial 25ms sleep floor)
- Total loop time: 60-75ms
- Unable to achieve target FPS even when processing was fast

### After Fix
- Effective FPS: Limited only by actual processing time
- With 40ms processing: Max ~25 FPS (1000ms / 40ms)
- With 35ms processing: Max ~28.6 FPS (can approach target)
- With 20ms processing: Can achieve 35 FPS target

### To Achieve Full 35 FPS

Based on the metrics, processing time needs to be reduced below 28.6ms:
- Current: `get_next_frame` ~20-28ms + `publish_frame` ~15-21ms = 35-49ms
- Target: Total processing < 28.6ms

Possible optimizations:
1. Optimize `get_next_frame` (currently 20-28ms)
2. Optimize `publish_frame` (currently 15-21ms)
3. Profile to find bottlenecks

But the pacing logic will no longer artificially limit FPS with minimum sleep enforcement.

## Testing

All tests pass:
- ✅ New test for deadline-based sleep logic
- ✅ Existing spool processor tests
- ✅ No regressions

## Files Changed

- `src/ros2_spool/spool_processor_node.py`: Simplified sleep logic (removed min_interval enforcement)
- `tests/test_spool_processor_improvements.py`: Updated test to match new logic

## Deployment Notes

After deploying this fix:
1. Monitor effective FPS - should be closer to target (limited by processing time)
2. Check performance metrics - total_loop_ms should be closer to processing time
3. If still not reaching 35 FPS, optimize `get_next_frame` and `publish_frame`

The pacing logic is now correct and will no longer artificially limit throughput.
