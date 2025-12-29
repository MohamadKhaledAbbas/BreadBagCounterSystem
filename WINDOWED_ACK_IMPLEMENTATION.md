# Windowed ACK / Backpressure Implementation Summary

## Overview

This document summarizes the implementation of configurable windowed ACK/backpressure in `SpoolProcessorNode` to improve throughput while maintaining accuracy and ordered processing.

## Problem Statement

The original `SpoolProcessorNode` enforced strict "one frame in flight" processing:
1. Read next spooled frame
2. Publish index + metadata + encoded frame
3. Wait for ACK
4. Only then continue

This strict ACK gating reduced throughput to the consumer's per-frame latency, causing the system to fall behind the spool disk/recorder and risking retention-based segment deletion.

## Solution: Windowed ACK Processing

The new implementation supports configurable parallelism via the `spool_inflight_window` setting:

- **`inflight_window=1`** (default): Backward compatible, strict serial processing
- **`inflight_window>1`** (e.g., 3-5): Multiple frames can be in-flight simultaneously

## Key Features

### 1. Backward Compatibility ✅

- Default `inflight_window=1` preserves existing strict serial behavior
- All existing tests pass without modification
- No changes to message formats, topics, or ROS2 QoS settings
- Configuration is DB-backed (consistent with existing spool settings)

### 2. Core Window Logic ✅

**In-flight Tracking:**
- Ordered deque of `InflightFrame` objects with metadata:
  - `seq`: Sequence number
  - `frame_index`: Frame index from spool
  - `segment_num`: Segment number
  - `publish_time`: Timestamp for timeout detection
  - `retry_count`: Number of retries so far
  - `acked`: Whether frame has been acknowledged
  - `frame_record`: Original frame data (for retry)

**Processing Flow:**
1. Processor maintains ordered queue of in-flight frames
2. Publishes frames up to `inflight_window` limit without blocking
3. ACK callback marks corresponding frame as acknowledged (handles out-of-order)
4. Window retirement removes acked frames from head in order
5. Timeout/retry logic per frame (doesn't block other frames)
6. After max retries, frame is skipped to avoid pipeline deadlock

### 3. Out-of-Order ACK Support ✅

- ACKs can arrive in any order (consumer may process frames in parallel)
- Each ACK marks its corresponding frame as acknowledged
- Frames are only retired from the head of the window when contiguously acked
- Maintains ordering correctness even with parallel consumer processing

### 4. Robust Timeout/Retry ✅

- Per-frame timeout tracking (independent of other frames)
- Retry logic doesn't block the entire pipeline
- Max retries exceeded → frame marked as acked and skipped
- Ensures pipeline never deadlocks on a single failed frame

### 5. Enhanced Observability ✅

**New Metrics:**
- `inflight`: Current number of frames in window
- `out_of_order_acks`: Count of ACKs received out of sequence
- `oldest_inflight_age`: Age of oldest frame in window (seconds)

**Regular Stats (10s interval):**
```
[SpoolProcessor] Stats: session=abc12345, seq=150, processed=145, 
  retried=2, skipped=0, timeouts=2, ack_rejected=0, out_of_order_acks=15,
  inflight=3/5, oldest_inflight_age=0.5s, segments=10, sps_pps_prepends=5
```

**Detailed Stats (2-minute interval):**
- Configuration (window, timeout, retry)
- ACK statistics (accepted, rejected, out-of-order)
- Frame processing (processed, retried, skipped, timeouts)
- Spool status (lag detection and warnings)

## Configuration

### Database Config

Add to `config` table (use `config.py` utility):

```bash
# Set inflight window (default: 1)
python config.py --key spool_inflight_window --value 3

# Other related settings
python config.py --key spool_ack_timeout --value 10.0
python config.py --key spool_retry_count --value 2
```

### Constants

Added to `src/constants.py`:
```python
spool_inflight_window = "spool_inflight_window"
```

### Code Files Modified

1. **`src/constants.py`**
   - Added `spool_inflight_window` constant

2. **`src/ros2_spool/spool_processor_node.py`**
   - Added `DEFAULT_INFLIGHT_WINDOW = 1`
   - Updated `ProcessorConfig` dataclass
   - Updated `load_config_from_db()` to load inflight_window
   - Added `InflightFrame` dataclass for tracking
   - Added `_inflight_frames` deque and `_inflight_lock`
   - Rewrote `_processor_loop()` for windowed processing
   - Updated `_ack_callback()` to handle out-of-order ACKs
   - Added helper methods:
     - `_retire_acked_frames()`
     - `_check_and_retry_timeouts()`
     - `_can_publish_frame()`
     - `_get_inflight_count()`
     - `_get_oldest_inflight_age()`
   - Updated stats logging with new metrics
   - Removed old `_process_frame_with_retry()` method

3. **`docs/ACCURACY_MODE_SPOOLING.md`**
   - Added windowed ACK feature documentation
   - Updated configuration table
   - Added monitoring guidance
   - Updated recommended production settings

## Testing

### Unit Tests ✅

Created `tests/test_spool_processor_window.py` with comprehensive coverage:

1. **`test_inflight_window_size_limit`**: Window respects max size
2. **`test_out_of_order_ack_handling`**: Out-of-order ACKs handled correctly
3. **`test_timeout_and_retry`**: Timeout detection and retry logic
4. **`test_max_retries_exceeded`**: Frames skipped after max retries
5. **`test_ordered_retirement`**: Frames retired in order from head
6. **`test_window_with_default_size_one`**: Backward compatibility with window=1
7. **`test_oldest_inflight_age`**: Age calculation for monitoring

**All tests pass:** ✅

### Existing Tests ✅

All existing tests pass without modification:
- `test_h264_nal.py` ✅
- `test_segment_io_roundtrip.py` ✅
- `test_retention_policy.py` ✅

### Code Quality ✅

- **Python syntax check**: ✅ Pass
- **Code review**: ✅ Completed, feedback addressed
- **Security scan (CodeQL)**: ✅ No issues found

## Production Deployment

### Recommended Settings

**Conservative (start here):**
```bash
python config.py --key spool_inflight_window --value 1  # Strict serial
python config.py --key spool_ack_timeout --value 10.0
python config.py --key spool_retry_count --value 2
```

**If experiencing spool lag:**
```bash
python config.py --key spool_inflight_window --value 3  # Increase parallelism
python config.py --key spool_ack_timeout --value 10.0   # Keep timeout reasonable
python config.py --key spool_retry_count --value 2      # Keep retries low
```

**For parallel consumer processing:**
```bash
python config.py --key spool_inflight_window --value 5  # Higher parallelism
```

### Monitoring

**Healthy System Indicators:**
- `inflight`: < `inflight_window` (window not full)
- `out_of_order_acks`: Any value (normal with window>1)
- `oldest_inflight_age`: < `ack_timeout` (e.g., <10s)
- `timeouts`: 0 or very low
- `skipped`: 0 (frames not being dropped)
- `spool_lag`: < 5 segments (processor keeping up)

**Warning Signs:**
- `inflight`: == `inflight_window` continuously (window always full)
- `oldest_inflight_age`: Approaching `ack_timeout` (frames timing out)
- `timeouts`: Increasing (consumer falling behind)
- `skipped`: > 0 (frames being dropped after retries)
- `spool_lag`: > 10 segments (processor falling behind recorder)

**Actions:**
1. If window always full but no timeouts → increase `inflight_window`
2. If many timeouts → investigate consumer performance
3. If frames skipped → check consumer health and logs
4. If spool lag increasing → increase `inflight_window` or optimize consumer

## Benefits

### Throughput Improvement

- **Before**: Throughput limited by consumer per-frame latency
  - Frame processing time: 50ms (detection + classification)
  - Max throughput: 20 FPS
  - With 30 FPS input: Falls behind by 10 FPS, spool lag increases

- **After (window=3)**:
  - 3 frames can be processed in parallel
  - Effective max throughput: ~60 FPS (if consumer has parallelism)
  - With 30 FPS input: Keeps up easily, no spool lag

### Reduced Risk

- Prevents retention-based segment deletion from spool lag
- Pipeline never deadlocks on single failed frame
- Graceful degradation under load

### Maintained Accuracy

- Ordered frame retirement ensures correctness
- Session-based ACK validation prevents stale ACKs
- No frame drops unless explicitly skipped after retries
- All ACKs properly correlated with frames

## Implementation Statistics

- **Lines of code added**: ~350
- **Lines of code modified**: ~100
- **Tests added**: 7 comprehensive unit tests
- **Files modified**: 3 (2 code, 1 doc)
- **Backward compatibility**: 100% (default window=1)
- **Test coverage**: All existing + new tests pass
- **Security issues**: 0 (CodeQL scan clean)

## Conclusion

The windowed ACK implementation successfully addresses the throughput limitation while maintaining all existing reliability and accuracy guarantees. The configurable window size provides a tunable knob for operators to balance throughput vs. backpressure based on their specific deployment needs.

Key strengths:
✅ Backward compatible (default preserves existing behavior)
✅ Production-ready (robust error handling, monitoring, logging)
✅ Well-tested (comprehensive unit tests + existing tests pass)
✅ Well-documented (code comments, docs, monitoring guide)
✅ Secure (CodeQL scan clean, no vulnerabilities)
✅ Maintainable (clear code structure, good naming, comments)

The implementation is ready for production deployment.
