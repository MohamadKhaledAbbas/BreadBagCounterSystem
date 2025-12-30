# Credit-Based Flow Control Migration - Implementation Summary

## Overview

This document summarizes the migration from blocking ACK design to credit-based flow control in the SpoolProcessor.

## Problem Statement

The original SpoolProcessor design used a blocking ACK model where:
- Each frame was published individually
- The processor waited for ACK before publishing the next frame
- This resulted in ~3-5 Hz throughput with ~1 second gaps
- The bottleneck was per-frame blocking on ACK wait

User requirement: Achieve best-effort processing at ~20 FPS while maintaining backpressure control.

## Solution: Credit-Based Flow Control

Implemented a bounded in-flight window design:

### Key Changes

1. **In-Flight Window Tracking**
   - `Dict[seq, InFlightFrame]` for fast lookup
   - `Deque[seq]` for FIFO timeout scanning
   - Bounded by `max_in_flight` (default: 10)

2. **Non-Blocking Publish Loop**
   - Continuously publishes while `in_flight < max_in_flight`
   - No per-frame ACK wait
   - Brief 5ms sleep to prevent CPU spinning
   - Naturally slows when window fills (backpressure)

3. **Asynchronous ACK Handling**
   - ACK callback frees credit immediately
   - Supports out-of-order ACKs
   - Session validation to reject stale ACKs

4. **Timeout-Based Recovery**
   - Periodic scan (every 500ms) for expired frames
   - Frames older than `ack_timeout` are marked expired
   - Credit is freed to prevent deadlock

5. **Enhanced Observability**
   - Publish rate (fps)
   - ACK rate (fps)
   - In-flight count (current/max)
   - Last ACK age
   - Backpressure warnings

## Files Modified

### 1. `src/constants.py`
Added configuration keys:
- `spool_max_in_flight`
- `spool_publish_idle_sleep_ms`
- `spool_empty_poll_interval`

### 2. `src/ros2_spool/spool_processor_node.py`
Major refactoring:
- Removed `_wait_for_ack()` blocking method
- Removed `_process_frame_with_retry()` retry logic
- Added `InFlightFrame` dataclass
- Implemented credit-based publish loop
- Rewrote `_ack_callback()` for credit release
- Added `_check_and_expire_timeouts()` method
- Updated `ProcessorState` enum (removed WAITING_FOR_ACK, added PUBLISHING/BACKPRESSURE)
- Enhanced stats logging with rates and in-flight metrics

## Files Created

### 1. `tests/test_credit_based_flow.py`
Comprehensive unit tests covering:
- In-flight window cap enforcement
- Out-of-order ACK handling
- Timeout-based credit release
- Backpressure behavior
- Mixed ACK and timeout scenarios

**Test Results:** 7/7 tests passing

### 2. `docs/CREDIT_BASED_FLOW_CONTROL.md`
Complete design documentation including:
- Architecture diagrams
- Data structure details
- Configuration guide
- Tuning recommendations
- Observability metrics
- Troubleshooting guide

### 3. `docs/ACCURACY_MODE_SPOOLING.md` (Updated)
Updated to reference credit-based design and new configuration keys.

## Configuration

### New Keys

| Key | Default | Description |
|-----|---------|-------------|
| `spool_max_in_flight` | `10` | Maximum frames in-flight |
| `spool_publish_idle_sleep_ms` | `5` | Sleep duration in publish loop |
| `spool_empty_poll_interval` | `1.0` | Poll interval when spool empty |
| `spool_ack_timeout` | `10.0` | Timeout for in-flight frames |

### Setting Configuration

```bash
python config.py --key spool_max_in_flight --value 10
python config.py --key spool_ack_timeout --value 10.0
python config.py --key spool_publish_idle_sleep_ms --value 5
```

## Expected Behavior

### Before (Blocking ACK)
- Throughput: ~3-5 Hz
- Gaps: ~1 second between frames during retry
- Blocking: Per-frame wait for ACK
- In-flight: Always 1

### After (Credit-Based)
- Throughput: Target 20 FPS
- Gaps: No 1-second gaps (continuous publishing)
- Non-blocking: ACK callback frees credit asynchronously
- In-flight: Configurable (default 10)

### Backpressure
When consumer slows or stops:
1. In-flight window fills to `max_in_flight`
2. Publishing naturally slows (credit exhausted)
3. Timeout scanner frees credit after `ack_timeout`
4. System recovers automatically

## Testing

### Unit Tests
```bash
cd /home/runner/work/BreadBagCounterSystem/BreadBagCounterSystem
python tests/test_credit_based_flow.py
```

Expected output:
```
============================================================
Running Credit-Based Flow Control Tests
============================================================
...
Test Results: 7 passed, 0 failed
============================================================
```

### Manual Validation

**Required for production:**
1. Test with consumer running normally
   - Check: `ros2 topic hz /spool_image_ch_0` should show ~20 Hz
   - Check: No 1-second gaps in logs
   
2. Test backpressure when consumer stops
   - Stop BagCounterApp
   - Observe: In-flight window fills to max
   - Observe: Backpressure warnings in logs
   - Observe: Timeout expiry after 10 seconds
   
3. Test session mismatch handling
   - Restart consumer mid-processing
   - Check: Stale ACKs rejected
   - Check: Processing continues with new session

## Observability

### Regular Stats (Every 10s)
```
[SpoolProcessor] Stats: session=d8489312, published=1250, acked=1240,
in_flight=8/10, pub_rate=18.5fps, ack_rate=18.2fps, state=publishing
```

### Detailed Stats (Every 2min)
```
================================================================================
[SpoolProcessor] 📊 Detailed Statistics (2-minute summary)
  Flow Control:
    - In-flight: 8/10
    - Publish rate: 19.2 fps
    - ACK rate: 19.0 fps
...
================================================================================
```

## Backward Compatibility

- Old config keys (`spool_retry_count`, `poll_interval`) are retained but marked deprecated
- Database config loading handles missing new keys gracefully (uses defaults)
- Session-based ACK protocol unchanged (compatible with existing consumer)

## Rollback Plan

If issues arise:
1. Revert to previous commit before this PR
2. Or: Set `spool_max_in_flight=1` to mimic old behavior (not recommended)

## Next Steps

1. **Manual Testing**: Validate on actual RDK X5 hardware
2. **Tune Configuration**: Adjust `max_in_flight` based on decoder capacity
3. **Monitor Metrics**: Watch publish rate, ACK rate, and in-flight saturation
4. **Production Deployment**: Use default configuration initially
5. **Iterate**: Tune based on observed performance

## References

- [docs/CREDIT_BASED_FLOW_CONTROL.md](../docs/CREDIT_BASED_FLOW_CONTROL.md) - Detailed design
- [docs/ACCURACY_MODE_SPOOLING.md](../docs/ACCURACY_MODE_SPOOLING.md) - Overall architecture
- [tests/test_credit_based_flow.py](../tests/test_credit_based_flow.py) - Unit tests

## Authors

- Implementation: GitHub Copilot
- Review: MohamadKhaledAbbas

---

**Status:** Implementation Complete ✅  
**Tests:** 7/7 Passing ✅  
**Documentation:** Complete ✅  
**Manual Validation:** Pending ⏳
