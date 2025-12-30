# ACK Flow Control Replacement: Monotonic Progress Tracking

## Executive Summary

This document explains why per-frame ACK-based flow control is **fundamentally incompatible** with real-time video processing and describes its production-grade replacement: **monotonic progress tracking with retention-safe backpressure**.

## The Problem with Per-Frame ACKs

### Why ACK-Based Designs Fail

ACK logic assumes: *"Every frame must be individually confirmed before moving on."*

This works for:
- Control packets
- Transactions
- RPC systems

It **does not work** for:
- Real-time vision
- Streaming media
- Event-driven analytics

### Concrete Failure Modes Observed

#### ❌ ACK Reordering

```
Processor published frame 105, waiting for ACK
ACK received for frame 102 (still processing in pipeline)
```

This happens because:
- ROS2 delivery is async
- Processing latency varies
- ACKs are not causally bound to frame publication

#### ❌ ACK Blocking

When ACK is delayed:
- Processor stops publishing
- Disk spool grows
- Retention deletes unprocessed segments
- Lag increases catastrophically

#### ❌ DDS QoS Incompatibility

ACK topics introduce:
- Reliability mismatches
- Silent DDS message rejection
- Hard-to-debug deadlocks

#### ❌ ACK ≠ Processing Guarantee

Even if ACK arrives:
- Frame may still be dropped later
- Event-level correctness is not guaranteed
- ACKs provide false confidence

## Industry Rule (Non-Negotiable)

> **Never use per-frame ACKs for real-time video processing.**

No production vision system (robotics, CCTV, factory inspection) uses frame-level ACKs. This is how Kafka, GStreamer, DeepStream, and ROS2 Nav stacks work.

## Production Replacement Architecture

### Core Principles

1. **Optimize for event accuracy, not frame accuracy**
2. **Never block on consumer feedback**
3. **Protect unprocessed data from deletion**
4. **Adapt under load instead of deadlock**

### Component 1: Monotonic Frame Index (Already Exists)

Every frame has a `frame_index` that is strictly increasing. This is the only ordering guarantee needed.

### Component 2: Processor Progress Marker

Replace ACK with a locally-tracked marker:

```python
last_committed_frame_index: int
```

Updated when:
- Frame is fully processed
- Relevant events are finalized

This marker is:
- **Monotonic** (only increases)
- **Written locally** (no round-trip)
- **Always consistent** (no race conditions)

### Component 3: Disk Retention Guard

Retention must obey:

```python
if segment.max_frame_index > last_committed_frame_index:
    # DO NOT DELETE - contains unprocessed data
    continue
```

**Never delete data the processor has not committed.**

This alone eliminates race conditions.

### Component 4: Sliding Window Processing

Processor behavior:
- Reads frames sequentially
- Processes continuously
- **Never waits** on consumer feedback

If lag grows beyond threshold:
- Skip intermediate frames
- **Never skip** entry/exit candidates
- Preserve event correctness

## Backpressure Without ACKs

Define hard limits:

```python
max_processing_lag_frames: int = 150  # 6 seconds at 25fps
max_frame_backlog: int = 300          # 12 seconds
```

When exceeded:
- Drop non-critical frames
- Reduce detection precision
- Increase spatial/temporal gating

**System adapts instead of blocking.**

## Event-Centric Commitment

The system cares about:
- **Events** (bag operations)
- **Transitions** (open → closing → closed)
- **Counts** (final tally)

**Not about:**
- Every individual frame

Correct commitment unit:
```
EVENT → COMMITTED (with frame_index)
```

**Not:**
```
FRAME → ACK
```

This aligns perfectly with the Event-Centric Monitor.

## Health Signals (Safe, Non-Blocking)

For visibility, use:
- Periodic heartbeat
- Lag metrics
- Queue depth stats

These are:
- **Observational** (don't affect processing)
- **Non-blocking** (no waiting)
- **Safe under failure** (graceful degradation)

## Expected Improvements

| Aspect | ACK-Based | Replacement |
|--------|-----------|-------------|
| Deadlocks | Possible | **Impossible** |
| Throughput | Limited | **Maximal** |
| Latency | Spiky | **Stable** |
| Debuggability | Hard | **Clear** |
| Production safety | ❌ | ✅ |
| Retention races | Possible | **Eliminated** |

## Implementation in V6

### EventCentricTracker

```python
# Track last processed frame for retention safety
self._last_processed_frame_index: int = 0

def update(...):
    # ... processing ...
    
    # Update progress marker
    self._last_processed_frame_index = frame_index
    
def get_last_processed_frame_index(self) -> int:
    """Get progress marker for retention guard."""
    return self._last_processed_frame_index
```

### RetentionPolicy

```python
class RetentionPolicy:
    def __init__(self, ..., retention_safety_enabled: bool = True):
        self.retention_safety_enabled = retention_safety_enabled
        self._last_processed_frame: int = 0
    
    def set_last_processed_frame(self, frame_index: int):
        """Update progress marker (called from processor)."""
        with self._progress_lock:
            if frame_index > self._last_processed_frame:
                self._last_processed_frame = frame_index
    
    def get_expired_segments(self):
        """Get segments eligible for deletion (respects progress)."""
        for segment in segments:
            # V6: Never delete unprocessed data
            if self.retention_safety_enabled:
                frame_range = self._get_segment_frame_range(segment.path)
                if frame_range and frame_range[1] > self._last_processed_frame:
                    # Segment contains unprocessed frames - protect it
                    continue
            
            expired.append(segment)
        
        return expired
```

### SpoolProcessor (Non-Blocking Mode)

```python
class SpoolProcessorNode:
    def _processor_loop(self):
        """Non-blocking processing loop."""
        while self._running:
            frame = self._get_next_frame()
            if frame is None:
                time.sleep(self.config.poll_interval)
                continue
            
            # Publish frame immediately (no waiting for ACK)
            self._publish_frame(frame)
            
            # Update progress marker
            self._update_progress(frame.index)
            
            # Check backpressure
            if self._should_skip_frames():
                # Intelligent frame skipping (preserve entry/exit)
                self._skip_non_critical_frames()
            
            # Log stats periodically
            self._maybe_log_stats()
```

## Migration Path

### Phase 1: Retention Safety (Completed in V6)
- ✅ Add `last_processed_frame_index` tracking
- ✅ Update RetentionPolicy to respect progress
- ✅ Document the new architecture

### Phase 2: Non-Blocking Processor (Next)
- Remove ACK waiting loop
- Implement sliding window processing
- Add adaptive backpressure

### Phase 3: Deprecate ACK Topics (Future)
- Keep ACK topics for backward compatibility (optional)
- Remove from critical path
- Use only for health monitoring

## Conclusion

The ACK-based frame coordination mechanism is **fundamentally incompatible** with real-time, event-driven video processing.

It is replaced with:
- **Monotonic progress marker** (local, fast, reliable)
- **Retention guards** (never delete unprocessed data)
- **Adaptive backpressure** (adapt instead of block)

This aligns the system with **industry-standard streaming architectures** used by Kafka, GStreamer, DeepStream, and ROS2 Nav stacks.

## One-Line Summary

> Replace per-frame ACK flow control with monotonic progress tracking and retention-safe backpressure for production-grade reliability.
