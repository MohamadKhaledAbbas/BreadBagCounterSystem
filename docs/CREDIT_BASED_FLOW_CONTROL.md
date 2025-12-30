# Credit-Based Flow Control Architecture

## Overview

The SpoolProcessor implements a **credit-based, best-effort, bounded in-flight window** design to maximize frame throughput while maintaining backpressure control. This replaces the previous blocking ACK design that limited throughput to ~3-5 Hz.

**Key Features:**
- **High Throughput**: Target 20 FPS when consumer can keep up
- **Bounded Window**: Configurable `max_in_flight` limit prevents overload
- **Non-Blocking ACK**: ACK callback frees credit asynchronously
- **Out-of-Order Support**: Any in-flight frame can be ACKed
- **Timeout Protection**: Expired frames automatically free credit
- **Natural Backpressure**: Publishing slows when in-flight window fills

## Architecture

### Credit-Based Flow Control Model

```
┌─────────────────────────────────────────────────────────────────────┐
│  SpoolProcessor (Publisher)                                          │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Publish Loop (Credit-Based)                                  │   │
│  │                                                               │   │
│  │  while running:                                              │   │
│  │    if in_flight < max_in_flight:  ◄─── Credit Available     │   │
│  │      frame = get_next_frame()                                │   │
│  │      publish_frame(frame)                                    │   │
│  │      track_in_flight(frame)       ◄─── Add to tracking      │   │
│  │      sleep(5ms)                   ◄─── Brief sleep          │   │
│  │    else:                                                     │   │
│  │      # Backpressure: wait for ACKs                           │   │
│  │      sleep(5ms)                                              │   │
│  │                                                               │   │
│  │    check_timeouts()               ◄─── Free credit for old  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ ACK Callback (Non-Blocking)                                  │   │
│  │                                                               │   │
│  │  on_ack_received(seq, session_id):                           │   │
│  │    if session_id == current_session:                         │   │
│  │      remove_from_in_flight(seq)   ◄─── Free credit          │   │
│  │      update_stats()                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Timeout Scanner (Periodic)                                   │   │
│  │                                                               │   │
│  │  every 500ms:                                                │   │
│  │    for frame in in_flight_order:  ◄─── FIFO order           │   │
│  │      if age > ack_timeout:                                   │   │
│  │        mark_expired(frame)                                   │   │
│  │        remove_from_in_flight()    ◄─── Free credit          │   │
│  │        break  ◄─── Stop on first non-expired (FIFO)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ In-Flight Tracking                                           │   │
│  │                                                               │   │
│  │  in_flight: Dict[seq -> InFlightFrame]                      │   │
│  │  in_flight_order: Deque[seq]      ◄─── FIFO for timeout     │   │
│  │                                                               │   │
│  │  Credit Available: len(in_flight) < max_in_flight            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

                            │
                            ▼
                  /spool_image_ch_0
                  /spool/current_frame_metadata
                            │
                            ▼
                            
┌─────────────────────────────────────────────────────────────────────┐
│  Consumer (BagCounterApp)                                            │
│                                                                       │
│  Decoder → Process Frame → Publish ACK(/processing_ack)             │
│                                                                       │
│  ACK includes: seq, frame_index, session_id                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Data Structures

**InFlightFrame:**
```python
@dataclass
class InFlightFrame:
    seq: int                 # Sequence number (unique per frame)
    frame_index: int         # Original frame index from spool
    sent_time: float         # Timestamp when published
    segment_num: int         # Segment number
    expired: bool = False    # Marked if timed out
```

**Tracking:**
- `in_flight: Dict[seq, InFlightFrame]` - Fast lookup by sequence number
- `in_flight_order: Deque[seq]` - FIFO order for timeout scanning

### Credit Management

**Publishing:**
1. Check: `len(in_flight) < max_in_flight`
2. If yes: Publish frame and add to tracking
3. If no: Sleep briefly (backpressure)

**ACK (Credit Release):**
1. Receive ACK with seq and session_id
2. Validate session_id (reject stale ACKs)
3. Remove seq from `in_flight` dict (credit freed)
4. Update statistics

**Timeout (Credit Recovery):**
1. Every 500ms, scan `in_flight_order` (FIFO)
2. For oldest frame: check if `age > ack_timeout`
3. If yes: mark expired, remove from tracking (credit freed)
4. If no: stop scanning (FIFO order means rest are newer)

## Configuration

### Database Configuration Keys

| Key | Default | Description |
|-----|---------|-------------|
| `spool_max_in_flight` | `10` | Maximum frames in-flight before backpressure |
| `spool_publish_idle_sleep_ms` | `5` | Milliseconds to sleep in publish loop |
| `spool_empty_poll_interval` | `1.0` | Seconds to wait when spool is empty |
| `spool_ack_timeout` | `10.0` | Seconds before frame is marked expired |

### Setting Configuration

Use `config.py` or database CLI:

```bash
# Set max in-flight (recommended: 10-20 for best throughput)
python config.py --key spool_max_in_flight --value 10

# Set timeout (lower = faster recovery from stuck frames)
python config.py --key spool_ack_timeout --value 10.0

# Set idle sleep (lower = higher CPU, higher throughput)
python config.py --key spool_publish_idle_sleep_ms --value 5

# Set empty poll interval
python config.py --key spool_empty_poll_interval --value 1.0
```

## Tuning for Production

### Maximizing Throughput

**Goal: Achieve 20 FPS throughput**

1. **Increase `max_in_flight`**: Higher window allows more frames in transit
   - Start with 10, increase to 20 if consumer can handle it
   - Monitor backpressure warnings in logs

2. **Reduce `publish_idle_sleep_ms`**: Less sleep = faster publishing
   - Default 5ms is reasonable
   - Can reduce to 1ms for maximum throughput (higher CPU)

3. **Monitor publish rate**: Check `pub_rate=X.Xfps` in logs
   - Should approach 20 FPS when consumer keeps up
   - If stuck at ~10 FPS, increase `max_in_flight`

4. **Check in-flight saturation**: Look for `in_flight=X/Y` in logs
   - If consistently at max, increase `max_in_flight`
   - If rarely at max, can reduce for safety

### Handling Slow Consumers

**Symptoms:**
- `in_flight` consistently at `max_in_flight`
- Backpressure state frequent
- ACK rate lower than publish rate

**Solutions:**
1. Optimize consumer (BagCounterApp) processing
2. Increase `ack_timeout` if ACKs are slow but valid
3. Reduce `max_in_flight` if decoder is unstable with high load
4. Check for hardware bottlenecks (CPU, memory)

### Preventing Timeout Spam

**Symptoms:**
- Many "Frame timeout" warnings
- High `timeouts` counter in stats
- Frames being expired unnecessarily

**Solutions:**
1. Increase `spool_ack_timeout` (e.g., 15.0 or 20.0 seconds)
2. Investigate why consumer is slow (check consumer logs)
3. Verify network/ROS2 connectivity is stable
4. Check if decoder is stuck or erroring

## Observability

### Regular Stats (Every 10 Seconds)

```
[SpoolProcessor] Stats: session=d8489312, published=1250, acked=1240, 
processed=1240, skipped=0, timeouts=2, ack_rejected=0, 
in_flight=8/10, pub_rate=18.5fps, ack_rate=18.2fps, state=publishing
```

**Key Metrics:**
- `published`: Total frames published
- `acked`: Total ACKs received
- `processed`: Frames successfully ACKed (same as acked)
- `in_flight=X/Y`: Current / Max in-flight
- `pub_rate`: Recent publish rate (frames/sec)
- `ack_rate`: Recent ACK rate (frames/sec)
- `state`: Current state (publishing, backpressure, spool_empty)

### Detailed Stats (Every 2 Minutes)

```
================================================================================
[SpoolProcessor] 📊 Detailed Statistics (2-minute summary)
  Session: d8489312-8ab4-4c76-b593-d9759133614d
  Flow Control:
    - In-flight: 8/10
    - Publish rate: 19.2 fps
    - ACK rate: 19.0 fps
    - Last ACK age: 0.5s
  ACK Statistics:
    - Accepted: 2450 (99.9%)
    - Rejected (stale): 2 (0.1%)
    - Total: 2452
  Frame Processing:
    - Published: 2500
    - Processed (ACKed): 2450
    - Skipped: 0
    - Timeouts: 5
  Spool Status:
    - Total segments: 36
    - Current segment: 1369
    - Oldest segment: 1360
    - Newest segment: 1372
    - Spool lag: 3 segments
    - SPS/PPS prepends: 36
  ✓ Spool lag is healthy (3 segments)
================================================================================
```

### Health Indicators

**Healthy System:**
- `pub_rate` near 20 FPS (if consumer keeps up)
- `ack_rate` close to `pub_rate`
- `in_flight` well below `max_in_flight` (e.g., 5/10)
- `timeouts` very low or zero
- `ack_rejected` very low or zero
- `spool_lag` < 5 segments

**Backpressure (Consumer Slow):**
- `in_flight` at or near `max_in_flight` (e.g., 10/10)
- `state=backpressure` frequent
- Backpressure warnings in logs
- `pub_rate` < 20 FPS

**Consumer Stuck:**
- `last_ack_age` > 60 seconds
- WATCHDOG warnings
- `timeouts` increasing rapidly
- `in_flight` fills then empties due to timeouts

**Session Mismatch:**
- `ack_rejected` > 0
- "ACK rejected: wrong session_id" warnings
- Restart both processor and consumer to resync

## Comparison: Blocking vs. Credit-Based

| Aspect | Blocking ACK | Credit-Based |
|--------|--------------|--------------|
| **Throughput** | ~3-5 Hz | ~20 FPS (target) |
| **ACK Waiting** | Blocks per frame | Non-blocking |
| **In-Flight** | Exactly 1 | Configurable (e.g., 10) |
| **Backpressure** | Per-frame timeout | Window fills naturally |
| **Out-of-Order ACK** | Not supported | Supported |
| **Timeout Recovery** | Retry with new seq | Auto-free credit |
| **CPU Efficiency** | Sleep 1s between frames | Brief sleep (5ms) |
| **Gap Behavior** | ~1s gaps during retry | No gaps (continuous) |

## Troubleshooting

### Low Publish Rate (< 10 FPS)

**Diagnose:**
```bash
# Check current rate
ros2 topic hz /spool_image_ch_0

# Check processor logs for in_flight status
journalctl -u breadcount-spool-processor -f | grep Stats
```

**Possible Causes:**
1. `max_in_flight` too low → Increase to 20
2. `spool_empty` → Check if recorder is running
3. Backpressure → Check consumer performance

### Backpressure Warnings

**Example:**
```
[SpoolProcessor] ⚠ BACKPRESSURE: In-flight window full (10/10) - consumer may be slower than publisher
```

**Actions:**
1. Check consumer logs for slow processing
2. Monitor `ack_rate` vs `pub_rate`
3. Verify decoder is not stuck
4. Consider increasing `max_in_flight` if consumer can handle it

### Timeout Spam

**Example:**
```
[SpoolProcessor] ⏱ Frame timeout: seq=42, frame_index=5432, age=10.5s, freeing credit
```

**Actions:**
1. Increase `spool_ack_timeout` (e.g., to 20.0)
2. Check consumer logs for errors
3. Verify ROS2 topics are connected:
   ```bash
   ros2 topic list
   ros2 topic info /processing_ack -v
   ```

### Session ID Mismatch

**Example:**
```
[SpoolProcessor] ⚠ ACK rejected: wrong session_id. Expected d8489312, got e5f6g7h8
```

**Actions:**
1. Restart both processor and consumer to resync
2. Verify READY topic is using TRANSIENT_LOCAL QoS
3. Check that both processes use the processor's session_id

## Best Practices

1. **Start Conservative**: Use default `max_in_flight=10`
2. **Monitor Throughput**: Check `ros2 topic hz /spool_image_ch_0`
3. **Tune Gradually**: Increase `max_in_flight` if backpressure frequent
4. **Watch Lag**: Keep `spool_lag` < 5 segments
5. **Log Analysis**: Review 2-minute detailed stats for trends
6. **Health Checks**: Monitor `ack_rate` vs `pub_rate` ratio
7. **Timeout Tuning**: Adjust `ack_timeout` based on consumer speed

## Testing

Unit tests for credit-based flow control are in `tests/test_credit_based_flow.py`:

```bash
# Run tests
python tests/test_credit_based_flow.py
```

**Test Coverage:**
- In-flight window cap enforcement
- Out-of-order ACK handling
- Timeout-based credit release
- Backpressure behavior
- Mixed ACK and timeout scenarios

## References

- [ACCURACY_MODE_SPOOLING.md](ACCURACY_MODE_SPOOLING.md) - Overall architecture
- `src/ros2_spool/spool_processor_node.py` - Implementation
- `src/constants.py` - Configuration keys
