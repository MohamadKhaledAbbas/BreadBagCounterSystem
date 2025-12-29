# Accuracy Mode Spooling Architecture

This document describes the Accuracy Mode feature for the BreadBag Counter System, which implements spool-to-disk of encoded H.264 frames and pull-based replay to ensure zero frame drops.

## Overview

In production environments, the RDK X5 edge device processes live RTSP streams with the following pipeline:

```
RTSP Source → H.264 Decode → NV12 Images → Detection → Classification → Counting
```

Under load, the pipeline may drop frames due to queue pressure, leading to missed bag counts. Accuracy Mode solves this by:

1. **Spooling H.264 frames to disk** before decoding
2. **Pull-based replay** where processing controls the pace
3. **Strict backpressure** ensuring exactly one frame in flight

## Architecture

### Process Model

Accuracy Mode uses a two-process architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Process 1: Recorder (always-on, minimal CPU)                           │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │ ROS2 Subscriber │───→│ Bounded Queue    │───→│ Segment Writer    │  │
│  │ /rtsp_image_ch_0│    │ (in-memory)      │    │ (disk I/O thread) │  │
│  └─────────────────┘    └──────────────────┘    └───────────────────┘  │
│                                                           │             │
│                                                           ▼             │
│                                              ┌────────────────────────┐ │
│                                              │ Spool Directory        │ │
│                                              │ seg_000001.bin         │ │
│                                              │ seg_000002.bin         │ │
│                                              │ ...                    │ │
│                                              └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Process 2: Processor (pull-paced replay)                               │
│  ┌───────────────────┐   ┌───────────────────┐   ┌──────────────────┐  │
│  │ Spool Reader      │──→│ Pump/Controller   │──→│ H.264 Publisher  │  │
│  │ (read segments)   │   │ (wait for ACK)    │   │ /spool_image_ch_0│  │
│  └───────────────────┘   └───────────────────┘   └──────────────────┘  │
│                                    ▲                       │            │
│                                    │                       ▼            │
│  ┌───────────────────┐            │           ┌──────────────────────┐ │
│  │ ACK Subscriber    │────────────┘           │ HW Decoder           │ │
│  │ /processing_ack   │                        │ (hobot_codec)        │ │
│  └───────────────────┘                        └──────────────────────┘ │
│                                                           │             │
│                                                           ▼             │
│                                              ┌──────────────────────┐   │
│                                              │ BagCounterApp        │   │
│                                              │ (detection, etc.)    │   │
│                                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Spool Recorder (`spool_recorder_node.py`)

The recorder subscribes to H.264 encoded frames and writes them to disk:

- **Input**: `/rtsp_image_ch_0` (img_msgs/msg/H26XFrame)
- **Output**: Segment files in spool directory

Features:
- Non-blocking ROS2 callback (enqueue to bounded queue)
- Background writer thread for disk I/O
- IDR-aligned segment rotation when possible
- SPS/PPS caching for segment boundaries
- Automatic retention policy enforcement

#### 2. Spool Processor (`spool_processor_node.py`)

The processor reads from the spool and publishes at controlled pace:

- **Input**: Segment files from spool directory
- **Output**: `/spool_image_ch_0` (img_msgs/msg/H26XFrame)

Features:
- Reads oldest closed segments in order
- **Configurable windowed ACK / backpressure** (see below)
- Out-of-order ACK handling
- Per-frame timeout and retry logic
- Graceful degradation (skip frames after max retries)

##### Windowed ACK / Backpressure

The processor supports configurable parallelism via the `spool_inflight_window` setting:

- **`inflight_window=1`** (default): Strict one-at-a-time processing (backward compatible)
  - Publishes one frame
  - Waits for ACK before publishing next frame
  - Maximum reliability, lower throughput

- **`inflight_window>1`** (e.g., 3-5): Multiple frames in flight
  - Publishes up to N frames without waiting for ACKs
  - ACKs can arrive out-of-order (frames marked individually)
  - Frames retired from window in order (maintains correctness)
  - Improves throughput when consumer has parallel stages
  - Reduces spool lag risk

**Configuration:**
```bash
# Set inflight window via database config
python config.py --key spool_inflight_window --value 3
```

**How it works:**
1. Processor maintains an ordered queue of in-flight frames
2. Publishes frames up to `inflight_window` limit
3. ACK callback marks corresponding frame as acknowledged
4. Window retirement removes acked frames from head (in order)
5. Timeout/retry logic per frame (doesn't block other frames)
6. After max retries, frame is skipped to avoid blocking pipeline

**Observability:**
- `inflight`: Current number of frames in window
- `out_of_order_acks`: Count of ACKs received out of sequence
- `oldest_inflight_age`: Age of oldest frame in window (for monitoring)

#### 3. Frame Index Tracking

Since NV12 decoded frames don't carry the original H.264 index, we use a side-channel:

```
SpoolProcessor → /spool/current_frame_metadata → BagCounterApp
      │                                                 │
      │                                                 ▼
      ▼                                          /processing_ready
 /spool_image_ch_0                                     ▼
      │                                          BagCounterApp processes
      ▼                                                 │
   Decoder                                              ▼
      │                                           /processing_ack
      ▼                                                 │
BagCounterApp ◄──────────────────────────────────────┘
```

### Production-Grade ACK Protocol

The ACK mechanism uses structured messages with session tracking to ensure reliability:

**Messages:**

1. **ProcessingReady** (JSON over std_msgs/String)
   - Published by BagCounterApp on startup
   - Topic: `/processing_ready`
   - QoS: RELIABLE, TRANSIENT_LOCAL (for late joiners)
   - Fields:
     - `session_id` (string): UUID of consumer session
     - `ready_time_sec` (int64): Timestamp seconds
     - `ready_time_nsec` (uint32): Timestamp nanoseconds

2. **FrameMetadata** (JSON over std_msgs/String)
   - Published by SpoolProcessor for each frame
   - Topic: `/spool/current_frame_metadata`
   - QoS: RELIABLE, depth=20
   - Fields:
     - `frame_index` (uint32): Frame index
     - `session_id` (string): Processor session UUID
     - `seq` (uint64): Monotonic sequence number
     - `sent_time_sec` (int64): Timestamp seconds
     - `sent_time_nsec` (uint32): Timestamp nanoseconds
     - `segment_num` (int32): Optional segment number (-1 if not set)

3. **ProcessingAck** (JSON over std_msgs/String)
   - Published by BagCounterApp after frame processing
   - Topic: `/processing_ack`
   - QoS: RELIABLE, depth=20
   - Fields: Same as FrameMetadata

**Protocol Flow:**

1. **Startup Handshake:**
   - BagCounterApp starts, generates `session_id`
   - Publishes `ProcessingReady` with its `session_id`
   - SpoolProcessor waits for `ProcessingReady` (timeout: 10s)

2. **Frame Processing:**
   - SpoolProcessor assigns `seq` number, publishes frame
   - Publishes `FrameMetadata` with frame context
   - BagCounterApp receives metadata, stores it
   - After consuming frame, publishes `ProcessingAck` with matching metadata
   - SpoolProcessor validates `session_id` and `seq` before accepting ACK

3. **Restart Safety:**
   - Stale ACKs with wrong `session_id` are rejected
   - Processor won't start until READY received for current session
   - Each restart gets new session IDs

### Topic-Based Pull Protocol

| Topic | Type | Direction | Purpose |
|-------|------|-----------|---------|
| `/spool_image_ch_0` | H26XFrame | Pub | Encoded frames to decoder |
| `/spool/current_frame_metadata` | String (JSON) | Pub | Frame metadata for ACK correlation |
| `/processing_ready` | String (JSON) | Pub | Consumer ready signal |
| `/processing_ack` | String (JSON) | Pub | ACK with full frame context |
| `/spool/request_next` | UInt32 | Sub | Optional: external pull request |

**QoS Configuration:**

- **READY topic**: RELIABLE + TRANSIENT_LOCAL (for late joiners)
- **ACK/Metadata topics**: RELIABLE + KEEP_LAST (depth=20)
- **Frame topic**: RELIABLE + KEEP_LAST (depth=10)


## Segment File Format

### Binary Format (Version 1)

```
┌─────────────────────────────────────┐
│ Header (8 bytes)                    │
│   Magic: "SPOOL1" (6 bytes)         │
│   Version: uint8 (1 byte)           │
│   Flags: uint8 (1 byte, reserved)   │
├─────────────────────────────────────┤
│ Record 1                            │
│   Magic: "FR" (2 bytes)             │
│   Index: uint32                     │
│   Width: uint32                     │
│   Height: uint32                    │
│   DTS seconds: int64                │
│   DTS nanoseconds: uint32           │
│   PTS seconds: int64                │
│   PTS nanoseconds: uint32           │
│   Encoding: 12 bytes (null-padded)  │
│   Data length: uint32               │
│   Data: raw bytes                   │
├─────────────────────────────────────┤
│ Record 2                            │
│   ...                               │
├─────────────────────────────────────┤
│ Record N                            │
│   ...                               │
└─────────────────────────────────────┘
```

### Atomic Writes

To prevent corruption during crashes:

1. Write to `seg_XXXXXX.tmp`
2. Flush and fsync
3. Rename to `seg_XXXXXX.bin`

This ensures only complete segments have the `.bin` extension.

### Metadata Files

Optional `seg_XXXXXX.meta.json` files contain:

```json
{
  "segment_number": 1,
  "start_time": 1703791200.123,
  "end_time": 1703791205.456,
  "frame_count": 125,
  "bytes_written": 524288,
  "first_frame_index": 1000,
  "last_frame_index": 1124,
  "has_idr": true
}
```

## Configuration

### Database Configuration (config table)

Configuration is stored in the SQLite database config table. Use `config.py` to set values:

| Key | Default | Description |
|-----|---------|-------------|
| `accuracy_mode_enabled` | `0` | Enable accuracy mode (`1` = enabled) |
| `spool_dir` | `/home/sunrise/BreadCounting/data/spool` | Spool directory |
| `spool_segment_duration` | `5.0` | Target segment duration (seconds) |
| `spool_retention_seconds` | `180` | Maximum segment age before deletion |
| `spool_ack_timeout` | `10.0` | Timeout waiting for ACK (seconds) |
| `spool_retry_count` | `2` | Retries before advancing |
| `spool_inflight_window` | `1` | Max frames in flight (1=strict serial, 3-5=parallel) |

Example configuration commands:

```bash
# Enable accuracy mode
python config.py --key accuracy_mode_enabled --value 1

# Set spool directory
python config.py --key spool_dir --value /home/sunrise/BreadCounting/data/spool

# Set retention to 5 minutes
python config.py --key spool_retention_seconds --value 300

# Configure windowed ACK (increase parallelism)
python config.py --key spool_inflight_window --value 3
```

### ROS2 Environment

Ensure these are set (as per existing launch configuration):

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///path/to/cyclonedds.xml
```

## Usage

### Using Supervisor (Recommended for Production)

Install the supervisor configuration:

```bash
sudo cp supervisor/breadcount-spool.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
```

Start/stop services:

```bash
# Start spool services
sudo supervisorctl start breadcount-spool-recorder breadcount-spool-processor

# Stop spool services
sudo supervisorctl stop breadcount-spool-recorder breadcount-spool-processor

# Check status
sudo supervisorctl status breadcount-spool-recorder breadcount-spool-processor
```

Or use the run_app.sh script (automatically starts spool services when accuracy_mode_enabled=1):

```bash
./run_app.sh
```

### Running Manually (Development)

```bash
# Start the recorder process (always-on)
python -m src.ros2_spool.spool_recorder_node
```

### Running the Processor

```bash
# Start the processor process (controls pace)
python -m src.ros2_spool.spool_processor_node
```

### Enabling Accuracy Mode in BagCounterApp

Enable via database configuration:

```bash
python config.py --key accuracy_mode_enabled --value 1
python main.py
```

## Disk and Storage Considerations

### Space Requirements

At typical H.264 bitrates (2-4 Mbps) and 180s retention:

- **Minimum**: ~45 MB (2 Mbps × 180s)
- **Typical**: ~90 MB (4 Mbps × 180s)
- **Safety margin**: 200 MB recommended

### SD Card Considerations

For SD card deployments:

1. **Wear leveling**: Segment rotation distributes writes
2. **Write amplification**: Large sequential writes are efficient
3. **Recommended**: Industrial-grade SD with high endurance (e.g., SLC/pSLC)

### Retention Tuning

The default 180s retention provides:
- Buffer for temporary processing slowdowns
- Recovery from brief network outages
- Minimal disk space usage

Increase retention if:
- Processing varies significantly in speed
- You need longer recovery windows

## Failure Modes and Recovery

### Spool Empty

If processing catches up to recording:
- Processor waits and polls for new segments
- No data loss, just temporary idle

### Retention-Induced Loss

If processing is too slow:
- Oldest segments are deleted to enforce retention
- Log warning: "Deleted expired segment"
- This indicates processing cannot keep up

### Crash Recovery

On restart:
1. Cleanup stale `.tmp` files (incomplete writes)
2. Resume from oldest complete segment
3. Continue recording new frames

### ACK Timeout

If BagCounterApp doesn't ACK:
1. Retry publishing same frame (configurable count)
2. After retries, log error and advance
3. Prevents deadlock on stuck frames

### Startup Synchronization

The processor includes production-grade startup handshake:
1. **Consumer startup**: BagCounterApp generates session_id and publishes READY
2. **Processor waits**: Waits up to 10 seconds for READY signal
3. **Session validation**: Only processes frames for the current session
4. **Late joiner support**: TRANSIENT_LOCAL durability allows processor to see READY even if it starts later

### Restart Scenarios

**Scenario 1: Processor restarts while Consumer running**
- Processor generates new session_id
- Waits for READY from consumer
- Consumer may still have old session_id - will continue with old session
- Processor ignores ACKs with old session_id
- **Resolution**: Restart consumer to sync session_ids, or processor accepts that consumer will keep old session

**Scenario 2: Consumer restarts while Processor running**
- Consumer generates new session_id
- Publishes READY with new session_id
- Processor sees READY and starts processing
- Consumer receives frames with processor's session_id
- Consumer ACKs with processor's session_id (from metadata)
- **Result**: Clean synchronization

**Scenario 3: Both restart simultaneously**
- Both generate new session_ids
- Consumer publishes READY first
- Processor sees READY and starts
- Both use processor's session_id from frame metadata
- **Result**: Clean synchronization

### Session ID Mismatch Recovery

If you see "ACK rejected: wrong session_id" warnings:
1. Check both processor and consumer logs for session_ids
2. Verify READY was published and received
3. If mismatch persists, restart both processes
4. Check TRANSIENT_LOCAL QoS is working (use `ros2 topic info -v`)

## Troubleshooting

### No Detections / Stuck Pipeline

**Symptoms:**
- ACK timeout logs for the same frame index repeatedly
- No detections in BagCounterApp
- `ros2 topic echo /nv12_images --once` takes 10-15 seconds

**Root Causes and Fixes:**

1. **Frame Index Correlation Issue**
   - **Cause**: ACK was published with wrong frame index due to race condition
   - **Fix**: ACK is now published immediately when frame is consumed from Ros2FrameServer
   - **Verification**: Check logs for matching frame indices in ACK and processor

2. **QoS Mismatch**
   - **Cause**: Decoder expects BEST_EFFORT but processor used RELIABLE
   - **Fix**: Frame publisher now uses BEST_EFFORT QoS (depth=5)
   - **Verification**: `ros2 topic info /spool_image_ch_0 -v` should show matching QoS

3. **Missing SPS/PPS at Segment Boundaries**
   - **Cause**: Decoder fails to initialize without SPS/PPS NAL units
   - **Fix**: Processor now caches and prepends SPS/PPS to first frame of each segment
   - **Verification**: Check logs for "Prepending cached SPS/PPS" messages

4. **Startup Race Condition**
   - **Cause**: Processor starts before consumer is ready
   - **Fix**: Added 10-second startup grace period with synchronization
   - **Verification**: Check for "Waiting for consumer startup" log messages

### Watchdog Warnings

If you see "WATCHDOG: No ACK received in X.Xs", this indicates:
- Consumer (BagCounterApp) may not be processing frames
- Detection pipeline may be stuck
- Network/ROS2 connectivity issues

**Actions:**
1. Check BagCounterApp logs for errors
2. Verify ROS2 topics are connected: `ros2 topic list`
3. Check detector initialization: look for model loading errors
4. Restart the pipeline if necessary

### Spool Lag Warnings

The processor monitors spool lag every 2 minutes and warns if falling behind:

**Warning Levels:**
- `> 10 segments`: Critical lag (~50s behind) - processing too slow
- `5-10 segments`: Borderline lag (~25-50s behind) - monitor closely
- `< 5 segments`: Healthy lag

**If you see SPOOL LAG WARNING:**
1. Processing is slower than recording
2. Spool directory will grow until retention limit
3. Old segments will be deleted, causing frame loss

**Solutions:**
- Reduce ACK timeout to process faster
- Optimize detection/classification pipeline
- Check consumer performance (CPU, memory)
- Increase recording retention window temporarily

## Observability

### Recorder Logs

```
[SpoolRecorder] Stats: received=1500, written=1500, dropped=0, queue=5/100
[SpoolRecorder] Retention: segments=36, size=85.2MB, oldest_age=175.3s
[SegmentWriter] Closed segment 36: 125 frames, 2.4MB
```

### Processor Logs

**Regular Stats (every 10 seconds):**
```
[SpoolProcessor] Stats: session=d8489312, seq=4129, processed=950, retried=2, skipped=0, timeouts=0, ack_rejected=0, segments=36, sps_pps_prepends=36, state=idle
[SpoolProcessor] Spool: segments=36, current_frame=5432, last_ack_age=0.5s
```

**Detailed Stats (every 2 minutes):**
```
================================================================================
[SpoolProcessor] 📊 Detailed Statistics (2-minute summary)
  Session: d8489312-8ab4-4c76-b593-d9759133614d
  ACK Statistics:
    - Accepted: 2450 (99.9%)
    - Rejected (stale): 2 (0.1%)
    - Total: 2452
  Frame Processing:
    - Processed: 2448
    - Retried: 5
    - Skipped: 0
    - Timeouts: 2
  Spool Status:
    - Total segments: 36
    - Current segment: 1369
    - Oldest segment: 1360
    - Newest segment: 1372
    - Spool lag: 3 segments
  ✓ Spool lag is healthy (3 segments)
================================================================================
```

**Frame Publishing (milestone every 100 frames):**
```
[SpoolProcessor] 📤 Milestone: published 4100 frames, current: index=5432, session=d8489312, segment=1369, data_len=12977
```

**ACK Matching (debug level):**
```
[SpoolProcessor] ✓ ACK matched: seq=4128, frame_index=5431, elapsed=0.027s
```

### Consumer Logs

**Startup:**
```
[BagCounterApp] Accuracy Mode: session_id=e5f6g7h8
[BagCounterApp] ✓ READY published: session_id=e5f6g7h8
```

**Frame Processing (debug level):**
```
[BagCounterApp] Frame metadata received: frame_index=5432, seq=4129, session_id=d8489312
[BagCounterApp] ✓ ACK published: frame_index=5432, seq=4129, session=d8489312
```

**Frame Index Mismatch (throttled - every 100th occurrence):**
```
[BagCounterApp] ⚠ Frame index mismatch: expected 0, metadata has 5431 (total mismatches: 3900)
```

Note: Frame index mismatches are expected when Ros2FrameServer falls back to index 0 due to timing. The ACK system uses the correct metadata, so processing continues normally.

### Key Metrics

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| Queue utilization | < 50% | > 80% | Increase queue size |
| Segment age | < 180s | Approaching retention | Speed up processing |
| Spool lag | < 5 segments | > 10 segments | Critical: processing too slow |
| ACK acceptance rate | > 99% | < 95% | Check session ID issues |
| ACK timeouts | 0 | > 0 | Check BagCounterApp |
| ACK rejected (stale) | 0 | > 0 | Session mismatch after restart |
| Frames dropped | 0 | > 0 | Check recorder queue |
| SPS/PPS prepends | Equal to segments | N/A | Normal behavior |
| Last ACK age | < 1s | > 60s | Consumer may be stuck |
| Session ID | Matches | Mismatch | Restart synchronization issue |

## Testing

Run the unit tests:

```bash
# Test NAL parsing
python tests/test_h264_nal.py

# Test segment I/O
python tests/test_segment_io_roundtrip.py

# Test retention policy
python tests/test_retention_policy.py
```

## Why BEST_EFFORT on /nv12_images is OK

The decoded NV12 images use BEST_EFFORT QoS because:

1. **One frame in flight**: Strict backpressure means only one frame is decoded at a time
2. **No queue pressure**: Decoder receives frames at controlled pace
3. **ACK correlation**: Frame index tracking ensures we know which frame was processed

The pull-based architecture eliminates the queue pressure that previously caused drops.

## Production Best Practices

### Recommended Configuration

For production deployments, consider these settings:

```bash
# Enable accuracy mode
python config.py --key accuracy_mode_enabled --value 1

# Set reasonable ACK timeout (10s default, 30s if detection is very slow)
python config.py --key spool_ack_timeout --value 10.0

# Set retry count (2-3 recommended)
python config.py --key spool_retry_count --value 3

# Set retention to match expected processing lag (180s default)
python config.py --key spool_retention_seconds --value 300

# Configure windowed ACK for better throughput
# Start with 1 (backward compatible), increase to 3-5 if spool lag occurs
python config.py --key spool_inflight_window --value 3
```

### Monitoring Checklist

1. **Health Metrics**: Monitor ACK timeouts, frame drops, queue utilization, inflight window size
2. **Log Analysis**: Watch for WATCHDOG warnings, ACK timeout logs, and spool lag warnings
3. **Windowed ACK Metrics**: Check `inflight`, `out_of_order_acks`, `oldest_inflight_age` in stats
4. **Disk Space**: Ensure spool directory has sufficient space (200MB+ recommended)
5. **CPU/Memory**: Monitor system resources for bottlenecks

**Key metrics to watch:**
- `inflight`: Should stay below `inflight_window` setting (healthy: <max)
- `out_of_order_acks`: Normal with window>1 (indicates parallel processing)
- `oldest_inflight_age`: Should be <ack_timeout (healthy: <10s)
- `timeouts`: Should be 0 or very low (high values indicate consumer lag)
- `skipped`: Should be 0 (>0 indicates frames dropped after retries)
