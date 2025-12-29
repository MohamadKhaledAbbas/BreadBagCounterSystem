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
- Publishes exactly one frame at a time
- Waits for ACK before next frame
- Timeout and retry logic

#### 3. Frame Index Tracking

Since NV12 decoded frames don't carry the original H.264 index, we use a side-channel:

```
SpoolProcessor → /spool/current_frame_index → Ros2FrameServer
                                                     │
                                                     ▼
                                               BagCounterApp
                                                     │
                                                     ▼
                       /processing_ack ← ACK with frame index
```

### Topic-Based Pull Protocol

Since this repository is not built with colcon/ament, we cannot add custom `.srv` types. Instead, we implement pull using standard message types:

| Topic | Type | Direction | Purpose |
|-------|------|-----------|---------|
| `/spool_image_ch_0` | H26XFrame | Pub | Encoded frames to decoder |
| `/spool/current_frame_index` | UInt32 | Pub | Frame index for correlation |
| `/processing_ack` | UInt32 | Pub | ACK with processed frame index |
| `/spool/request_next` | UInt32 | Sub | Optional: external pull request |

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
| `spool_ack_timeout` | `30.0` | Timeout waiting for ACK (seconds) |
| `spool_retry_count` | `2` | Retries before advancing |

Example configuration commands:

```bash
# Enable accuracy mode
python config.py --key accuracy_mode_enabled --value 1

# Set spool directory
python config.py --key spool_dir --value /home/sunrise/BreadCounting/data/spool

# Set retention to 5 minutes
python config.py --key spool_retention_seconds --value 300
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

## Observability

### Recorder Logs

```
[SpoolRecorder] Stats: received=1500, written=1500, dropped=0, queue=5/100
[SpoolRecorder] Retention: segments=36, size=85.2MB, oldest_age=175.3s
[SegmentWriter] Closed segment 36: 125 frames, 2.4MB
```

### Processor Logs

```
[SpoolProcessor] Stats: processed=1000, retried=2, skipped=0, timeouts=0
[SpoolProcessor] Spool: segments=36, current_frame=1000
```

### Key Metrics

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| Queue utilization | < 50% | > 80% | Increase queue size |
| Segment age | < 180s | Approaching retention | Speed up processing |
| ACK timeouts | 0 | > 0 | Check BagCounterApp |
| Frames dropped | 0 | > 0 | Check recorder queue |

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
