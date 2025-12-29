# Accuracy Mode Troubleshooting Guide

## ACK Timeout Issues

### Symptom
```
[SpoolProcessor] ACK timeout for frame XXXXX, retry 1/2
[Ros2FrameServer] Received frame index: XXXXX, queue_size=8
[SpoolProcessor] ACK timeout for frame XXXXX, retry 2/2
[Ros2FrameServer] Received frame index: XXXXX, queue_size=9
[SpoolProcessor] Frame XXXXX skipped after retries
```

### Diagnosis Checklist

#### 1. Check Decoder Status

The most common cause is the decoder (`hobot_codec_republish`) not running or not decoding frames.

```bash
# Check if decoder node is running
ros2 node list | grep codec

# Expected output:
# /hobot_codec_republish
```

If decoder is not listed, check ROS2 service:
```bash
sudo supervisorctl status breadcount-ros2
```

#### 2. Verify Topic Connections

**Check decoder subscription:**
```bash
ros2 topic info /spool_image_ch_0 -v

# Expected: Should show hobot_codec_republish as subscriber
```

**Check decoder publication:**
```bash
ros2 topic info /nv12_images -v

# Expected: Should show hobot_codec_republish as publisher
```

**Check frame rate:**
```bash
# Check if frames are being published by SpoolProcessor
ros2 topic hz /spool_image_ch_0

# Check if frames are being decoded
ros2 topic hz /nv12_images

# If /nv12_images shows 0 Hz or no output, decoder is not working
```

#### 3. Check Frame Index Correlation

**Monitor Ros2FrameServer:**
```bash
# Look for these debug messages
grep "Correlated decoded frame" /home/sunrise/BreadCounting/data/logs/*.log

# If NO messages found, NV12 frames are not arriving at BagCounterApp
```

**Check pending queue:**
```bash
# Look for Ros2FrameServer stats
grep "pending_indices" /home/sunrise/BreadCounting/data/logs/*.log

# Healthy: pending_indices=0-5
# Problem: pending_indices growing (8, 9, 10, 11, ...)
```

If `pending_indices` keeps growing, it means:
- Frame indices ARE being received ✓
- But NV12 decoded frames are NOT arriving ✗

#### 4. Check BagCounterApp Status

```bash
# Check if BagCounterApp is running
ps aux | grep BagCounterApp

# Check if it's processing frames
grep "Frame acquisition stats" /home/sunrise/BreadCounting/data/logs/*.log

# If NO output, BagCounterApp is not receiving frames
```

#### 5. Inspect Decoder Logs

Check for decoder initialization errors:
```bash
# Check ROS2 launch logs
sudo supervisorctl tail -f breadcount-ros2

# Look for errors like:
# - "Failed to initialize codec"
# - "Invalid format"
# - "QoS mismatch"
```

### Common Issues and Fixes

#### Issue 1: Decoder Not Running

**Symptoms:**
- `ros2 node list` doesn't show `/hobot_codec_republish`
- No output from `ros2 topic hz /nv12_images`

**Fix:**
```bash
# Restart ROS2 service
sudo supervisorctl restart breadcount-ros2

# Wait 10 seconds for initialization
sleep 10

# Verify decoder is running
ros2 node list | grep codec
```

#### Issue 2: Decoder Not Initializing (No SPS/PPS)

**Symptoms:**
- Decoder node is running ✓
- Subscribed to `/spool_image_ch_0` ✓
- But NOT publishing to `/nv12_images` ✗
- `sps_pps_prepends=0` in SpoolProcessor stats
- `ros2 topic hz /spool_image_ch_0` shows 0.033 Hz (one frame per 30s)

**Root Cause:**
Decoder can't initialize without SPS/PPS (Sequence Parameter Set / Picture Parameter Set) NAL units. If the first frame sent doesn't contain SPS/PPS and none are cached, decoder fails silently.

**Fix:**
This is now fixed automatically via pre-scanning (commit 2da2c7c). SpoolProcessor scans up to 100 frames on startup to find and cache SPS/PPS before sending any frames to the decoder.

**Verification:**
```bash
# Check pre-scan logs
grep "Pre-scanning for SPS/PPS" /home/sunrise/BreadCounting/data/logs/spool-processor.log
# Should see: "Found and cached SPS/PPS"

# Check that SPS/PPS prepending is working
grep "sps_pps_prepends" /home/sunrise/BreadCounting/data/logs/spool-processor.log
# Should show sps_pps_prepends > 0

# Verify decoder is now publishing
ros2 topic hz /nv12_images
# Should show ~30 Hz
```

If pre-scan fails to find SPS/PPS:
```bash
# Check if segments contain any IDR frames
python3 -c "
from src.spool.segment_io import SegmentReader
from src.spool.h264_nal import extract_sps_pps
reader = SegmentReader('/home/sunrise/BreadCounting/data/spool')
for frame in reader.read_frames():
    sps, pps = extract_sps_pps(frame.data)
    if sps or pps:
        print(f'Frame {frame.index}: SPS={bool(sps)}, PPS={bool(pps)}')
        break
"
```

#### Issue 3: QoS Mismatch

**Symptoms:**
- Decoder runs but doesn't receive frames
- `ros2 topic info /spool_image_ch_0 -v` shows no subscribers

**Fix:**
QoS should already be BEST_EFFORT. If mismatch occurs, check:
```bash
# Check SpoolProcessor QoS
grep "best_effort_qos" /home/sunrise/BreadCounting/src/ros2_spool/spool_processor_node.py

# Should see depth=5, BEST_EFFORT
```

#### Issue 4: Missing SPS/PPS (Legacy - Fixed)

**Symptoms:**
- Decoder receives frames but doesn't decode
- Logs show "Invalid NAL unit" or "No SPS/PPS"

**Fix:**
This is now handled automatically by pre-scan and prepending logic. If you still see issues:
```bash
# Verify pre-scan is enabled
grep "prescan_for_sps_pps" /home/sunrise/BreadCounting/src/ros2_spool/spool_processor_node.py

# Check prepending is working
grep "Prepending cached SPS" /home/sunrise/BreadCounting/data/logs/*.log
```

#### Issue 5: Wrong Topic Configuration

**Symptoms:**
- Everything appears to run but no frames flow

**Fix:**
Verify topic configuration in launcher:
```bash
cat /home/sunrise/BreadCounting/src/ros2/Ros2PipelineLauncher.py | grep -A5 "sub_topic"

# Should show:
# 'sub_topic': '/spool_image_ch_0',
```

### Recovery Procedure

If ACK timeouts persist after diagnostics:

**Step 1: Full Restart**
```bash
# Stop all services
sudo supervisorctl stop breadcount-ros2 breadcount-main breadcount-spool-recorder breadcount-spool-processor

# Wait for clean shutdown
sleep 5

# Start in order
sudo supervisorctl start breadcount-ros2
sleep 10  # Wait for decoder initialization

sudo supervisorctl start breadcount-spool-recorder
sudo supervisorctl start breadcount-spool-processor
sleep 5

sudo supervisorctl start breadcount-main
```

**Step 2: Monitor Startup**
```bash
# Watch all logs
tail -f /home/sunrise/BreadCounting/data/logs/*.log

# Look for:
# - "Waiting for consumer startup" (SpoolProcessor)
# - "Correlated decoded frame" (Ros2FrameServer)
# - "Frame acquisition stats" (BagCounterApp)
```

**Step 3: Verify Flow**
```bash
# 1. Check SpoolProcessor is publishing
ros2 topic hz /spool_image_ch_0
# Should show ~30 Hz

# 2. Check decoder is decoding
ros2 topic hz /nv12_images
# Should show ~30 Hz

# 3. Check ACKs are being sent
ros2 topic hz /processing_ack
# Should show ~30 Hz
```

### Debug Output Interpretation

**Healthy System:**
```
[SpoolProcessor] Published frame 22724
[Ros2FrameServer] Received frame index: 22724, queue_size=2
[Ros2FrameServer] Correlated decoded frame with index 22724, pending_queue=1
[BagCounterApp] Published ACK for frame 22724
[SpoolProcessor] Received ACK for frame 22724
[SpoolProcessor] Published frame 22725
```

**Broken System (Decoder Not Initialized - No SPS/PPS):**
```
[SpoolProcessor] Published frame 22724
[SpoolProcessor] ACK timeout for frame 22724, retry 1/2 ← 30 seconds later
[SpoolProcessor] Published frame 22724
[SpoolProcessor] ACK timeout for frame 22724, retry 2/2 ← 30 seconds later
[SpoolProcessor] Frame 22724 skipped after retries
[SpoolProcessor] Stats: sps_pps_prepends=0 ← No SPS/PPS prepending!
```

Key indicators:
- Frames published every 30 seconds (= retry timeout)
- `sps_pps_prepends=0` means decoder never received SPS/PPS
- NO "Correlated decoded frame" messages
- `ros2 topic hz /nv12_images` shows "not published"

**Fixed System (With Pre-scan):**
```
[SpoolProcessor] Pre-scanning for SPS/PPS NAL units...
[SpoolProcessor] Found and cached SPS from frame 22724
[SpoolProcessor] Found and cached PPS from frame 22724
[SpoolProcessor] Pre-scan complete: found SPS/PPS after scanning 1 frames
[SpoolProcessor] Published frame 22724 (data_len=7834)
[Ros2FrameServer] Correlated decoded frame with index 22724
[BagCounterApp] Published ACK for frame 22724
[SpoolProcessor] Stats: sps_pps_prepends=1, processed=1
```

### Performance Metrics

**Normal Operation:**
```
[Ros2FrameServer] Stats: received=1500, processed=1500, dropped=0,
  drop_rate=0.00%, queue_util=20.0%, pending_indices=2, fallbacks=0

[SpoolProcessor] Stats: processed=1000, retried=0, skipped=0, 
  timeouts=0, segments=36
```

**Decoder Failure:**
```
[Ros2FrameServer] Stats: received=0, processed=0, dropped=0,
  pending_indices=15 ← Growing!

[SpoolProcessor] Stats: processed=0, retried=6, skipped=3,
  timeouts=9, segments=0 ← No successful processing!
```

### Contact Support

If issues persist after following this guide:

1. Capture full diagnostic output:
   ```bash
   ./docs/collect_diagnostics.sh > diagnostics.txt
   ```

2. Provide:
   - Full logs from all services
   - Output of `ros2 node list`
   - Output of `ros2 topic list`
   - Output of diagnostic script

## Related Documentation

- [ACCURACY_MODE_SPOOLING.md](ACCURACY_MODE_SPOOLING.md) - Architecture overview
- [ACK_RELIABILITY_FIX.md](ACK_RELIABILITY_FIX.md) - FIFO queue implementation details
