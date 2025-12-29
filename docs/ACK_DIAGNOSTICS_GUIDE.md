# Enhanced ACK Diagnostics Guide

## Changes Made (Commit a8d5af8 + new enhancements)

### Enhanced Logging for ACK Flow Visibility

To diagnose ACK timeout issues, we've added enhanced INFO-level logging at critical points in the ACK flow. This makes it easy to see where ACKs are being published, received, and matched.

### What to Look For in Logs

#### **Healthy ACK Flow**

You should see this pattern repeating every ~33ms (at 30 FPS):

```
[SpoolProcessor] Published frame 22724 (data_len=7834)
[SpoolProcessor] Waiting for ACK for frame 22724 (timeout=10.0s)
[Ros2FrameServer] Correlated decoded frame with index 22724, pending_queue=2
[BagCounterApp] ✓ Published ACK for frame 22724
[SpoolProcessor] ✓ ACK callback triggered for frame 22724
[SpoolProcessor] ✓ ACK matched for frame 22724 (elapsed=0.045s)
```

**Key indicators of healthy operation:**
- `✓` symbols indicate successful operations
- ACK arrives within <100ms typically
- Frame indices match throughout the flow
- No mismatch warnings
- No timeout warnings

#### **Problem Pattern 1: ACK Never Published**

```
[SpoolProcessor] Published frame 22724 (data_len=7834)
[SpoolProcessor] Waiting for ACK for frame 22724 (timeout=10.0s)
[Ros2FrameServer] Received frame index: 22724, queue_size=8
... 10 seconds pass with NO other logs ...
[SpoolProcessor] ⏱ ACK timeout for frame 22724 after 10.0s - no ACK received
```

**Missing:**
- NO "Correlated decoded frame" message → Decoder not decoding
- NO "Published ACK" message → BagCounterApp not processing frames

**Root cause:** Decoder initialization issue (see commit 2da2c7c fix for SPS/PPS pre-scan)

**Action:** 
1. Check if decoder is publishing: `ros2 topic hz /nv12_images`
2. Check pre-scan logs: `grep "Pre-scanning for SPS/PPS" logs/spool-processor.log`
3. Restart SpoolProcessor: `sudo supervisorctl restart breadcount-spool-processor`

#### **Problem Pattern 2: ACK Published but Not Received**

```
[SpoolProcessor] Published frame 22724 (data_len=7834)
[SpoolProcessor] Waiting for ACK for frame 22724 (timeout=10.0s)
[Ros2FrameServer] Correlated decoded frame with index 22724, pending_queue=2
[BagCounterApp] ✓ Published ACK for frame 22724
... 10 seconds pass with NO ACK callback ...
[SpoolProcessor] ⏱ ACK timeout for frame 22724 after 10.0s - no ACK received
```

**Missing:**
- NO "ACK callback triggered" message → ACK not reaching SpoolProcessor

**Root cause:** ROS2 communication issue (QoS mismatch, network, or subscription not active)

**Action:**
1. Verify both sides use RELIABLE QoS: ✓ Already configured
2. Check ROS2 node list: `ros2 node list`
3. Check topic connections: `ros2 topic info /processing_ack`
4. Restart ROS2: `sudo supervisorctl restart breadcount-ros2`

#### **Problem Pattern 3: ACK Received but Index Mismatch**

```
[SpoolProcessor] Published frame 22724 (data_len=7834)
[SpoolProcessor] Waiting for ACK for frame 22724 (timeout=10.0s)
[BagCounterApp] ✓ Published ACK for frame 22723
[SpoolProcessor] ✓ ACK callback triggered for frame 22723
[SpoolProcessor] ⚠ ACK mismatch: expected 22724, got 22723 (waiting 9.5s more)
```

**Root cause:** Frame index correlation issue (FIFO queue problem or race condition)

**Action:**
1. Check `pending_indices` metric in Ros2FrameServer stats
2. Verify FIFO queue is working: should see "Correlated decoded frame" messages
3. Check if multiple ACKs arriving out of order

#### **Problem Pattern 4: State Machine Inversion** (from new requirement)

If you see this pattern, there's a critical bug:

```
[SpoolProcessor] Waiting for ACK for frame 22724 (timeout=10.0s)
[SpoolProcessor] Published frame 22724 (data_len=7834)
```

**WRONG ORDER!** Waiting happens BEFORE publishing.

**Root cause:** State machine bug - setting `WAITING_FOR_ACK` before publishing frame.

**Status:** ✅ Verified correct in current code (lines 683-690 of spool_processor_node.py)

### How to Use These Diagnostics

#### **Step 1: Enable INFO-level logging**

Ensure logs are set to INFO level (not just WARNING). Check your logging configuration.

#### **Step 2: Monitor real-time logs**

```bash
# Follow all logs
tail -f /home/sunrise/BreadCounting/data/logs/*.log

# Filter for ACK flow
tail -f /home/sunrise/BreadCounting/data/logs/*.log | grep -E "Published frame|Waiting for ACK|Published ACK|ACK callback|ACK matched"

# Filter for problems
tail -f /home/sunrise/BreadCounting/data/logs/*.log | grep -E "timeout|mismatch|⏱|⚠"
```

#### **Step 3: Identify pattern**

Match the log output against the patterns above to identify which issue you're experiencing.

#### **Step 4: Apply targeted fix**

Use the "Action" listed for your specific pattern.

### QoS Configuration Verification

Both publisher and subscriber use RELIABLE QoS with depth=10 (SpoolProcessor) and depth=1 (BagCounterApp):

**SpoolProcessor (subscriber):**
```python
reliable_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10  # Buffering for reliability
)
```

**BagCounterApp (publisher):**
```python
reliable_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1  # Single ACK at a time
)
```

This configuration ensures:
- ACKs are not lost under normal operation
- Messages are delivered in order
- Subscriber has buffer to handle bursts

### Testing Checklist

After restarting with the enhanced logging:

- [ ] See "✓ Published ACK" messages in BagCounterApp logs
- [ ] See "✓ ACK callback triggered" messages in SpoolProcessor logs
- [ ] See "✓ ACK matched" messages with elapsed time <1s
- [ ] See "Correlated decoded frame" messages in Ros2FrameServer logs
- [ ] `sps_pps_prepends > 0` in SpoolProcessor stats
- [ ] `pending_indices` stays low (0-5) in Ros2FrameServer stats
- [ ] NO timeout warnings
- [ ] NO mismatch warnings

If any of these fail, use the pattern matching above to diagnose.

### Summary of Fixes

1. **Commit aacf3fd**: FIFO queue for frame index correlation
2. **Commit 2da2c7c**: SPS/PPS pre-scan for decoder initialization
3. **Commit a8d5af8**: Memory leak fix and documentation updates
4. **This commit**: Enhanced logging for ACK flow visibility

All known issues from the new requirement analysis have been addressed:
- ✅ State machine order is correct
- ✅ ACK correlation uses matching types (UInt32)
- ✅ QoS is RELIABLE on both sides
- ✅ Subscription created before publishing
- ✅ Enhanced logging to diagnose any remaining issues
