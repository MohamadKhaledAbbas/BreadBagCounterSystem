# V4 Performance Optimization - Configuration Guide

## Overview
This guide provides detailed configuration instructions for the V4 performance optimization features designed to achieve 20-25ms frame processing time for 25+ FPS throughput.

---

## Phase 1: Detection Results Queue

### Purpose
Decouple detection from monitor processing to allow parallel execution. Detection runs at full BPU speed without blocking on monitor.

### Configuration Parameters

#### `detection_queue_enabled`
- **Type**: Boolean
- **Default**: `True`
- **Environment Variable**: `DETECTION_QUEUE_ENABLED`
- **Description**: Enable detection results queue for parallel processing
- **Values**:
  - `True`: Detection and monitor run in separate threads (recommended)
  - `False`: Legacy serial processing

**Example**:
```bash
export DETECTION_QUEUE_ENABLED=true
```

#### `detection_queue_size`
- **Type**: Integer
- **Default**: `10`
- **Environment Variable**: `DETECTION_QUEUE_SIZE`
- **Range**: 5 - 30
- **Description**: Maximum number of detection results to buffer
- **Tuning**:
  - Lower values (5-8): Less memory, faster backpressure response
  - Higher values (15-30): More buffering, handles burst loads better
- **Memory Impact**: ~4MB per frame at 720p × queue size

**Example**:
```bash
export DETECTION_QUEUE_SIZE=10
```

#### `detection_queue_warning_threshold`
- **Type**: Float
- **Default**: `0.7` (70%)
- **Environment Variable**: `DETECTION_QUEUE_WARNING_THRESHOLD`
- **Range**: 0.5 - 0.9
- **Description**: Queue utilization threshold to trigger warnings
- **Interpretation**: Warning logged when queue exceeds this threshold

**Example**:
```bash
export DETECTION_QUEUE_WARNING_THRESHOLD=0.7
```

### Expected Performance
- **Baseline**: 35ms detection + 25ms monitor = 60ms total (serial)
- **Optimized**: ~35ms total (parallel, overlap)
- **Gain**: ~40% improvement

### Monitoring
Check logs for these metrics:
```
[QueueStats] Detection: 4/10 (40.0% full, drops=0)
[MonitorThread] Frame 300: detect=32.5ms, monitor=23.1ms, queue_size=4/10
```

---

## Phase 2: BPU Batch Inference

### Purpose
Process multiple frames in a single BPU forward pass for 40-60% speedup. YOLOv8n achieves 220 FPS with batching vs 140 FPS single-frame.

### Configuration Parameters

#### `detection_batch_enabled`
- **Type**: Boolean
- **Default**: `True`
- **Environment Variable**: `DETECTION_BATCH_ENABLED`
- **Description**: Enable batch inference for detection
- **Values**:
  - `True`: Accumulate frames and process in batches (recommended)
  - `False`: Single-frame processing (legacy)

**Example**:
```bash
export DETECTION_BATCH_ENABLED=true
```

#### `detection_batch_size`
- **Type**: Integer
- **Default**: `2`
- **Environment Variable**: `DETECTION_BATCH_SIZE`
- **Range**: 2 - 4
- **Description**: Number of frames to process per batch
- **Tuning Guide**:
  - **Batch=2**: Conservative, ~40% speedup, minimal latency impact (recommended start)
  - **Batch=3**: Balanced, ~50% speedup, slight latency increase
  - **Batch=4**: Aggressive, ~60% speedup, may increase latency

**Performance vs Latency Tradeoff**:
| Batch Size | Speedup | Time/Frame | Latency Impact |
|------------|---------|------------|----------------|
| 1 (baseline) | 1.0x | 35ms | 0ms |
| 2 | 1.4x | 25ms | +5ms |
| 3 | 1.5x | 23ms | +10ms |
| 4 | 1.6x | 22ms | +15ms |

**Example**:
```bash
export DETECTION_BATCH_SIZE=2  # Start conservative
```

#### `detection_batch_timeout_ms`
- **Type**: Float
- **Default**: `5.0` milliseconds
- **Environment Variable**: `DETECTION_BATCH_TIMEOUT_MS`
- **Range**: 2.0 - 10.0 ms
- **Description**: Maximum time to wait for batch to fill
- **Purpose**: Prevents latency spikes when frame rate drops
- **Tuning**:
  - Lower values (2-3ms): More responsive, may reduce batch efficiency
  - Higher values (7-10ms): More efficient batching, may increase latency

**Example**:
```bash
export DETECTION_BATCH_TIMEOUT_MS=5.0
```

### Expected Performance
- **Baseline**: 35ms per frame (single)
- **Optimized (batch=2)**: 18-22ms per frame
- **Gain**: 40-50% speedup

### Monitoring
Check logs for these metrics:
```
[BpuDetector] Batch inference stats (50 batches): avg_batch_size=2.0, avg_time_per_frame=21.35ms (speedup=1.64x vs 35ms baseline)
```

### Tuning Workflow
1. **Start Conservative**: `DETECTION_BATCH_SIZE=2`
2. **Measure**: Check `avg_time_per_frame_ms` in logs
3. **If < 25ms**: Increase to `3` for more throughput
4. **If < 20ms**: Increase to `4` for maximum throughput
5. **Monitor**: Watch for latency spikes in real-world scenarios

---

## Phase 3: Monitor Processing Optimization

### Purpose
Reduce monitor processing time through lazy ROI cropping and vectorized IoU calculations.

### Configuration Parameters

#### `lazy_roi_cropping_enabled`
- **Type**: Boolean
- **Default**: `True`
- **Environment Variable**: `LAZY_ROI_CROPPING_ENABLED`
- **Description**: Enable lazy ROI cropping (crop on-demand, not immediately)
- **Benefits**:
  - Reduces memory bandwidth (no immediate cropping)
  - Reduces CPU overhead (only crop what's needed)
  - Events that expire never trigger cropping
- **Expected Gain**: 30-50% reduction in monitor time

**Example**:
```bash
export LAZY_ROI_CROPPING_ENABLED=true
```

#### `vectorized_iou_enabled`
- **Type**: Boolean
- **Default**: `True`
- **Environment Variable**: `VECTORIZED_IOU_ENABLED`
- **Description**: Enable numpy vectorized IoU calculations
- **Benefits**:
  - Replaces O(n*m) loops with O(1) vectorized ops
  - 2-3x faster association for multiple events
  - Better performance with many active events
- **Expected Gain**: 30-40% reduction in association time

**Example**:
```bash
export VECTORIZED_IOU_ENABLED=true
```

### Expected Performance
- **Baseline**: 25ms monitor time
- **Optimized**: 10-15ms monitor time
- **Gain**: 40-60% improvement

---

## Phase 4: Classification Batch Processing (FUTURE)

### Purpose
Batch multiple events' ROIs in single classifier call. Lower priority as classification already runs async.

### Configuration Parameters

#### `classification_batch_enabled`
- **Type**: Boolean
- **Default**: `False` (not implemented)
- **Environment Variable**: `CLASSIFICATION_BATCH_ENABLED`
- **Description**: Enable batch classification for multiple events
- **Note**: Only enable if classification becomes bottleneck after Phases 1-3

#### `classification_batch_size`
- **Type**: Integer
- **Default**: `4`
- **Environment Variable**: `CLASSIFICATION_BATCH_SIZE`
- **Range**: 2 - 8
- **Description**: Number of events to batch for classification

---

## Recommended Production Configuration

### For Maximum Performance (RDK X5)
```bash
# Phase 1: Detection Queue
export DETECTION_QUEUE_ENABLED=true
export DETECTION_QUEUE_SIZE=10
export DETECTION_QUEUE_WARNING_THRESHOLD=0.7

# Phase 2: Batch Inference (aggressive)
export DETECTION_BATCH_ENABLED=true
export DETECTION_BATCH_SIZE=4
export DETECTION_BATCH_TIMEOUT_MS=5.0

# Phase 3: Monitor Optimization
export LAZY_ROI_CROPPING_ENABLED=true
export VECTORIZED_IOU_ENABLED=true

# Phase 4: Classification (disabled, not needed)
export CLASSIFICATION_BATCH_ENABLED=false
```

**Expected**: 15-20ms per frame, 50+ FPS

### For Balanced Performance (Recommended Start)
```bash
# Phase 1: Detection Queue
export DETECTION_QUEUE_ENABLED=true
export DETECTION_QUEUE_SIZE=10
export DETECTION_QUEUE_WARNING_THRESHOLD=0.7

# Phase 2: Batch Inference (conservative)
export DETECTION_BATCH_ENABLED=true
export DETECTION_BATCH_SIZE=2
export DETECTION_BATCH_TIMEOUT_MS=5.0

# Phase 3: Monitor Optimization
export LAZY_ROI_CROPPING_ENABLED=true
export VECTORIZED_IOU_ENABLED=true

# Phase 4: Classification (disabled)
export CLASSIFICATION_BATCH_ENABLED=false
```

**Expected**: 20-25ms per frame, 40+ FPS

### For Testing/Debugging (Legacy Mode)
```bash
# Disable all optimizations
export DETECTION_QUEUE_ENABLED=false
export DETECTION_BATCH_ENABLED=false
export LAZY_ROI_CROPPING_ENABLED=false
export VECTORIZED_IOU_ENABLED=false
export CLASSIFICATION_BATCH_ENABLED=false
```

**Expected**: 60ms per frame, 16 FPS (baseline)

---

## Troubleshooting

### Issue: Detection Queue Drops
**Symptoms**: `[DetectionQueuePressure] High queue utilization`

**Solutions**:
1. Check monitor thread CPU usage - may be bottleneck
2. Increase `detection_queue_size` to 15-20
3. Enable Phase 3 optimizations (lazy ROI, vectorized IoU)
4. Check for slow disk I/O if ROI saving is enabled

### Issue: Batch Inference Not Achieving Speedup
**Symptoms**: `avg_time_per_frame_ms` not improving

**Solutions**:
1. Check if hobot_dnn native batch API is available
2. Verify `detection_batch_size` > 1
3. Check for CPU preprocessing bottleneck
4. Monitor `preprocess_time_ms` in logs

### Issue: Latency Spikes
**Symptoms**: Occasional very high frame processing times

**Solutions**:
1. Reduce `detection_batch_size` from 4 to 2
2. Reduce `detection_batch_timeout_ms` from 5ms to 3ms
3. Check for degraded mode activation
4. Monitor system CPU and memory

### Issue: Accuracy Degradation
**Symptoms**: Missed detections or incorrect counts

**Solutions**:
1. Verify batch inference produces same results as single-frame
2. Check for frame drops in detection queue
3. Ensure lazy ROI cropping doesn't affect classification
4. Run accuracy validation tests

---

## Performance Monitoring

### Key Metrics to Watch

1. **Detection Time**:
   ```
   [BpuDetector] Avg timing: inference=18.5ms
   ```
   - Target: <22ms per frame (with batch=2)

2. **Monitor Time**:
   ```
   [MonitorThread] detect=32.5ms, monitor=12.3ms
   ```
   - Target: <15ms

3. **Queue Utilization**:
   ```
   [QueueStats] Detection: 4/10 (40.0% full, drops=0)
   ```
   - Target: <80% utilization, 0 drops

4. **End-to-End FPS**:
   ```
   [Frame 300] FPS: 42.3
   ```
   - Target: ≥25 FPS

### Structured Logging
V4 optimizations emit structured JSON logs for analysis:

```json
{
  "event": "batch_inference_stats",
  "avg_batch_size": 2.0,
  "avg_time_per_frame_ms": 21.35,
  "speedup_factor": 1.64,
  "target_speedup": "1.4-1.6x"
}
```

Use tools like `jq` to analyze:
```bash
grep "batch_inference_stats" app.log | jq '.avg_time_per_frame_ms'
```

---

## Migration from V3 to V4

### Backward Compatibility
All V4 optimizations are **opt-in** with feature flags defaulting to **enabled**. To run in legacy V3 mode, disable all flags:

```bash
export DETECTION_QUEUE_ENABLED=false
export DETECTION_BATCH_ENABLED=false
export LAZY_ROI_CROPPING_ENABLED=false
export VECTORIZED_IOU_ENABLED=false
```

### Gradual Migration
1. **Week 1**: Enable Phase 1 only (detection queue)
2. **Week 2**: Enable Phase 2 (batch inference with batch=2)
3. **Week 3**: Enable Phase 3 (monitor optimizations)
4. **Week 4**: Tune batch_size to 3 or 4 for maximum performance

### Validation
After enabling each phase:
1. Run 1-hour stability test
2. Compare counts vs baseline
3. Check for queue drops or errors
4. Monitor memory usage

---

## References
- [RDK X5 Python API Manual](https://developer.d-robotics.cc/rdk_doc/en/rdk_s/Algorithm_Application/python-api/)
- [YOLOv8 220 FPS Batch Inference Demo](https://my.cytron.io/tutorial/rdk-x5-yolov8n-220-fps-object-detection-end-to-end-220-fps)
- [hobot_dnn GitHub](https://github.com/D-Robotics/hobot_dnn)
- [V4 Implementation Status](./V4_PERFORMANCE_OPTIMIZATION_STATUS.md)

---

**Last Updated**: 2025-12-27
**Version**: V4.0
