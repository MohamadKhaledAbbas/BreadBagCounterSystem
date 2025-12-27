# V4 Performance Optimization - Implementation Status

## Overview
This document tracks the implementation status of the V4 performance optimization initiative aimed at achieving 20-25ms frame processing time (currently 60-130ms) for 25+ FPS throughput.

## Target Performance Gains
| Optimization | Current | Target | Expected Gain |
|--------------|---------|--------|---------------|
| **Phase 1**: Detection Queue | 35ms detect + 25ms monitor (serial) | ~35ms (parallel) | 40% |
| **Phase 2**: BPU Batch (batch=2) | 35ms/frame | 18-22ms/frame | 40-50% |
| **Phase 3**: Monitor Vectorization | 25ms | 10-15ms | 40-60% |
| **TOTAL PIPELINE** | **60ms** | **20-25ms** | **2.5-3x** |
| **Effective FPS** | **16 FPS** | **40-50 FPS** | **Exceeds 25 FPS target ✅** |

---

## Phase 1: Detection Results Queue ⏳ IN PROGRESS

### Status: 70% Complete

### Completed ✅
1. **Configuration Parameters** (`src/config/tracking_config.py`)
   - ✅ `detection_queue_enabled: bool = True`
   - ✅ `detection_queue_size: int = 10`
   - ✅ `detection_queue_warning_threshold: float = 0.7`
   - ✅ Environment variable overrides (`DETECTION_QUEUE_ENABLED`, etc.)

2. **BagCounterApp Initialization** (`src/counting/BagCounterApp.py`)
   - ✅ Detection queue instantiation
   - ✅ Monitor thread initialization variables
   - ✅ Detection queue drop tracking

3. **Monitor Thread Loop**
   - ✅ `_monitor_thread_loop()` implementation
   - ✅ Queue consumption with timeout
   - ✅ Monitor processing (EventCentricStateMonitor.update())
   - ✅ Classification enqueueing
   - ✅ Error handling and structured logging

4. **Queue Statistics Logging**
   - ✅ Detection queue metrics in `_log_queue_stats()`
   - ✅ Utilization warnings
   - ✅ Drop tracking

### Remaining Work ⏳
1. **Logic Thread Integration**
   - [ ] Modify `_logic_thread_loop()` to enqueue detection results instead of calling monitor directly
   - [ ] Handle detection queue full scenario (backpressure)
   - [ ] Update timing logs to reflect decoupled architecture

2. **Thread Lifecycle Management**
   - [ ] Start monitor thread in `run()` method
   - [ ] Graceful shutdown in `shutdown_procedure()`
   - [ ] Thread join with timeout

3. **Testing**
   - [ ] Unit tests for detection queue
   - [ ] Integration tests with real video
   - [ ] Performance benchmarking

---

## Phase 2: BPU Batch Inference ⏳ IN PROGRESS

### Status: 60% Complete

### Completed ✅
1. **Configuration Parameters** (`src/config/tracking_config.py`)
   - ✅ `detection_batch_enabled: bool = True`
   - ✅ `detection_batch_size: int = 2`
   - ✅ `detection_batch_timeout_ms: float = 5.0`
   - ✅ Environment variable overrides

2. **BpuDetector.predict_batch()** (`src/detection/BpuDetector.py`)
   - ✅ Batch preprocessing (vectorized)
   - ✅ Batch inference attempt (note: hobot_dnn may need native batch API)
   - ✅ Per-frame postprocessing
   - ✅ Comprehensive timing metrics logging
   - ✅ Speedup factor calculation vs baseline
   - ✅ Fallback to single-frame on errors

3. **Metrics & Logging**
   - ✅ Batch counter and timing accumulation
   - ✅ Structured JSON logging every 50 batches
   - ✅ Speedup factor reporting

### Remaining Work ⏳
1. **Logic Thread Integration**
   - [ ] Frame accumulation in logic thread
   - [ ] Timeout-based batch flushing (max 5ms wait)
   - [ ] Call `predict_batch()` instead of `predict()`
   - [ ] Handle partial batches gracefully

2. **BPU Native Batch Support**
   - [ ] Investigate hobot_dnn batch API
   - [ ] Optimize 4D tensor input if supported
   - [ ] May require hobot_dnn documentation review

3. **Testing**
   - [ ] Benchmark single-frame vs batch (target: 40-60% speedup)
   - [ ] Validate detection accuracy (no degradation)
   - [ ] Test with different batch sizes (2, 3, 4)

---

## Phase 3: Monitor Processing Optimization ⏸️ PLANNED

### Status: 0% Complete (Configuration Added)

### Completed ✅
1. **Configuration Parameters** (`src/config/tracking_config.py`)
   - ✅ `lazy_roi_cropping_enabled: bool = True`
   - ✅ `vectorized_iou_enabled: bool = True`

### Remaining Work ⏳
1. **Lazy ROI Cropping**
   - [ ] Modify `EventCentricTracker` to store metadata only
   - [ ] Implement on-demand cropping when event is ready
   - [ ] Memory and CPU metrics

2. **Vectorized IoU**
   - [ ] Implement `compute_iou_batch()` function
   - [ ] Replace loop-based IoU in association logic
   - [ ] Performance benchmarking

3. **Testing**
   - [ ] Validate no accuracy degradation
   - [ ] Measure monitor time reduction (target: 30-50%)

---

## Phase 4: Classification Batch Processing ⏸️ LOWER PRIORITY

### Status: 0% Complete (Configuration Added)

### Note
Classification already runs async (Phase 2: Multiple workers). Only implement if it becomes a bottleneck after Phases 1-3.

### Completed ✅
1. **Configuration Parameters** (`src/config/tracking_config.py`)
   - ✅ `classification_batch_enabled: bool = False`
   - ✅ `classification_batch_size: int = 4`

---

## Implementation Priorities

### Critical Path (Must Complete)
1. ✅ **Configuration**: All phase parameters added
2. ⏳ **Phase 1 Integration**: Complete logic thread and thread lifecycle (Est: 2-3 hours)
3. ⏳ **Phase 2 Integration**: Complete batch accumulation and flushing (Est: 2-3 hours)
4. ⏳ **Testing**: Benchmark and validate (Est: 1-2 hours)

### Optional Enhancements
- Phase 3: Lazy ROI + Vectorized IoU (Est: 3-4 hours)
- Phase 4: Classification batching (Only if needed)

---

## Next Steps

### Immediate (Complete Phase 1)
1. **Modify `_logic_thread_loop()`**:
   ```python
   # After detection
   if self.detection_queue_enabled:
       # Enqueue detection result
       detection_result = (current_frame_detections, frame_count, frame, detect_time)
       try:
           self.detection_queue.put_nowait(detection_result)
       except queue.Full:
           # Handle backpressure
           with self.stats_lock:
               self.detection_queue_drops += 1
   else:
       # Legacy: Process monitor inline
       ready_events = self.monitor.update(...)
   ```

2. **Start Monitor Thread in `run()`**:
   ```python
   if self.detection_queue_enabled:
       self._monitor_running = True
       self._monitor_thread = threading.Thread(
           target=self._monitor_thread_loop,
           daemon=True,
           name="MonitorThread"
       )
       self._monitor_thread.start()
       logger.info("[BagCounterApp] Monitor thread started (V4 Phase 1)")
   ```

3. **Shutdown in `shutdown_procedure()`**:
   ```python
   if self.detection_queue_enabled and self._monitor_thread:
       self._monitor_running = False
       if self._monitor_thread.is_alive():
           self._monitor_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
   ```

### Next (Complete Phase 2)
1. **Frame Accumulation**:
   ```python
   if self.batch_inference_enabled:
       self._frame_batch.append(frame)
       if self._batch_start_time is None:
           self._batch_start_time = time.perf_counter()
       
       # Check if batch is full or timeout exceeded
       batch_ready = (
           len(self._frame_batch) >= tracking_config.detection_batch_size or
           (time.perf_counter() - self._batch_start_time) * 1000 >= tracking_config.detection_batch_timeout_ms
       )
       
       if batch_ready:
           # Process batch
           detections_batch = self.detector.predict_batch(self._frame_batch)
           # ... process results ...
           self._frame_batch = []
           self._batch_start_time = None
   ```

---

## Testing Plan

### Unit Tests
- [ ] `test_detection_queue.py`: Queue operations, backpressure, drops
- [ ] `test_batch_inference.py`: Batch preprocessing, inference, postprocessing
- [ ] `test_monitor_thread.py`: Thread lifecycle, queue consumption

### Integration Tests
- [ ] `test_detection_queue_integration.py`: End-to-end with detection queue
- [ ] `test_batch_inference_integration.py`: Batch inference in full pipeline
- [ ] `test_performance_regression.py`: Ensure no accuracy degradation

### Performance Benchmarks
- [ ] Single-frame baseline: Target ~35ms detection
- [ ] Batch inference (batch=2): Target ~18-22ms per frame
- [ ] Detection queue parallel: Target ~35ms total (vs 60ms serial)
- [ ] End-to-end: Target <25ms P95 latency

---

## Documentation Updates Required

- [ ] `docs/AUDIT_REPORT.md`: Add V4 performance optimization section
- [ ] `docs/BATCH_INFERENCE_GUIDE.md`: Create new guide with tuning instructions
- [ ] `README.md`: Update performance section with new benchmarks
- [ ] `docs/CONFIGURATION.md`: Document all new parameters
- [ ] `CHANGES_SUMMARY.md`: Add V4 optimization details

---

## Known Issues & Limitations

### Phase 2: BPU Batch Inference
- **hobot_dnn Native Batch API**: Current implementation processes frames individually in a loop. True batch processing requires hobot_dnn's native batch API which may differ from current usage. Consult RDK X5 documentation:
  - https://developer.d-robotics.cc/rdk_doc/en/rdk_s/Algorithm_Application/python-api/
  - https://github.com/D-Robotics/hobot_dnn

- **Fallback Behavior**: If batch API is not available or fails, gracefully falls back to single-frame processing

### Phase 1: Detection Queue
- **Memory Overhead**: Detection queue stores frame copies. With queue size of 10 at 720p, this is ~40MB additional RAM. Acceptable for RDK X5 but monitor on low-memory systems.

---

## Success Metrics

### Phase 1 Success Criteria
- ✅ Detection and monitor run in separate threads
- ✅ Detection time not impacted by monitor time
- ✅ Detection queue utilization < 80% under normal load
- ✅ No detection accuracy degradation

### Phase 2 Success Criteria
- ✅ Batch inference achieves 40-60% speedup vs single-frame
- ✅ Average time per frame < 22ms with batch=2
- ✅ No detection accuracy degradation
- ✅ Graceful handling of batch size variations

### Overall V4 Success Criteria
- ✅ Frame processing time: **<25ms (P95)**
- ✅ Sustained FPS: **≥25 FPS** for 1-hour test
- ✅ Detection accuracy: **No degradation** vs baseline
- ✅ Event completion rate: **≥95%**
- ✅ Memory usage: **<20% increase** vs baseline
- ✅ Graceful degradation: No crashes under 2x target load

---

**Last Updated**: 2025-12-27
**Status**: 🟡 In Progress (70% Phase 1, 60% Phase 2)
