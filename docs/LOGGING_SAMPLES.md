# Structured Logging Samples - BreadBag Counter System

This document provides representative log samples demonstrating the enhanced production-level structured logging implemented in the BreadBag Counter System. All logs are emitted in both human-readable format (console/file) and structured JSON format for automated analysis.

## Table of Contents

1. [INFO Level Logs - Normal Operation](#info-level-logs---normal-operation)
2. [WARNING Level Logs - Anomalies and Issues](#warning-level-logs---anomalies-and-issues)
3. [ERROR Level Logs - Critical Failures](#error-level-logs---critical-failures)
4. [Log Field Reference](#log-field-reference)
5. [Using Logs for Debugging](#using-logs-for-debugging)

---

## INFO Level Logs - Normal Operation

These logs track the normal flow of events through the pipeline, providing visibility into every step where values are created, changed, or merged.

### 1. Event Creation

**Human-Readable Format:**
```
13:43:55.170 | INFO | [EVENT_CREATED] id=12345, conf=0.870, frame=1523
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170314Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[EVENT_CREATED] id=12345, conf=0.870, frame=1523",
  "component": "BagStateMonitor",
  "data": {
    "event_id": 12345,
    "confidence": 0.87,
    "box": [100, 200, 300, 400],
    "frame_index": 1523,
    "state": "detecting_open"
  }
}
```

**Context:** A new bag event is created when an "open" detection appears that doesn't match any existing events. This log captures the initial detection confidence, bounding box coordinates, and frame index.

**Debugging Use:** If bags are being under-counted, check if events are being created for all visible bags. Low confidence values may indicate detection model issues.

---

### 2. ROI Addition

**Human-Readable Format:**
```
13:43:55.170 | DEBUG | [ROI_ADDED] event=12345, type=OPEN, sharpness=1850.3, frame=1523, conf=0.87, total=3
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170450Z",
  "level": "DEBUG",
  "logger": "BreadCounter",
  "message": "[ROI_ADDED] event=12345, type=OPEN, sharpness=1850.3, frame=1523, conf=0.87, total=3",
  "component": "BagEvent",
  "data": {
    "event_id": 12345,
    "roi_type": "OPEN",
    "is_open": true,
    "sharpness": 1850.3,
    "frame_index": 1523,
    "confidence": 0.87,
    "total_rois": 3,
    "bbox_area": 40000
  }
}
```

**Context:** Each time a bag is detected, an ROI (Region of Interest) is extracted and added to the event's collection. This log tracks quality metrics like sharpness (Laplacian variance) which determines which frames are used for classification.

**Debugging Use:** If classifications are poor quality, check sharpness values. Low sharpness (<400) indicates blurry frames. If total_rois stays low, bags may be moving too fast or detection is spotty.

---

### 3. State Transition

**Human-Readable Format:**
```
13:43:55.170 | INFO | [STATE_TRANSITION] id=12345, detecting_open -> detecting_closed, trigger=min_open_frames_reached
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170610Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[STATE_TRANSITION] id=12345, detecting_open -> detecting_closed, trigger=min_open_frames_reached",
  "component": "BagStateMonitor",
  "data": {
    "event_id": 12345,
    "old_state": "detecting_open",
    "new_state": "detecting_closed",
    "trigger": "min_open_frames_reached",
    "open_hits": 5,
    "closed_hits": 1,
    "iou": 0.78,
    "frame_index": 1528
  }
}
```

**Context:** Events progress through states: `detecting_open` → `detecting_closed` → `counted`. This log captures all state changes with the trigger condition and current hit counts.

**Debugging Use:** If bags aren't reaching the "counted" state, trace the state transitions. Check if open_hits/closed_hits are reaching required thresholds (default: 5 open, 3 closed).

---

### 4. Classification Candidate

**Human-Readable Format:**
```
13:43:55.170 | DEBUG | [CANDIDATE] track=12345, idx=0, label=Whole_Wheat, conf=0.920, sharpness=1850.3, time=0.85, contrib=0.784
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170737Z",
  "level": "DEBUG",
  "logger": "BreadCounter",
  "message": "[CANDIDATE] track=12345, idx=0, label=Whole_Wheat, conf=0.920, sharpness=1850.3, time=0.85, contrib=0.784",
  "component": "ClassifierService",
  "data": {
    "track_id": 12345,
    "candidate_idx": 0,
    "label": "Whole_Wheat",
    "confidence": 0.92,
    "sharpness": 1850.3,
    "relative_time": 0.85,
    "contribution": 0.784,
    "frame_index": 1540
  }
}
```

**Context:** During classification, each candidate ROI is classified individually. The contribution score combines confidence, sharpness, and temporal position (later frames weighted higher).

**Debugging Use:** If final classifications are wrong, examine individual candidate results. Look for disagreement between candidates or consistently low confidence values.

---

### 5. Classification Result

**Human-Readable Format:**
```
13:43:55.170 | INFO | [CLASSIFICATION] track=12345, label=Whole_Wheat, conf=0.920, ratio=8.92
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170866Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[CLASSIFICATION] track=12345, label=Whole_Wheat, conf=0.920, ratio=8.92",
  "component": "ClassifierService",
  "data": {
    "track_id": 12345,
    "label": "Whole_Wheat",
    "confidence": 0.92,
    "candidates": 5,
    "used_voting": true,
    "rejection_reason": null,
    "evidence_scores": {
      "Whole_Wheat": {
        "score": 3.854,
        "count": 5,
        "best_confidence": 0.92
      },
      "White": {
        "score": 0.432,
        "count": 2,
        "best_confidence": 0.48
      }
    },
    "winner_ratio": 8.92,
    "processing_time_ms": 145.3
  }
}
```

**Context:** Final classification decision using evidence accumulation. The winner_ratio (winner_score / runner_up_score) indicates confidence in the decision. Higher ratio = more confident.

**Debugging Use:** For misclassifications, check:
- Evidence scores for competing labels
- Winner ratio (low ratio = ambiguous decision)
- Number of candidates (fewer candidates = less reliable)
- Processing time (slow = potential performance issue)

---

### 6. Count Update

**Human-Readable Format:**
```
13:43:55.170 | INFO | [COUNT_UPDATE] type=Whole_Wheat, count=42, track=12345, conf=0.920
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.170988Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[COUNT_UPDATE] type=Whole_Wheat, count=42, track=12345, conf=0.920",
  "component": "BagCounterApp",
  "data": {
    "bag_type": "Whole_Wheat",
    "new_count": 42,
    "track_id": 12345,
    "confidence": 0.92,
    "phash": "a8f7e3c2d1b9",
    "candidates_evaluated": 5
  }
}
```

**Context:** The final step - a bag has been classified and the count is incremented. This is the money shot for detecting under/over counting.

**Debugging Use:** 
- **Under-counting:** Missing COUNT_UPDATE logs indicate events not completing the pipeline
- **Over-counting:** Duplicate COUNT_UPDATE logs for same bag (check phash and track_id)
- Compare count increments to expected production rate

---

### 7. Frame Processing

**Human-Readable Format:**
```
13:43:55.171 | DEBUG | [FRAME] id=1540, detect=32.5ms, monitor=5.3ms, total=42.8ms, dets=6, ready=1
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171109Z",
  "level": "DEBUG",
  "logger": "BreadCounter",
  "message": "[FRAME] id=1540, detect=32.5ms, monitor=5.3ms, total=42.8ms, dets=6, ready=1",
  "component": "BagCounterApp",
  "data": {
    "frame_id": 1540,
    "detection_time_ms": 32.5,
    "monitor_time_ms": 5.3,
    "total_time_ms": 42.8,
    "detections_count": 6,
    "events_ready": 1,
    "queue_sizes": {
      "input": 8,
      "classification": 2
    },
    "fps": 23.4
  }
}
```

**Context:** Performance metrics for each frame processed. Target is 25 FPS (40ms per frame).

**Debugging Use:** 
- Detection time > 35ms indicates detection bottleneck
- Queue sizes growing indicate backpressure
- Low FPS indicates system can't keep up with input stream

---

## WARNING Level Logs - Anomalies and Issues

These logs indicate the system is functioning but experiencing issues that may lead to accuracy problems.

### 1. Event Expiration

**Human-Readable Format:**
```
13:43:55.171 | WARNING | [EVENT_EXPIRED] id=12346, state=detecting_closed, frames=25, open_hits=4, closed_hits=2, idle=15
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171233Z",
  "level": "WARNING",
  "logger": "BreadCounter",
  "message": "[EVENT_EXPIRED] id=12346, state=detecting_closed, frames=25, open_hits=4, closed_hits=2, idle=15",
  "component": "BagStateMonitor",
  "data": {
    "event_id": 12346,
    "state": "detecting_closed",
    "frames_tracked": 25,
    "open_hits": 4,
    "closed_hits": 2,
    "frames_since_update": 15,
    "avg_motion": 3.2,
    "avg_confidence": 0.62
  }
}
```

**Context:** An event was tracked but expired before reaching the "counted" state. This is the #1 cause of under-counting.

**Debugging Use:** 
- **open_hits < 5:** Bag didn't appear open long enough (too fast, occlusion)
- **closed_hits < 3:** Bag didn't appear closed long enough (conveyor issues, bags sticking)
- **High frames_since_update:** Bag left field of view
- **Low avg_confidence:** Poor detection quality

**Action Items:**
- If many events expire in `detecting_open`: Lower min_open_frames threshold or improve open detection
- If many events expire in `detecting_closed`: Lower min_closed_frames threshold or improve closed detection
- If avg_motion is high: Bags may be moving too fast for the pipeline

---

### 2. Classification Unknown - Low Evidence

**Human-Readable Format:**
```
13:43:55.171 | WARNING | [CLASSIFICATION] track=12347, label=Unknown, conf=0.450, reason=low_evidence (1.234 < 2.0), ratio=1.25
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171358Z",
  "level": "WARNING",
  "logger": "BreadCounter",
  "message": "[CLASSIFICATION] track=12347, label=Unknown, conf=0.450, reason=low_evidence (1.234 < 2.0), ratio=1.25",
  "component": "ClassifierService",
  "data": {
    "track_id": 12347,
    "label": "Unknown",
    "confidence": 0.45,
    "candidates": 3,
    "used_voting": true,
    "rejection_reason": "low_evidence (1.234 < 2.0)",
    "evidence_scores": {
      "Whole_Wheat": {
        "score": 1.234,
        "count": 2,
        "best_confidence": 0.65
      },
      "White": {
        "score": 0.987,
        "count": 1,
        "best_confidence": 0.55
      }
    },
    "winner_ratio": 1.25,
    "processing_time_ms": 98.2
  }
}
```

**Context:** Classification resulted in "Unknown" because total evidence score was below threshold (default: 2.0).

**Debugging Use:**
- Few candidates (< 3): Track was too short or ROI quality poor
- Low best_confidence: Classifier model uncertain
- Check if this happens for specific bag types (model training issue)

**Action Items:**
- Lower min_total_evidence threshold (but may reduce accuracy)
- Improve ROI quality (better lighting, sharper images)
- Retrain classifier model on production data

---

### 3. Classification Unknown - Ambiguous

**Human-Readable Format:**
```
13:43:55.171 | WARNING | [CLASSIFICATION] track=12348, label=Unknown, conf=0.780, reason=ambiguous (1.35 < 1.8), ratio=1.35
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171509Z",
  "level": "WARNING",
  "logger": "BreadCounter",
  "message": "[CLASSIFICATION] track=12348, label=Unknown, conf=0.780, reason=ambiguous (1.35 < 1.8), ratio=1.35",
  "component": "ClassifierService",
  "data": {
    "track_id": 12348,
    "label": "Unknown",
    "confidence": 0.78,
    "candidates": 6,
    "used_voting": true,
    "rejection_reason": "ambiguous (1.35 < 1.8)",
    "evidence_scores": {
      "Whole_Wheat": {
        "score": 2.456,
        "count": 3,
        "best_confidence": 0.78
      },
      "Bran": {
        "score": 1.819,
        "count": 3,
        "best_confidence": 0.72
      }
    },
    "winner_ratio": 1.35,
    "processing_time_ms": 156.7
  }
}
```

**Context:** Classification resulted in "Unknown" because winner/runner-up ratio was too low (default: 1.8). Two labels were too close in score.

**Debugging Use:**
- Check which labels are frequently confused (e.g., Whole_Wheat vs Bran)
- Similar scores indicate visually similar bags or model confusion

**Action Items:**
- If specific label pairs are always confused: Retrain model with more distinctive features
- If happens across many label pairs: Lighting or image quality issue
- Consider lowering ratio_threshold (but may increase misclassifications)

---

### 4. Queue Backpressure

**Human-Readable Format:**
```
13:43:55.171 | WARNING | [BACKPRESSURE] queue=input_queue, util=85.0%, drops=23, action=skip_frame
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171640Z",
  "level": "WARNING",
  "logger": "BreadCounter",
  "message": "[BACKPRESSURE] queue=input_queue, util=85.0%, drops=23, action=skip_frame",
  "component": "BagCounterApp",
  "data": {
    "queue_name": "input_queue",
    "utilization": 0.85,
    "drops": 23,
    "action": "skip_frame",
    "avg_detection_time_ms": 42.3,
    "frames_skipped": 50
  }
}
```

**Context:** The system can't keep up with the input frame rate. Frames are being dropped or skipped to prevent queue overflow.

**Debugging Use:**
- High drops: System is falling behind (CPU/GPU overload)
- High avg_detection_time_ms: Detection model too slow
- Check if this correlates with counting errors (dropped frames = missed bags)

**Action Items:**
- Reduce input frame rate
- Optimize detection model (lighter model, lower resolution)
- Upgrade hardware (faster GPU)
- Tune adaptive skip threshold

---

### 5. Event Suppression

**Human-Readable Format:**
```
13:43:55.171 | INFO | [EVENT_SUPPRESSED] new_id=-1, reason=duplicate_spatial_overlap, iou=0.82, conflict_with=12345
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171764Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[EVENT_SUPPRESSED] new_id=-1, reason=duplicate_spatial_overlap, iou=0.82, conflict_with=12345",
  "component": "BagStateMonitor",
  "data": {
    "event_id": -1,
    "reason": "duplicate_spatial_overlap",
    "iou": 0.82,
    "conflicting_event_id": 12345,
    "center_distance": 15.3,
    "frame_index": 1545,
    "detection_confidence": 0.73
  }
}
```

**Context:** A new detection was suppressed because it overlaps with a recently counted event. This prevents over-counting the same bag.

**Debugging Use:**
- **Over-counting:** Too few suppressions may indicate lockout window too short
- **Under-counting:** Too many suppressions may indicate lockout window too long
- Check IoU and center_distance values to understand spatial overlap

---

## ERROR Level Logs - Critical Failures

These logs indicate critical errors that prevent normal operation.

### 1. Frame Processing Error

**Human-Readable Format:**
```
13:43:55.171 | ERROR | [ERROR] component=LogicThread, op=frame_processing, type=ValueError, msg=Invalid detection box coordinates: [100, 200, 90, 400], affected=[1567]
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.171888Z",
  "level": "ERROR",
  "logger": "BreadCounter",
  "message": "[ERROR] component=LogicThread, op=frame_processing, type=ValueError, msg=Invalid detection box coordinates: [100, 200, 90, 400], affected=[1567]",
  "component": "LogicThread",
  "data": {
    "operation": "frame_processing",
    "error_type": "ValueError",
    "error_message": "Invalid detection box coordinates: [100, 200, 90, 400]",
    "affected_ids": [1567],
    "upstream_context": {
      "detections_count": 5,
      "active_events": 3,
      "input_queue_size": 15,
      "classification_queue_size": 4
    },
    "traceback": "Traceback (most recent call last):\n  File \"BagCounterApp.py\", line 485, in _logic_thread_loop\n    ..."
  }
}
```

**Context:** A critical error occurred during frame processing. The affected_ids field shows which frame(s) were impacted. The upstream_context provides the system state at the time of error.

**Debugging Use:**
- Error message shows the specific problem (invalid coordinates: x2 < x1)
- Upstream context shows system was processing normally (5 detections, 3 active events)
- Frame 1567 is lost, but system should recover
- If this error repeats frequently, indicates detection model producing invalid outputs

---

### 2. Classification Inference Error

**Human-Readable Format:**
```
13:43:55.172 | ERROR | [ERROR] component=ClassifierService, op=track_classification, type=RuntimeError, msg=Classifier model inference failed: CUDA out of memory, affected=[12349]
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.172009Z",
  "level": "ERROR",
  "logger": "BreadCounter",
  "message": "[ERROR] component=ClassifierService, op=track_classification, type=RuntimeError, msg=Classifier model inference failed: CUDA out of memory, affected=[12349]",
  "component": "ClassifierService",
  "data": {
    "operation": "track_classification",
    "error_type": "RuntimeError",
    "error_message": "Classifier model inference failed: CUDA out of memory",
    "affected_ids": [12349],
    "upstream_context": {
      "candidates_count": 7,
      "event_stats": {
        "total_frames_tracked": 35,
        "track_duration_frames": 28,
        "avg_sharpness": 1654.2
      }
    },
    "traceback": "Traceback (most recent call last):\n  File \"ClassifierService.py\", line 398, in process\n    ..."
  }
}
```

**Context:** Classification failed for track 12349 due to GPU memory exhaustion. This bag will likely be counted as "Unknown" or not counted at all.

**Debugging Use:**
- CUDA out of memory: GPU memory exhausted (too many concurrent classifications or large batch)
- Track 12349 is affected - this bag's count may be lost
- Event had good stats (35 frames, 7 candidates, decent sharpness)
- If repeated: Need to reduce batch size, use smaller model, or upgrade GPU

---

### 3. Memory Error

**Human-Readable Format:**
```
13:43:55.172 | ERROR | [ERROR] component=ClassificationThread, op=classification_processing, type=MemoryError, msg=Unable to allocate memory for ROI processing, affected=[12350, 12351, 12352]
```

**JSON Format:**
```json
{
  "timestamp": "2025-12-14T13:43:55.172128Z",
  "level": "ERROR",
  "logger": "BreadCounter",
  "message": "[ERROR] component=ClassificationThread, op=classification_processing, type=MemoryError, msg=Unable to allocate memory for ROI processing, affected=[12350, 12351, 12352]",
  "component": "ClassificationThread",
  "data": {
    "operation": "classification_processing",
    "error_type": "MemoryError",
    "error_message": "Unable to allocate memory for ROI processing",
    "affected_ids": [12350, 12351, 12352],
    "upstream_context": {
      "candidates_count": 0,
      "classification_queue_size": 18
    },
    "traceback": "Traceback (most recent call last):\n  File \"BagCounterApp.py\", line 369, in _classification_thread_loop\n    ..."
  }
}
```

**Context:** System RAM exhausted during classification processing. Multiple tracks affected (12350, 12351, 12352).

**Debugging Use:**
- Multiple affected IDs: cascading failure affecting multiple bags
- Classification queue at 18 items: backlog building up before failure
- System likely needs restart to recover
- If repeated: Memory leak or insufficient RAM for workload

---

## Log Field Reference

### Common Fields (All Logs)

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO8601 | UTC timestamp of log event |
| `level` | string | Log level: DEBUG, INFO, WARNING, ERROR |
| `logger` | string | Logger name (usually "BreadCounter") |
| `message` | string | Human-readable log message |
| `component` | string | Component generating the log |
| `data` | object | Structured data specific to log type |

### Event Tracking Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | integer | Unique event identifier |
| `track_id` | integer | Unique track identifier (same as event_id) |
| `frame_index` | integer | Frame number in video stream |
| `confidence` | float | Detection or classification confidence (0-1) |
| `box` | [x1,y1,x2,y2] | Bounding box coordinates |
| `state` | string | Event state: detecting_open, detecting_closed, counted |
| `open_hits` | integer | Number of "open" detections |
| `closed_hits` | integer | Number of "closed" detections |
| `frames_tracked` | integer | Total frames this event was tracked |
| `frames_since_update` | integer | Frames since last detection match |

### Classification Fields

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Classified bag type (or "Unknown") |
| `candidates` | integer | Number of ROI candidates evaluated |
| `evidence_scores` | object | Evidence scores per label |
| `winner_ratio` | float | Winner score / runner-up score |
| `rejection_reason` | string | Why classification resulted in Unknown |
| `sharpness` | float | Laplacian variance (higher = sharper) |
| `relative_time` | float | Position in track (0=start, 1=end) |
| `contribution` | float | Candidate's contribution to evidence |

### Performance Fields

| Field | Type | Description |
|-------|------|-------------|
| `detection_time_ms` | float | Detection inference time |
| `monitor_time_ms` | float | Event monitoring time |
| `total_time_ms` | float | Total frame processing time |
| `processing_time_ms` | float | Classification processing time |
| `fps` | float | Current frames per second |
| `queue_sizes` | object | Current queue utilizations |

### Error Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `operation` | string | Operation that failed |
| `error_type` | string | Exception type (ValueError, RuntimeError, etc) |
| `error_message` | string | Error message |
| `affected_ids` | array | IDs of affected events/frames/tracks |
| `upstream_context` | object | System state at time of error |
| `traceback` | string | Full Python traceback |

---

## Using Logs for Debugging

### Debugging Under-Counting

**Symptom:** Actual count lower than expected

**Investigation Steps:**

1. **Check COUNT_UPDATE frequency:**
   ```bash
   grep "COUNT_UPDATE" app.json.log | jq .data.new_count
   ```
   Compare increment rate to expected production rate.

2. **Look for EVENT_EXPIRED warnings:**
   ```bash
   grep "EVENT_EXPIRED" app.json.log | jq .data
   ```
   - Many expired events = bags leaving before classification completes
   - Check `state`, `open_hits`, `closed_hits` to see where events are failing

3. **Check for event creation:**
   ```bash
   grep "EVENT_CREATED" app.json.log | wc -l
   ```
   If few events created, detection may be missing bags.

4. **Look for frame drops:**
   ```bash
   grep "BACKPRESSURE" app.json.log | jq .data.drops
   ```
   Dropped frames = missed bags.

5. **Check detection confidence:**
   ```bash
   grep "EVENT_CREATED" app.json.log | jq .data.confidence | sort -n
   ```
   Many low confidence values suggest detection model issues.

### Debugging Over-Counting

**Symptom:** Actual count higher than expected

**Investigation Steps:**

1. **Check for duplicate tracks:**
   ```bash
   grep "COUNT_UPDATE" app.json.log | jq .data.track_id | sort | uniq -d
   ```
   Duplicate track IDs = same bag counted twice.

2. **Check phash values:**
   ```bash
   grep "COUNT_UPDATE" app.json.log | jq '.data | {track: .track_id, phash: .phash, time: .timestamp}'
   ```
   Same phash appearing multiple times = same bag counted multiple times.

3. **Check EVENT_SUPPRESSED frequency:**
   ```bash
   grep "EVENT_SUPPRESSED" app.json.log | jq .data
   ```
   Too few suppressions = lockout window may be too short.

4. **Check IoU values at suppression:**
   ```bash
   grep "EVENT_SUPPRESSED" app.json.log | jq .data.iou | sort -n
   ```
   Low IoU values suggest suppression threshold may need adjustment.

### Debugging Poor Classification

**Symptom:** Wrong bag types being counted

**Investigation Steps:**

1. **Check Unknown rate:**
   ```bash
   grep "CLASSIFICATION.*Unknown" app.json.log | wc -l
   total=$(grep "CLASSIFICATION" app.json.log | wc -l)
   echo "Unknown rate: $(awk "BEGIN {print ($unknown_count / $total) * 100}")%"
   ```

2. **Analyze rejection reasons:**
   ```bash
   grep "CLASSIFICATION.*Unknown" app.json.log | jq .data.rejection_reason | sort | uniq -c
   ```
   
3. **Check evidence scores for misclassifications:**
   ```bash
   grep "CLASSIFICATION" app.json.log | jq '.data | {track: .track_id, label: .label, scores: .evidence_scores, ratio: .winner_ratio}'
   ```
   Look for low ratios (< 2.0) or disagreement in scores.

4. **Check candidate quality:**
   ```bash
   grep "CANDIDATE" app.json.log | jq '.data | {track: .track_id, label: .label, conf: .confidence, sharpness: .sharpness}'
   ```
   Low sharpness or confidence indicates poor input quality.

### Performance Troubleshooting

1. **Check frame processing time:**
   ```bash
   grep "FRAME" app.json.log | jq .data.total_time_ms | awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'
   ```
   Target: < 40ms (for 25 FPS)

2. **Identify bottleneck component:**
   ```bash
   grep "FRAME" app.json.log | jq '{det: .data.detection_time_ms, mon: .data.monitor_time_ms}' | head -20
   ```
   
3. **Check queue utilization:**
   ```bash
   grep "FRAME" app.json.log | jq .data.queue_sizes
   ```
   Queues consistently full = system overloaded

4. **Monitor classification throughput:**
   ```bash
   grep "CLASSIFICATION" app.json.log | jq .data.processing_time_ms | awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'
   ```

---

## Summary

The enhanced structured logging provides complete visibility into the BreadBag Counter pipeline:

1. **Every value creation/change is logged** with sufficient context to understand what happened
2. **Every decision point is logged** with the reasoning (state transitions, classifications, suppressions)
3. **Every error is logged** with affected IDs and system context for debugging
4. **JSON format** enables automated analysis and alerting
5. **Human-readable format** provides quick insights during development

Use these logs to:
- **Diagnose under/over-counting** by tracing event lifecycle
- **Identify performance bottlenecks** through timing metrics
- **Debug classification issues** through evidence scores
- **Monitor system health** through KPI metrics and warnings
- **Root cause failures** through error context and tracebacks
