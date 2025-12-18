# Structured Logging Schema - BreadBag Counter System

## Overview

The BreadBag Counter System uses structured JSON logging for machine-parseable observability. All logs are written to `data/logs/app.json.log` with automatic rotation.

## Log Format

Each log entry is a single-line JSON object with the following structure:

```json
{
  "timestamp": "2025-12-18T04:43:39.531862Z",
  "level": "INFO",
  "logger": "BreadCounter",
  "message": "[EVENT_CREATED] id=12345, conf=0.870, frame=1523",
  "module": "BagStateMonitor",
  "function": "process",
  "line": 123,
  "component": "BagStateMonitor",
  "data": {
    "event_id": 12345,
    "confidence": 0.87,
    "box": [100, 200, 300, 400],
    "frame_index": 1523
  }
}
```

## Standard Fields

All log entries include these fields:

- **timestamp**: ISO8601 UTC timestamp with 'Z' suffix
- **level**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **logger**: Logger name (usually "BreadCounter")
- **message**: Human-readable message (often includes structured tags like `[EVENT_CREATED]`)
- **module**: Python module name
- **function**: Function name
- **line**: Line number
- **component**: Logical component name (e.g., "BagStateMonitor", "ClassifierService")
- **data**: Dictionary with structured context data (optional)

## Event Types and Schemas

### 1. Event Lifecycle Events

#### EVENT_CREATED
Logged when a new tracking event is created for a detected bag.

**Message Pattern**: `[EVENT_CREATED] id={event_id}, conf={confidence:.3f}, frame={frame_index}`

**Data Fields**:
```json
{
  "event_id": 12345,
  "confidence": 0.87,
  "box": [100, 200, 300, 400],
  "frame_index": 1523,
  "state": "detecting_open"
}
```

#### EVENT_COMMITTED
Logged when an event is finalized and ready for counting.

**Message Pattern**: `[EVENT_COMMITTED] id={event_id}, lifespan={lifespan_ms:.0f}ms, open_ev={open_evidence}, closed_ev={closed_evidence}, rois={roi_count}, reason={commit_reason}`

**Data Fields**:
```json
{
  "event_id": 12345,
  "lifespan_ms": 1234.5,
  "state": "CLOSED",
  "open_evidence": 5,
  "closed_evidence": 3,
  "roi_count": 8,
  "commit_reason": "ghost_timeout",
  "detection_gaps": [123, 456]
}
```

#### EVENT_EXPIRED
Logged when an event is terminated without counting.

**Message Pattern**: `[EVENT_EXPIRED] id={event_id}, state={state}, frames={frames_tracked}, open_hits={open_hits}, closed_hits={closed_hits}, idle={frames_since_update}`

**Data Fields**:
```json
{
  "event_id": 12346,
  "state": "detecting_closed",
  "frames_tracked": 25,
  "open_hits": 4,
  "closed_hits": 2,
  "frames_since_update": 15,
  "avg_motion": 3.2,
  "avg_confidence": 0.62
}
```

#### EVENT_SUPPRESSED
Logged when a new event creation is suppressed (anti-double-counting).

**Message Pattern**: `[EVENT_SUPPRESSED] new_id={event_id}, reason={reason}, iou={iou:.2f}, conflict_with={conflicting_event_id}`

**Data Fields**:
```json
{
  "event_id": -1,
  "reason": "duplicate_spatial_overlap",
  "iou": 0.82,
  "conflicting_event_id": 12345,
  "center_distance": 15.3,
  "frame_index": 1545,
  "detection_confidence": 0.73
}
```

#### STATE_TRANSITION
Logged when an event transitions between states (OPEN → CLOSING → CLOSED → COMMITTED).

**Message Pattern**: `[STATE_TRANSITION] id={event_id}, {old_state} -> {new_state}, trigger={trigger}`

**Data Fields**:
```json
{
  "event_id": 12345,
  "old_state": "detecting_open",
  "new_state": "detecting_closed",
  "trigger": "min_open_frames_reached",
  "open_hits": 5,
  "closed_hits": 1,
  "iou": 0.78,
  "frame_index": 1528
}
```

### 2. ROI (Region of Interest) Events

#### ROI_ADDED
Logged when a new ROI sample is collected for classification.

**Message Pattern**: `[ROI_ADDED] event={event_id}, type={roi_type}, sharpness={sharpness:.1f}, frame={frame_index}, conf={confidence:.2f}, total={total_rois}`

**Data Fields**:
```json
{
  "event_id": 12345,
  "roi_type": "OPEN",
  "is_open": true,
  "sharpness": 1850.3,
  "frame_index": 1523,
  "confidence": 0.87,
  "total_rois": 3,
  "bbox_area": 40000
}
```

#### ROI_REJECTED
Logged when an ROI sample is rejected due to quality issues.

**Message Pattern**: `[ROI_REJECTED] event={event_id}, reason={reason}, sharpness={sharpness:.1f}, dims={dimensions}, brightness={brightness:.1f}`

**Data Fields**:
```json
{
  "event_id": 12345,
  "reason": "too_blurry",
  "sharpness": 123.4,
  "dimensions": [160, 175],
  "brightness": 45.2
}
```

### 3. Classification Events

#### CLASSIFICATION
Logged when a bag is classified by type.

**Message Pattern**: `[CLASSIFICATION] track={track_id}, label={label}, conf={confidence:.3f}, ratio={winner_ratio:.2f}`

**Data Fields**:
```json
{
  "track_id": 12345,
  "label": "Whole_Wheat",
  "confidence": 0.92,
  "candidates": 5,
  "used_voting": true,
  "rejection_reason": null,
  "evidence_scores": {
    "Whole_Wheat": {"score": 3.854, "count": 5, "best_confidence": 0.92},
    "White": {"score": 0.432, "count": 2, "best_confidence": 0.48}
  },
  "winner_ratio": 8.92,
  "processing_time_ms": 145.3
}
```

#### CANDIDATE
Logged for each ROI candidate evaluated during classification.

**Message Pattern**: `[CANDIDATE] track={track_id}, idx={candidate_idx}, label={label}, conf={confidence:.3f}, sharpness={sharpness:.1f}, time={relative_time:.2f}, contrib={contribution:.3f}`

**Data Fields**:
```json
{
  "track_id": 12345,
  "candidate_idx": 0,
  "label": "Whole_Wheat",
  "confidence": 0.92,
  "sharpness": 1850.3,
  "relative_time": 0.85,
  "contribution": 0.784,
  "frame_index": 1540
}
```

### 4. Counting Events

#### COUNT_UPDATE
Logged when a bag is successfully counted.

**Message Pattern**: `[COUNT_UPDATE] type={bag_type}, count={new_count}, track={track_id}, conf={confidence:.3f}`

**Data Fields**:
```json
{
  "bag_type": "Whole_Wheat",
  "new_count": 42,
  "track_id": 12345,
  "confidence": 0.92,
  "phash": "a8f7e3c2d1b9",
  "candidates_evaluated": 5
}
```

### 5. Frame Processing Events

#### FRAME
Logged for each processed frame with performance metrics.

**Message Pattern**: `[FRAME] id={frame_id}, detect={detection_time_ms:.1f}ms, monitor={monitor_time_ms:.1f}ms, total={total_time_ms:.1f}ms, dets={detections_count}, ready={events_ready}`

**Data Fields**:
```json
{
  "frame_id": 1540,
  "detection_time_ms": 32.5,
  "monitor_time_ms": 5.3,
  "total_time_ms": 42.8,
  "detections_count": 6,
  "events_ready": 1,
  "queue_sizes": {"input": 8, "classification": 2},
  "fps": 23.4
}
```

### 6. Backpressure Events

#### BACKPRESSURE
Logged when the system is overloaded and dropping frames.

**Message Pattern**: `[BACKPRESSURE] queue={queue_name}, util={utilization:.1%}, drops={drops}, action={action}`

**Data Fields**:
```json
{
  "queue_name": "input_queue",
  "utilization": 0.85,
  "drops": 23,
  "action": "skip_frame",
  "avg_detection_time_ms": 42.3,
  "frames_skipped": 50
}
```

### 7. Error Events

#### ERROR
Logged for pipeline errors with full context.

**Message Pattern**: `[ERROR] component={component}, op={operation}, type={error_type}, msg={error_message}, affected={affected_ids}`

**Data Fields**:
```json
{
  "operation": "frame_processing",
  "error_type": "ValueError",
  "error_message": "Invalid detection box coordinates",
  "affected_ids": [1567],
  "upstream_context": {
    "detections_count": 5,
    "active_events": 3
  },
  "traceback": "Traceback (most recent call last)..."
}
```

## Frame-Based Thresholds

The system uses **frame-based thresholds** for consistent behavior across different processing speeds:

| Threshold | Default (frames) | Time @ 25 FPS |
|-----------|------------------|---------------|
| `ghost_timeout_frames` | 25 | 1000ms (1 second) |
| `temporal_cooldown_frames` | 10 | 400ms |
| `suppression_duration_frames` | 38 | 1520ms (1.5 seconds) |
| `open_to_closing_frames` | 3 | 120ms |
| `closing_stability_frames` | 4 | 160ms |
| `closed_stability_frames` | 5 | 200ms |

## Using the Logs

### Log Analysis Tool

Run the log analyzer to generate HTML and JSON reports:

```bash
python tools/log_analyzer.py --log-dir data/logs --day 2025-12-18
```

### Querying Logs

Extract specific event types:

```bash
# Get all event creations
cat data/logs/app.json.log | grep 'EVENT_CREATED' | jq '.data'

# Get all classification results
cat data/logs/app.json.log | grep 'CLASSIFICATION' | jq 'select(.data.label != "Unknown")'

# Get frame processing times
cat data/logs/app.json.log | grep 'FRAME' | jq '.data | {frame_id, total_time_ms, fps}'

# Get all errors
cat data/logs/app.json.log | jq 'select(.level == "ERROR")'
```

### Common Queries

**Find events with long lifetimes:**
```bash
cat data/logs/app.json.log | grep 'EVENT_COMMITTED' | jq 'select(.data.lifespan_ms > 2000)'
```

**Check ROI rejection rate:**
```bash
cat data/logs/app.json.log | grep 'ROI_' | jq -s 'group_by(.message | contains("REJECTED")) | map({rejected: .[0].message, count: length})'
```

**Analyze suppression reasons:**
```bash
cat data/logs/app.json.log | grep 'EVENT_SUPPRESSED' | jq '.data.reason' | sort | uniq -c
```

## Best Practices

1. **Enable JSON logging**: Set `ENABLE_JSON_LOGGING=true` in environment
2. **Rotate logs regularly**: Logs auto-rotate at 50MB (configurable via `LOG_FILE_MAX_BYTES`)
3. **Monitor disk space**: Keep at least 25 backup files (configurable via `LOG_FILE_BACKUP_COUNT`)
4. **Use log analyzer daily**: Generate reports to track trends and detect issues early
5. **Set appropriate log level**: Use `LOG_LEVEL=INFO` for production, `DEBUG` for troubleshooting

## Log Configuration

Environment variables for logging:

- `LOG_DIR`: Directory for log files (default: `data/logs`)
- `LOG_LEVEL`: Minimum log level (default: `DEBUG`)
- `ENABLE_JSON_LOGGING`: Enable structured JSON logs (default: `true`)
- `LOG_FILE_MAX_BYTES`: Max size before rotation (default: 50MB)
- `LOG_FILE_BACKUP_COUNT`: Number of backup files to keep (default: 25)
