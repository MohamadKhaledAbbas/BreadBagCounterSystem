# Event-Centric Tracking System Architecture

## Overview

This document describes the event-centric tracking system designed to achieve **≥99.9% counting reliability** (≤1 error per 1000 bags) for bread bag counting in human-operated table environments.

## Design Principles

### Event-Centric vs Object-Centric Tracking

Traditional trackers (SORT, ByteTrack, DeepSORT) track **objects** across frames using:
- Visual appearance features
- IoU (Intersection over Union) matching
- Re-identification networks

Our system tracks **Events** - the physical operation of a bread bag being tied:
- Survives detection loss
- Survives track fragmentation  
- Survives hand occlusion
- Uses only spatial/temporal association

### Why This Approach?

In human-operated table environments:
- Workers **rotate** bags 360°
- Workers **flip** bags upside down
- Workers **occlude** bags with hands while tying
- Bags **deform** significantly during manipulation

IoU-based tracking fails because:
- Box overlap drops to zero during rotation
- Appearance changes dramatically during flipping
- Detections disappear during occlusion

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DETECTION LAYER                             │
│                                                                   │
│   YOLO Model #1: bread-bag-opened / bread-bag-closed             │
│   (outputs confidence per frame - EVIDENCE, not final state)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EVENT-CENTRIC TRACKER                           │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   EVENTS    │◄───│ ASSOCIATION │◄───│ DETECTIONS  │          │
│  │             │    │             │    │             │          │
│  │ State:      │    │ Centroid    │    │ Centroid    │          │
│  │ OPEN        │    │ Distance    │    │ Class ID    │          │
│  │ CLOSING     │    │ Time Gap    │    │ Confidence  │          │
│  │ CLOSED      │    │ NO IoU!     │    │ Box         │          │
│  │ COMMITTED   │    │             │    │             │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                   │
│  Ghost Event Support: Keep event alive during detection gaps     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COUNTING RULE                                │
│                                                                   │
│   Count ONLY when:                                               │
│   1. Event state == CLOSED                                       │
│   2. No detections for exit_timeout_ms                          │
│   3. Last centroid near scene exit boundary                     │
│                                                                   │
│   ✗ NOT counted at moment of closure                            │
│   ✗ NOT counted if still in work zone                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION LAYER                           │
│                                                                   │
│   YOLO Model #2 + Temporal Voting                               │
│   - Classify ROIs collected during CLOSED state                 │
│   - Require X% vote agreement                                    │
│   - Require confidence margin between top classes               │
│   - Output UNKNOWN if ambiguous (precision over recall)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## State Machine Definition

```
     ┌───────────────────────────────────────┐
     │                                       │
     │         Event State Machine           │
     │                                       │
     └───────────────────────────────────────┘


    [Detection: Open Bag]
           │
           ▼
    ┌──────────┐
    │   OPEN   │◄────────────────────────────┐
    │          │                             │
    │ Collect  │                             │ (open evidence
    │ evidence │                             │  resumes)
    └────┬─────┘                             │
         │                                   │
         │ (min_open_evidence reached        │
         │  + closed detection seen          │
         │  + open_to_closing_time_ms)       │
         │                                   │
         ▼                                   │
    ┌──────────┐                             │
    │ CLOSING  │─────────────────────────────┘
    │          │
    │ Monitor  │
    │ stability│
    └────┬─────┘
         │
         │ (closing_stability_time_ms
         │  + min_closed_evidence
         │  + centroid_stability_px)
         │
         ▼
    ┌──────────┐
    │  CLOSED  │
    │          │
    │ Collect  │
    │ ROIs for │
    │ classify │
    └────┬─────┘
         │
         │ (ghost_timeout_ms expired
         │  + MUST be near exit boundary)
         │
         ▼
    ┌──────────┐
    │COMMITTED │ ──► COUNT + CLASSIFY
    │          │
    │ Final    │
    │ state    │
    └──────────┘
```

**CRITICAL COUNTING RULE:**
A bag is ONLY counted when its centroid is within `exit_boundary_margin_px` of the frame edge.
If the bag is in the center of the scene, it will NOT be counted even after timeout - it must
physically move to the exit boundary first.

### State Transition Requirements

| Transition | Requirements |
|------------|--------------|
| OPEN → CLOSING | `min_open_evidence_count` reached + closed detection + `open_to_closing_time_ms` in OPEN |
| CLOSING → OPEN | 2+ open detections in last 3 frames |
| CLOSING → CLOSED | `closing_stability_time_ms` + `min_closed_evidence_count` + centroid stable |
| CLOSED → COMMITTED | `ghost_timeout_ms` without detection + **MUST be near exit boundary** |

---

## Event Association Rules

### Centroid-Based Association (NO IoU)

For each detection, compute centroid and associate to active Event if:

```python
# D: Max distance in pixels
distance = sqrt((det_centroid_x - event_centroid_x)^2 + 
                (det_centroid_y - event_centroid_y)^2)

# T: Max time gap in milliseconds
time_gap = detection_timestamp_ms - event.last_detection_time_ms

if distance < D and time_gap < T:
    associate detection to event
```

### Ghost Event Handling

```python
# G: Ghost timeout in milliseconds
if event has no detection:
    time_since_detection = current_time_ms - event.last_detection_time_ms
    
    if time_since_detection < G:
        # Keep event alive (ghost state)
        # Detection may reappear (hand occlusion)
    else:
        # Ghost timeout exceeded
        if event.state == CLOSED:
            # Eligible for commit (count)
            check exit boundary
        else:
            # Event expires (not counted)
            expire event
```

---

## Tuning Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **D** `association_distance_px` | 100.0 | 50-200 | Max centroid distance for association |
| **T** `association_time_ms` | 400.0 | 200-800 | Max time gap for association |
| **G** `ghost_timeout_ms` | 1000.0 | 500-2000 | Keep event alive without detections |
| `exit_timeout_ms` | 800.0 | 500-1500 | Time in CLOSED before commit |
| `exit_boundary_margin_px` | 50 | 30-100 | Distance from edge for "near exit" |
| `open_to_closing_time_ms` | 100.0 | 50-300 | Min time in OPEN before CLOSING |
| `closing_stability_time_ms` | 150.0 | 100-400 | Closed detections must persist |
| `centroid_stability_px` | 30.0 | 10-50 | Max movement for "stable" |
| `min_open_evidence_count` | 3 | 2-10 | Min open detections before state change |
| `min_closed_evidence_count` | 2 | 1-5 | Min closed detections for CLOSED |

### V5.1 Additional Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `velocity_scaling_enabled` | true | - | Enable velocity-based distance scaling |
| `velocity_scale_factor` | 2.5 | 1.5-4.0 | Max multiplier for association distance |
| `max_association_distance_px` | 250.0 | 150-400 | Absolute max association distance |
| `allow_center_commit` | true | - | Allow counting bags that don't exit to edge |
| `center_commit_idle_frames` | 25 | 15-50 | Frames idle before center commit |
| `center_commit_min_closed_ratio` | 0.3 | 0.2-0.6 | Min closed/total ratio for center commit |
| `closing_revert_open_count` | 3 | 2-5 | Open detections needed to revert CLOSING→OPEN |
| `closing_revert_window_size` | 5 | 3-8 | Window size for revert check |

### Tuning Guidelines

**D (association_distance_px)**
- Based on expected bag movement per frame
- At 25fps with typical human speed: 100px works well
- Too low: Loses track during fast movement
- Too high: May merge nearby bags

**T (association_time_ms)**
- Should be longer than typical detection flicker
- Should be shorter than time between different bags
- 400ms handles momentary detection drops

**G (ghost_timeout_ms)**
- Should cover typical hand occlusion duration
- 1000ms (1 second) handles most tying scenarios
- Too short: Counts bag while worker still holding
- Too long: Delays counting unnecessarily

**Velocity Scaling (NEW in V5.1)**
- When enabled, association distance scales based on bag velocity
- Helps maintain tracking during bag flipping/throwing
- Uses predicted centroid position for association
- Set `max_association_distance_px` to prevent over-association

**Center Commit (NEW in V5.1)**
- Enable when bags don't exit to frame edge after closing
- Requires bag to be idle for `center_commit_idle_frames` frames
- Requires minimum closed evidence ratio to avoid false commits
- Useful for table-based operations where bags are placed down

**Anti-Oscillation (NEW in V5.1)**
- Prevents rapid OPEN↔CLOSING state changes during noisy detections
- Requires `closing_revert_open_count` open detections SINCE entering CLOSING
- Uses evidence window SINCE state entry, not global history

---

## Failure Modes and Handling

### Failure Mode 1: Dropped Events
**Symptom**: Event expires without reaching CLOSED
**Cause**: Insufficient closed detections or ghost timeout
**Detection**: `state_transitions` shows no CLOSED transition
**Mitigation**: Increase `ghost_timeout_ms` or lower `min_closed_evidence_count`

### Failure Mode 2: False Splits
**Symptom**: Single bag creates multiple events
**Cause**: Detection gap exceeds time threshold
**Detection**: Multiple events with overlapping spatial trajectory
**Mitigation**: Increase `association_time_ms` or `ghost_timeout_ms`

### Failure Mode 3: Premature Commits
**Symptom**: Count triggers while bag still in frame
**Cause**: Exit timeout too short or exit boundary too wide
**Detection**: `commit_reason` shows "exit_timeout" but bag visible
**Mitigation**: Increase `exit_timeout_ms` or reduce `exit_boundary_margin_px`

### Failure Mode 4: Merged Events
**Symptom**: Two bags counted as one
**Cause**: Association distance too large
**Detection**: Single event with abnormally long lifespan
**Mitigation**: Reduce `association_distance_px`

---

## Debugging & Metrics

### Per-Event Logging

Each event logs:
```json
{
    "event_id": 12345678,
    "lifespan_ms": 2500.0,
    "created_at_ms": 0.0,
    "last_detection_ms": 2400.0,
    "detection_gaps": [[100, 300], [800, 1000]],
    "state_transitions": [
        {"timestamp_ms": 0, "to_state": "OPEN", "trigger": "event_created"},
        {"timestamp_ms": 200, "to_state": "CLOSING", "trigger": "closed_evidence_detected"},
        {"timestamp_ms": 400, "to_state": "CLOSED", "trigger": "closing_stable"},
        {"timestamp_ms": 2500, "to_state": "COMMITTED", "trigger": "exit_boundary"}
    ],
    "roi_count": 5,
    "open_evidence_count": 8,
    "closed_evidence_count": 12,
    "commit_reason": "exit_boundary"
}
```

### Tracker Statistics

```json
{
    "events_created": 150,
    "events_committed": 147,
    "events_expired": 3,
    "events_suppressed": 5,
    "completion_rate": 0.98,
    "total_detections_processed": 12500
}
```

### Analysis Queries

**Find dropped events:**
```python
events_with_progress = [e for e in expired_events 
                        if e.open_evidence_count >= 3]
```

**Find potential splits:**
```python
# Events created within D pixels and T ms of each other
for i, e1 in enumerate(events):
    for e2 in events[i+1:]:
        if distance(e1.centroid, e2.centroid) < D:
            if abs(e1.created_at - e2.created_at) < T:
                flag_potential_split(e1, e2)
```

---

## Integration with Existing Pipeline

The `EventCentricStateMonitor` provides drop-in compatibility:

```python
# Before (legacy IoU-based)
from src.counting.BagStateMonitor import BagStateMonitor
monitor = BagStateMonitor(open_id, closed_id)

# After (event-centric)
from src.counting.EventCentricStateMonitor import EventCentricStateMonitor
monitor = EventCentricStateMonitor(open_id, closed_id)
```

Both use the same `update()` interface:
```python
ready_events = monitor.update(detections, {"frame_count": n, "frame": img})
```

The `use_event_centric_tracking` config flag controls which system is used.

---

## What This System Does NOT Use

Per requirements, this implementation explicitly **excludes**:

- ❌ Visual appearance embeddings
- ❌ DeepSORT / ByteTrack
- ❌ IoU-based matching
- ❌ Re-identification networks
- ❌ Frame-based counting logic
- ❌ Additional YOLO models

---

## Summary

This event-centric tracking system achieves robust counting in challenging human-operated environments by:

1. **Treating events, not objects** - An Event survives what destroys traditional tracks
2. **Using centroid distance, not IoU** - Rotation-invariant association
3. **Using milliseconds, not frames** - Deterministic timing
4. **Counting at exit, not closure** - Ensures bag has left scene
5. **Providing full explainability** - Every decision is logged for debugging
