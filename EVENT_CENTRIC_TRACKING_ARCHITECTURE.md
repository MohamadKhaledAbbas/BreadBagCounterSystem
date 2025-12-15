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
│   Count ONLY when (Timeout-Based Commitment):                    │
│   1. Event state == CLOSED                                       │
│   2. No detections for commit_idle_frames                        │
│   3. Minimum closed evidence ratio met                           │
│                                                                   │
│   ✗ NOT counted at moment of closure                            │
│   ✗ Exit boundary logic REMOVED for simplicity                  │
│                                                                   │
│   Anti-Double-Counting:                                          │
│   - Suppress new events near recently committed locations        │
│   - Configurable suppression distance and duration               │
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
         │  + commit_idle_frames reached
         │  + min_closed_ratio met)
         │
         ▼
    ┌──────────┐
    │COMMITTED │ ──► COUNT + CLASSIFY
    │          │
    │ Final    │
    │ state    │
    └──────────┘
```

**TIMEOUT-BASED COUNTING RULE:**
A bag is counted when it has been undetected for `commit_idle_frames` after entering CLOSED state,
with a minimum `commit_min_closed_ratio` of closed evidence. Exit boundary logic has been removed
for simplicity and robustness. Anti-double-counting is achieved through spatial suppression of
new events near recently committed locations.

### State Transition Requirements

| Transition | Requirements |
|------------|--------------|
| OPEN → CLOSING | `min_open_evidence_count` reached + closed detection + `open_to_closing_time_ms` in OPEN |
| CLOSING → OPEN | `closing_revert_open_count` open detections since entering CLOSING |
| CLOSING → CLOSED | `closing_stability_time_ms` + `min_closed_evidence_count` + centroid stable |
| CLOSED → COMMITTED | `ghost_timeout_ms` + `commit_idle_frames` + `commit_min_closed_ratio` met |

---

## Event Association Rules

### Parallel Hybrid Association (Centroid + IoU)

Association uses **parallel hybrid** evaluation where both metrics are ALWAYS computed:

1. **Centroid distance** - with velocity-based scaling for fast movements
2. **IoU (Intersection over Union)** - for robustness during flips/spins

**Key Design Choice:** Both metrics are computed for every association attempt, regardless of 
whether one criterion already passes. This provides:
- **Robustness during flips/spins**: Centroid may jump but IoU remains high
- **Robustness during fast slides**: IoU may drop but centroid stays close  
- **Detailed debugging**: All metrics logged for every association attempt

A detection associates if EITHER criterion is met:

```python
# Compute BOTH metrics in parallel
distance = sqrt((det_centroid_x - event_centroid_x)^2 + 
                (det_centroid_y - event_centroid_y)^2)
iou_value = compute_iou(event.last_box, detection.box)

# Check time gap (fails both if exceeded)
if time_gap > T:
    reject with reason "time_gap_exceeded" (still logs both metrics)
    return False

# Check both criteria
centroid_match = distance <= D (velocity-scaled)
iou_match = iou_value >= IoU_threshold

# Associate if EITHER matches
if centroid_match and iou_match:
    return True, "both_match"
elif centroid_match:
    return True, "centroid_match"  # Typical for fast slides
elif iou_match:
    return True, "iou_match"       # Typical for flips/spins
else:
    return False, "no_match"
```

### Association Result Types

| Result Type | Centroid | IoU | Typical Scenario |
|-------------|----------|-----|------------------|
| `both_match` | ✓ | ✓ | Normal small movement |
| `centroid_match` | ✓ | ✗ | Fast slide, box shape change |
| `iou_match` | ✗ | ✓ | Flip/spin, centroid jumps |
| `no_match` | ✗ | ✗ | Different detection, false match |
| `time_exceeded` | N/A | N/A | Detection too late |

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
            # Eligible for commit if idle timeout met
            if frames_without_detection >= commit_idle_frames:
                commit event (count)
        else:
            # Event expires (not counted)
            expire event
```

---

## Tuning Parameters

### Core Association Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **D** `association_distance_px` | 100.0 | 50-200 | Max centroid distance for association |
| **T** `association_time_ms` | 400.0 | 200-800 | Max time gap for association |
| **G** `ghost_timeout_ms` | 1000.0 | 500-2000 | Keep event alive without detections |
| `open_to_closing_time_ms` | 100.0 | 50-300 | Min time in OPEN before CLOSING |
| `closing_stability_time_ms` | 150.0 | 100-400 | Closed detections must persist |
| `centroid_stability_px` | 30.0 | 10-50 | Max movement for "stable" |
| `min_open_evidence_count` | 3 | 2-10 | Min open detections before state change |
| `min_closed_evidence_count` | 2 | 1-5 | Min closed detections for CLOSED |

### IoU-Based Association Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `iou_association_enabled` | true | - | Enable IoU as complementary association criterion |
| `iou_association_threshold` | 0.3 | 0.2-0.5 | Min IoU to associate when centroid fails |

### Velocity-Based Association Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `velocity_scaling_enabled` | true | - | Enable velocity-based distance scaling |
| `velocity_scale_factor` | 2.5 | 1.5-4.0 | Max multiplier for association distance |
| `max_association_distance_px` | 250.0 | 150-400 | Absolute max association distance |

### Timeout-Based Commitment Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `commit_idle_frames` | 25 | 15-50 | Frames without detection before commit |
| `commit_min_closed_ratio` | 0.3 | 0.2-0.6 | Min closed/total ratio for commit |

### Anti-Double-Counting Suppression Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `suppression_distance_px` | 150.0 | 100-250 | Distance to suppress new events near commits |
| `suppression_duration_ms` | 1000.0 | 500-2000 | Duration to suppress after commit |

### Anti-Oscillation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
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

**IoU Association**
- Provides robustness during partial occlusion
- Helps when centroid shifts but boxes still overlap
- Set threshold based on expected overlap during manipulation

**Velocity Scaling**
- When enabled, association distance scales based on bag velocity
- Helps maintain tracking during bag flipping/throwing
- Uses predicted centroid position for association
- Set `max_association_distance_px` to prevent over-association

**Timeout-Based Commitment**
- Bags are committed after being undetected for `commit_idle_frames`
- Works regardless of bag position in frame (no exit boundary required)
- Requires minimum closed evidence ratio to avoid false commits

**Anti-Double-Counting Suppression**
- Prevents re-counting bags that are temporarily re-detected after commitment
- `suppression_distance_px` should be larger than `association_distance_px`
- `suppression_duration_ms` should cover time for bag to leave scene

**Anti-Oscillation**
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
**Mitigation**: Increase `association_time_ms` or `ghost_timeout_ms`, or enable IoU association

### Failure Mode 3: Premature Commits
**Symptom**: Count triggers while bag still in frame
**Cause**: Idle timeout too short
**Detection**: `commit_reason` shows "timeout_commit" but bag visible
**Mitigation**: Increase `commit_idle_frames` or `ghost_timeout_ms`

### Failure Mode 4: Merged Events
**Symptom**: Two bags counted as one
**Cause**: Association distance too large or IoU threshold too low
**Detection**: Single event with abnormally long lifespan
**Mitigation**: Reduce `association_distance_px` or increase `iou_association_threshold`

### Failure Mode 5: Double Counting
**Symptom**: Same bag counted multiple times
**Cause**: Suppression parameters too restrictive, bag re-detected after commit
**Detection**: Events created near recently committed locations
**Mitigation**: Increase `suppression_distance_px` or `suppression_duration_ms`

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
- ❌ Re-identification networks
- ❌ Frame-based counting logic
- ❌ Additional YOLO models
- ❌ Exit boundary-based commitment (removed for simplicity)

---

## Summary

This event-centric tracking system achieves robust counting in challenging human-operated environments by:

1. **Treating events, not objects** - An Event survives what destroys traditional tracks
2. **Using parallel hybrid association** - Both centroid distance AND IoU computed for every attempt
3. **Using milliseconds, not frames** - Deterministic timing
4. **Timeout-based commitment** - Count after idle timeout, not at boundary
5. **Anti-double-counting** - Suppression of new events near recent commits
6. **Providing full explainability** - Every decision is logged with both metrics for debugging

### Parallel Hybrid Association Benefits

The parallel hybrid approach provides significant robustness improvements:

| Scenario | Centroid Only | Hybrid Approach |
|----------|---------------|-----------------|
| Bag flip/spin | ❌ Fails (centroid jumps) | ✅ IoU rescues |
| Fast slide | ✅ Works | ✅ Works (both metrics) |
| Partial occlusion | ⚠️ May fail | ✅ IoU handles overlap |
| Normal movement | ✅ Works | ✅ Both metrics match |

Every association attempt logs:
- Centroid distance and threshold
- IoU value and threshold  
- Time gap
- Which metric(s) matched
- Detection and event centroids

---

## Bug Fix: Hybrid Event Selection (December 2025)

### Issue Description

While the `can_associate()` method correctly computed both centroid distance and IoU for association decisions, the tracker's event selection algorithm had a critical bug: when multiple events could associate with a detection, it **only considered centroid distance** to pick the "best" event, completely ignoring IoU values.

**Impact:** Even if Event A had high IoU (0.8) with a detection but larger centroid distance (150px), and Event B had zero IoU (0.0) but smaller centroid distance (100px), Event B would be selected. This defeated the hybrid association approach and caused IoU to be 0.00 in most cases.

### Root Cause

```python
# BEFORE (buggy code):
for event in active_events:
    can_assoc, distance, reason = event.can_associate(evidence)
    if can_assoc and distance < best_distance:  # ❌ Only considers distance!
        best_event = event
        best_distance = distance
```

### Fix: Hybrid Scoring Algorithm

Implemented adaptive scoring that weighs both IoU and centroid distance:

```python
# AFTER (fixed code):
for event in active_events:
    can_assoc, distance, reason, iou_value = event.can_associate(evidence)
    if not can_assoc:
        continue
    
    # Normalize distance to 0-1 range (1 = closest)
    normalized_distance = max(0, 1.0 - (distance / max_distance))
    
    # Adaptive weighting based on IoU magnitude:
    if iou_value >= 0.5:
        # High IoU: Trust it heavily (80% IoU, 20% distance)
        score = 0.8 * iou_value + 0.2 * normalized_distance
    elif iou_value >= 0.3:
        # Moderate IoU: Balance both (60% IoU, 40% distance)
        score = 0.6 * iou_value + 0.4 * normalized_distance
    else:
        # Low IoU: Trust distance more (30% IoU, 70% distance)
        score = 0.3 * iou_value + 0.7 * normalized_distance
    
    if score > best_score:
        best_event = event
        best_score = score
```

### Scoring Rationale

The adaptive weighting reflects confidence in each metric:

| IoU Range | Weight Distribution | Reasoning |
|-----------|---------------------|-----------|
| ≥ 0.5 | 80% IoU, 20% distance | High overlap = very likely same object |
| 0.3-0.5 | 60% IoU, 40% distance | Moderate overlap = balance both signals |
| < 0.3 | 30% IoU, 70% distance | Low overlap = rely on spatial proximity |

**Why not always 50/50?** High IoU is a stronger signal of object identity than centroid proximity. When boxes overlap significantly (IoU ≥ 0.5), they're almost certainly the same object even if centroids moved (e.g., due to bag rotation/flip).

### Testing

Added comprehensive tests validating the fix:
- `test_high_iou_wins_over_close_centroid`: Verifies high IoU event is selected over closer event with zero IoU
- `test_scoring_weights_adapt_to_iou`: Validates adaptive weighting works correctly
- `test_multiple_events_best_score_wins`: Tests complete selection logic with multiple competing events

All 42 tests pass (39 original + 3 new).
