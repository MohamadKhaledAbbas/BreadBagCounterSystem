# Tracking System Fixes Summary

This document summarizes the fixes applied to address three critical tracking issues in the Event-Centric Tracking System.

## Issue #1: Active Events "Merge/Teleport" After Throw/Fast Movement

### Problem
During fast bag movements or crowded scenes with multiple bags on the table, events would sometimes "teleport" to distant detections. This occurred when:
- Velocity scaling or expanded IoU allowed associations to detections far from the event's last position
- Multiple events competed for the same detection, causing incorrect associations
- The system didn't enforce hard limits on how far an event could jump per frame

### Solution

#### 1. Hard Cap on Centroid Jump Distance
**Parameter:** `max_jump_distance_px` (default: 200.0)

Added an absolute hard limit on centroid movement per association. Even if IoU or expanded IoU criteria are satisfied, associations are rejected if the centroid moves more than this threshold.

```python
# In can_associate() method
if distance_to_last > self.config.max_jump_distance_px:
    return False, distance_to_last, "jump_distance_exceeded", iou_value
```

**Configuration:**
```python
max_jump_distance_px: float = 200.0
"""Hard cap on centroid jump distance per association."""
```

#### 2. Centroid Proximity Requirement for Expanded IoU
**Parameters:**
- `require_centroid_proximity_for_expanded_iou` (default: True)
- `max_centroid_distance_for_expanded_iou` (default: 250.0)

Expanded IoU associations (used during flip/spin scenarios) now require the centroid to remain within reasonable proximity. This prevents expanded IoU from matching detections that are too far away.

```python
# In can_associate() method
if self.config.require_centroid_proximity_for_expanded_iou and expanded_iou_match:
    if distance_to_last > self.config.max_centroid_distance_for_expanded_iou:
        expanded_iou_match = False  # Reject expanded IoU due to distance
```

#### 3. One-to-One Greedy Matching
Implemented global greedy assignment to ensure each detection associates with at most one event per frame.

**Before:** Detections were greedily associated in iteration order, potentially allowing multiple detections to update the same event or causing suboptimal matches.

**After:** 
1. Build all possible (detection, event, score) associations
2. Sort by score (best first)
3. Assign greedily, ensuring one-to-one mapping

```python
# Build candidates
association_candidates = []
for det_idx, evidence in enumerate(detection_evidences):
    for event in self.active_events.values():
        if can_associate:
            association_candidates.append((det_idx, event.id, score, ...))

# Sort and assign greedily
association_candidates.sort(key=lambda x: x[2], reverse=True)
assigned_detections = set()
assigned_events = set()
for det_idx, event_id, score, ... in association_candidates:
    if det_idx not in assigned_detections and event_id not in assigned_events:
        # Assign this pair
        ...
```

### Testing
- `test_max_jump_distance_rejects_far_detection`: Validates hard cap rejection
- `test_max_jump_distance_allows_within_threshold`: Validates acceptance within threshold
- `test_expanded_iou_requires_centroid_proximity`: Validates expanded IoU proximity requirement
- `test_one_to_one_matching_in_crowded_scene`: Validates greedy assignment logic

---

## Issue #2: Events Appear Outside Work Zone and Don't Expire as Expected

### Problem
Work zone filtering was only applied at event creation time. Events could:
- Continue tracking detections that drifted outside the work zone
- Remain active indefinitely when bags moved to areas outside the designated work area
- Not expire even when clearly outside the valid tracking region

### Solution

#### 1. Work Zone Enforcement During Associations
**Parameter:** `enforce_work_zone_associations` (default: True)

Added work zone filtering during the association phase, not just during event creation.

```python
# In update() method, before building association candidates
if self.config.work_zone_enabled and self.config.enforce_work_zone_associations:
    if not self._is_in_work_zone(evidence.centroid_x, evidence.centroid_y):
        continue  # Skip this detection for association
```

**Configuration:**
```python
enforce_work_zone_associations: bool = True
"""Prevent associations for detections outside work zone."""
```

#### 2. Out-of-Zone Tracking and Faster Expiration
**Parameter:** `out_of_zone_grace_frames` (default: 5)

Events now track how long they've been outside the work zone and expire faster when the grace period is exceeded.

**New Event Fields:**
```python
self.frames_out_of_zone = 0
self.out_of_zone_since_ms: Optional[float] = None
```

**Logic in update_ghost_state():**
```python
if self.config.work_zone_enabled and self.config.enforce_work_zone_associations:
    in_zone = (... check if centroid is in zone ...)
    if not in_zone:
        if self.frames_out_of_zone >= self.config.out_of_zone_grace_frames:
            return False, 'expire'  # Expire faster
```

**Reset on Detection:**
```python
# In add_detection() method
self.out_of_zone_since_ms = None
self.frames_out_of_zone = 0
```

### Testing
- `test_work_zone_association_filtering`: Validates association filtering
- `test_out_of_zone_grace_period_expiration`: Validates faster expiration
- `test_out_of_zone_tracking_reset_when_back_in_zone`: Validates reset behavior

---

## Issue #3: Suppression After Commit Blocks New Event Creation Too Aggressively

### Problem
The anti-double-counting suppression mechanism used only centroid proximity (`suppression_distance_px`) and time (`suppression_duration_ms`) to determine if a new detection should be suppressed. This caused issues when:
- Workers removed a counted bag and immediately started a new one at the same location
- The new bag was incorrectly suppressed because its centroid was close to the last committed event
- Fast workflows were unnecessarily slowed down by overly conservative suppression

### Solution

#### 1. Conditional Suppression with Box Overlap
**Parameters:**
- `suppression_require_box_overlap` (default: True)
- `suppression_iou_threshold` (default: 0.15)

Suppression now requires BOTH centroid proximity AND box overlap with the last committed box (when enabled).

**Before:**
```python
# Only checked centroid distance
if distance < self.config.suppression_distance_px:
    return True  # Suppress
```

**After:**
```python
# Check centroid distance
if distance >= self.config.suppression_distance_px:
    continue  # Too far, no suppression

# If box overlap required, check IoU
if self.config.suppression_require_box_overlap:
    if 'box' in rc:
        iou = self._compute_iou_static(rc['box'], evidence.box)
        if iou < self.config.suppression_iou_threshold:
            continue  # No overlap, allow new event
            
# Suppress only if both proximity and overlap conditions met
return True
```

#### 2. Store Last Committed Box
Modified the recently_committed structure to include the box:

```python
self.recently_committed.append({
    'centroid': event.last_centroid,
    'box': event.last_box,  # NEW: Store box for overlap check
    'timestamp_ms': timestamp_ms,
    'event_id': event_id
})
```

#### 3. New Helper Method
Added `_compute_iou_static()` to compute IoU for suppression checks without needing an event instance.

### Use Cases

**Scenario 1: Worker Removes Bag and Starts New One**
- Last committed bag: centroid (640, 360), box (590, 310, 690, 410)
- Worker removes bag, places new bag at similar location
- New detection: centroid (645, 365), box (700, 320, 800, 420)
- Result: **Not suppressed** (centroid close but no box overlap)

**Scenario 2: Same Bag Temporarily Lost and Re-detected**
- Last committed bag: centroid (640, 360), box (590, 310, 690, 410)
- Bag not actually removed, just detection gap
- New detection: centroid (650, 365), box (600, 315, 700, 415)
- Result: **Suppressed** (centroid close AND significant box overlap)

### Configuration

```python
suppression_require_box_overlap: bool = True
"""When True, suppression requires BOTH centroid proximity and box overlap."""

suppression_iou_threshold: float = 0.15
"""Minimum IoU with last committed box to trigger suppression."""
```

### Testing
- `test_suppression_with_box_overlap_blocks_new_event`: Validates suppression with overlap
- `test_suppression_without_box_overlap_allows_new_event`: Validates allowance without overlap
- `test_suppression_respects_iou_threshold`: Validates threshold checking
- `test_suppression_without_overlap_requirement`: Validates legacy distance-only mode

---

## Configuration Changes Summary

### New Parameters in EventConfig

```python
# Issue #1: Teleportation Prevention
max_jump_distance_px: float = 200.0
require_centroid_proximity_for_expanded_iou: bool = True
max_centroid_distance_for_expanded_iou: float = 250.0

# Issue #2: Work Zone Enforcement
enforce_work_zone_associations: bool = True
out_of_zone_grace_frames: int = 5

# Issue #3: Conditional Suppression
suppression_require_box_overlap: bool = True
suppression_iou_threshold: float = 0.15
```

### Updated Files

1. **src/tracking/EventCentricTracker.py**
   - Added hard jump distance check in `can_associate()`
   - Implemented one-to-one greedy matching in `update()`
   - Added out-of-zone tracking in `BreadBagEvent`
   - Updated `_should_suppress()` with conditional logic
   - Added `_compute_iou_static()` helper method

2. **src/config/tracking_config.py**
   - Added new configuration parameters to `TrackingConfig`
   - Updated `get_event_config()` to pass new parameters
   - Added comprehensive documentation for each parameter

3. **src/test/test_event_centric_tracker.py**
   - Added `TestTeleportationPrevention` test class (4 tests)
   - Added `TestWorkZoneEnforcement` test class (3 tests)
   - Added `TestConditionalSuppression` test class (4 tests)

---

## Migration Guide

### For Existing Deployments

The new parameters have sensible defaults that maintain backward compatibility while enabling the fixes:

1. **Issue #1 Fixes (Teleportation)**
   - `max_jump_distance_px=200.0`: Slightly higher than `max_association_distance_px` (180.0)
   - `require_centroid_proximity_for_expanded_iou=True`: Adds safety without breaking existing behavior
   - One-to-one matching: Improves accuracy, no breaking changes

2. **Issue #2 Fixes (Work Zone)**
   - `enforce_work_zone_associations=True`: Matches the intent of the existing `work_zone_enabled`
   - `out_of_zone_grace_frames=5`: ~200ms at 25fps, reasonable grace period

3. **Issue #3 Fixes (Suppression)**
   - `suppression_require_box_overlap=True`: More intelligent suppression
   - `suppression_iou_threshold=0.15`: Conservative threshold

### Tuning Recommendations

#### For Crowded Scenes (Multiple Bags on Table)
```python
# Tighten teleportation constraints
max_jump_distance_px = 150.0
max_centroid_distance_for_expanded_iou = 200.0
```

#### For Fast Workflows
```python
# Allow faster new event creation after commit
suppression_duration_ms = 500.0  # Reduce from 1000.0
suppression_iou_threshold = 0.20  # Require more overlap to suppress
```

#### For Strict Work Zone Enforcement
```python
# Expire out-of-zone events quickly
out_of_zone_grace_frames = 3
enforce_work_zone_associations = True
```

#### For Lenient Work Zone (Edge Cases)
```python
# Allow more grace before expiration
out_of_zone_grace_frames = 10
# Or disable association filtering
enforce_work_zone_associations = False
```

---

## Expected Improvements

1. **Reduced False Associations**: Hard caps prevent events from jumping to distant detections
2. **Better Work Zone Compliance**: Events stay within designated area or expire quickly
3. **Faster Workflows**: Workers can start new bags immediately without suppression delays
4. **No Increase in Double-Counting**: Conditional suppression maintains accuracy while improving speed

---

## Testing Instructions

### Running Unit Tests

```bash
# Install dependencies
pip install pytest numpy opencv-python

# Run all tracking tests
python -m pytest src/test/test_event_centric_tracker.py -v

# Run specific issue tests
python -m pytest src/test/test_event_centric_tracker.py::TestTeleportationPrevention -v
python -m pytest src/test/test_event_centric_tracker.py::TestWorkZoneEnforcement -v
python -m pytest src/test/test_event_centric_tracker.py::TestConditionalSuppression -v
```

### Validation Checklist

- [ ] Run existing test suite to ensure no regressions
- [ ] Test with real production data to validate improvements
- [ ] Monitor double-counting rate (should not increase)
- [ ] Monitor event expiration rate (may increase for out-of-zone events)
- [ ] Monitor suppression rate (should decrease with conditional suppression)
- [ ] Verify behavior in crowded scenes (multiple bags on table)
- [ ] Verify behavior in fast workflows (rapid bag turnover)

---

## Future Enhancements

Potential areas for further improvement:

1. **Adaptive Thresholds**: Adjust parameters based on scene density or velocity
2. **Zone Transition Smoothing**: Add hysteresis for zone boundary crossings
3. **Suppression Decay**: Gradually reduce suppression strength over time
4. **Multi-Zone Support**: Different parameters for different zones in the frame
5. **Velocity-Based Suppression**: Consider bag velocity in suppression decisions

---

## Questions or Issues?

If you encounter any issues with these changes or have questions about tuning parameters for your specific use case, please refer to:

- `EVENT_CENTRIC_TRACKING_ARCHITECTURE.md`: Overall system architecture
- `src/config/tracking_config.py`: Detailed parameter documentation
- `src/test/test_event_centric_tracker.py`: Test cases demonstrating expected behavior
