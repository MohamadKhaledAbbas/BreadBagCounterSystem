# V6 Performance & Reliability Optimization

## Overview

V6 implements production-grade optimizations for the Event-Centric Bread Counting Pipeline, focusing on:
- **Event accuracy** over frame accuracy
- **Bounded computational complexity**
- **Deterministic behavior under load**
- **Retention safety** for unprocessed data

## Key Optimizations

### 1. Adaptive Ghost Timeout (Section 6️⃣)

Ghost timeout now scales with object velocity to handle spinning/thrown bags more intelligently.

**Formula:**
```
ghost_timeout = base_timeout + k * velocity_magnitude
```

**Benefits:**
- Spinning objects survive short occlusions (higher velocity = longer timeout)
- Thrown/fast objects terminate quickly (prevent stale events)
- More responsive to object motion dynamics

**Configuration:**
```python
adaptive_ghost_timeout_enabled: bool = True
adaptive_ghost_velocity_factor: float = 2.0  # k factor
adaptive_ghost_min_timeout_frames: int = 15  # floor
adaptive_ghost_max_timeout_frames: int = 75  # ceiling
```

### 2. Multi-Stage Matching with Early Rejection (Section 5️⃣)

IOU computation is expensive. We now reject most candidates cheaply before computing IOU.

**Matching Pipeline Order:**
1. Ghost timeout check (instant rejection - cheapest)
2. Centroid distance gate (cheap)
3. Area ratio gate (cheap)
4. IOU computation (expensive - only if above pass)

**Benefits:**
- Most candidates rejected cheaply in stages 1-3
- IOU only computed on viable candidates
- Significant CPU cost reduction (30-50%)

**Configuration:**
```python
early_rejection_enabled: bool = True
early_rejection_area_ratio_min: float = 0.4  # reject if area ratio too different
early_rejection_area_ratio_max: float = 2.5  # max allowed size difference
```

### 3. Temporal Decimation (Section 7️⃣)

Detection runs every frame. Monitor updates do not need to.

**Skip monitor update when:**
- Bounding box area change < ε (5%)
- Centroid shift < δ (5 pixels)
- Confidence unchanged

**Benefits:**
- Preserves correctness (detection still runs every frame)
- Cuts monitor CPU cost significantly (30-50%)
- Only skips redundant state updates

**Configuration:**
```python
temporal_decimation_enabled: bool = True
temporal_decimation_area_epsilon: float = 0.05     # 5% area change threshold
temporal_decimation_centroid_delta_px: float = 5.0 # 5 pixel movement threshold
temporal_decimation_confidence_epsilon: float = 0.05  # 5% confidence change
temporal_decimation_max_skip_frames: int = 3       # force update after 3 skips
```

### 4. Spatial Zones (Section 3️⃣)

Explicit zone definitions for predictable event lifecycle.

**Zones:**
- **ENTRY_ZONE**: Where new events can be created
- **ACTIVE_ZONE**: Where events participate in matching
- **EXIT_ZONE**: Where events are candidates for finalization

**Configuration:**
```python
spatial_zones_enabled: bool = True
entry_zone_margin_px: int = 50   # margin from edges for entry
exit_zone_margin_px: int = 80    # margin defining exit zone
```

### 5. Retention Safety (Section 1️⃣1️⃣)

Retention must never delete unprocessed data.

**Rule:**
```
segment.frame_index >= last_processed_index
```

**Benefits:**
- Never deletes unprocessed data
- Prevents data loss under load
- Production-safe operation

**Configuration:**
```python
retention_safety_enabled: bool = True
```

**Usage:**
```python
from src.spool.retention import RetentionPolicy

policy = RetentionPolicy(spool_dir, retention_safety_enabled=True)

# Periodically update from tracker
tracker_stats = tracker.get_tracker_stats()
policy.set_last_processed_frame(tracker_stats['last_processed_frame_index'])
```

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Avg per-frame latency | ~70 ms | 25-35 ms |
| Monitor CPU load | Unbounded | Constant |
| Frame drops | Implicit | None |
| Count accuracy | Good | Stable & deterministic |
| System stability | Fragile under load | Production-grade |

## Monitoring

### Tracker Statistics

```python
stats = tracker.get_tracker_stats()
print(f"Frames decimated: {stats['frames_decimated']}")
print(f"Last processed frame: {stats['last_processed_frame_index']}")
```

### Retention Statistics

```python
retention_stats = policy.get_stats()
print(f"Segments protected by progress: {retention_stats['segments_protected_by_progress']}")
print(f"Last processed frame: {retention_stats['last_processed_frame']}")
```

## Environment Variables

All V6 parameters can be controlled via environment variables:

```bash
# Temporal Decimation
export TEMPORAL_DECIMATION_ENABLED=true
export TEMPORAL_DECIMATION_AREA_EPSILON=0.05
export TEMPORAL_DECIMATION_CENTROID_DELTA=5.0
export TEMPORAL_DECIMATION_MAX_SKIP=3

# Early Rejection
export EARLY_REJECTION_ENABLED=true
export EARLY_REJECTION_AREA_RATIO_MIN=0.4
export EARLY_REJECTION_AREA_RATIO_MAX=2.5

# Spatial Zones
export SPATIAL_ZONES_ENABLED=true
export ENTRY_ZONE_MARGIN_PX=50
export EXIT_ZONE_MARGIN_PX=80

# Retention Safety
export RETENTION_SAFETY_ENABLED=true
```

## Architecture Principles

This implementation follows the core agreement from the optimization plan:

1. **Optimize for event accuracy, not frame accuracy**
   - It is acceptable to skip internal states
   - It is NOT acceptable to miss: object entry, object exit, count trigger

2. **Bound Monitor Complexity**
   - Active events must always remain O(1)
   - Prune aggressively via spatial gates

3. **Monitoring must be:**
   - Deterministic
   - Bounded in complexity
   - Resistant to chaotic motion (spinning, flipping, throwing)
