# Configuration Guide

This document describes key configuration options for the BreadBag Counter System.

## Environment Variables

### Logging Configuration

- `LOG_LEVEL`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `ENABLE_JSON_LOGGING`: Enable structured JSON logging (`1` or `0`)
- `LOG_DIR`: Directory for log files (default: `./data/logs`)

### Unknown Bag Handling

- `ENABLE_UNKNOWN_PHASH_CLUSTERING`: Enable legacy pHash-based Unknown clustering
  - `0` (default): All Unknowns grouped into single "Unknown" bag type
  - `1`: Create separate `unknown_bag_N` types based on pHash similarity
  - **Recommendation**: Keep disabled (0) for cleaner analytics

## Tracking Configuration

Located in `src/config/tracking_config.py` - key parameters below:

### Classification Confidence Thresholds

```python
high_confidence_threshold: float = 0.5
```
- Threshold separating "high" vs "low" confidence classifications
- Used for confidence tier reporting in analytics
- Range: 0.3 - 0.7
- **Recommendation**: 0.5 provides good separation

### Degraded Mode (Overload Protection)

```python
degraded_mode_enabled: bool = True
```
- Enable automatic degraded mode under overload
- **Recommendation**: Keep enabled for production

```python
degraded_mode_queue_threshold: float = 0.7
```
- Queue utilization (0.0-1.0) to trigger degraded mode
- 0.7 = 70% queue full
- Range: 0.5 - 0.9
- **Recommendation**: 0.7 balances latency vs reliability

```python
degraded_mode_delay_threshold_ms: float = 100.0
```
- Average queue delay (milliseconds) to trigger degraded mode
- Range: 50 - 300
- **Recommendation**: 100ms for responsive activation

```python
degraded_mode_disable_roi_saving: bool = True
```
- Disable ROI image saving in degraded mode to reduce disk I/O
- **Recommendation**: True (saves I/O, snapshots continue if enabled)

```python
degraded_mode_disable_visualization: bool = False
```
- Disable visualization rendering in degraded mode
- **Recommendation**: False (keep UI visible for monitoring)
- Set to True if maximum throughput is needed

```python
degraded_mode_skip_low_detection_frames: bool = True
```
- Skip frames with no detections when in degraded mode
- Safe because empty frames don't contribute to tracking
- **Recommendation**: True (reduces wasted processing)

### Evidence-Based Classification

```python
min_total_evidence_score: float = 0.3
```
- Minimum evidence score to accept a classification
- Below this → Unknown
- Range: 0.1 - 1.0
- **Recommendation**: 0.3 provides good filtering

```python
evidence_ratio_threshold: float = 1.5
```
- Winner/runner-up ratio for acceptance
- Lower → more permissive, higher → stricter
- Range: 1.1 - 3.0
- **Recommendation**: 1.5 balances accuracy vs Unknown rate

```python
classifier_reject_labels: tuple = ('Rejected',)
```
- Labels to skip/reject during voting and aggregation
- Predictions with these labels are excluded from evidence accumulation
- They don't count toward minimum candidates threshold
- If ALL predictions are rejected, track is classified as "Unknown"
- **Default**: ('Rejected',)
- **Example**: Add more labels like ('Rejected', 'LowQuality', 'Ambiguous')
- **Recommendation**: Use this to filter out low-quality or uncertain classifier predictions

**Behavior Details:**
- ROIs with reject labels are completely skipped during evidence accumulation
- Rejection count is tracked and reported in statistics
- Other classes in the same probability vector are still used (only the reject label itself is skipped)
- This improves classification quality by ignoring frames the classifier marked as unreliable

## Database Schema Changes

### Confidence Tier Column

The system automatically adds a `confidence_tier` column to `bag_events` table on first run:

```sql
ALTER TABLE bag_events 
ADD COLUMN confidence_tier TEXT DEFAULT 'high'
```

This is **backward compatible**:
- Existing rows get default value 'high'
- No data migration needed
- No manual intervention required

## Analytics Configuration

### Confidence Tier Display

Analytics UI automatically shows confidence breakdown:
- **Total count** per bag type
- **High confidence count** (green badge)
- **Low confidence count** (gold badge)

No configuration needed - it reads from database automatically.

## Performance Tuning

### For High Throughput (30+ FPS)

```python
# In tracking_config.py
degraded_mode_queue_threshold = 0.6  # Enter degraded mode earlier
degraded_mode_disable_visualization = True  # Skip visualization
```

### For Maximum Accuracy (Lower FPS OK)

```python
# In tracking_config.py
degraded_mode_queue_threshold = 0.8  # Tolerate higher queue
min_total_evidence_score = 0.4  # Stricter evidence requirement
evidence_ratio_threshold = 2.0  # Require clearer winner
```

### For Debugging (No Degraded Mode)

```python
# In tracking_config.py
degraded_mode_enabled = False
```

Or via environment:
```bash
export ENABLE_UNKNOWN_PHASH_CLUSTERING=1  # See individual Unknown types
```

## Troubleshooting Configuration Issues

### Too Many Unknown Cards in Analytics

**Problem**: Many `unknown_bag_1`, `unknown_bag_2` cards  
**Solution**: Ensure `ENABLE_UNKNOWN_PHASH_CLUSTERING` is not set (or set to `0`)

### Low Confidence Counts Seem Wrong

**Problem**: Most bags show as "low confidence"  
**Solution**: Adjust `high_confidence_threshold` in tracking_config.py
- Lower it (e.g., 0.4) if your model produces lower confidence scores
- Check your classifier model confidence distribution first

### System Enters Degraded Mode Too Often

**Problem**: Frequent degraded mode activation messages  
**Solution**: 
1. Check if camera FPS can be reduced (e.g., 25 → 20 FPS)
2. Increase `degraded_mode_queue_threshold` to 0.8
3. Increase `degraded_mode_delay_threshold_ms` to 150

### Frames Being Dropped

**Problem**: High frame drop count in logs  
**Solution**:
1. Enable degraded mode if disabled
2. Reduce camera FPS
3. Check detection model performance (should be < 35ms per frame)

## Migration from Previous Versions

### From v3.x → v4.x (Confidence Tiering)

**No action required** - schema migration is automatic:
- Database adds `confidence_tier` column on startup
- Existing events get 'high' as default tier
- Analytics UI automatically shows new breakdowns

### From v2.x → v3.x (Degraded Mode)

**Recommended**:
1. Review `tracking_config.py` for new degraded mode parameters
2. Test with current load to see if degraded mode activates
3. Adjust thresholds based on your hardware and FPS target

## Best Practices

1. **Production Deployments**:
   - Enable degraded mode
   - Disable Unknown pHash clustering
   - Monitor degraded mode activation frequency
   - Adjust thresholds based on your load patterns

2. **Development/Testing**:
   - Can disable degraded mode for deterministic behavior
   - Enable Unknown pHash clustering to study classifier failures
   - Lower confidence thresholds to see more classifications

3. **Analytics Review**:
   - Check confidence tier breakdowns weekly
   - High "low confidence" rate indicates need for model retraining
   - Unknown rate > 10% indicates detection or environmental issues
