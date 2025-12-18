# Classification Stability Heuristics

## Overview

This document describes the production-grade classification stability heuristics added to improve counting reliability in the BreadBagCounterSystem. These features help reduce false positives/negatives by leveraging historical context and tracking classification stability over time.

## Features

### 1. Previous-Label Reuse

**Purpose**: Reuse previous classification when current confidence is low but historical evidence is strong.

**How it works**:
- When a classification has confidence below `LOW_CONF_THRESHOLD` (default: 0.65)
- AND there's a strong historical streak (≥ `STREAK_MIN_LENGTH`, default: 3 consecutive same-label classifications)
- AND no higher-confidence conflicting candidate exists
- AND the previous label matches burst dominance if available (≥ `BURST_DOMINANCE_MIN_RATIO`, default: 0.75)
- THEN the system will reuse the previous label instead of accepting the low-confidence classification

**Safety**: This feature is **disabled by default** and must be explicitly enabled via configuration.

### 2. Track Label Volatility Monitoring

**Purpose**: Detect and flag tracks where classification labels change frequently, indicating potential instability.

**How it works**:
- For each track, calculates: `volatility = (number of label changes) / (track lifespan in bags)`
- Tracks exceeding `TRACK_VOLATILITY_THRESHOLD` (default: 0.3) are flagged
- High-volatility tracks are logged with full label history
- Volatility metrics are included in analyzer reports

**Benefits**:
- Identifies classification instability patterns
- Helps detect model quality issues
- Flags tracks that may need manual review

## Configuration

All parameters are configurable via environment variables or tracking_config.py:

### Environment Variables

```bash
# Feature flags
ENABLE_LABEL_REUSE=false              # Enable/disable label reuse (default: false)
ENABLE_VOLATILITY_LOGGING=true        # Enable/disable volatility logging (default: true)

# Label reuse thresholds
LOW_CONF_THRESHOLD=0.65               # Confidence below which reuse is considered
STREAK_MIN_LENGTH=3                   # Minimum streak length for reuse
BURST_DOMINANCE_MIN_RATIO=0.75        # Minimum burst dominance ratio
BURST_WINDOW_SIZE=10                  # Burst window size (number of classifications)

# Volatility thresholds
TRACK_VOLATILITY_THRESHOLD=0.3        # Volatility threshold for flagging
```

### Configuration File

Parameters are defined in `src/config/tracking_config.py` with detailed comments:

```python
from src.config.tracking_config import tracking_config

# Access configuration values
print(f"Label reuse enabled: {tracking_config.enable_label_reuse}")
print(f"Low confidence threshold: {tracking_config.low_conf_threshold}")
print(f"Volatility threshold: {tracking_config.track_volatility_threshold}")
```

## Usage

### Enabling Label Reuse

**For production testing:**
```bash
export ENABLE_LABEL_REUSE=true
export LOW_CONF_THRESHOLD=0.65
python main.py
```

**For development:**
```python
# In your code
from src.config.tracking_config import tracking_config
tracking_config.enable_label_reuse = True
```

### Tuning Thresholds

#### LOW_CONF_THRESHOLD (0.5 - 0.8)
- **Lower (0.5-0.6)**: More aggressive reuse, may mask genuine label changes
- **Higher (0.7-0.8)**: Conservative reuse, only for borderline cases
- **Recommended**: Start with 0.65 and adjust based on Unknown rate

#### STREAK_MIN_LENGTH (2 - 10)
- **Lower (2-3)**: More responsive to short-term patterns
- **Higher (5-10)**: Only trust very stable long-term patterns
- **Recommended**: Start with 3 for single-variety scenarios

#### BURST_DOMINANCE_MIN_RATIO (0.6 - 0.9)
- **Lower (0.6-0.7)**: Allow reuse in more diverse scenarios
- **Higher (0.8-0.9)**: Only allow reuse in very homogeneous bursts
- **Recommended**: Start with 0.75 for production

#### TRACK_VOLATILITY_THRESHOLD (0.2 - 0.5)
- **Lower (0.2)**: Flag more tracks, stricter stability requirement
- **Higher (0.4-0.5)**: Only flag very unstable tracks
- **Recommended**: 0.3 (one change per 3 bags)

## Structured Logging

### Log Formats

#### Label Reuse Override
```
[LABEL_REUSE] track=12345, prev=Bran, new=Green_Yellow(0.620), streak=3, dom=Bran(0.85), reason=low_confidence_with_strong_streak
```

**Fields**:
- `track`: Track ID
- `prev`: Previous label (from streak)
- `new`: Current low-confidence label
- `streak`: Streak length
- `dom`: Dominant label in burst (label and ratio)
- `reason`: Override reason

#### High Volatility Flag
```
[HIGH_VOLATILITY] track=12346, changes=4, lifespan=10, volatility=0.400
```

**Fields**:
- `track`: Track ID
- `changes`: Number of label changes
- `lifespan`: Track lifespan (number of classifications)
- `volatility`: Volatility score

### JSON Logging

All structured logs are also written to `data/logs/app.json.log`:

```json
{
  "timestamp": "2025-12-18T14:30:00.123Z",
  "level": "INFO",
  "component": "ClassifierService",
  "message": "[LABEL_REUSE] ...",
  "data": {
    "track_id": 12345,
    "prev_label": "Bran",
    "new_label": "Green_Yellow",
    "new_confidence": 0.62,
    "streak_len": 3,
    "dominance_label": "Bran",
    "dominance_ratio": 0.85,
    "candidate_tops": [["Green_Yellow", 0.62], ["Bran", 0.58]],
    "reuse_reason": "low_confidence_with_strong_streak"
  }
}
```

## Log Analysis

### Using log_analyzer.py

The log analyzer automatically parses and visualizes stability heuristics:

```bash
python tools/log_analyzer.py \
  --log-dir data/logs \
  --day 2025-12-18 \
  --output reports
```

### Report Sections

The HTML report includes:

1. **Classification Stability Heuristics** section:
   - Label reuse count and rate
   - High volatility track count
   - Average/max volatility scores

2. **Recent Label Reuse Events** table:
   - Track IDs with reuse
   - Previous vs. new labels
   - Confidence and streak details
   - Burst dominance information

3. **High Volatility Tracks** table:
   - Track IDs with high volatility
   - Label change counts
   - Lifespan and volatility scores

### JSON Summary

Statistics are also exported to `reports/YYYY-MM-DD/summary.json`:

```json
{
  "classification": {
    "stability_heuristics": {
      "label_reuse_count": 5,
      "label_reuse_rate": 0.05,
      "label_reuse_events": [...],
      "high_volatility_tracks": 2,
      "avg_volatility": 0.25,
      "max_volatility": 0.45,
      "volatility_details": [...]
    }
  }
}
```

## Best Practices

### 1. Start Conservative
- Keep label reuse **disabled** initially
- Monitor volatility metrics for baseline understanding
- Enable reuse only after establishing stable patterns

### 2. Tuning Strategy
1. Analyze current Unknown rate and confusion pairs
2. Enable label reuse with default thresholds
3. Monitor reuse events in logs
4. Adjust `LOW_CONF_THRESHOLD` based on results
5. Fine-tune streak and dominance ratios as needed

### 3. Production Rollout
- Use feature flag to enable/disable without code changes
- Test in parallel (shadow mode) before full deployment
- Monitor volatility metrics for regressions
- Set up alerts for high volatility track rates

### 4. Debugging
- Review label reuse logs for unexpected overrides
- Check high volatility tracks for patterns
- Use analyzer reports to identify trends
- Compare metrics before/after enabling reuse

## Safety Guarantees

1. **Feature-flagged**: Label reuse disabled by default, must be explicitly enabled
2. **No breaking changes**: Existing classification API unchanged
3. **Backward compatible**: Works with existing logs and configuration
4. **Auditable**: All decisions logged with structured data
5. **Reversible**: Can disable reuse at any time without side effects

## Performance Impact

- **Negligible CPU overhead**: Simple calculations on small data structures
- **Memory**: O(N) per track, where N is classification history size (typically < 10)
- **Logging**: Structured logs add ~200 bytes per event (JSON format)
- **Analysis**: Log analyzer processes new formats efficiently

## Troubleshooting

### Issue: Too many label reuse events
**Cause**: `LOW_CONF_THRESHOLD` too high or streak requirements too lenient
**Solution**: Increase threshold to 0.7 or increase `STREAK_MIN_LENGTH` to 5

### Issue: Not enough reuse (still high Unknown rate)
**Cause**: Thresholds too strict or burst dominance requirement too strong
**Solution**: Lower `LOW_CONF_THRESHOLD` to 0.6 or `BURST_DOMINANCE_MIN_RATIO` to 0.7

### Issue: High volatility tracks not being flagged
**Cause**: `TRACK_VOLATILITY_THRESHOLD` too high
**Solution**: Lower threshold to 0.2 to catch more cases

### Issue: Too many volatility warnings
**Cause**: Threshold too low for your use case
**Solution**: Increase `TRACK_VOLATILITY_THRESHOLD` to 0.4

## References

- Configuration: `src/config/tracking_config.py`
- Implementation: `src/classifier/ClassifierService.py`
- Logging: `src/utils/AppLogging.py`
- Analysis: `tools/log_analyzer.py`

## Support

For questions or issues:
1. Check logs in `data/logs/app.json.log`
2. Review analyzer reports in `reports/`
3. Consult configuration comments in `tracking_config.py`
4. Open an issue on GitHub with log excerpts
