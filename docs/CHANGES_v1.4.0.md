# Version 1.4.0 - Classification Stability Heuristics

**Release Date**: 2025-12-18  
**Type**: Feature Enhancement

## Summary

This release adds production-grade classification stability heuristics to improve counting reliability by leveraging historical context and tracking classification stability over time.

## New Features

### 1. Previous-Label Reuse System

Intelligently reuses previous classifications when current confidence is low but historical evidence is strong.

**Key Points**:
- ✅ Feature-flagged (disabled by default)
- ✅ Requires strong evidence: streak ≥ 3, burst dominance ≥ 75%
- ✅ Guards against conflicts: checks for higher-confidence alternatives
- ✅ Fully audited: all decisions logged with structured data

**Configuration**:
```bash
export ENABLE_LABEL_REUSE=true
export LOW_CONF_THRESHOLD=0.65
export STREAK_MIN_LENGTH=3
export BURST_DOMINANCE_MIN_RATIO=0.75
```

### 2. Track Label Volatility Monitoring

Automatically detects and flags tracks with unstable classifications.

**Key Points**:
- Calculates volatility: (label changes) / (track lifespan)
- Flags tracks exceeding threshold (default: 0.3)
- Provides full label history for flagged tracks
- Surfaces in analyzer reports

**Configuration**:
```bash
export TRACK_VOLATILITY_THRESHOLD=0.3
export ENABLE_VOLATILITY_LOGGING=true
```

### 3. Enhanced Structured Logging

New log events for audit and analysis:

**Label Reuse Override**:
```
[LABEL_REUSE] track=12345, prev=Bran, new=Green_Yellow(0.620), streak=3, dom=Bran(0.85)
```

**High Volatility Flag**:
```
[HIGH_VOLATILITY] track=12346, changes=4, lifespan=10, volatility=0.400
```

### 4. Log Analyzer Enhancements

Updated `log_analyzer.py` with:
- Parsing support for new log formats
- HTML report sections for stability heuristics
- JSON export of volatility and reuse metrics

## Configuration Parameters

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LABEL_REUSE` | `false` | Enable/disable label reuse feature |
| `LOW_CONF_THRESHOLD` | `0.65` | Confidence below which reuse is considered |
| `STREAK_MIN_LENGTH` | `3` | Minimum consecutive same-label classifications |
| `BURST_DOMINANCE_MIN_RATIO` | `0.75` | Minimum burst dominance ratio for reuse |
| `BURST_WINDOW_SIZE` | `10` | Number of recent classifications for burst analysis |
| `TRACK_VOLATILITY_THRESHOLD` | `0.3` | Volatility score threshold for flagging |
| `ENABLE_VOLATILITY_LOGGING` | `true` | Enable/disable volatility logging |

### Rationale for Defaults

- **`LOW_CONF_THRESHOLD=0.65`**: Between low (< 0.5) and high (≥ 0.7) confidence tiers
- **`STREAK_MIN_LENGTH=3`**: Ensures pattern is not just noise (requires 3+ consecutive)
- **`BURST_DOMINANCE_MIN_RATIO=0.75`**: Guards against reuse in mixed-variety scenarios
- **`TRACK_VOLATILITY_THRESHOLD=0.3`**: Flags tracks changing more than once per 3 bags

## Files Changed

- `src/config/tracking_config.py`: Added 7 new parameters with env var support
- `src/classifier/ClassifierService.py`: Implemented reuse logic and volatility tracking (200+ lines)
- `src/utils/AppLogging.py`: Added 2 new structured logging methods
- `tools/log_analyzer.py`: Updated parser and HTML report generator
- `docs/CLASSIFICATION_STABILITY_HEURISTICS.md`: Comprehensive documentation (9KB)

## Testing

### Unit Tests
- ✅ Configuration defaults and environment variables
- ✅ Volatility calculation logic
- ✅ Streak detection
- ✅ Burst dominance calculation

### Integration Tests
- ✅ Structured logging methods
- ✅ Log analyzer parsing
- ✅ End-to-end configuration flow

### Security
- ✅ CodeQL scan: 0 alerts found
- ✅ Safe dictionary access patterns
- ✅ Input validation for all parameters

## Migration Guide

### Existing Users

**No action required**. All new features are:
- Disabled by default (label reuse)
- Backward compatible
- Non-breaking

**To enable label reuse**:
1. Review documentation: `docs/CLASSIFICATION_STABILITY_HEURISTICS.md`
2. Test in development with: `ENABLE_LABEL_REUSE=true python main.py`
3. Monitor logs for label reuse events
4. Adjust thresholds based on results
5. Deploy to production when confident

### New Users

Start with default configuration. Volatility monitoring is enabled by default and provides valuable insights without changing behavior.

## Performance Impact

- **CPU**: Negligible overhead (< 1% additional processing time)
- **Memory**: O(N) per track, where N is typically < 10 classifications
- **Disk**: ~200 bytes per structured log event (JSON format)
- **Network**: No impact (local processing only)

## Known Limitations

1. **Label reuse disabled by default**: Must be explicitly enabled for production use
2. **Requires classification history**: Reuse logic needs ≥ 3 recent classifications
3. **Single-variety optimization**: Burst dominance works best in single-variety scenarios
4. **No cross-session persistence**: History resets on application restart

## Future Enhancements

Potential improvements for future releases:
- Persistent classification history across restarts
- Adaptive threshold tuning based on accuracy metrics
- Machine learning-based volatility prediction
- Real-time alerting for high volatility rates

## Rollback

If issues arise, disable the feature:
```bash
export ENABLE_LABEL_REUSE=false
# Restart application
```

Volatility logging can also be disabled:
```bash
export ENABLE_VOLATILITY_LOGGING=false
# Restart application
```

## Support

- **Documentation**: `docs/CLASSIFICATION_STABILITY_HEURISTICS.md`
- **Configuration**: `src/config/tracking_config.py`
- **Logs**: `data/logs/app.json.log`
- **Reports**: `reports/YYYY-MM-DD/report.html`

## Credits

Implemented as part of production-grade reliability improvements for the BreadBagCounterSystem.

## License

Same as parent project.
