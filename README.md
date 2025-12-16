# BreadBag Counter System

A production-grade computer vision system for counting bread bags on a conveyor belt using YOLO-based detection and classification.

## Features

- **Real-time Bag Detection**: YOLO-based detection of open and closed bread bags
- **Multi-class Classification**: Classifies bags by type (Whole Wheat, White, Bran, etc.)
- **Event-based Tracking**: Robust event lifecycle management with state transitions
- **Production Logging**: Structured JSON logging for analysis and debugging
- **Log Analysis Tool**: Windows-friendly log analyzer with HTML report generation

## System Architecture

```
Frame Source → Detection (YOLO A) → Event Tracking → Classification (YOLO B) → Counting
                                         ↓
                                  State Management
                                         ↓
                               Database & Logging
```

## Quick Start

### Running the Application

```bash
python main.py
```

### Analyzing Logs

The log analyzer tool helps diagnose counting issues and system performance:

```powershell
# Windows - Analyze today's logs
python tools/log_analyzer.py --log-dir C:\Users\Khaled\Desktop\OrabiLogs\logs

# Linux/macOS - Analyze today's logs
python tools/log_analyzer.py --log-dir ./data/logs

# Analyze specific day
python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-16

# Analyze specific time range
python tools/log_analyzer.py --log-dir ./data/logs --from 2025-12-16T08:00:00Z --to 2025-12-16T16:00:00Z
```

See [docs/LOG_ANALYZER.md](docs/LOG_ANALYZER.md) for detailed documentation.

## Documentation

- **[AUDIT_REPORT.md](AUDIT_REPORT.md)**: Comprehensive system audit and accuracy roadmap
- **[LOGGING_REFACTOR_SUMMARY.md](LOGGING_REFACTOR_SUMMARY.md)**: Logging architecture overview
- **[LOGGING_SAMPLES.md](LOGGING_SAMPLES.md)**: Log format reference and debugging guide
- **[EVENT_CENTRIC_TRACKING_ARCHITECTURE.md](EVENT_CENTRIC_TRACKING_ARCHITECTURE.md)**: Event tracking design
- **[PLATFORM_COMPATIBILITY.md](PLATFORM_COMPATIBILITY.md)**: Platform support information
- **[docs/LOG_ANALYZER.md](docs/LOG_ANALYZER.md)**: Log analyzer tool documentation

## Key Components

### Detection & Tracking
- `src/detection/`: YOLO-based detection modules
- `src/tracking/`: Object tracking with ByteTrack
- `src/counting/BagStateMonitor.py`: Event lifecycle management
- `src/counting/BagCounterApp.py`: Main application logic

### Classification
- `src/classifier/ClassifierService.py`: Multi-candidate classification with voting
- Evidence-based decision making with rejection thresholds

### Logging & Analysis
- `src/utils/AppLogging.py`: Structured JSON logging
- `tools/log_analyzer.py`: Log analysis and HTML report generation

## Log Analysis

The log analyzer provides:

- **Performance Metrics**: Frame processing times, FPS, detection/monitor breakdown
- **Counting Accuracy**: Event lifecycle, expiration analysis, duplicate detection
- **Classification Quality**: Unknown rate, rejection reasons, confidence distribution
- **Issue Detection**: Automatic identification of 7 common problems with remediation advice
- **HTML Reports**: Self-contained reports with charts (Chart.js via CDN)
- **Time Series**: Per-minute trends for errors, warnings, FPS, unknown rate

## Configuration

Key configuration files:
- `config.py`: Main application configuration
- `src/config/tracking_config.py`: Event tracking parameters
- Environment variables: `LOG_LEVEL`, `ENABLE_JSON_LOGGING`, `LOG_DIR`, `ENABLE_UNKNOWN_PHASH_CLUSTERING`

### Unknown Bag Handling

The system now uses **stable Unknown aggregation** to avoid creating many noisy Unknown bag types:

- **Default Behavior**: All Unknown bags are grouped into a single `"Unknown"` bag type in the database
- **Legacy Mode**: Set `ENABLE_UNKNOWN_PHASH_CLUSTERING=1` to enable pHash-based clustering (creates `unknown_bag_1`, `unknown_bag_2`, etc.)
- **Benefit**: Cleaner analytics with one Unknown card instead of many

Unknown bags are classified with a machine-readable reason:
- `structural`: Too few ROIs, track too short, or no valid classifications
- `low_evidence`: Insufficient evidence score
- `ambiguous`: Multiple classes with similar scores

### Confidence Tiering

Classification results are now tracked with **confidence tiers** for better analytics visibility:

- **High Confidence**: Classification confidence >= 0.5 (configurable via `tracking_config.high_confidence_threshold`)
- **Low Confidence**: Classification confidence < 0.5

Analytics UI displays both high and low counts per bag type:
- **Green badge**: High confidence detections
- **Gold badge**: Low confidence detections

This helps identify which bag types need better training data or improved detection conditions.

### Degraded Mode (Overload Protection)

When the system detects overload, it automatically reduces non-critical work to maintain tracking reliability:

**Triggers** (either condition activates degraded mode):
- Queue utilization > 70% (default, configurable via `degraded_mode_queue_threshold`)
- Average queue delay > 100ms (default, configurable via `degraded_mode_delay_threshold_ms`)

**Degraded Mode Actions**:
- Skip ROI image saving to reduce disk I/O
- Skip frames with no detections and no active events
- Optionally disable visualization (controlled by `degraded_mode_disable_visualization`)
- Continue all tracking and counting operations

**Configuration** (in `src/config/tracking_config.py`):
```python
degraded_mode_enabled = True  # Enable/disable feature
degraded_mode_queue_threshold = 0.7  # Queue % to trigger
degraded_mode_delay_threshold_ms = 100.0  # Avg delay to trigger
degraded_mode_disable_roi_saving = True  # Disable ROI saving
degraded_mode_disable_visualization = False  # Keep UI visible
degraded_mode_skip_low_detection_frames = True  # Skip empty frames
```

**Philosophy**: The system prioritizes **counting accuracy over latency**. It prefers buffering and delay rather than dropping frames that could contain bags.

## Troubleshooting

### Under-counting

1. Check `EVENT_EXPIRED` logs for events not completing the pipeline
2. Review `event_expired_by_state` to see where events are failing
3. Check frame drop statistics for missed bags
4. Use log analyzer to identify patterns: `python tools/log_analyzer.py --log-dir ./data/logs`

### Over-counting

1. Check for duplicate `track_id` in `COUNT_UPDATE` logs
2. Review `EVENT_SUPPRESSED` frequency
3. Check `phash` values for duplicate detections
4. Use log analyzer's overcount detection: look for "Duplicate Track IDs" issue

### Poor Classification

1. Check Unknown rate in classification logs
2. Review `rejection_reason` breakdown (low_evidence vs ambiguous)
3. Examine ROI quality metrics (sharpness values)
4. Use log analyzer's classification quality section

### Performance Issues

1. Check frame processing times (target: <40ms for 25 FPS)
2. Review backpressure events and frame drops
3. Monitor queue utilization
4. Use log analyzer's performance charts to identify bottlenecks

## Requirements

- Python 3.7+
- Standard library only (for log analyzer)
- See individual module imports for additional dependencies

## Development

### Running Tests

Tests can be generated using the log sample generator:

```bash
python tools/generate_test_logs.py --output-dir ./test_logs --num-entries 1000
python tools/log_analyzer.py --log-dir ./test_logs
```

### Contributing

When adding new features:
1. Add structured logging for key events
2. Update log analyzer if new metrics are needed
3. Document changes in relevant markdown files
4. Test with log analyzer to verify metrics

## License

(Add license information here)

## Authors

(Add author information here)

## Version History

- **v3.0**: Performance optimization for 25fps at 720p
- **v2.0**: Enhanced structured logging and production-grade features
- **v1.0**: Initial release with detection, tracking, and classification
