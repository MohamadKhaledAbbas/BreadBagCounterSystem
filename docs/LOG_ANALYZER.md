# Log Analyzer - BreadBag Counter System

## Overview

The Log Analyzer is a Windows-friendly tool that parses rotated JSON log files from the BreadBag Counter System and generates comprehensive HTML reports with actionable diagnostics.

## Features

- ✅ **Windows-Compatible**: Works seamlessly with Windows paths (e.g., `C:\Users\Khaled\Desktop\OrabiLogs\logs`)
- 📊 **Per-Day Analysis**: Analyze logs for specific UTC days
- 🔄 **Rotation-Aware**: Automatically discovers and processes rotated log files (app.json.log.*)
- 📈 **Rich Metrics**: Comprehensive metrics including errors, performance, counting accuracy, and classification quality
- 🎨 **HTML Reports**: Self-contained HTML reports with embedded charts (Chart.js via CDN)
- 🚨 **Issue Detection**: Automatic detection of common issues with suggested fixes
- ⚡ **Streaming Parser**: Memory-efficient streaming for multi-GB log files
- 🎯 **Frame-Based Analysis**: Tracks frame-based thresholds with FPS-aware conversions
- 📊 **Enhanced Metrics**: Event lifecycle, ROI quality, classification details, and system health

## Installation

No additional dependencies required beyond Python 3.7+. The tool uses only standard library modules.

## Usage

### Basic Usage (Windows PowerShell)

Analyze today's logs (UTC):

```powershell
python tools/log_analyzer.py --log-dir C:\Users\Khaled\Desktop\OrabiLogs\logs
```

### Analyze Specific Day

```powershell
python tools/log_analyzer.py --log-dir C:\Users\Khaled\Desktop\OrabiLogs\logs --day 2025-12-16
```

### Analyze Custom Time Range

```powershell
python tools/log_analyzer.py --log-dir C:\Users\Khaled\Desktop\OrabiLogs\logs --from 2025-12-16T00:00:00Z --to 2025-12-16T23:59:59Z
```

### Specify Output Directory

```powershell
python tools/log_analyzer.py --log-dir C:\Users\Khaled\Desktop\OrabiLogs\logs --day 2025-12-16 --output C:\Users\Khaled\Desktop\Reports
```

### Linux/macOS Usage

```bash
python tools/log_analyzer.py --log-dir ./data/logs
python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-16
```

## Command-Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--log-dir` | Directory containing app.json.log and rotated backups | Yes | - |
| `--day` | Analyze logs for specific day (YYYY-MM-DD, UTC) | No | Today (UTC) |
| `--from` | Start timestamp (ISO8601 format with Z) | No | - |
| `--to` | End timestamp (ISO8601 format with Z) | No | - |
| `--output` | Output directory for reports | No | `./reports` |

## Output Files

The analyzer generates two files in the output directory:

### 1. HTML Report (`reports/YYYY-MM-DD/report.html`)

A comprehensive, self-contained HTML report including:

- **Summary KPIs**: 9 cards showing total entries, errors, bags counted, suppressed events, FPS, unknown rate, events created/committed, average event lifetime
- **Frame-Based Threshold Configuration**: Table showing frame-based thresholds with FPS-aware time conversions
- **Event Suppression Analysis**: Breakdown by suppression type (spatial, temporal, active event exclusion)
- **Event Lifecycle Metrics**: Created/committed/expired counts, average and max lifetime in frames and seconds
- **ROI Collection Metrics**: Added/rejected counts, rejection rate, sharpness distribution, average ROIs per event
- **Classification Quality**: Unknown rate, confidence stats, candidates per classification, voting rate, processing times
- **Bag Type Distribution**: Counts by bag type
- **Issue Findings**: Automatic detection of problems with severity levels, descriptions, likely causes, and remediation advice
- **Frame Performance**: Charts and tables showing processing times (avg, p50, p95, max)
- **Error/Warning Analysis**: Top repeating errors and warnings grouped by component
- **Time Series Charts**: Per-minute trends for events created/committed/expired, errors, warnings, FPS, ROI activity

The HTML report can be opened directly in any web browser (Chrome, Firefox, Edge, Safari).

### 2. JSON Summary (`reports/YYYY-MM-DD/summary.json`)

A machine-readable JSON file containing all computed metrics and statistics for further processing or integration with other tools.

## Metrics Computed

### Metadata
- App start time and version
- Report generation timestamp

### Error & Warning Analysis
- Total error/warning counts
- Top repeating messages by (component, message)
- Pipeline errors grouped by (component, operation, error_type)

### Backpressure Metrics
- Total frame drops
- Total frames skipped
- Queue utilization statistics (avg, max)

### Frame Performance
- Detection time: avg, p50, p95, max, min
- Monitor time: avg, p50, p95, max, min
- Total frame time: avg, p50, p95, max, min
- FPS: avg, p50, p95, max, min

### Frame-Based Thresholds
- Ghost timeout (frames → ms conversion)
- Temporal cooldown (frames → ms conversion)
- Suppression duration (frames → ms conversion)
- Average FPS for conversions

### Event Lifecycle
- Total created, committed, expired
- Expired by state breakdown
- Total suppressed
- Average lifetime (frames and seconds)
- Lifetime statistics (avg, p50, p95, max, min)

### Event Creation Blockers
- Total blocked
- Breakdown by reason (covered_by_active_event, suppression_spatial, suppression_temporal, active_event_exclusion)
- Suppression distance statistics
- Cooldown time statistics

### Counting Signals
- Total bags counted
- Bag type distribution

### Track Statistics
- Total created, duplicates, expired
- Average lifetime in frames
- Unique tracks counted
- Duplicate track_id and phash indicators

### ROI Quality
- Total added/rejected
- Rejection rate
- Rejection reasons breakdown
- Sharpness statistics (avg, p50, p95, max, min)
- Average ROIs per event

### Classification Quality
- Total classifications
- Unknown count and rate
- Rejection reason breakdown
- Confidence distribution (avg, p50, p95, max, min)
- Average candidates per classification
- Voting usage rate
- Average processing time

### Time Series
Per-minute aggregation of:
- Errors
- Warnings
- Backpressure drops
- Unknown classifications
- Average FPS
- Events created, committed, expired
- Suppressed events
- ROI added, ROI rejected

## Issue Detection

The analyzer automatically detects common issues and provides actionable diagnostics with frame-based threshold awareness:

### 1. High Event Suppression Rate (>5%)
- **Severity**: Warning
- **Likely Cause**: Temporal cooldown or suppression thresholds may be too aggressive
- **Where to Look**: Review `temporal_cooldown_frames` in config (default: 10 frames). Consider reducing to 5-8 frames for faster workflows
- **Recommendation**: Frame-based thresholds scale naturally with FPS. Adjust thresholds in frame units, not milliseconds

### 2. High Unknown Classification Rate (>10%)
- **Severity**: Warning
- **Likely Cause**: Poor ROI quality, model uncertainty, or inadequate training data
- **Where to Look**: Check rejection_reasons breakdown (shows top 3), review ROI sharpness and rejection rate
- **Recommendation**: Includes ROI metrics showing average sharpness and rejection rate

### 3. Frame Drops Due to Backpressure
- **Severity**: Error
- **Likely Cause**: System cannot keep up with input frame rate (CPU/GPU overload)
- **Where to Look**: Check frame processing times, consider reducing input FPS or optimizing models

### 4. High Frame Processing Time (P95 > 50ms)
- **Severity**: Warning
- **Likely Cause**: Detection or monitoring bottleneck, hardware limitations
- **Where to Look**: Compare detection_time_ms vs monitor_time_ms to identify bottleneck

### 5. Low FPS Throughput (<20 FPS)
- **Severity**: Warning
- **Likely Cause**: System overload, slow detection model, or hardware limitations
- **Where to Look**: Check backpressure events, frame processing times, and queue utilization

### 4. Low FPS Throughput (<20 FPS)
- **Severity**: Warning
- **Likely Cause**: System overload, slow detection model, or hardware limitations
- **Where to Look**: Check backpressure events, frame processing times, queue utilization

### 5. High Event Expiration Rate (>30%)
- **Severity**: Error
- **Likely Cause**: Bags not tracked long enough (too fast), insufficient hits for state transitions
- **Where to Look**: Check event_expired_by_state breakdown, consider lowering min_open_frames or min_closed_frames

### 6. Duplicate Track IDs in COUNT_UPDATE
- **Severity**: Error
- **Likely Cause**: Same bag counted more than once, suppression lockout too short
- **Where to Look**: Review specific track IDs, check EVENT_SUPPRESSED logs and lockout_window setting

### 7. High Error Count (>100)
- **Severity**: Error
- **Likely Cause**: System instability, invalid data, or recurring bugs
- **Where to Look**: Check top error types and pipeline_error_groups for patterns

## Log File Format

The analyzer expects JSON logs in the format produced by `src/utils/AppLogging.py`:

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

Key fields:
- `timestamp`: ISO8601 timestamp with 'Z' suffix (UTC)
- `level`: Log level (INFO, WARNING, ERROR, DEBUG)
- `component`: Component generating the log
- `message`: Human-readable message
- `data`: Structured data with metric values

## Log File Discovery

The analyzer automatically discovers:
- Current log file: `app.json.log`
- Rotated backups: `app.json.log.1`, `app.json.log.2`, ..., `app.json.log.N`

Files do not need to be in chronological order. The analyzer filters entries by timestamp, not file order.

## Performance

- **Streaming Parser**: Processes logs line-by-line without loading entire files into memory
- **Efficient**: Can handle multi-GB log files (tested up to 10GB+)
- **Fast**: Typical processing speed: 50,000+ lines per second on modern hardware

## Troubleshooting

### Issue: "No log files found in {directory}"

**Solution**: Ensure the `--log-dir` points to the directory containing `app.json.log`. Check that:
- The path is correct (use forward slashes or escaped backslashes in Windows)
- The log file is named `app.json.log` (not `app.log`)
- JSON logging is enabled (`ENABLE_JSON_LOGGING=true` in environment)

### Issue: "Total Entries: 0"

**Solution**: No log entries matched the selected time range. Check:
- The `--day` or `--from`/`--to` parameters are correct
- Log entries have valid `timestamp` fields in ISO8601 format with 'Z' suffix
- The selected day has log data (logs may be from a different day)

### Issue: HTML report opens but charts are blank

**Solution**: Ensure you have internet access when opening the report. Charts use Chart.js loaded from CDN (cdn.jsdelivr.net). If offline:
- Download Chart.js and modify the HTML to use a local copy
- Or view the data in the JSON summary instead

### Issue: "Permission denied" on Windows

**Solution**: Run PowerShell as Administrator or ensure you have write permissions to the output directory.

## Integration with Other Tools

### Using JSON Summary for Further Analysis

The JSON summary can be used with other tools:

```python
import json

with open('reports/2025-12-16/summary.json', 'r') as f:
    stats = json.load(f)

# Extract specific metrics
unknown_rate = stats['classification']['unknown_rate']
avg_fps = stats['fps']['avg']
issues = stats['issues']

print(f"Unknown Rate: {unknown_rate:.1%}")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Issues: {len(issues)}")
```

### Automated Monitoring

Set up a scheduled task (Windows Task Scheduler) or cron job (Linux) to run the analyzer daily:

**Windows (PowerShell script)**:
```powershell
# analyze_logs.ps1
$logDir = "C:\Users\Khaled\Desktop\OrabiLogs\logs"
$outputDir = "C:\Users\Khaled\Desktop\Reports"

python tools/log_analyzer.py --log-dir $logDir --output $outputDir

# Open report in browser
$date = Get-Date -Format "yyyy-MM-dd"
Start-Process "$outputDir\$date\report.html"
```

**Linux/macOS (bash script)**:
```bash
#!/bin/bash
# analyze_logs.sh
LOG_DIR="./data/logs"
OUTPUT_DIR="./reports"

python tools/log_analyzer.py --log-dir "$LOG_DIR" --output "$OUTPUT_DIR"

# Open report (macOS)
DATE=$(date +%Y-%m-%d)
open "$OUTPUT_DIR/$DATE/report.html"
```

## Examples

### Example 1: Daily Production Analysis

Analyze yesterday's production logs every morning:

```powershell
$yesterday = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
python tools/log_analyzer.py --log-dir C:\Production\logs --day $yesterday --output C:\Reports
```

### Example 2: Shift Analysis

Analyze logs for a specific 8-hour shift (8 AM to 4 PM UTC):

```powershell
python tools/log_analyzer.py --log-dir C:\Production\logs --from 2025-12-16T08:00:00Z --to 2025-12-16T16:00:00Z
```

### Example 3: Multi-Day Batch Analysis

Analyze multiple days in a loop:

```powershell
$dates = @("2025-12-14", "2025-12-15", "2025-12-16")
foreach ($date in $dates) {
    Write-Host "Analyzing $date..."
    python tools/log_analyzer.py --log-dir C:\Production\logs --day $date --output C:\Reports
}
```

## Related Documentation

- [LOGGING_SAMPLES.md](LOGGING_SAMPLES.md) - Log format reference and examples
- [LOGGING_REFACTOR_SUMMARY.md](LOGGING_REFACTOR_SUMMARY.md) - Logging architecture overview
- [AUDIT_REPORT.md](AUDIT_REPORT.md) - System architecture and metrics

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review log files for error messages
3. Open an issue on the GitHub repository with:
   - Command used
   - Error message or unexpected behavior
   - Sample log entries (if relevant)

## Version History

- **v1.0** (2025-12-16): Initial release
  - Windows-compatible path handling
  - Per-day analysis with UTC support
  - HTML report generation with Chart.js
  - JSON summary export
  - Issue detection and diagnostics
  - Streaming parser for large files
