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

- **Summary KPIs**: Total entries, errors, warnings, unknown rate, FPS, bags counted
- **Time Range Information**: Start/end times and duration
- **Issue Findings**: Automatic detection of problems with severity levels, descriptions, likely causes, and remediation advice
- **Frame Performance**: Charts and tables showing processing times (avg, p50, p95, max)
- **Counting Metrics**: Event lifecycle statistics, bag type distribution
- **Classification Quality**: Unknown rate, rejection reasons, confidence statistics
- **Error/Warning Analysis**: Top repeating errors and warnings
- **Time Series Charts**: Per-minute trends for errors, warnings, FPS, unknown rate

The HTML report can be opened directly in any web browser (Chrome, Firefox, Edge, Safari).

### 2. JSON Summary (`reports/YYYY-MM-DD/summary.json`)

A machine-readable JSON file containing all computed metrics and statistics for further processing or integration with other tools.

## Metrics Computed

### Error & Warning Analysis
- Total error/warning counts
- Top repeating messages by (component, message)
- Pipeline errors grouped by (component, operation, error_type)

### Backpressure Metrics
- Total frame drops
- Total frames skipped
- Queue utilization statistics

### Frame Performance
- Detection time: avg, p50, p95, max
- Monitor time: avg, p50, p95, max
- Total frame time: avg, p50, p95, max
- FPS: avg, p50, p95, max

### Counting Signals
- EVENT_CREATED count
- EVENT_EXPIRED count (by state)
- EVENT_SUPPRESSED count
- COUNT_UPDATE count
- Bag type distribution

### Overcount Indicators
- Duplicate track_id in COUNT_UPDATE events
- Repeated phash values

### Classification Quality
- Total classifications
- Unknown count and rate
- Rejection reason breakdown
- Confidence distribution (avg, p50, p95)

### Time Series
Per-minute aggregation of:
- Errors
- Warnings
- Backpressure drops
- Unknown classifications
- Average FPS

## Issue Detection

The analyzer automatically detects common issues and provides actionable diagnostics:

### 1. High Unknown Classification Rate (>10%)
- **Severity**: Warning
- **Likely Cause**: Poor ROI quality, model uncertainty, or inadequate training data
- **Where to Look**: Check rejection_reasons breakdown, review ROI sharpness values

### 2. Frame Drops Due to Backpressure
- **Severity**: Error
- **Likely Cause**: System cannot keep up with input frame rate (CPU/GPU overload)
- **Where to Look**: Check frame processing times, consider reducing input FPS or optimizing models

### 3. High Frame Processing Time (P95 > 50ms)
- **Severity**: Warning
- **Likely Cause**: Detection or monitoring bottleneck, hardware limitations
- **Where to Look**: Compare detection_time_ms vs monitor_time_ms to identify bottleneck

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

- [LOGGING_SAMPLES.md](../LOGGING_SAMPLES.md) - Log format reference and examples
- [LOGGING_REFACTOR_SUMMARY.md](../LOGGING_REFACTOR_SUMMARY.md) - Logging architecture overview
- [AUDIT_REPORT.md](../AUDIT_REPORT.md) - System architecture and metrics

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
