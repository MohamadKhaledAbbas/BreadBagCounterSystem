#!/usr/bin/env python3
"""
Windows-friendly Log Analyzer for BreadBagCounterSystem

Parses rotated JSON log files and generates per-day HTML reports with actionable diagnostics.

Usage:
    python tools/log_analyzer.py --log-dir C:\\Users\\Khaled\\Desktop\\OrabiLogs\\logs
    python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-16
    python tools/log_analyzer.py --log-dir ./data/logs --from 2025-12-16T00:00:00Z --to 2025-12-16T23:59:59Z
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import statistics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze BreadBagCounterSystem JSON logs and generate HTML reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze today's logs (UTC)
  python tools/log_analyzer.py --log-dir C:\\Users\\Khaled\\Desktop\\OrabiLogs\\logs
  
  # Analyze specific day
  python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-16
  
  # Analyze specific time range
  python tools/log_analyzer.py --log-dir ./data/logs --from 2025-12-16T00:00:00Z --to 2025-12-16T23:59:59Z
  
  # Specify output directory
  python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-16 --output ./reports
        """
    )
    
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory containing app.json.log and rotated backups (e.g., app.json.log.*)"
    )
    
    parser.add_argument(
        "--day",
        help="Analyze logs for specific day (YYYY-MM-DD, UTC). Defaults to today."
    )
    
    parser.add_argument(
        "--from",
        dest="from_time",
        help="Start timestamp (ISO8601 format, e.g., 2025-12-16T00:00:00Z)"
    )
    
    parser.add_argument(
        "--to",
        dest="to_time",
        help="End timestamp (ISO8601 format, e.g., 2025-12-16T23:59:59Z)"
    )
    
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory for reports (default: ./reports)"
    )
    
    return parser.parse_args()


def get_time_range(args) -> Tuple[datetime, datetime]:
    """
    Determine the time range to analyze based on arguments.
    Returns (start_time, end_time) as timezone-aware datetime objects in UTC.
    """
    if args.from_time and args.to_time:
        # Custom range
        start = datetime.fromisoformat(args.from_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(args.to_time.replace('Z', '+00:00'))
    elif args.day:
        # Specific day (00:00:00 to 23:59:59.999999 UTC)
        day = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start = day
        end = day + timedelta(days=1, microseconds=-1)
    else:
        # Default to today UTC
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = today
        end = today + timedelta(days=1, microseconds=-1)
    
    return start, end


def discover_log_files(log_dir: str) -> List[str]:
    """
    Discover all relevant log files in the directory.
    Finds app.json.log and all rotated backups (app.json.log.1, app.json.log.2, etc.)
    """
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Find main log file and rotated backups
    log_files = []
    
    # Main log file
    main_log = log_dir_path / "app.json.log"
    if main_log.exists():
        log_files.append(str(main_log))
    
    # Rotated backups (app.json.log.1, app.json.log.2, etc.)
    rotated_logs = sorted(glob.glob(str(log_dir_path / "app.json.log.*")))
    log_files.extend(rotated_logs)
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")
    
    return log_files


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single JSON log line.
    Returns parsed dict or None if malformed.
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        entry = json.loads(line)
        return entry
    except json.JSONDecodeError:
        return None


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse ISO8601 timestamp string to timezone-aware datetime.
    Handles both 'Z' suffix and '+00:00' timezone format.
    """
    try:
        # Replace 'Z' with '+00:00' for proper parsing
        ts_str = ts_str.replace('Z', '+00:00')
        return datetime.fromisoformat(ts_str)
    except (ValueError, AttributeError):
        return None


def is_in_time_range(entry: Dict[str, Any], start: datetime, end: datetime) -> bool:
    """Check if log entry timestamp falls within the specified time range."""
    ts_str = entry.get("timestamp")
    if not ts_str:
        return False
    
    ts = parse_timestamp(ts_str)
    if not ts:
        return False
    
    return start <= ts <= end


def stream_log_entries(log_files: List[str], start: datetime, end: datetime):
    """
    Stream log entries from all files that fall within the time range.
    Yields (entry_dict, file_path, line_number, is_valid) tuples.
    """
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    entry = parse_log_line(line)
                    if entry is None:
                        yield (None, log_file, line_num, False)
                        continue
                    
                    # Check if entry is in time range
                    if is_in_time_range(entry, start, end):
                        yield (entry, log_file, line_num, True)
        except Exception as e:
            print(f"Warning: Error reading {log_file}: {e}", file=sys.stderr)


class LogAnalyzer:
    """Main analyzer class that computes metrics and generates reports."""
    
    def __init__(self, start_time: datetime, end_time: datetime):
        self.start_time = start_time
        self.end_time = end_time
        
        # Counters and aggregators
        self.total_entries = 0
        self.skipped_entries = 0
        self.entries_by_level = Counter()
        self.entries_by_component = Counter()
        
        # Error and warning tracking
        self.errors = []  # List of (component, message, data)
        self.warnings = []
        self.error_counts = Counter()  # (component, message) -> count
        self.warning_counts = Counter()
        
        # Pipeline errors
        self.pipeline_errors = []  # List of full error entries
        self.pipeline_error_groups = Counter()  # (component, operation, error_type) -> count
        
        # Backpressure metrics
        self.backpressure_events = []
        self.total_drops = 0
        self.total_frames_skipped = 0
        self.queue_utilization_samples = []
        
        # Frame performance
        self.frame_times = []  # total_time_ms values
        self.detection_times = []
        self.monitor_times = []
        self.fps_values = []
        
        # Counting signals
        self.event_created_count = 0
        self.event_expired_count = 0
        self.event_expired_by_state = Counter()
        self.event_suppressed_count = 0
        self.count_update_count = 0
        
        # Overcount indicators
        self.count_update_track_ids = []
        self.count_update_phashes = []
        
        # Classification quality
        self.classification_count = 0
        self.unknown_count = 0
        self.rejection_reasons = Counter()
        self.confidence_values = []
        self.bag_type_counts = Counter()
        
        # Time series (per-minute buckets)
        self.time_buckets = defaultdict(lambda: {
            "errors": 0,
            "warnings": 0,
            "backpressure_drops": 0,
            "unknown_classifications": 0,
            "fps_samples": []
        })
        
    def analyze_entry(self, entry: Dict[str, Any]):
        """Analyze a single log entry and update metrics."""
        self.total_entries += 1
        
        level = entry.get("level", "")
        component = entry.get("component", "Unknown")
        message = entry.get("message", "")
        data = entry.get("data", {})
        timestamp_str = entry.get("timestamp", "")
        
        self.entries_by_level[level] += 1
        self.entries_by_component[component] += 1
        
        # Get minute bucket for time series
        ts = parse_timestamp(timestamp_str)
        if ts:
            minute_key = ts.replace(second=0, microsecond=0).isoformat()
        else:
            minute_key = None
        
        # Error tracking
        if level == "ERROR":
            self.errors.append((component, message, data))
            self.error_counts[(component, message)] += 1
            if minute_key:
                self.time_buckets[minute_key]["errors"] += 1
            
            # Pipeline errors with structured data
            if data:
                operation = data.get("operation", "unknown")
                error_type = data.get("error_type", "unknown")
                self.pipeline_errors.append(entry)
                self.pipeline_error_groups[(component, operation, error_type)] += 1
        
        # Warning tracking
        if level == "WARNING":
            self.warnings.append((component, message, data))
            self.warning_counts[(component, message)] += 1
            if minute_key:
                self.time_buckets[minute_key]["warnings"] += 1
        
        # Backpressure detection
        if "BACKPRESSURE" in message or "backpressure" in message.lower():
            self.backpressure_events.append(entry)
            drops = data.get("drops", 0)
            frames_skipped = data.get("frames_skipped", 0)
            utilization = data.get("utilization", 0)
            
            self.total_drops += drops
            self.total_frames_skipped += frames_skipped
            if utilization:
                self.queue_utilization_samples.append(utilization)
            
            if minute_key:
                self.time_buckets[minute_key]["backpressure_drops"] += drops
        
        # Frame performance
        if "FRAME" in message or data.get("frame_id") is not None:
            if "total_time_ms" in data:
                self.frame_times.append(data["total_time_ms"])
            if "detection_time_ms" in data:
                self.detection_times.append(data["detection_time_ms"])
            if "monitor_time_ms" in data:
                self.monitor_times.append(data["monitor_time_ms"])
            if "fps" in data:
                fps = data["fps"]
                self.fps_values.append(fps)
                if minute_key:
                    self.time_buckets[minute_key]["fps_samples"].append(fps)
        
        # Event lifecycle tracking
        if "EVENT_CREATED" in message:
            self.event_created_count += 1
        
        if "EVENT_EXPIRED" in message:
            self.event_expired_count += 1
            state = data.get("state", "unknown")
            self.event_expired_by_state[state] += 1
        
        if "EVENT_SUPPRESSED" in message:
            self.event_suppressed_count += 1
        
        # Count updates
        if "COUNT_UPDATE" in message:
            self.count_update_count += 1
            track_id = data.get("track_id")
            phash = data.get("phash")
            bag_type = data.get("bag_type")
            
            if track_id is not None:
                self.count_update_track_ids.append(track_id)
            if phash is not None:
                self.count_update_phashes.append(phash)
            if bag_type:
                self.bag_type_counts[bag_type] += 1
        
        # Classification tracking
        if "CLASSIFICATION" in message:
            self.classification_count += 1
            label = data.get("label", "")
            confidence = data.get("confidence")
            rejection_reason = data.get("rejection_reason")
            
            if label == "Unknown":
                self.unknown_count += 1
                if minute_key:
                    self.time_buckets[minute_key]["unknown_classifications"] += 1
                if rejection_reason:
                    self.rejection_reasons[rejection_reason] += 1
            
            if confidence is not None:
                self.confidence_values.append(confidence)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics and metrics."""
        stats = {
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_hours": (self.end_time - self.start_time).total_seconds() / 3600
            },
            "parsing": {
                "total_entries": self.total_entries,
                "skipped_entries": self.skipped_entries,
                "entries_by_level": dict(self.entries_by_level),
                "entries_by_component": dict(self.entries_by_component)
            },
            "errors": {
                "total": len(self.errors),
                "top_errors": self.error_counts.most_common(10),
                "pipeline_error_groups": self.pipeline_error_groups.most_common(10)
            },
            "warnings": {
                "total": len(self.warnings),
                "top_warnings": self.warning_counts.most_common(10)
            },
            "backpressure": {
                "total_events": len(self.backpressure_events),
                "total_drops": self.total_drops,
                "total_frames_skipped": self.total_frames_skipped,
                "avg_queue_utilization": statistics.mean(self.queue_utilization_samples) if self.queue_utilization_samples else 0
            },
            "frame_performance": self._compute_percentile_stats(self.frame_times, "total_time_ms"),
            "detection_performance": self._compute_percentile_stats(self.detection_times, "detection_time_ms"),
            "monitor_performance": self._compute_percentile_stats(self.monitor_times, "monitor_time_ms"),
            "fps": self._compute_percentile_stats(self.fps_values, "fps"),
            "counting": {
                "event_created": self.event_created_count,
                "event_expired": self.event_expired_count,
                "event_expired_by_state": dict(self.event_expired_by_state),
                "event_suppressed": self.event_suppressed_count,
                "count_update": self.count_update_count,
                "bag_type_counts": dict(self.bag_type_counts)
            },
            "overcount_indicators": {
                "duplicate_track_ids": self._find_duplicates(self.count_update_track_ids),
                "duplicate_phashes": self._find_duplicates(self.count_update_phashes)
            },
            "classification": {
                "total": self.classification_count,
                "unknown_count": self.unknown_count,
                "unknown_rate": self.unknown_count / self.classification_count if self.classification_count > 0 else 0,
                "rejection_reasons": dict(self.rejection_reasons),
                "confidence_stats": self._compute_percentile_stats(self.confidence_values, "confidence")
            },
            "time_series": self._compute_time_series(),
            "issues": self._detect_issues()
        }
        
        return stats
    
    def _compute_percentile_stats(self, values: List[float], name: str) -> Dict[str, Any]:
        """Compute avg, p50, p95, max for a list of values."""
        if not values:
            return {
                "name": name,
                "count": 0,
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "max": 0,
                "min": 0
            }
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        # Proper percentile calculation: position = percentile * (n-1)
        p95_index = int(0.95 * (n - 1))
        
        return {
            "name": name,
            "count": n,
            "avg": statistics.mean(values),
            "p50": sorted_values[n // 2],
            "p95": sorted_values[p95_index],
            "max": max(values),
            "min": min(values)
        }
    
    def _find_duplicates(self, items: List) -> List[Tuple[Any, int]]:
        """Find items that appear more than once and return (item, count) pairs."""
        counts = Counter(items)
        return [(item, count) for item, count in counts.items() if count > 1]
    
    def _compute_time_series(self) -> List[Dict[str, Any]]:
        """Convert time bucket data to sorted time series."""
        series = []
        for minute_key in sorted(self.time_buckets.keys()):
            bucket = self.time_buckets[minute_key]
            avg_fps = statistics.mean(bucket["fps_samples"]) if bucket["fps_samples"] else 0
            series.append({
                "timestamp": minute_key,
                "errors": bucket["errors"],
                "warnings": bucket["warnings"],
                "backpressure_drops": bucket["backpressure_drops"],
                "unknown_classifications": bucket["unknown_classifications"],
                "avg_fps": avg_fps
            })
        return series
    
    def _detect_issues(self) -> List[Dict[str, Any]]:
        """Detect issues based on thresholds and return actionable findings."""
        issues = []
        
        # Issue: High unknown rate
        if self.classification_count > 0:
            unknown_rate = self.unknown_count / self.classification_count
            if unknown_rate > 0.10:  # > 10%
                issues.append({
                    "severity": "warning",
                    "title": "High Unknown Classification Rate",
                    "description": f"Unknown rate is {unknown_rate:.1%} (threshold: 10%)",
                    "likely_cause": "Poor ROI quality, model uncertainty, or inadequate training data",
                    "where_to_look": "Check rejection_reasons breakdown, review ROI sharpness values in logs"
                })
        
        # Issue: Backpressure drops
        if self.total_drops > 0:
            issues.append({
                "severity": "error",
                "title": "Frame Drops Due to Backpressure",
                "description": f"Total frames dropped: {self.total_drops}",
                "likely_cause": "System cannot keep up with input frame rate (CPU/GPU overload)",
                "where_to_look": "Check frame processing times (detection_time_ms, total_time_ms), consider reducing input FPS or optimizing models"
            })
        
        # Issue: High frame processing time / low FPS
        if self.frame_times:
            avg_frame_time = statistics.mean(self.frame_times)
            p95_frame_time = sorted(self.frame_times)[int(len(self.frame_times) * 0.95)] if len(self.frame_times) > 20 else max(self.frame_times)
            
            if p95_frame_time > 50:  # > 50ms (slower than 20 FPS)
                issues.append({
                    "severity": "warning",
                    "title": "High Frame Processing Time",
                    "description": f"P95 frame time: {p95_frame_time:.1f}ms (avg: {avg_frame_time:.1f}ms)",
                    "likely_cause": "Detection or monitoring bottleneck, hardware limitations",
                    "where_to_look": "Compare detection_time_ms vs monitor_time_ms to identify bottleneck component"
                })
        
        if self.fps_values:
            avg_fps = statistics.mean(self.fps_values)
            if avg_fps < 20:  # < 20 FPS
                issues.append({
                    "severity": "warning",
                    "title": "Low FPS Throughput",
                    "description": f"Average FPS: {avg_fps:.1f} (target: 25+)",
                    "likely_cause": "System overload, slow detection model, or hardware limitations",
                    "where_to_look": "Check backpressure events, frame processing times, and queue utilization"
                })
        
        # Issue: High event expiration rate
        if self.event_created_count > 0:
            expiration_rate = self.event_expired_count / self.event_created_count
            if expiration_rate > 0.30:  # > 30%
                issues.append({
                    "severity": "error",
                    "title": "High Event Expiration Rate (Under-counting Risk)",
                    "description": f"Expiration rate: {expiration_rate:.1%} ({self.event_expired_count}/{self.event_created_count})",
                    "likely_cause": "Bags not tracked long enough (too fast), insufficient hits for state transitions",
                    "where_to_look": f"Check event_expired_by_state breakdown: {dict(self.event_expired_by_state)}. Consider lowering min_open_frames or min_closed_frames thresholds."
                })
        
        # Issue: Duplicate track IDs (over-counting indicator)
        duplicate_tracks = self._find_duplicates(self.count_update_track_ids)
        if duplicate_tracks:
            issues.append({
                "severity": "error",
                "title": "Duplicate Track IDs in COUNT_UPDATE (Over-counting)",
                "description": f"Found {len(duplicate_tracks)} track IDs that were counted multiple times",
                "likely_cause": "Same bag counted more than once, suppression lockout too short",
                "where_to_look": f"Review track IDs: {duplicate_tracks[:5]} (showing first 5). Check EVENT_SUPPRESSED logs and lockout_window setting."
            })
        
        # Issue: Many errors
        if len(self.errors) > 100:
            issues.append({
                "severity": "error",
                "title": "High Error Count",
                "description": f"Total errors: {len(self.errors)}",
                "likely_cause": "System instability, invalid data, or recurring bugs",
                "where_to_look": "Check top error types and pipeline_error_groups for patterns"
            })
        
        return issues


def generate_html_report(stats: Dict[str, Any], output_path: str):
    """Generate HTML report with embedded CSS and Chart.js visualizations."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadBag Counter Log Analysis - {stats['time_range']['start'][:10]}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .kpi-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .kpi-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        
        .kpi-card.warning {{
            background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        }}
        
        .kpi-card.error {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        
        .kpi-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .issue-card {{
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }}
        
        .issue-card.warning {{
            background: #fff3cd;
            border-left-color: #ffc107;
        }}
        
        .issue-card.error {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        
        .issue-card h4 {{
            margin-bottom: 10px;
            color: #333;
        }}
        
        .issue-card p {{
            margin: 5px 0;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }}
        
        .badge.success {{
            background: #28a745;
            color: white;
        }}
        
        .badge.warning {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge.error {{
            background: #dc3545;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🍞 BreadBag Counter - Log Analysis Report</h1>
        
        <div class="summary-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Log Entries</div>
                <div class="kpi-value">{stats['parsing']['total_entries']:,}</div>
            </div>
            <div class="kpi-card {'error' if stats['errors']['total'] > 100 else 'warning' if stats['errors']['total'] > 10 else 'success'}">
                <div class="kpi-label">Errors</div>
                <div class="kpi-value">{stats['errors']['total']}</div>
            </div>
            <div class="kpi-card {'warning' if stats['warnings']['total'] > 100 else 'success'}">
                <div class="kpi-label">Warnings</div>
                <div class="kpi-value">{stats['warnings']['total']}</div>
            </div>
            <div class="kpi-card {'warning' if stats['classification']['unknown_rate'] > 0.1 else 'success'}">
                <div class="kpi-label">Unknown Rate</div>
                <div class="kpi-value">{stats['classification']['unknown_rate']:.1%}</div>
            </div>
            <div class="kpi-card {'success' if stats['fps']['count'] > 0 and stats['fps']['avg'] >= 20 else 'warning'}">
                <div class="kpi-label">Avg FPS</div>
                <div class="kpi-value">{stats['fps']['avg'] if stats['fps']['count'] > 0 else 0:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Bags Counted</div>
                <div class="kpi-value">{stats['counting']['count_update']}</div>
            </div>
        </div>
        
        <h2>📊 Time Range</h2>
        <table>
            <tr>
                <th>Start Time (UTC)</th>
                <th>End Time (UTC)</th>
                <th>Duration</th>
            </tr>
            <tr>
                <td>{stats['time_range']['start']}</td>
                <td>{stats['time_range']['end']}</td>
                <td>{stats['time_range']['duration_hours']:.2f} hours</td>
            </tr>
        </table>
        
        <h2>🚨 Issue Findings</h2>
        {'<p><strong>✅ No issues detected!</strong> All metrics are within acceptable thresholds.</p>' if not stats['issues'] else ''}
        {''.join([f'''
        <div class="issue-card {issue['severity']}">
            <h4><span class="badge {issue['severity']}">{issue['severity'].upper()}</span> {issue['title']}</h4>
            <p><strong>Description:</strong> {issue['description']}</p>
            <p><strong>Likely Cause:</strong> {issue['likely_cause']}</p>
            <p><strong>Where to Look:</strong> {issue['where_to_look']}</p>
        </div>
        ''' for issue in stats['issues']])}
        
        <h2>📈 Frame Performance</h2>
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
                <th>Avg</th>
                <th>P50</th>
                <th>P95</th>
                <th>Max</th>
            </tr>
            <tr>
                <td>Total Frame Time</td>
                <td>{stats['frame_performance']['count']}</td>
                <td>{stats['frame_performance']['avg']:.1f} ms</td>
                <td>{stats['frame_performance']['p50']:.1f} ms</td>
                <td>{stats['frame_performance']['p95']:.1f} ms</td>
                <td>{stats['frame_performance']['max']:.1f} ms</td>
            </tr>
            <tr>
                <td>Detection Time</td>
                <td>{stats['detection_performance']['count']}</td>
                <td>{stats['detection_performance']['avg']:.1f} ms</td>
                <td>{stats['detection_performance']['p50']:.1f} ms</td>
                <td>{stats['detection_performance']['p95']:.1f} ms</td>
                <td>{stats['detection_performance']['max']:.1f} ms</td>
            </tr>
            <tr>
                <td>Monitor Time</td>
                <td>{stats['monitor_performance']['count']}</td>
                <td>{stats['monitor_performance']['avg']:.1f} ms</td>
                <td>{stats['monitor_performance']['p50']:.1f} ms</td>
                <td>{stats['monitor_performance']['p95']:.1f} ms</td>
                <td>{stats['monitor_performance']['max']:.1f} ms</td>
            </tr>
            <tr>
                <td>FPS</td>
                <td>{stats['fps']['count']}</td>
                <td>{stats['fps']['avg']:.1f}</td>
                <td>{stats['fps']['p50']:.1f}</td>
                <td>{stats['fps']['p95']:.1f}</td>
                <td>{stats['fps']['max']:.1f}</td>
            </tr>
        </table>
        
        <h2>🎯 Counting Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>Events Created</td>
                <td>{stats['counting']['event_created']}</td>
            </tr>
            <tr>
                <td>Events Expired</td>
                <td>{stats['counting']['event_expired']}</td>
            </tr>
            <tr>
                <td>Events Suppressed</td>
                <td>{stats['counting']['event_suppressed']}</td>
            </tr>
            <tr>
                <td>Count Updates (Bags Counted)</td>
                <td>{stats['counting']['count_update']}</td>
            </tr>
        </table>
        
        <h3>Bag Type Distribution</h3>
        <div class="chart-container">
            <canvas id="bagTypeChart"></canvas>
        </div>
        
        <h3>Event Expiration by State</h3>
        <table>
            <tr>
                <th>State</th>
                <th>Count</th>
            </tr>
            {''.join([f'<tr><td>{state}</td><td>{count}</td></tr>' for state, count in stats['counting']['event_expired_by_state'].items()])}
        </table>
        
        <h2>🔍 Classification Quality</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Classifications</td>
                <td>{stats['classification']['total']}</td>
            </tr>
            <tr>
                <td>Unknown Classifications</td>
                <td>{stats['classification']['unknown_count']} ({stats['classification']['unknown_rate']:.1%})</td>
            </tr>
            <tr>
                <td>Avg Confidence</td>
                <td>{stats['classification']['confidence_stats']['avg'] if stats['classification']['confidence_stats']['count'] > 0 else 0:.3f}</td>
            </tr>
        </table>
        
        <h3>Rejection Reasons</h3>
        <table>
            <tr>
                <th>Reason</th>
                <th>Count</th>
            </tr>
            {''.join([f'<tr><td>{reason}</td><td>{count}</td></tr>' for reason, count in stats['classification']['rejection_reasons'].items()])}
        </table>
        
        <h2>⚠️ Top Errors</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Message</th>
                <th>Count</th>
            </tr>
            {''.join([f'<tr><td>{comp}</td><td>{msg}</td><td>{count}</td></tr>' for (comp, msg), count in stats['errors']['top_errors'][:10]])}
        </table>
        
        <h2>⚠️ Top Warnings</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Message</th>
                <th>Count</th>
            </tr>
            {''.join([f'<tr><td>{comp}</td><td>{msg}</td><td>{count}</td></tr>' for (comp, msg), count in stats['warnings']['top_warnings'][:10]])}
        </table>
        
        <h2>⏱️ Time Series</h2>
        <div class="chart-container">
            <canvas id="timeSeriesChart"></canvas>
        </div>
        
        <div class="footer">
            <p><strong>Report Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p>
            <p><strong>Total Entries Parsed:</strong> {stats['parsing']['total_entries']:,} | <strong>Skipped:</strong> {stats['parsing']['skipped_entries']}</p>
            <p><strong>Analyzer Version:</strong> 1.0</p>
        </div>
    </div>
    
    <script>
        // Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {{
            type: 'bar',
            data: {{
                labels: ['Avg', 'P50', 'P95', 'Max'],
                datasets: [{{
                    label: 'Total Frame Time (ms)',
                    data: [{stats['frame_performance']['avg']:.1f}, {stats['frame_performance']['p50']:.1f}, {stats['frame_performance']['p95']:.1f}, {stats['frame_performance']['max']:.1f}],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }}, {{
                    label: 'Detection Time (ms)',
                    data: [{stats['detection_performance']['avg']:.1f}, {stats['detection_performance']['p50']:.1f}, {stats['detection_performance']['p95']:.1f}, {stats['detection_performance']['max']:.1f}],
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Time (ms)'
                        }}
                    }}
                }}
            }}
        }});
        
        // Bag Type Chart
        const bagTypeCtx = document.getElementById('bagTypeChart').getContext('2d');
        new Chart(bagTypeCtx, {{
            type: 'pie',
            data: {{
                labels: {json.dumps(list(stats['counting']['bag_type_counts'].keys()))},
                datasets: [{{
                    data: {json.dumps(list(stats['counting']['bag_type_counts'].values()))},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
        
        // Time Series Chart
        const tsData = {json.dumps(stats['time_series'])};
        const tsLabels = tsData.map(d => d.timestamp.substring(11, 16)); // HH:MM
        const tsCtx = document.getElementById('timeSeriesChart').getContext('2d');
        new Chart(tsCtx, {{
            type: 'line',
            data: {{
                labels: tsLabels,
                datasets: [{{
                    label: 'Errors',
                    data: tsData.map(d => d.errors),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    yAxisID: 'y'
                }}, {{
                    label: 'Warnings',
                    data: tsData.map(d => d.warnings),
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.1)',
                    yAxisID: 'y'
                }}, {{
                    label: 'Unknown Classifications',
                    data: tsData.map(d => d.unknown_classifications),
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    yAxisID: 'y'
                }}, {{
                    label: 'Avg FPS',
                    data: tsData.map(d => d.avg_fps),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Count'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'FPS'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get time range
    start_time, end_time = get_time_range(args)
    print(f"Analyzing logs from {start_time.isoformat()} to {end_time.isoformat()}")
    
    # Discover log files
    log_files = discover_log_files(args.log_dir)
    print(f"Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  - {f}")
    
    # Initialize analyzer
    analyzer = LogAnalyzer(start_time, end_time)
    
    # Stream and analyze entries
    print("\nAnalyzing log entries...")
    for entry, file_path, line_num, is_valid in stream_log_entries(log_files, start_time, end_time):
        if is_valid:
            analyzer.analyze_entry(entry)
        else:
            analyzer.skipped_entries += 1
    
    print(f"Processed {analyzer.total_entries} entries (skipped {analyzer.skipped_entries} malformed lines)")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = analyzer.compute_statistics()
    
    # Create output directory
    output_dir = Path(args.output) / start_time.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML report
    html_path = output_dir / "report.html"
    print(f"\nGenerating HTML report: {html_path}")
    generate_html_report(stats, str(html_path))
    
    # Also save JSON summary
    json_path = output_dir / "summary.json"
    print(f"Generating JSON summary: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print("\n✅ Analysis complete!")
    print(f"   HTML Report: {html_path}")
    print(f"   JSON Summary: {json_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Entries: {stats['parsing']['total_entries']:,}")
    print(f"Errors: {stats['errors']['total']}")
    print(f"Warnings: {stats['warnings']['total']}")
    print(f"Bags Counted: {stats['counting']['count_update']}")
    print(f"Unknown Rate: {stats['classification']['unknown_rate']:.1%}")
    if stats['fps']['count'] > 0:
        print(f"Average FPS: {stats['fps']['avg']:.1f}")
    print(f"\nIssues Detected: {len(stats['issues'])}")
    for issue in stats['issues']:
        print(f"  [{issue['severity'].upper()}] {issue['title']}")
    print("="*60)


if __name__ == "__main__":
    main()
