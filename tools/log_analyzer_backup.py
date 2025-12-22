#!/usr/bin/env python3
"""
Windows-friendly Log Analyzer for BreadBagCounterSystem

Parses rotated JSON log files and generates per-day HTML reports with actionable diagnostics.
"""

import argparse
import json
import os
import sys
import glob
import re
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
        help="Analyze logs for specific day (YYYY-MM-DD, UTC).Defaults to today."
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
    """Determine the time range to analyze based on arguments."""
    if args.from_time and args.to_time:
        start = datetime.fromisoformat(args.from_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(args.to_time.replace('Z', '+00:00'))
    elif args.day:
        day = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start = day
        end = day + timedelta(days=1, microseconds=-1)
    else:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = today
        end = today + timedelta(days=1, microseconds=-1)

    return start, end


def discover_log_files(log_dir: str) -> List[str]:
    """Discover all relevant log files in the directory."""
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    log_files = []
    main_log = log_dir_path / "app.json.log"
    if main_log.exists():
        log_files.append(str(main_log))

    rotated_logs = sorted(glob.glob(str(log_dir_path / "app.json.log.*")))
    log_files.extend(rotated_logs)

    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    return log_files


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line (JSON or text format)."""
    line = line.strip()
    if not line:
        return None

    # Try JSON first
    try:
        entry = json.loads(line)
        return entry
    except json.JSONDecodeError:
        pass
    
    # Fall back to regex parsing for text format logs
    return parse_text_log_line(line)


def parse_text_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse text format log line using regex patterns."""
    # Pattern: 2025-12-18 07:55:39.043 | INFO | BreadCounter | [MESSAGE] ...
    log_pattern = r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(.+)$'
    match = re.match(log_pattern, line)
    
    if not match:
        return None
    
    timestamp_str, level, logger, message = match.groups()
    
    # Convert timestamp to ISO format
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)
        timestamp_iso = dt.isoformat()
    except ValueError:
        return None
    
    # Extract data fields from message using patterns
    data = {}
    component = "Unknown"
    
    # Extract frame data: [FRAME] id=22740, detect=50.1ms, monitor=1.1ms, total=52.3ms, dets=5, ready=0
    if "[FRAME]" in message:
        component = "BagCounterApp"
        frame_pattern = r'id=(\d+).*?detect=([\d.]+)ms.*?monitor=([\d.]+)ms.*?total=([\d.]+)ms.*?dets=(\d+).*?ready=(\d+)'
        m = re.search(frame_pattern, message)
        if m:
            data = {
                "frame_id": int(m.group(1)),
                "detection_time_ms": float(m.group(2)),
                "monitor_time_ms": float(m.group(3)),
                "total_time_ms": float(m.group(4)),
                "detections_count": int(m.group(5)),
                "events_ready": int(m.group(6))
            }
    
    # Extract event created: [EVENT_CREATED] id=3390011057, conf=0.418, frame=22731
    elif "[EVENT_CREATED]" in message:
        component = "EventCentricTracker"
        event_pattern = r'id=(\d+).*?conf=([\d.]+).*?frame=(\d+)'
        m = re.search(event_pattern, message)
        if m:
            data = {
                "event_id": int(m.group(1)),
                "confidence": float(m.group(2)),
                "frame_id": int(m.group(3))
            }
    
    # Extract event expired: [EVENT_EXPIRED] id=3390011057, state=OPEN, frames=1, open_hits=1, closed_hits=0, idle=40
    elif "[EVENT_EXPIRED]" in message:
        component = "EventCentricTracker"
        exp_pattern = r'id=(\d+).*?state=(\w+).*?frames=(\d+).*?open_hits=(\d+).*?closed_hits=(\d+).*?idle=(\d+)'
        m = re.search(exp_pattern, message)
        if m:
            data = {
                "event_id": int(m.group(1)),
                "state": m.group(2),
                "frames_tracked": int(m.group(3)),
                "open_hits": int(m.group(4)),
                "closed_hits": int(m.group(5)),
                "idle_frames": int(m.group(6))
            }
    
    # Extract forced close: [EVENT_FORCED_CLOSE]
    elif "[EVENT_FORCED_CLOSE]" in message:
        component = "EventCentricTracker"
        forced_pattern = r'id=(\d+).*?state=(\w+).*?reason=([\w_]+)'
        m = re.search(forced_pattern, message)
        if m:
            data = {
                "event_id": int(m.group(1)),
                "state": m.group(2),
                "forced_close_reason": m.group(3)
            }
    
    # Extract classification: [CLASSIFICATION]
    elif "[CLASSIFICATION]" in message or "CLASSIFICATION" in message:
        component = "ClassifierService"
        # Pattern varies, extract what we can
        conf_pattern = r'conf(?:idence)?[=:]?\s*([\d.]+)'
        label_pattern = r'label[=:]?\s*(\w+)'
        m_conf = re.search(conf_pattern, message)
        m_label = re.search(label_pattern, message)
        if m_conf:
            data["confidence"] = float(m_conf.group(1))
        if m_label:
            data["label"] = m_label.group(1)
    
    # Extract count update: [COUNT_UPDATE]
    elif "[COUNT_UPDATE]" in message:
        component = "BagCounterApp"
        count_pattern = r'track[_-]?id[=:]?\s*(\d+)'
        bag_pattern = r'bag[_-]?type[=:]?\s*(\w+)'
        m_track = re.search(count_pattern, message)
        m_bag = re.search(bag_pattern, message)
        if m_track:
            data["track_id"] = int(m_track.group(1))
        if m_bag:
            data["bag_type"] = m_bag.group(1)
    
    # Extract queue stats: [QueueStats]
    elif "[QueueStats]" in message:
        component = "BagCounterApp"
        queue_pattern = r'Input:\s*(\d+)/(\d+).*?drops=(\d+).*?Classification:\s*(\d+)/(\d+).*?drops=(\d+)'
        m = re.search(queue_pattern, message)
        if m:
            data = {
                "input_queue_size": int(m.group(1)),
                "input_queue_capacity": int(m.group(2)),
                "input_drops": int(m.group(3)),
                "classification_queue_size": int(m.group(4)),
                "classification_queue_capacity": int(m.group(5)),
                "classification_drops": int(m.group(6))
            }
    
    # Extract KPI alerts: [PipelineMetrics]
    elif "[PipelineMetrics]" in message:
        component = "PipelineMetrics"
        # PipelineMetrics messages are complex summaries - we track them via level (WARNING/INFO)
    
    # Extract ROI added: [ROI_ADDED]
    elif "[ROI_ADDED]" in message:
        component = "EventCentricTracker"
        sharp_pattern = r'sharpness[=:]?\s*([\d.]+)'
        m = re.search(sharp_pattern, message)
        if m:
            data["sharpness"] = float(m.group(1))
    
    # Extract ROI rejected: [ROI_REJECTED]
    elif "[ROI_REJECTED]" in message:
        component = "EventCentricTracker"
        reason_pattern = r'reason[=:]?\s*(\w+)'
        m = re.search(reason_pattern, message)
        if m:
            data["reason"] = m.group(1)
    
    # Extract state transition: [STATE_TRANSITION]
    elif "[STATE_TRANSITION]" in message:
        component = "BagStateMonitor"
        trans_pattern = r'id=(\d+).*?(\w+)\s*->\s*(\w+).*?trigger=([^,\(]+)'
        m = re.search(trans_pattern, message)
        if m:
            data = {
                "event_id": int(m.group(1)),
                "old_state": m.group(2),
                "new_state": m.group(3),
                "trigger": m.group(4).strip()
            }
    
    # Extract label reuse: [LABEL_REUSE]
    elif "[LABEL_REUSE]" in message:
        component = "ClassifierService"
        reuse_pattern = r'track=(\d+).*?prev=(\w+).*?new=(\w+)\(([\d.]+)\).*?streak=(\d+)'
        m = re.search(reuse_pattern, message)
        if m:
            data = {
                "track_id": int(m.group(1)),
                "prev_label": m.group(2),
                "new_label": m.group(3),
                "new_confidence": float(m.group(4)),
                "streak_len": int(m.group(5))
            }
            # Extract dominance if present
            dom_pattern = r'dom=(\w+)\(([\d.]+)\)'
            m_dom = re.search(dom_pattern, message)
            if m_dom:
                data["dominance_label"] = m_dom.group(1)
                data["dominance_ratio"] = float(m_dom.group(2))
    
    # Extract high volatility: [HIGH_VOLATILITY]
    elif "[HIGH_VOLATILITY]" in message:
        component = "ClassifierService"
        vol_pattern = r'track=(\d+).*?changes=(\d+).*?lifespan=(\d+).*?volatility=([\d.]+)'
        m = re.search(vol_pattern, message)
        if m:
            data = {
                "track_id": int(m.group(1)),
                "label_changes": int(m.group(2)),
                "lifespan": int(m.group(3)),
                "volatility_score": float(m.group(4))
            }
    
    return {
        "timestamp": timestamp_iso,
        "level": level,
        "logger": logger,
        "message": message,
        "component": component,
        "data": data
    }


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO8601 timestamp string to timezone-aware datetime."""
    try:
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
    """Stream log entries from all files that fall within the time range."""
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    entry = parse_log_line(line)
                    if entry is None:
                        yield (None, log_file, line_num, False)
                        continue

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
        self.errors = []
        self.warnings = []
        self.error_counts = Counter()
        self.warning_counts = Counter()

        # Pipeline errors
        self.pipeline_errors = []
        self.pipeline_error_groups = Counter()

        # Backpressure metrics
        self.backpressure_events = []
        self.total_drops = 0
        self.total_frames_skipped = 0
        self.queue_utilization_samples = []

        # Frame performance
        self.frame_times = []
        self.detection_times = []
        self.monitor_times = []
        self.fps_values = []
        self.frame_indices = []

        # Event lifecycle tracking (frame-based)
        self.event_created_count = 0
        self.event_committed_count = 0
        self.event_expired_count = 0
        self.event_expired_by_state = Counter()
        self.event_suppressed_count = 0
        self.count_update_count = 0
        
        # Event lifetime analysis (frame and time based)
        self.event_lifetimes_frames = []
        self.event_lifetimes_ms = []
        self.event_states_at_close = Counter()
        
        # Frame-based threshold tracking
        self.ghost_timeout_frames_observed = []
        self.temporal_cooldown_frames_observed = []
        self.suppression_duration_frames_observed = []
        
        # Event creation blockers
        self.event_creation_blockers = {
            "covered_by_active_event": 0,
            "suppression_spatial": 0,
            "suppression_temporal": 0,
            "active_event_exclusion": 0,
        }
        self.suppression_distances = []
        self.cooldown_times_ms = []
        self.suppression_frames_remaining = []

        # Track statistics
        self.track_created_count = 0
        self.track_duplicate_count = 0
        self.track_lifetime_frames = []
        self.track_expired_count = 0
        
        # Overcount indicators
        self.count_update_track_ids = []
        self.count_update_phashes = []

        # Classification quality
        self.classification_count = 0
        self.unknown_count = 0
        self.rejection_reasons = Counter()
        self.confidence_values = []
        self.bag_type_counts = Counter()
        self.candidates_count = []
        self.voting_used_count = 0
        self.classification_times_ms = []
        
        # Enhanced classification reliability tracking
        self.per_label_confidences = defaultdict(list)  # label -> [confidence values]
        self.per_label_counts = Counter()  # label -> count
        self.low_confidence_by_label = Counter()  # label -> count (conf < 0.7)
        self.confusion_pairs = Counter()  # (prev_label, new_label) -> count
        self.track_label_history = defaultdict(list)  # track_id -> [(frame, label, conf)]
        
        # Streak tracking and burst anomaly detection
        self.classification_streaks = []  # [(label, start_frame, end_frame, count, min_conf)]
        self.current_streak = None  # (label, start_frame, count, confidences)
        self.burst_anomalies = []  # low-confidence flips on long streaks
        
        # Minute-level dominant label analysis
        self.minute_label_distribution = defaultdict(Counter)  # minute_key -> {label: count}
        
        # V6: Stability heuristics tracking
        self.label_reuse_count = 0
        self.label_reuse_events = []  # detailed reuse events
        self.high_volatility_tracks = []  # tracks with high volatility
        self.volatility_scores = []  # all volatility scores
        
        # V8: Probability adjustment tracking
        self.prob_adjustment_count = 0
        self.prob_adjustment_applied = 0
        self.prob_adjustment_samples = []  # sample adjustments for analysis
        
        # V8: Evidence accumulation tracking
        self.evidence_accumulation_used_count = 0
        self.gate_passed_count = 0
        self.gate_failed_count = 0
        self.gate_failure_reasons = Counter()
        self.trust_stats_samples = []  # sample trust stats for analysis
        self.inertia_applied_count = 0
        
        # V8: Disambiguation tracking
        self.disambiguation_applied_count = 0
        self.disambiguation_samples = []  # sample disambiguation details
        
        # Forced closes and lifecycle details
        self.forced_close_count = 0
        self.forced_close_reasons = Counter()
        self.idle_commit_count = 0
        self.detection_gap_closures = []
        self.expiry_details = []  # detailed expiry info

        # ROI statistics
        self.roi_added_count = 0
        self.roi_rejected_count = 0
        self.roi_reject_reasons = Counter()
        self.roi_sharpness_values = []
        self.roi_per_event = []
        
        # System metrics
        self.app_start_time = None
        self.app_version = None
        
        # Time series (per-minute buckets)
        self.time_buckets = defaultdict(lambda: {
            "errors": 0,
            "warnings": 0,
            "backpressure_drops": 0,
            "unknown_classifications": 0,
            "fps_samples": [],
            "event_created": 0,
            "event_committed": 0,
            "event_expired": 0,
            "suppressed": 0,
            "skip_creation": 0,
            "roi_added": 0,
            "roi_rejected": 0,
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

        # App metadata tracking
        if "Initialized" in message or "AppLogging" in message:
            if not self.app_start_time:
                self.app_start_time = timestamp_str
            # Try to extract version if present
            if "version" in data:
                self.app_version = data.get("version")

        # Error tracking
        if level == "ERROR":
            self.errors.append((component, message, data))
            self.error_counts[(component, message)] += 1
            if minute_key:
                self.time_buckets[minute_key]["errors"] += 1

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
        if "BACKPRESSURE" in message:
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
        
        # Queue stats (extract drops from QueueStats message)
        # Note: QueueStats logs cumulative drops, so we only count them once per unique log entry
        if "QueueStats" in message or "[QueueStats]" in message:
            input_drops = data.get("input_drops", 0)
            class_drops = data.get("classification_drops", 0)
            # Only add drops if this appears to be a periodic stats dump (not double-counted with BACKPRESSURE events)
            if input_drops > 0 or class_drops > 0:
                self.total_drops += (input_drops + class_drops)

        # Frame performance
        if "FRAME" in message:
            frame_id = data.get("frame_id")
            if frame_id is not None:
                self.frame_indices.append(frame_id)
            
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

        # Event lifecycle tracking (frame-based)
        if "EVENT_CREATED" in message:
            self.event_created_count += 1
            if minute_key:
                self.time_buckets[minute_key]["event_created"] += 1

        if "EVENT_COMMITTED" in message:
            self.event_committed_count += 1
            # Track event lifetime
            lifespan_ms = data.get("lifespan_ms")
            if lifespan_ms is not None:
                self.event_lifetimes_ms.append(lifespan_ms)
            # Track ROI count per event
            roi_count = data.get("roi_count")
            if roi_count is not None:
                self.roi_per_event.append(roi_count)
            if minute_key:
                self.time_buckets[minute_key]["event_committed"] += 1

        if "EVENT_EXPIRED" in message:
            self.event_expired_count += 1
            state = data.get("state", "unknown")
            self.event_expired_by_state[state] += 1
            # Track lifetime
            frames_tracked = data.get("frames_tracked")
            if frames_tracked:
                self.event_lifetimes_frames.append(frames_tracked)
            # Detailed expiry tracking
            self.expiry_details.append({
                "event_id": data.get("event_id"),
                "state": state,
                "frames_tracked": frames_tracked,
                "reason": data.get("expiration_reason", "unknown")
            })
            if minute_key:
                self.time_buckets[minute_key]["event_expired"] += 1
        
        # Forced close tracking
        if "EVENT_FORCED_CLOSE" in message or "FORCED_CLOSE" in message:
            self.forced_close_count += 1
            reason = data.get("forced_close_reason", data.get("reason", "unknown"))
            self.forced_close_reasons[reason] += 1
        
        # Idle commit tracking (from "idle threshold" or similar messages)
        if "idle" in message.lower() and "commit" in message.lower():
            self.idle_commit_count += 1
        
        # Detection gap closure
        if "Detection gap closed" in message or "gap closed" in message.lower():
            gap_ms = data.get("gap_ms")
            if gap_ms is None:
                # Try to extract from message with more specific pattern
                gap_pattern = r'gap[^:]*:\s*([\d.]+)\s*ms'
                m = re.search(gap_pattern, message, re.IGNORECASE)
                if m:
                    gap_ms = float(m.group(1))
            if gap_ms:
                self.detection_gap_closures.append(gap_ms)

        if "EVENT_SUPPRESSED" in message:
            self.event_suppressed_count += 1
            reason = data.get("reason", "unknown")
            if "spatial" in reason.lower() or "distance" in reason.lower():
                self.event_creation_blockers["suppression_spatial"] += 1
                distance = data.get("center_distance")
                if distance:
                    self.suppression_distances.append(distance)
            elif "temporal" in reason.lower() or "cooldown" in reason.lower():
                self.event_creation_blockers["suppression_temporal"] += 1
            elif "overlap" in reason.lower() or "iou" in reason.lower():
                self.event_creation_blockers["covered_by_active_event"] += 1
            if minute_key:
                self.time_buckets[minute_key]["suppressed"] += 1

        # ROI tracking
        if "ROI_ADDED" in message:
            self.roi_added_count += 1
            sharpness = data.get("sharpness")
            if sharpness is not None:
                self.roi_sharpness_values.append(sharpness)
            if minute_key:
                self.time_buckets[minute_key]["roi_added"] += 1
        
        if "ROI_REJECTED" in message:
            self.roi_rejected_count += 1
            reason = data.get("reason", "unknown")
            self.roi_reject_reasons[reason] += 1
            if minute_key:
                self.time_buckets[minute_key]["roi_rejected"] += 1

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
            candidates = data.get("candidates")
            used_voting = data.get("used_voting")
            processing_time = data.get("processing_time_ms")
            track_id = data.get("track_id")
            frame_id = data.get("frame_id")
            
            # V8: Extract metadata (structured field in classification logs)
            metadata = data.get("metadata", {})
            
            # V8: Track evidence accumulation usage
            evidence_used = metadata.get("evidence_accumulation_used", False)
            if evidence_used:
                self.evidence_accumulation_used_count += 1
                
                # Track gate passing
                gate_passed = metadata.get("gate_passed")
                if gate_passed is True:
                    self.gate_passed_count += 1
                elif gate_passed is False:
                    self.gate_failed_count += 1
                    gate_reason = metadata.get("gate_failure_reason")
                    if gate_reason:
                        self.gate_failure_reasons[gate_reason] += 1
                
                # Track trust stats
                trust_stats = metadata.get("trust_stats")
                if trust_stats and len(self.trust_stats_samples) < 100:
                    self.trust_stats_samples.append({
                        "track_id": track_id,
                        "label": label,
                        "trust_stats": trust_stats,
                        "rois_trusted": metadata.get("rois_trusted", 0)
                    })
                
                # Track inertia/class-switch penalty
                if metadata.get("class_switch_penalty_applied"):
                    self.inertia_applied_count += 1
            
            # V8: Track probability adjustments
            if metadata.get("probability_adjustment_applied"):
                self.prob_adjustment_applied += 1
                self.prob_adjustment_count += metadata.get("probability_adjustment_count", 0)
                
                # Store sample adjustments for analysis
                samples = metadata.get("probability_adjustment_samples", [])
                for sample in samples:
                    if len(self.prob_adjustment_samples) < 50:
                        self.prob_adjustment_samples.append({
                            "track_id": track_id,
                            "label": label,
                            **sample
                        })
            
            # V8: Track disambiguation
            if metadata.get("disambiguation_applied"):
                self.disambiguation_applied_count += 1
                self.disambiguation_count_val = metadata.get("disambiguation_count", 0)
                
                # Store sample disambiguation details
                if len(self.disambiguation_samples) < 50:
                    self.disambiguation_samples.append({
                        "track_id": track_id,
                        "label": label,
                        "count": self.disambiguation_count_val
                    })

            if label == "Unknown":
                self.unknown_count += 1
                if minute_key:
                    self.time_buckets[minute_key]["unknown_classifications"] += 1
                if rejection_reason:
                    self.rejection_reasons[rejection_reason] += 1

            if confidence is not None:
                self.confidence_values.append(confidence)
                
                # Per-label confidence tracking
                if label:
                    self.per_label_confidences[label].append(confidence)
                    self.per_label_counts[label] += 1
                    
                    # Low-confidence tracking (threshold: 0.7)
                    if confidence < 0.7:
                        self.low_confidence_by_label[label] += 1
                    
                    # Track label history for confusion pairs
                    if track_id:
                        history = self.track_label_history[track_id]
                        if history:
                            prev_label = history[-1][1]
                            if prev_label != label:
                                self.confusion_pairs[(prev_label, label)] += 1
                        history.append((frame_id or 0, label, confidence))
                    
                    # Streak tracking
                    if self.current_streak and self.current_streak[0] == label:
                        # Continue current streak
                        self.current_streak = (
                            label,
                            self.current_streak[1],  # start_frame
                            self.current_streak[2] + 1,  # count
                            self.current_streak[3] + [confidence]  # confidences
                        )
                    else:
                        # End current streak and start new one
                        if self.current_streak:
                            label_s, start_f, count, confs = self.current_streak
                            min_conf = min(confs) if confs else 0
                            end_f = frame_id or start_f + count
                            self.classification_streaks.append((label_s, start_f, end_f, count, min_conf))
                            
                            # Detect burst anomaly: long streak with low-confidence flip
                            if count >= 10 and min_conf < 0.7:
                                self.burst_anomalies.append({
                                    "label": label_s,
                                    "start_frame": start_f,
                                    "end_frame": end_f,
                                    "count": count,
                                    "min_confidence": min_conf
                                })
                        
                        self.current_streak = (label, frame_id or 0, 1, [confidence])
                    
                    # Minute-level label distribution
                    if minute_key:
                        self.minute_label_distribution[minute_key][label] += 1
            
            if candidates is not None:
                self.candidates_count.append(candidates)
            
            if used_voting:
                self.voting_used_count += 1
            
            if processing_time:
                self.classification_times_ms.append(processing_time)

        # Classification candidate details
        if "CANDIDATE" in message:
            # Track individual candidate contributions
            pass  # Already captured in classification aggregates
        
        # V6: Label reuse tracking
        if "LABEL_REUSE" in message:
            self.label_reuse_count += 1
            if data:
                self.label_reuse_events.append({
                    "track_id": data.get("track_id"),
                    "prev_label": data.get("prev_label"),
                    "new_label": data.get("new_label"),
                    "new_confidence": data.get("new_confidence"),
                    "streak_len": data.get("streak_len"),
                    "dominance_label": data.get("dominance_label"),
                    "dominance_ratio": data.get("dominance_ratio")
                })
        
        # V6: High volatility tracking
        if "HIGH_VOLATILITY" in message:
            if data:
                volatility_score = data.get("volatility_score")
                if volatility_score is not None:
                    self.volatility_scores.append(volatility_score)
                
                self.high_volatility_tracks.append({
                    "track_id": data.get("track_id"),
                    "label_changes": data.get("label_changes"),
                    "lifespan": data.get("lifespan"),
                    "volatility_score": volatility_score
                })
        
        # Track statistics (if we add track-level logging)
        # For now, track_id in COUNT_UPDATE gives us track lifecycle info

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics and metrics with frame-based threshold analysis."""
        
        # Compute average FPS for frame-to-time conversions
        avg_fps = statistics.mean(self.fps_values) if self.fps_values else 25.0
        
        stats = {
            "metadata": {
                "app_start_time": self.app_start_time,
                "app_version": self.app_version or "unknown",
                "report_generated": datetime.now(timezone.utc).isoformat(),
            },
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_hours": (self.end_time - self.start_time).total_seconds() / 3600,
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "parsing": {
                "total_entries": self.total_entries,
                "skipped_entries": self.skipped_entries,
                "entries_by_level": dict(self.entries_by_level),
                "entries_by_component": dict(self.entries_by_component)
            },
            "errors": {
                "total": len(self.errors),
                "top_errors": [(f"{comp}::{msg}", count) for (comp, msg), count in self.error_counts.most_common(10)],
                "pipeline_error_groups": [(f"{comp}/{op}/{etype}", count) for (comp, op, etype), count in self.pipeline_error_groups.most_common(10)]
            },
            "warnings": {
                "total": len(self.warnings),
                "top_warnings": [(f"{comp}::{msg}", count) for (comp, msg), count in self.warning_counts.most_common(10)]
            },
            "backpressure": {
                "total_events": len(self.backpressure_events),
                "total_drops": self.total_drops,
                "total_frames_skipped": self.total_frames_skipped,
                "avg_queue_utilization": statistics.mean(self.queue_utilization_samples) if self.queue_utilization_samples else 0,
                "max_queue_utilization": max(self.queue_utilization_samples) if self.queue_utilization_samples else 0
            },
            "frame_performance": self._compute_percentile_stats(self.frame_times, "total_time_ms"),
            "detection_performance": self._compute_percentile_stats(self.detection_times, "detection_time_ms"),
            "monitor_performance": self._compute_percentile_stats(self.monitor_times, "monitor_time_ms"),
            "fps": self._compute_percentile_stats(self.fps_values, "fps"),
            "events": {
                "total_created": self.event_created_count,
                "total_committed": self.event_committed_count,
                "total_expired": self.event_expired_count,
                "expired_by_state": dict(self.event_expired_by_state),
                "total_suppressed": self.event_suppressed_count,
                "avg_lifetime_frames": statistics.mean(self.event_lifetimes_frames) if self.event_lifetimes_frames else 0,
                "avg_lifetime_ms": statistics.mean(self.event_lifetimes_ms) if self.event_lifetimes_ms else 0,
                "avg_lifetime_seconds": statistics.mean(self.event_lifetimes_ms) / 1000.0 if self.event_lifetimes_ms else 0,
                "lifetime_frames_stats": self._compute_percentile_stats(self.event_lifetimes_frames, "lifetime_frames"),
                "lifetime_ms_stats": self._compute_percentile_stats(self.event_lifetimes_ms, "lifetime_ms"),
            },
            "counting": {
                "total_bags_counted": self.count_update_count,
                "bag_type_distribution": dict(self.bag_type_counts)
            },
            "event_creation_blockers": {
                "total_blocked": sum(self.event_creation_blockers.values()),
                "by_reason": dict(self.event_creation_blockers),
                "suppression_distance_stats": self._compute_percentile_stats(self.suppression_distances, "suppression_distance_px"),
                "cooldown_time_ms_stats": self._compute_percentile_stats(self.cooldown_times_ms, "cooldown_time_ms"),
            },
            "frame_based_thresholds": {
                "note": "System uses frame-based thresholds. Typical values @ 25fps:",
                "ghost_timeout_frames": "25 frames (1000ms)",
                "temporal_cooldown_frames": "10 frames (400ms)",
                "suppression_duration_frames": "38 frames (1520ms)",
                "avg_fps": avg_fps,
                "frame_to_ms_conversion": f"1 frame = {1000.0/avg_fps:.1f}ms @ {avg_fps:.1f}fps"
            },
            "tracks": {
                "total_created": self.track_created_count,
                "total_duplicates": self.track_duplicate_count,
                "total_expired": self.track_expired_count,
                "avg_lifetime_frames": statistics.mean(self.track_lifetime_frames) if self.track_lifetime_frames else 0,
                "unique_tracks_counted": len(set(self.count_update_track_ids)),
                "duplicate_track_ids": self._find_duplicates(self.count_update_track_ids),
                "duplicate_phashes": self._find_duplicates(self.count_update_phashes)
            },
            "classification": {
                "total": self.classification_count,
                "unknown_count": self.unknown_count,
                "unknown_rate": self.unknown_count / self.classification_count if self.classification_count > 0 else 0,
                "rejection_reasons": dict(self.rejection_reasons),
                "confidence_stats": self._compute_percentile_stats(self.confidence_values, "confidence"),
                "avg_candidates_per_classification": statistics.mean(self.candidates_count) if self.candidates_count else 0,
                "voting_used_count": self.voting_used_count,
                "voting_rate": self.voting_used_count / self.classification_count if self.classification_count > 0 else 0,
                "avg_processing_time_ms": statistics.mean(self.classification_times_ms) if self.classification_times_ms else 0,
                # Enhanced reliability metrics
                "per_label_stats": self._compute_per_label_stats(),
                "low_confidence_rate_by_label": {
                    label: self.low_confidence_by_label[label] / self.per_label_counts[label]
                    for label in self.per_label_counts.keys()
                },
                "confusion_pairs": dict(self.confusion_pairs.most_common(20)),
                "top_label_flips": [(f"{l1}→{l2}", count) for (l1, l2), count in self.confusion_pairs.most_common(10)],
                # V6: Stability heuristics
                "stability_heuristics": {
                    "label_reuse_count": self.label_reuse_count,
                    "label_reuse_rate": self.label_reuse_count / self.classification_count if self.classification_count > 0 else 0,
                    "label_reuse_events": self.label_reuse_events[:20],  # Top 20 for report
                    "high_volatility_tracks": len(self.high_volatility_tracks),
                    "avg_volatility": statistics.mean(self.volatility_scores) if self.volatility_scores else 0,
                    "max_volatility": max(self.volatility_scores) if self.volatility_scores else 0,
                    "volatility_details": self.high_volatility_tracks[:20],  # Top 20 for report
                },
                # V8: Evidence accumulation metrics
                "evidence_accumulation": {
                    "used_count": self.evidence_accumulation_used_count,
                    "usage_rate": self.evidence_accumulation_used_count / self.classification_count if self.classification_count > 0 else 0,
                    "gate_passed_count": self.gate_passed_count,
                    "gate_failed_count": self.gate_failed_count,
                    "gate_pass_rate": self.gate_passed_count / self.evidence_accumulation_used_count if self.evidence_accumulation_used_count > 0 else 0,
                    "gate_failure_reasons": dict(self.gate_failure_reasons),
                    "inertia_applied_count": self.inertia_applied_count,
                    "inertia_rate": self.inertia_applied_count / self.evidence_accumulation_used_count if self.evidence_accumulation_used_count > 0 else 0,
                    "trust_stats_samples": self.trust_stats_samples[:10],  # Top 10 samples
                },
                # V8: Disambiguation metrics
                "disambiguation": {
                    "applied_count": self.disambiguation_applied_count,
                    "application_rate": self.disambiguation_applied_count / self.classification_count if self.classification_count > 0 else 0,
                    "samples": self.disambiguation_samples[:10],  # Top 10 samples
                },
                # V8: Probability adjustment metrics
                "probability_adjustment": {
                    "applied_tracks": self.prob_adjustment_applied,
                    "total_adjustments": self.prob_adjustment_count,
                    "application_rate": self.prob_adjustment_applied / self.classification_count if self.classification_count > 0 else 0,
                    "samples": self.prob_adjustment_samples[:10],  # Top 10 samples
                },
            },
            "streak_analysis": {
                "total_streaks": len(self.classification_streaks),
                "burst_anomalies": self.burst_anomalies,
                "avg_streak_length": statistics.mean([s[3] for s in self.classification_streaks]) if self.classification_streaks else 0,
                "longest_streak": max([s[3] for s in self.classification_streaks], default=0),
            },
            "minute_level_analysis": self._analyze_minute_level_labels(),
            "lifecycle_details": {
                "forced_closes": {
                    "total": self.forced_close_count,
                    "by_reason": dict(self.forced_close_reasons),
                },
                "idle_commits": self.idle_commit_count,
                "detection_gap_closures": {
                    "total": len(self.detection_gap_closures),
                    "avg_gap_ms": statistics.mean(self.detection_gap_closures) if self.detection_gap_closures else 0,
                    "max_gap_ms": max(self.detection_gap_closures) if self.detection_gap_closures else 0,
                },
                "expiry_details": {
                    "total": len(self.expiry_details),
                    "by_state": Counter([e["state"] for e in self.expiry_details]),
                },
            },
            "roi": {
                "total_added": self.roi_added_count,
                "total_rejected": self.roi_rejected_count,
                "rejection_rate": self.roi_rejected_count / (self.roi_added_count + self.roi_rejected_count) if (self.roi_added_count + self.roi_rejected_count) > 0 else 0,
                "reject_reasons": dict(self.roi_reject_reasons),
                "avg_sharpness": statistics.mean(self.roi_sharpness_values) if self.roi_sharpness_values else 0,
                "sharpness_stats": self._compute_percentile_stats(self.roi_sharpness_values, "roi_sharpness"),
                "avg_rois_per_event": statistics.mean(self.roi_per_event) if self.roi_per_event else 0,
            },
            "time_series": self._compute_time_series(),
            "risk_heuristics": self._compute_risk_heuristics(),
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
    
    def _compute_per_label_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute per-label confidence statistics."""
        stats = {}
        for label, confidences in self.per_label_confidences.items():
            if confidences:
                stats[label] = {
                    "count": len(confidences),
                    "avg_confidence": statistics.mean(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                    "p50_confidence": sorted(confidences)[len(confidences) // 2],
                }
        return stats
    
    def _analyze_minute_level_labels(self) -> Dict[str, Any]:
        """Analyze minute-level label distributions to detect out-of-pattern labels."""
        analysis = {
            "total_minutes": len(self.minute_label_distribution),
            "dominant_label_per_minute": {},
            "out_of_pattern_labels": [],
        }
        
        # Compute dominant label per minute
        for minute_key, label_counts in self.minute_label_distribution.items():
            if label_counts:
                dominant_label = label_counts.most_common(1)[0]
                analysis["dominant_label_per_minute"][minute_key] = {
                    "label": dominant_label[0],
                    "count": dominant_label[1],
                    "total": sum(label_counts.values()),
                    "percentage": dominant_label[1] / sum(label_counts.values()) * 100
                }
        
        # Detect out-of-pattern: if we see a different label in a minute where one label dominates
        # (e.g., in single-variant runs)
        for minute_key, label_counts in self.minute_label_distribution.items():
            total = sum(label_counts.values())
            if total > 5:  # Only consider minutes with enough samples
                dominant = label_counts.most_common(1)[0]
                if dominant[1] / total > 0.9:  # If one label is >90% dominant
                    # Check for outlier labels
                    for label, count in label_counts.items():
                        if label != dominant[0] and count > 0:
                            analysis["out_of_pattern_labels"].append({
                                "minute": minute_key,
                                "dominant_label": dominant[0],
                                "dominant_count": dominant[1],
                                "outlier_label": label,
                                "outlier_count": count,
                            })
        
        return analysis
    
    def _compute_risk_heuristics(self) -> Dict[str, Any]:
        """Compute risk heuristics for undercount and overcount."""
        undercount_risk = 0
        overcount_risk = 0
        risk_factors = []
        
        # Undercount risk factors
        total_potential_events = self.event_created_count + sum(self.event_creation_blockers.values())
        if total_potential_events > 0:
            suppression_rate = sum([
                self.event_creation_blockers.get("suppression_spatial", 0),
                self.event_creation_blockers.get("suppression_temporal", 0)
            ]) / total_potential_events
            
            if suppression_rate > 0.10:
                undercount_risk += 30
                risk_factors.append(f"High suppression rate: {suppression_rate:.1%}")
        
        # High expiry rate (events not committed)
        if self.event_created_count > 0:
            expiry_rate = self.event_expired_count / self.event_created_count
            if expiry_rate > 0.10:
                undercount_risk += 20
                risk_factors.append(f"High expiry rate: {expiry_rate:.1%}")
        
        # Queue drops (missed frames)
        if self.total_drops > 0:
            undercount_risk += min(30, self.total_drops)
            risk_factors.append(f"Frame drops: {self.total_drops}")
        
        # Forced closes (potential missed counts)
        if self.forced_close_count > 0:
            undercount_risk += min(20, self.forced_close_count * 2)
            risk_factors.append(f"Forced closes: {self.forced_close_count}")
        
        # Overcount risk factors
        
        # Duplicate track IDs
        duplicate_tracks = self._find_duplicates(self.count_update_track_ids)
        if duplicate_tracks:
            overcount_risk += min(30, len(duplicate_tracks) * 5)
            risk_factors.append(f"Duplicate track IDs: {len(duplicate_tracks)}")
        
        # Duplicate phashes
        duplicate_phashes = self._find_duplicates(self.count_update_phashes)
        if duplicate_phashes:
            overcount_risk += min(20, len(duplicate_phashes) * 3)
            risk_factors.append(f"Duplicate phashes: {len(duplicate_phashes)}")
        
        # Low average event lifetime (possible double-counting)
        avg_lifetime_frames = statistics.mean(self.event_lifetimes_frames) if self.event_lifetimes_frames else 0
        if avg_lifetime_frames > 0 and avg_lifetime_frames < 10:
            overcount_risk += 15
            risk_factors.append(f"Very short avg event lifetime: {avg_lifetime_frames:.1f} frames")
        
        # High classification unknown rate (misclassification risk)
        if self.classification_count > 0:
            unknown_rate = self.unknown_count / self.classification_count
            if unknown_rate > 0.15:
                # This affects both under and over count
                undercount_risk += 10
                overcount_risk += 10
                risk_factors.append(f"High unknown classification rate: {unknown_rate:.1%}")
        
        # Confusion pairs (label flips indicate instability)
        if len(self.confusion_pairs) > 10:
            overcount_risk += 10
            undercount_risk += 5
            risk_factors.append(f"High label confusion: {len(self.confusion_pairs)} flip patterns")
        
        return {
            "undercount_risk_score": min(100, undercount_risk),
            "overcount_risk_score": min(100, overcount_risk),
            "risk_factors": risk_factors,
            "risk_level": self._assess_risk_level(undercount_risk, overcount_risk),
        }
    
    def _assess_risk_level(self, undercount_risk: int, overcount_risk: int) -> str:
        """Assess overall risk level based on scores."""
        max_risk = max(undercount_risk, overcount_risk)
        if max_risk >= 50:
            return "HIGH"
        elif max_risk >= 25:
            return "MEDIUM"
        else:
            return "LOW"

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
                "avg_fps": avg_fps,
                "event_created": bucket["event_created"],
                "event_committed": bucket["event_committed"],
                "event_expired": bucket["event_expired"],
                "suppressed": bucket["suppressed"],
                "skip_creation": bucket["skip_creation"],
                "roi_added": bucket["roi_added"],
                "roi_rejected": bucket["roi_rejected"]
            })
        return series

    def _detect_issues(self) -> List[Dict[str, Any]]:
        """Detect issues based on thresholds and return actionable findings (frame-based)."""
        issues = []
        
        avg_fps = statistics.mean(self.fps_values) if self.fps_values else 25.0

        # Event suppression analysis (frame-based)
        total_potential_events = self.event_created_count + sum(self.event_creation_blockers.values())

        if total_potential_events > 0:
            total_suppressed = (self.event_creation_blockers.get("suppression_spatial", 0) +
                               self.event_creation_blockers.get("suppression_temporal", 0))
            suppression_rate = (total_suppressed / total_potential_events * 100)

            if suppression_rate > 5:
                # Convert frame thresholds to ms for recommendations
                temporal_cooldown_frames = 10  # Default from config
                temporal_cooldown_ms = temporal_cooldown_frames * (1000.0 / avg_fps)
                
                issues.append({
                    "severity": "warning",
                    "title": "High Event Suppression Rate (Potential Undercounting)",
                    "description": f"Suppression rate: {suppression_rate:.1f}% ({total_suppressed} of {total_potential_events} detections suppressed)",
                    "likely_cause": f"Temporal cooldown ({temporal_cooldown_frames} frames @ {avg_fps:.1f}fps = {temporal_cooldown_ms:.0f}ms) may be too aggressive",
                    "where_to_look": f"Review temporal_cooldown_frames in config. Current: ~{temporal_cooldown_frames} frames. Consider reducing to 5-8 frames for faster workflows.",
                    "recommendation": {
                        "temporal_cooldown_frames_current": temporal_cooldown_frames,
                        "temporal_cooldown_frames_recommended": "5-8",
                        "suppression_distance_px_current": 100,
                        "suppression_distance_px_recommended": "80-100",
                        "note": "Frame-based thresholds scale naturally with FPS"
                    }
                })

        # High unknown rate
        if self.classification_count > 0:
            unknown_rate = self.unknown_count / self.classification_count
            if unknown_rate > 0.10:
                top_reasons = dict(Counter(self.rejection_reasons).most_common(3))
                issues.append({
                    "severity": "warning",
                    "title": "High Unknown Classification Rate",
                    "description": f"Unknown rate is {unknown_rate:.1%} ({self.unknown_count}/{self.classification_count} classifications)",
                    "likely_cause": "Poor ROI quality, model uncertainty, or inadequate training data",
                    "where_to_look": f"Check rejection_reasons: {top_reasons}. Review ROI sharpness and collection quality.",
                    "recommendation": {
                        "top_rejection_reasons": top_reasons,
                        "avg_roi_sharpness": statistics.mean(self.roi_sharpness_values) if self.roi_sharpness_values else 0,
                        "roi_rejection_rate": f"{self.roi_rejected_count / (self.roi_added_count + self.roi_rejected_count) * 100:.1f}%" if (self.roi_added_count + self.roi_rejected_count) > 0 else "N/A"
                    }
                })

        # Backpressure drops
        if self.total_drops > 0:
            issues.append({
                "severity": "error",
                "title": "Frame Drops Due to Backpressure",
                "description": f"Total frames dropped: {self.total_drops}",
                "likely_cause": "System cannot keep up with input frame rate (CPU/GPU overload)",
                "where_to_look": "Check frame processing times (detection_time_ms, total_time_ms), consider reducing input FPS or optimizing models"
            })

        # High frame processing time
        if self.frame_times:
            avg_frame_time = statistics.mean(self.frame_times)
            p95_frame_time = sorted(self.frame_times)[int(len(self.frame_times) * 0.95)] if len(
                self.frame_times) > 20 else max(self.frame_times)

            if p95_frame_time > 50:
                issues.append({
                    "severity": "warning",
                    "title": "High Frame Processing Time",
                    "description": f"P95 frame time: {p95_frame_time:.1f}ms (avg: {avg_frame_time:.1f}ms)",
                    "likely_cause": "Detection or monitoring bottleneck, hardware limitations",
                    "where_to_look": "Compare detection_time_ms vs monitor_time_ms to identify bottleneck component"
                })

        # Low FPS
        if self.fps_values:
            avg_fps = statistics.mean(self.fps_values)
            if avg_fps < 20:
                issues.append({
                    "severity": "warning",
                    "title": "Low FPS Throughput",
                    "description": f"Average FPS: {avg_fps:.1f} (target: 25+)",
                    "likely_cause": "System overload, slow detection model, or hardware limitations",
                    "where_to_look": "Check backpressure events, frame processing times, and queue utilization"
                })
        
        # Forced closes
        if self.forced_close_count > 0:
            issues.append({
                "severity": "warning",
                "title": "Forced Event Closes Detected",
                "description": f"Total forced closes: {self.forced_close_count}",
                "likely_cause": "Events stuck in CLOSED state exceeding max duration",
                "where_to_look": f"Review forced_close_reasons: {dict(self.forced_close_reasons)}. Check max_closed_state_frames threshold.",
                "recommendation": {
                    "forced_close_reasons": dict(self.forced_close_reasons),
                    "note": "Forced closes may indicate premature commits or missed transitions"
                }
            })
        
        # Burst anomalies (low-confidence flips on long streaks)
        if self.burst_anomalies:
            issues.append({
                "severity": "warning",
                "title": "Burst Anomalies Detected",
                "description": f"Found {len(self.burst_anomalies)} classification streaks with low-confidence flips",
                "likely_cause": "Inconsistent classification on similar bags or poor model quality",
                "where_to_look": f"Review burst_anomalies in report. Top affected label: {self.burst_anomalies[0]['label'] if self.burst_anomalies else 'N/A'}",
                "recommendation": {
                    "burst_count": len(self.burst_anomalies),
                    "note": "Long streaks with low confidence suggest model instability"
                }
            })
        
        # High confusion (label flips)
        if len(self.confusion_pairs) > 10:
            top_confusion = self.confusion_pairs.most_common(3)
            issues.append({
                "severity": "warning",
                "title": "High Label Confusion",
                "description": f"Detected {len(self.confusion_pairs)} distinct label flip patterns",
                "likely_cause": "Model struggling to distinguish between certain bag types",
                "where_to_look": f"Top confusion pairs: {[(f'{l1}→{l2}', c) for (l1, l2), c in top_confusion]}",
                "recommendation": {
                    "top_confusion_pairs": [(f"{l1}→{l2}", c) for (l1, l2), c in top_confusion],
                    "note": "Consider retraining model with more diverse samples for confused classes"
                }
            })

        return issues


def _generate_per_label_rows(stats: Dict[str, Any]) -> str:
    """Generate HTML table rows for per-label statistics."""
    rows = []
    for label, label_stats in stats['classification']['per_label_stats'].items():
        low_conf_rate = stats['classification']['low_confidence_rate_by_label'].get(label, 0)
        row = (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td>{label_stats['count']}</td>"
            f"<td>{label_stats['avg_confidence']:.3f}</td>"
            f"<td>{label_stats['min_confidence']:.3f}</td>"
            f"<td>{low_conf_rate:.1%}</td>"
            f"</tr>"
        )
        rows.append(row)
    return ''.join(rows)


def _generate_reuse_events_table(stats: Dict[str, Any]) -> str:
    """Generate HTML table for label reuse events."""
    events = stats['classification']['stability_heuristics']['label_reuse_events'][:10]
    
    rows = []
    for evt in events:
        dom_label = evt.get('dominance_label', 'N/A')
        dom_ratio = evt.get('dominance_ratio', 0)
        row = (
            f"<tr>"
            f"<td>{evt['track_id']}</td>"
            f"<td>{evt['prev_label']}</td>"
            f"<td>{evt['new_label']}</td>"
            f"<td>{evt['new_confidence']:.3f}</td>"
            f"<td>{evt['streak_len']}</td>"
            f"<td>{dom_label} ({dom_ratio:.2f})</td>"
            f"</tr>"
        )
        rows.append(row)
    
    table = (
        '<table>'
        '<tr><th>Track ID</th><th>Prev Label</th><th>New Label</th>'
        '<th>Confidence</th><th>Streak</th><th>Dominance</th></tr>'
        f"{''.join(rows)}"
        '</table>'
    )
    return table


def _generate_volatility_table(stats: Dict[str, Any]) -> str:
    """Generate HTML table for high volatility tracks."""
    tracks = stats['classification']['stability_heuristics']['volatility_details'][:10]
    
    rows = []
    for track in tracks:
        row = (
            f"<tr>"
            f"<td>{track['track_id']}</td>"
            f"<td>{track['label_changes']}</td>"
            f"<td>{track['lifespan']}</td>"
            f"<td>{track['volatility_score']:.3f}</td>"
            f"</tr>"
        )
        rows.append(row)
    
    table = (
        '<table>'
        '<tr><th>Track ID</th><th>Label Changes</th><th>Lifespan</th><th>Volatility Score</th></tr>'
        f"{''.join(rows)}"
        '</table>'
    )
    return table


def generate_html_report(stats: Dict[str, Any], output_path: str):
    """Generate HTML report with embedded CSS and Chart.js visualizations."""

    html_content = f"""<! DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadBag Counter Log Analysis - {stats['time_range']['start'][: 10]}</title>
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
            background:  #f5f5f5;
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
            margin-bottom:  20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}

        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom:  15px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .kpi-card {{
            background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding:  20px;
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
            font-weight:  bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin:  20px 0;
            background: white;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background: #3498db;
            color:  white;
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

        .recommendation {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: monospace;
            font-size:  12px;
            border:  1px solid #b3d9e8;
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
            border-radius:  4px;
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
            <div class="kpi-card">
                <div class="kpi-label">Bags Counted</div>
                <div class="kpi-value">{stats['counting']['total_bags_counted']}</div>
            </div>
            <div class="kpi-card {'warning' if stats['event_creation_blockers']['total_blocked'] > stats['events']['total_created'] * 0.05 else 'success'}">
                <div class="kpi-label">Suppressed Events</div>
                <div class="kpi-value">{stats['event_creation_blockers']['total_blocked']}</div>
            </div>
            <div class="kpi-card {'success' if stats['fps']['count'] > 0 and stats['fps']['avg'] >= 20 else 'warning'}">
                <div class="kpi-label">Avg FPS</div>
                <div class="kpi-value">{stats['fps']['avg'] if stats['fps']['count'] > 0 else 0:.1f}</div>
            </div>
            <div class="kpi-card {'warning' if stats['classification']['unknown_rate'] > 0.1 else 'success'}">
                <div class="kpi-label">Unknown Rate</div>
                <div class="kpi-value">{stats['classification']['unknown_rate']:.1%}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Events Created</div>
                <div class="kpi-value">{stats['events']['total_created']}</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-label">Events Committed</div>
                <div class="kpi-value">{stats['events']['total_committed']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Event Lifetime</div>
                <div class="kpi-value">{stats['events']['avg_lifetime_seconds']:.1f}s</div>
            </div>
        </div>

        <h2>⚙️ Frame-Based Threshold Configuration</h2>
        <div class="kpi-card">
            <p><strong>System uses frame-based thresholds for consistent behavior across different processing speeds.</strong></p>
            <p>FPS: {stats['fps']['avg']:.1f} | Frame Duration: {1000.0/stats['fps']['avg'] if stats['fps']['avg'] > 0 else 0:.1f}ms</p>
        </div>
        <table>
            <tr>
                <th>Threshold</th>
                <th>Default (frames)</th>
                <th>Time @ {stats['fps']['avg']:.1f} FPS</th>
            </tr>
            <tr>
                <td>Ghost Timeout</td>
                <td>25 frames</td>
                <td>{25 * 1000.0 / stats['fps']['avg'] if stats['fps']['avg'] > 0 else 0:.0f} ms (~1 second)</td>
            </tr>
            <tr>
                <td>Temporal Cooldown</td>
                <td>10 frames</td>
                <td>{10 * 1000.0 / stats['fps']['avg'] if stats['fps']['avg'] > 0 else 0:.0f} ms (~400ms)</td>
            </tr>
            <tr>
                <td>Suppression Duration</td>
                <td>38 frames</td>
                <td>{38 * 1000.0 / stats['fps']['avg'] if stats['fps']['avg'] > 0 else 0:.0f} ms (~1.5 seconds)</td>
            </tr>
        </table>

        <h2>🚨 Event Suppression Analysis</h2>
        <table>
            <tr>
                <th>Suppression Type</th>
                <th>Count</th>
                <th>% of Total</th>
            </tr>
            <tr>
                <td>Covered by Active Event</td>
                <td>{stats['event_creation_blockers']['by_reason'].get('covered_by_active_event', 0)}</td>
                <td>{stats['event_creation_blockers']['by_reason'].get('covered_by_active_event', 0) / stats['events']['total_created'] * 100 if stats['events']['total_created'] > 0 else 0:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Suppression: Spatial (Distance)</strong></td>
                <td><strong>{stats['event_creation_blockers']['by_reason'].get('suppression_spatial', 0)}</strong></td>
                <td><strong>{stats['event_creation_blockers']['by_reason'].get('suppression_spatial', 0) / stats['events']['total_created'] * 100 if stats['events']['total_created'] > 0 else 0:.1f}%</strong></td>
            </tr>
            <tr>
                <td><strong>Suppression: Temporal (Cooldown)</strong></td>
                <td><strong>{stats['event_creation_blockers']['by_reason'].get('suppression_temporal', 0)}</strong></td>
                <td><strong>{stats['event_creation_blockers']['by_reason'].get('suppression_temporal', 0) / stats['events']['total_created'] * 100 if stats['events']['total_created'] > 0 else 0:.1f}%</strong></td>
            </tr>
            <tr>
                <td>Active Event Exclusion</td>
                <td>{stats['event_creation_blockers']['by_reason'].get('active_event_exclusion', 0)}</td>
                <td>{stats['event_creation_blockers']['by_reason'].get('active_event_exclusion', 0) / stats['events']['total_created'] * 100 if stats['events']['total_created'] > 0 else 0:.1f}%</td>
            </tr>
        </table>

        <h3>Suppression Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Suppression Distance</td>
                <td>{stats['event_creation_blockers']['suppression_distance_stats']['avg']:.1f} px</td>
            </tr>
            <tr>
                <td>P95 Suppression Distance</td>
                <td>{stats['event_creation_blockers']['suppression_distance_stats']['p95']:.1f} px</td>
            </tr>
            <tr>
                <td>Average Cooldown Time</td>
                <td>{stats['event_creation_blockers']['cooldown_time_ms_stats']['avg']:.0f} ms</td>
            </tr>
            <tr>
                <td>P95 Cooldown Time</td>
                <td>{stats['event_creation_blockers']['cooldown_time_ms_stats']['p95']:.0f} ms</td>
            </tr>
        </table>

        <h3>Event Lifecycle Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Events Created</td>
                <td>{stats['events']['total_created']}</td>
            </tr>
            <tr>
                <td>Total Events Committed (Counted)</td>
                <td>{stats['events']['total_committed']}</td>
            </tr>
            <tr>
                <td>Total Events Expired</td>
                <td>{stats['events']['total_expired']}</td>
            </tr>
            <tr>
                <td>Average Event Lifetime</td>
                <td>{stats['events']['avg_lifetime_frames']:.1f} frames ({stats['events']['avg_lifetime_seconds']:.2f}s)</td>
            </tr>
            <tr>
                <td>Max Event Lifetime</td>
                <td>{stats['events']['lifetime_frames_stats']['max']:.0f} frames ({stats['events']['lifetime_ms_stats']['max'] / 1000:.2f}s)</td>
            </tr>
        </table>

        <h3>ROI Collection Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total ROIs Added</td>
                <td>{stats['roi']['total_added']}</td>
            </tr>
            <tr>
                <td>Total ROIs Rejected</td>
                <td>{stats['roi']['total_rejected']}</td>
            </tr>
            <tr>
                <td>ROI Rejection Rate</td>
                <td>{stats['roi']['rejection_rate']:.1%}</td>
            </tr>
            <tr>
                <td>Average Sharpness</td>
                <td>{stats['roi']['avg_sharpness']:.1f}</td>
            </tr>
            <tr>
                <td>Avg ROIs per Event</td>
                <td>{stats['roi']['avg_rois_per_event']:.1f}</td>
            </tr>
        </table>

        <h2>🎯 Classification Quality</h2>
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
                <td>Unknown Count</td>
                <td>{stats['classification']['unknown_count']}</td>
            </tr>
            <tr>
                <td>Unknown Rate</td>
                <td>{stats['classification']['unknown_rate']:.1%}</td>
            </tr>
            <tr>
                <td>Average Confidence</td>
                <td>{stats['classification']['confidence_stats']['avg']:.3f}</td>
            </tr>
            <tr>
                <td>Avg Candidates per Classification</td>
                <td>{stats['classification']['avg_candidates_per_classification']:.1f}</td>
            </tr>
            <tr>
                <td>Voting Usage Rate</td>
                <td>{stats['classification']['voting_rate']:.1%}</td>
            </tr>
            <tr>
                <td>Avg Classification Time</td>
                <td>{stats['classification']['avg_processing_time_ms']:.1f} ms</td>
            </tr>
        </table>

        <h3>Bag Type Distribution</h3>
        <table>
            <tr>
                <th>Bag Type</th>
                <th>Count</th>
            </tr>
            {''.join([f"<tr><td>{bag_type}</td><td>{count}</td></tr>" for bag_type, count in sorted(stats['counting']['bag_type_distribution'].items(), key=lambda x: x[1], reverse=True)])}
        </table>

        <h3>📊 Per-Label Confidence Statistics</h3>
        <table>
            <tr>
                <th>Label</th>
                <th>Count</th>
                <th>Avg Confidence</th>
                <th>Min Confidence</th>
                <th>Low-Conf Rate (&lt;0.7)</th>
            </tr>
            {_generate_per_label_rows(stats)}
        </table>

        <h3>🔄 Top Label Confusion Pairs</h3>
        <table>
            <tr>
                <th>Transition</th>
                <th>Count</th>
            </tr>
            {''.join([f"<tr><td>{flip}</td><td>{count}</td></tr>" for flip, count in stats['classification']['top_label_flips']])}
        </table>

        <h2>🛡️ Classification Stability Heuristics (V6)</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Label Reuse Events</td>
                <td>{stats['classification']['stability_heuristics']['label_reuse_count']}</td>
            </tr>
            <tr>
                <td>Label Reuse Rate</td>
                <td>{stats['classification']['stability_heuristics']['label_reuse_rate']:.1%}</td>
            </tr>
            <tr>
                <td>High Volatility Tracks</td>
                <td><strong>{stats['classification']['stability_heuristics']['high_volatility_tracks']}</strong></td>
            </tr>
            <tr>
                <td>Average Track Volatility</td>
                <td>{stats['classification']['stability_heuristics']['avg_volatility']:.3f}</td>
            </tr>
            <tr>
                <td>Max Track Volatility</td>
                <td>{stats['classification']['stability_heuristics']['max_volatility']:.3f}</td>
            </tr>
        </table>

        {'<h3>📋 Recent Label Reuse Events</h3>' if stats['classification']['stability_heuristics']['label_reuse_events'] else ''}
        {_generate_reuse_events_table(stats) if stats['classification']['stability_heuristics']['label_reuse_events'] else ''}

        {'<h3>⚠️ High Volatility Tracks</h3>' if stats['classification']['stability_heuristics']['volatility_details'] else ''}
        {_generate_volatility_table(stats) if stats['classification']['stability_heuristics']['volatility_details'] else ''}

        <h2>📈 Streak Analysis & Burst Anomalies</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Classification Streaks</td>
                <td>{stats['streak_analysis']['total_streaks']}</td>
            </tr>
            <tr>
                <td>Average Streak Length</td>
                <td>{stats['streak_analysis']['avg_streak_length']:.1f}</td>
            </tr>
            <tr>
                <td>Longest Streak</td>
                <td>{stats['streak_analysis']['longest_streak']}</td>
            </tr>
            <tr>
                <td>Burst Anomalies Detected</td>
                <td><strong>{len(stats['streak_analysis']['burst_anomalies'])}</strong></td>
            </tr>
        </table>

        {'<h3>⚠️ Burst Anomaly Details</h3>' if stats['streak_analysis']['burst_anomalies'] else ''}
        {'<table><tr><th>Label</th><th>Frames</th><th>Count</th><th>Min Confidence</th></tr>' + ''.join([f"<tr><td>{anom['label']}</td><td>{anom['start_frame']}-{anom['end_frame']}</td><td>{anom['count']}</td><td>{anom['min_confidence']:.3f}</td></tr>" for anom in stats['streak_analysis']['burst_anomalies'][:10]]) + '</table>' if stats['streak_analysis']['burst_anomalies'] else ''}

        <h2>📊 Lifecycle Details</h2>
        <h3>Forced Closes</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Forced Closes</td>
                <td>{stats['lifecycle_details']['forced_closes']['total']}</td>
            </tr>
            <tr>
                <td>Idle Commits</td>
                <td>{stats['lifecycle_details']['idle_commits']}</td>
            </tr>
        </table>

        {'<h4>Forced Close Reasons</h4><table><tr><th>Reason</th><th>Count</th></tr>' + ''.join([f"<tr><td>{reason}</td><td>{count}</td></tr>" for reason, count in stats['lifecycle_details']['forced_closes']['by_reason'].items()]) + '</table>' if stats['lifecycle_details']['forced_closes']['by_reason'] else ''}

        <h3>Detection Gap Closures</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Gap Closures</td>
                <td>{stats['lifecycle_details']['detection_gap_closures']['total']}</td>
            </tr>
            <tr>
                <td>Average Gap Duration</td>
                <td>{stats['lifecycle_details']['detection_gap_closures']['avg_gap_ms']:.1f} ms</td>
            </tr>
            <tr>
                <td>Max Gap Duration</td>
                <td>{stats['lifecycle_details']['detection_gap_closures']['max_gap_ms']:.1f} ms</td>
            </tr>
        </table>

        <h2>⚠️ Risk Heuristics</h2>
        <div class="summary-grid">
            <div class="kpi-card {'error' if stats['risk_heuristics']['risk_level'] == 'HIGH' else 'warning' if stats['risk_heuristics']['risk_level'] == 'MEDIUM' else 'success'}">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value">{stats['risk_heuristics']['risk_level']}</div>
            </div>
            <div class="kpi-card {'warning' if stats['risk_heuristics']['undercount_risk_score'] >= 25 else 'success'}">
                <div class="kpi-label">Undercount Risk</div>
                <div class="kpi-value">{stats['risk_heuristics']['undercount_risk_score']}/100</div>
            </div>
            <div class="kpi-card {'warning' if stats['risk_heuristics']['overcount_risk_score'] >= 25 else 'success'}">
                <div class="kpi-label">Overcount Risk</div>
                <div class="kpi-value">{stats['risk_heuristics']['overcount_risk_score']}/100</div>
            </div>
        </div>

        <h3>Risk Factors</h3>
        <ul>
            {''.join([f"<li>{factor}</li>" for factor in stats['risk_heuristics']['risk_factors']])}
        </ul>

        <h2>🚨 Issue Findings</h2>
        {'<p><strong>✅ No critical issues detected! </strong></p>' if not stats['issues'] else ''}
        {''.join([f'''
        <div class="issue-card {issue['severity']}">
            <h4><span class="badge {issue['severity']}">{issue['severity'].upper()}</span> {issue['title']}</h4>
            <p><strong>Description:</strong> {issue['description']}</p>
            <p><strong>Likely Cause:</strong> {issue['likely_cause']}</p>
            <p><strong>Where to Look:</strong> {issue['where_to_look']}</p>
            {f'<div class="recommendation"><strong>Recommendation:</strong><br>' + '<br>'.join([f'{k}: {v}' for k, v in issue.get('recommendation', {}).items()]) + '</div>' if 'recommendation' in issue else ''}
        </div>
        ''' for issue in stats['issues']])}

        <h2>📈 Time Series - Event Creation vs Suppression</h2>
        <div class="chart-container">
            <canvas id="suppressionChart"></canvas>
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
        </table>

        <div class="footer">
            <p><strong>Report Generated:</strong> {stats['metadata']['report_generated']}</p>
            <p><strong>App Started:</strong> {stats['metadata'].get('app_start_time', 'N/A')} | <strong>Version:</strong> {stats['metadata']['app_version']}</p>
            <p><strong>Total Entries Parsed:</strong> {stats['parsing']['total_entries']:,} | <strong>Skipped:</strong> {stats['parsing']['skipped_entries']}</p>
        </div>
    </div>

    <script>
        // Suppression Timeline Chart
        const tsData = {json.dumps(stats['time_series'])};
        const tsLabels = tsData.map(d => d.timestamp.substring(11, 16));
        const tsCtx = document.getElementById('suppressionChart').getContext('2d');
        new Chart(tsCtx, {{
            type: 'line',
            data: {{
                labels: tsLabels,
                datasets: [{{
                    label: 'Events Created',
                    data: tsData.map(d => d.event_created),
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }}, {{
                    label: 'Suppressed (Cooldown + Distance)',
                    data: tsData.map(d => d.suppressed),
                    borderColor:  'rgba(220, 53, 69, 1)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    yAxisID: 'y',
                    tension:  0.4
                }}, {{
                    label: 'Skip Creation (Active Event)',
                    data: tsData.map(d => d.skip_creation),
                    borderColor: 'rgba(255, 193, 7, 1)',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                scales:  {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position:  'left',
                        title: {{
                            display: true,
                            text: 'Count per Minute'
                        }}
                    }}
                }}
            }}
        }});

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
                            text:  'Time (ms)'
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

    start_time, end_time = get_time_range(args)
    print(f"Analyzing logs from {start_time.isoformat()} to {end_time.isoformat()}")

    log_files = discover_log_files(args.log_dir)
    print(f"Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  - {f}")

    analyzer = LogAnalyzer(start_time, end_time)

    print("\nAnalyzing log entries...")
    for entry, file_path, line_num, is_valid in stream_log_entries(log_files, start_time, end_time):
        if is_valid:
            analyzer.analyze_entry(entry)
        else:
            analyzer.skipped_entries += 1

    print(f"Processed {analyzer.total_entries} entries (skipped {analyzer.skipped_entries} malformed lines)")

    print("\nComputing statistics...")
    stats = analyzer.compute_statistics()

    output_dir = Path(args.output) / start_time.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "report.html"
    print(f"\nGenerating HTML report:  {html_path}")
    generate_html_report(stats, str(html_path))

    json_path = output_dir / "summary.json"
    print(f"Generating JSON summary: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    print("\n✅ Analysis complete!")
    print(f"   HTML Report: {html_path}")
    print(f"   JSON Summary:  {json_path}")

    print("\n" + "=" * 70)
    print("SUMMARY - Frame-Based Threshold Analysis")
    print("=" * 70)
    print(f"Total Bags Counted: {stats['counting']['total_bags_counted']}")
    print(f"Events Created: {stats['events']['total_created']}")
    print(f"Events Committed: {stats['events']['total_committed']}")
    print(f"Events Expired: {stats['events']['total_expired']}")
    print(f"Average Event Lifetime: {stats['events']['avg_lifetime_frames']:.1f} frames ({stats['events']['avg_lifetime_seconds']:.2f}s)")
    print(f"\nSuppression Breakdown:")
    print(f"  - Covered by Active Event: {stats['event_creation_blockers']['by_reason'].get('covered_by_active_event', 0)}")
    print(f"  - Suppressed (Spatial): {stats['event_creation_blockers']['by_reason'].get('suppression_spatial', 0)}")
    print(f"  - Suppressed (Temporal): {stats['event_creation_blockers']['by_reason'].get('suppression_temporal', 0)}")
    print(f"  - Active Event Exclusion: {stats['event_creation_blockers']['by_reason'].get('active_event_exclusion', 0)}")
    
    total_suppressed = stats['event_creation_blockers']['total_blocked']
    suppression_pct = total_suppressed / stats['events']['total_created'] * 100 if stats['events']['total_created'] > 0 else 0
    print(f"\nTotal Suppression Rate: {suppression_pct:.1f}% ({total_suppressed} blocked)")
    
    avg_fps = stats['fps']['avg'] if stats['fps']['avg'] > 0 else 25.0
    print(f"\n🔧 Frame-Based Thresholds @ {avg_fps:.1f} FPS:")
    print(f"   - Ghost Timeout: 25 frames ({25 * 1000.0 / avg_fps:.0f}ms)")
    print(f"   - Temporal Cooldown: 10 frames ({10 * 1000.0 / avg_fps:.0f}ms)")
    print(f"   - Suppression Duration: 38 frames ({38 * 1000.0 / avg_fps:.0f}ms)")
    
    if suppression_pct > 5:
        print(f"\n⚠️  High suppression rate detected!")
        print(f"   Consider reducing temporal_cooldown_frames from 10 to 5-8")
    
    print("=" * 70)


if __name__ == "__main__":
    main()