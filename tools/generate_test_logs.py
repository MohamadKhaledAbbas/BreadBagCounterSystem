#!/usr/bin/env python3
"""
Generate sample JSON logs for testing the log analyzer.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


def generate_sample_logs(output_dir: str, num_entries: int = 1000):
    """Generate sample JSON log entries for testing."""
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / "app.json.log"
    
    base_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    entries = []
    
    # Generate various log types
    for i in range(num_entries):
        timestamp = base_time + timedelta(seconds=i * 0.5)
        ts_str = timestamp.isoformat().replace('+00:00', 'Z')
        
        # Frame processing logs (most common)
        if i % 3 == 0:
            entry = {
                "timestamp": ts_str,
                "level": "DEBUG",
                "logger": "BreadCounter",
                "message": f"[FRAME] id={1000 + i}, detect=32.5ms, monitor=5.3ms, total=42.8ms, dets=6, ready=1",
                "component": "BagCounterApp",
                "data": {
                    "frame_id": 1000 + i,
                    "detection_time_ms": 32.5 + (i % 10),
                    "monitor_time_ms": 5.3 + (i % 3),
                    "total_time_ms": 42.8 + (i % 15),
                    "detections_count": 6,
                    "events_ready": 1,
                    "fps": 23.4 + (i % 5)
                }
            }
        
        # Event creation
        elif i % 10 == 1:
            entry = {
                "timestamp": ts_str,
                "level": "INFO",
                "logger": "BreadCounter",
                "message": f"[EVENT_CREATED] id={10000 + i}, conf=0.870, frame={1000 + i}",
                "component": "BagStateMonitor",
                "data": {
                    "event_id": 10000 + i,
                    "confidence": 0.87 - (i % 5) * 0.02,
                    "box": [100, 200, 300, 400],
                    "frame_index": 1000 + i,
                    "state": "detecting_open"
                }
            }
        
        # Classification
        elif i % 10 == 2:
            labels = ["Whole_Wheat", "White", "Bran", "Unknown"]
            label = labels[i % len(labels)]
            entry = {
                "timestamp": ts_str,
                "level": "WARNING" if label == "Unknown" else "INFO",
                "logger": "BreadCounter",
                "message": f"[CLASSIFICATION] track={10000 + i}, label={label}, conf=0.920, ratio=8.92",
                "component": "ClassifierService",
                "data": {
                    "track_id": 10000 + i,
                    "label": label,
                    "confidence": 0.92 if label != "Unknown" else 0.45,
                    "candidates": 5,
                    "used_voting": True,
                    "rejection_reason": "low_evidence (1.234 < 2.0)" if label == "Unknown" else None,
                    "winner_ratio": 8.92 if label != "Unknown" else 1.25
                }
            }
        
        # Count update
        elif i % 10 == 3:
            bag_types = ["Whole_Wheat", "White", "Bran"]
            bag_type = bag_types[i % len(bag_types)]
            entry = {
                "timestamp": ts_str,
                "level": "INFO",
                "logger": "BreadCounter",
                "message": f"[COUNT_UPDATE] type={bag_type}, count={i // 10}, track={10000 + i}, conf=0.920",
                "component": "BagCounterApp",
                "data": {
                    "bag_type": bag_type,
                    "new_count": i // 10,
                    "track_id": 10000 + i,
                    "confidence": 0.92,
                    "phash": f"a8f7e3c2d1b{i % 100}",
                    "candidates_evaluated": 5
                }
            }
        
        # Event expiration (warning)
        elif i % 20 == 4:
            entry = {
                "timestamp": ts_str,
                "level": "WARNING",
                "logger": "BreadCounter",
                "message": f"[EVENT_EXPIRED] id={10000 + i}, state=detecting_closed, frames=25, open_hits=4, closed_hits=2, idle=15",
                "component": "BagStateMonitor",
                "data": {
                    "event_id": 10000 + i,
                    "state": "detecting_closed",
                    "frames_tracked": 25,
                    "open_hits": 4,
                    "closed_hits": 2,
                    "frames_since_update": 15
                }
            }
        
        # Backpressure warning (occasional)
        elif i % 50 == 5:
            entry = {
                "timestamp": ts_str,
                "level": "WARNING",
                "logger": "BreadCounter",
                "message": "[BACKPRESSURE] queue=input_queue, util=85.0%, drops=23, action=skip_frame",
                "component": "BagCounterApp",
                "data": {
                    "queue_name": "input_queue",
                    "utilization": 0.85,
                    "drops": 23,
                    "action": "skip_frame",
                    "frames_skipped": 50
                }
            }
        
        # Error (rare)
        elif i % 100 == 6:
            entry = {
                "timestamp": ts_str,
                "level": "ERROR",
                "logger": "BreadCounter",
                "message": "[ERROR] component=LogicThread, op=frame_processing, type=ValueError, msg=Invalid detection box coordinates",
                "component": "LogicThread",
                "data": {
                    "operation": "frame_processing",
                    "error_type": "ValueError",
                    "error_message": "Invalid detection box coordinates: [100, 200, 90, 400]",
                    "affected_ids": [1000 + i]
                }
            }
        
        # Event suppression
        elif i % 15 == 7:
            entry = {
                "timestamp": ts_str,
                "level": "INFO",
                "logger": "BreadCounter",
                "message": "[EVENT_SUPPRESSED] new_id=-1, reason=duplicate_spatial_overlap, iou=0.82, conflict_with=12345",
                "component": "BagStateMonitor",
                "data": {
                    "event_id": -1,
                    "reason": "duplicate_spatial_overlap",
                    "iou": 0.82,
                    "conflicting_event_id": 12345
                }
            }
        
        # Default: info log
        else:
            entry = {
                "timestamp": ts_str,
                "level": "INFO",
                "logger": "BreadCounter",
                "message": f"Processing frame {1000 + i}",
                "component": "BagCounterApp",
                "data": {}
            }
        
        entries.append(entry)
    
    # Write to log file
    with open(log_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {num_entries} log entries in {log_file}")
    return str(log_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample logs for testing")
    parser.add_argument("--output-dir", default="./test_logs", help="Output directory")
    parser.add_argument("--num-entries", type=int, default=1000, help="Number of entries to generate")
    
    args = parser.parse_args()
    
    generate_sample_logs(args.output_dir, args.num_entries)
