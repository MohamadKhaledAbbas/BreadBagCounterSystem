#!/usr/bin/env python3
"""
Generate sample log outputs to demonstrate the enhanced structured logging.
This script simulates various pipeline events without running the full application.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.AppLogging import logger, structured_logger

def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def generate_info_samples():
    """Generate INFO level log samples."""
    print_separator("INFO LEVEL LOGS - Normal Pipeline Operation")
    
    # Event creation
    structured_logger.event_created(
        event_id=12345,
        confidence=0.87,
        box=[100, 200, 300, 400],
        frame_index=1523,
        state='detecting_open'
    )
    
    # ROI addition
    structured_logger.roi_added(
        event_id=12345,
        is_open=True,
        sharpness=1850.3,
        frame_index=1523,
        confidence=0.87,
        total_rois=3,
        bbox_area=40000
    )
    
    # State transition
    structured_logger.event_state_transition(
        event_id=12345,
        old_state='detecting_open',
        new_state='detecting_closed',
        trigger='min_open_frames_reached',
        open_hits=5,
        closed_hits=1,
        iou=0.78,
        frame_index=1528
    )
    
    # Classification candidate
    structured_logger.classification_candidate(
        track_id=12345,
        candidate_idx=0,
        label='Whole_Wheat',
        confidence=0.92,
        sharpness=1850.3,
        relative_time=0.85,
        contribution=0.784,
        frame_index=1540
    )
    
    # Classification result
    structured_logger.classification_result(
        track_id=12345,
        label='Whole_Wheat',
        confidence=0.92,
        candidates=5,
        used_voting=True,
        rejection_reason=None,
        evidence_scores={
            'Whole_Wheat': {'score': 3.854, 'count': 5, 'best_confidence': 0.92},
            'White': {'score': 0.432, 'count': 2, 'best_confidence': 0.48}
        },
        winner_ratio=8.92,
        processing_time_ms=145.3
    )
    
    # Count update
    structured_logger.count_updated(
        bag_type='Whole_Wheat',
        new_count=42,
        track_id=12345,
        confidence=0.92,
        phash='a8f7e3c2d1b9',
        candidates_evaluated=5
    )
    
    # Frame processing
    structured_logger.frame_processed(
        frame_id=1540,
        detection_time_ms=32.5,
        monitor_time_ms=5.3,
        total_time_ms=42.8,
        detections_count=6,
        events_ready=1,
        queue_sizes={'input': 8, 'classification': 2},
        fps=23.4
    )

def generate_warning_samples():
    """Generate WARNING level log samples."""
    print_separator("WARNING LEVEL LOGS - Anomalies and Issues")
    
    # Event expiration
    structured_logger.event_expired(
        event_id=12346,
        state='detecting_closed',
        frames_tracked=25,
        open_hits=4,
        closed_hits=2,
        frames_since_update=15,
        avg_motion=3.2,
        avg_confidence=0.62
    )
    
    # Classification unknown (low evidence)
    structured_logger.classification_result(
        track_id=12347,
        label='Unknown',
        confidence=0.45,
        candidates=3,
        used_voting=True,
        rejection_reason='low_evidence (1.234 < 2.0)',
        evidence_scores={
            'Whole_Wheat': {'score': 1.234, 'count': 2, 'best_confidence': 0.65},
            'White': {'score': 0.987, 'count': 1, 'best_confidence': 0.55}
        },
        winner_ratio=1.25,
        processing_time_ms=98.2
    )
    
    # Classification unknown (ambiguous)
    structured_logger.classification_result(
        track_id=12348,
        label='Unknown',
        confidence=0.78,
        candidates=6,
        used_voting=True,
        rejection_reason='ambiguous (1.35 < 1.8)',
        evidence_scores={
            'Whole_Wheat': {'score': 2.456, 'count': 3, 'best_confidence': 0.78},
            'Bran': {'score': 1.819, 'count': 3, 'best_confidence': 0.72}
        },
        winner_ratio=1.35,
        processing_time_ms=156.7
    )
    
    # Queue backpressure
    structured_logger.queue_backpressure(
        queue_name='input_queue',
        utilization=0.85,
        drops=23,
        action='skip_frame',
        avg_detection_time_ms=42.3,
        frames_skipped=50
    )
    
    # Event suppression
    structured_logger.event_suppressed(
        event_id=-1,
        reason='duplicate_spatial_overlap',
        iou=0.82,
        conflicting_event_id=12345,
        center_distance=15.3,
        frame_index=1545,
        detection_confidence=0.73
    )

def generate_error_samples():
    """Generate ERROR level log samples."""
    print_separator("ERROR LEVEL LOGS - Critical Failures")
    
    # Pipeline error in frame processing
    structured_logger.pipeline_error(
        component='LogicThread',
        operation='frame_processing',
        error_type='ValueError',
        error_message='Invalid detection box coordinates: [100, 200, 90, 400]',
        affected_ids=[1567],
        context={
            'detections_count': 5,
            'active_events': 3,
            'input_queue_size': 15,
            'classification_queue_size': 4
        },
        traceback='Traceback (most recent call last):\n  File "BagCounterApp.py", line 485, in _logic_thread_loop\n    ...'
    )
    
    # Pipeline error in classification
    structured_logger.pipeline_error(
        component='ClassifierService',
        operation='track_classification',
        error_type='RuntimeError',
        error_message='Classifier model inference failed: CUDA out of memory',
        affected_ids=[12349],
        context={
            'candidates_count': 7,
            'event_stats': {
                'total_frames_tracked': 35,
                'track_duration_frames': 28,
                'avg_sharpness': 1654.2
            }
        },
        traceback='Traceback (most recent call last):\n  File "ClassifierService.py", line 398, in process\n    ...'
    )
    
    # Pipeline error with multiple affected events
    structured_logger.pipeline_error(
        component='ClassificationThread',
        operation='classification_processing',
        error_type='MemoryError',
        error_message='Unable to allocate memory for ROI processing',
        affected_ids=[12350, 12351, 12352],
        context={
            'candidates_count': 0,
            'classification_queue_size': 18
        },
        traceback='Traceback (most recent call last):\n  File "BagCounterApp.py", line 369, in _classification_thread_loop\n    ...'
    )

def main():
    """Generate all log samples."""
    print("\n" + "#" * 80)
    print("# STRUCTURED LOGGING SAMPLES FOR BREADBAG COUNTER SYSTEM")
    print("# These samples demonstrate INFO, WARNING, and ERROR level logs")
    print("# with full context for production-level observability")
    print("#" * 80)
    
    generate_info_samples()
    generate_warning_samples()
    generate_error_samples()
    
    print("\n" + "=" * 80)
    print(" Log samples generated successfully!")
    print(" Check console output above and data/logs/app.json.log for JSON format")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
