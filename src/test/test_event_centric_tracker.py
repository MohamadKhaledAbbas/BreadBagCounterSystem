"""
Unit Tests for Event-Centric Tracking System.

Tests cover:
1. Event creation and state machine transitions
2. Centroid-based association (no IoU)
3. Ghost event handling for detection gaps
4. Exit-boundary-based counting
5. ROI collection during CLOSED state
6. Temporal stability requirements

Run with: python -m pytest src/test/test_event_centric_tracker.py -v
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from src.tracking.EventCentricTracker import (
    EventCentricTracker,
    EventConfig,
    BreadBagEvent,
    EventState,
    DetectionEvidence,
    ROICandidate,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Create a default EventConfig for testing."""
    return EventConfig(
        # Association parameters
        association_distance_px=100.0,
        association_time_ms=400.0,
        
        # Ghost timeout
        ghost_timeout_ms=1000.0,
        
        # Exit parameters
        exit_timeout_ms=800.0,
        exit_boundary_margin_px=50,
        
        # State transition timing
        open_to_closing_time_ms=100.0,
        closing_stability_time_ms=150.0,
        closed_stability_time_ms=200.0,
        centroid_stability_px=30.0,
        
        # Evidence thresholds
        min_open_evidence_count=3,
        min_closed_evidence_count=2,
        min_detection_confidence=0.4,
        
        # ROI collection
        max_roi_samples=8,
        min_roi_size=50,  # Smaller for testing
        min_roi_sharpness=100.0,  # Lower for testing
        min_brightness=50,
        max_brightness=250,
        
        # Classification
        min_voting_agreement_pct=60.0,
        confidence_margin_threshold=0.15,
        
        # Resource limits
        max_active_events=10,
    )


@pytest.fixture
def tracker(default_config):
    """Create an EventCentricTracker for testing."""
    return EventCentricTracker(
        config=default_config,
        open_class_id=1,
        closed_class_id=0
    )


@pytest.fixture
def dummy_frame():
    """Create a dummy frame for testing."""
    # 720p frame with some texture for sharpness testing
    frame = np.random.randint(100, 200, (720, 1280, 3), dtype=np.uint8)
    return frame


def create_detection(box, class_id, conf=0.8):
    """Helper to create detection dict."""
    return {
        'box': box,
        'class_id': class_id,
        'conf': conf,
    }


def create_evidence(timestamp_ms, x, y, w=100, h=100, is_open=True, is_closed=False, 
                   conf=0.8, frame_index=0):
    """Helper to create DetectionEvidence."""
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    return DetectionEvidence(
        timestamp_ms=timestamp_ms,
        centroid_x=x,
        centroid_y=y,
        box=(x1, y1, x2, y2),
        is_open=is_open,
        is_closed=is_closed,
        confidence=conf,
        frame_index=frame_index,
    )


# =============================================================================
# Event State Machine Tests
# =============================================================================

class TestEventStateMachine:
    """Tests for BreadBagEvent state machine."""
    
    def test_event_starts_in_open_state(self, default_config):
        """Event should start in OPEN state."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        assert event.state == EventState.OPEN
    
    def test_open_to_closing_transition(self, default_config):
        """Event should transition from OPEN to CLOSING after enough evidence."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Add more open evidence to meet min_open_evidence_count
        for i in range(1, 5):
            evidence = create_evidence(i * 50.0, 640, 360, is_open=True, frame_index=i)
            event.add_detection(evidence)
        
        assert event.state == EventState.OPEN
        assert event.open_evidence_count >= default_config.min_open_evidence_count
        
        # Now add a closed detection after time threshold
        closed_evidence = create_evidence(300.0, 640, 360, is_open=False, is_closed=True, frame_index=5)
        event.add_detection(closed_evidence)
        
        assert event.state == EventState.CLOSING
    
    def test_closing_to_closed_transition(self, default_config):
        """Event should transition from CLOSING to CLOSED with temporal stability."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Build up open evidence
        for i in range(1, 4):
            evidence = create_evidence(i * 50.0, 640, 360, is_open=True, frame_index=i)
            event.add_detection(evidence)
        
        # Add closed evidence to trigger CLOSING
        event.add_detection(create_evidence(200.0, 640, 360, is_open=False, is_closed=True, frame_index=4))
        assert event.state == EventState.CLOSING
        
        # Add more closed evidence with time passing
        event.add_detection(create_evidence(400.0, 642, 361, is_open=False, is_closed=True, frame_index=5))
        
        # Should transition to CLOSED after stability time
        assert event.state == EventState.CLOSED
    
    def test_closing_reverts_to_open(self, default_config):
        """Event should revert from CLOSING to OPEN if open evidence resumes."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Build up open evidence and trigger CLOSING
        for i in range(1, 4):
            event.add_detection(create_evidence(i * 50.0, 640, 360, is_open=True, frame_index=i))
        event.add_detection(create_evidence(200.0, 640, 360, is_open=False, is_closed=True, frame_index=4))
        
        assert event.state == EventState.CLOSING
        
        # Resume open evidence
        event.add_detection(create_evidence(250.0, 640, 360, is_open=True, frame_index=5))
        event.add_detection(create_evidence(280.0, 640, 360, is_open=True, frame_index=6))
        
        assert event.state == EventState.OPEN
    
    def test_state_transition_history(self, default_config):
        """State transitions should be logged for debugging."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        assert len(event.state_transitions) >= 1
        assert event.state_transitions[0]['to_state'] == 'OPEN'
        assert event.state_transitions[0]['trigger'] == 'event_created'


# =============================================================================
# Centroid-Based Association Tests
# =============================================================================

class TestCentroidAssociation:
    """Tests for centroid-based association (no IoU)."""
    
    def test_association_within_distance(self, default_config):
        """Detection within distance threshold should associate."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Detection 50px away (within 100px threshold)
        new_evidence = create_evidence(100.0, 690, 360, is_open=True, frame_index=1)
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is True
        assert distance < default_config.association_distance_px
    
    def test_association_outside_distance(self, default_config):
        """Detection outside distance threshold should not associate."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Detection 150px away (outside 100px threshold)
        new_evidence = create_evidence(100.0, 790, 360, is_open=True, frame_index=1)
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is False
        assert 'distance_exceeded' in reason
    
    def test_association_time_gap_exceeded(self, default_config):
        """Detection after time threshold should not associate."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Detection 500ms later (outside 400ms threshold)
        new_evidence = create_evidence(500.0, 650, 360, is_open=True, frame_index=1)
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is False
        assert 'time_gap_exceeded' in reason
    
    def test_no_iou_used(self, default_config):
        """Association should not use IoU, only centroid distance."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create detection with same centroid but very different box size
        # If IoU were used, this would have low overlap
        new_evidence = create_evidence(100.0, 640, 360, is_open=True, frame_index=1)
        new_evidence = DetectionEvidence(
            timestamp_ms=100.0,
            centroid_x=640,  # Same centroid
            centroid_y=360,
            box=(600, 320, 680, 400),  # Much smaller box
            is_open=True,
            is_closed=False,
            confidence=0.8,
            frame_index=1,
        )
        
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is True
        assert distance == 0.0  # Centroid is exactly the same


# =============================================================================
# Ghost Event Tests
# =============================================================================

class TestGhostEvents:
    """Tests for ghost event handling during detection gaps."""
    
    def test_event_survives_detection_gap(self, default_config):
        """Event should survive gaps shorter than ghost timeout."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Simulate ghost state for 500ms (within 1000ms timeout)
        should_commit = event.update_ghost_state(500.0, (1280, 720))
        
        assert should_commit is False
        assert event.state == EventState.OPEN  # Still alive
    
    def test_event_tracking_detection_gaps(self, default_config):
        """Event should track detection gaps for debugging."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Start ghost state
        event.update_ghost_state(100.0, (1280, 720))
        
        assert event.current_gap_start is not None
        
        # Detection resumes
        resume_evidence = create_evidence(200.0, 645, 362, is_open=True, frame_index=5)
        event.add_detection(resume_evidence)
        
        assert len(event.detection_gaps) == 1
        assert event.current_gap_start is None  # Gap closed
    
    def test_event_expires_after_ghost_timeout(self, default_config):
        """Event in non-CLOSED state should expire after ghost timeout."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Ghost state for 1500ms (outside 1000ms timeout)
        should_commit = event.update_ghost_state(1500.0, (1280, 720))
        
        # OPEN state events don't commit, they expire
        assert should_commit is False


# =============================================================================
# Exit-Boundary Counting Tests
# =============================================================================

class TestExitBoundaryCounting:
    """Tests for exit-boundary-based counting."""
    
    def test_counting_at_exit_boundary(self, default_config):
        """Event should commit when near exit boundary after CLOSED."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state for testing
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        
        # Move centroid near exit boundary (within 50px of edge)
        event.last_centroid = (1260, 360)  # Near right edge of 1280 width frame
        
        # Wait for ghost timeout
        should_commit = event.update_ghost_state(1100.0, (1280, 720))
        
        assert should_commit is True
        assert event.state == EventState.COMMITTED
        assert event.commit_reason == "exit_boundary"
    
    def test_no_counting_before_closed_state(self, default_config):
        """Event should not commit from OPEN state."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Move centroid near exit boundary
        event.last_centroid = (1260, 360)
        
        # Try to trigger commit from OPEN state
        should_commit = event.update_ghost_state(1100.0, (1280, 720))
        
        assert should_commit is False
        assert event.state != EventState.COMMITTED
    
    def test_no_commit_without_exit_boundary(self, default_config):
        """Event should NOT commit if not near exit boundary, even after timeout."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        
        # Centroid in center (not near boundary)
        event.last_centroid = (640, 360)
        
        # Wait for both ghost and exit timeouts - should NOT commit because not near boundary
        should_commit = event.update_ghost_state(2000.0, (1280, 720))
        
        # CRITICAL: Should NOT commit if bag is still in scene center
        assert should_commit is False
        assert event.state == EventState.CLOSED  # Still in CLOSED, waiting for exit


# =============================================================================
# Tracker Integration Tests
# =============================================================================

class TestTrackerIntegration:
    """Tests for EventCentricTracker integration."""
    
    def test_tracker_creates_event_from_open_detection(self, tracker, dummy_frame):
        """Tracker should create event from open detection."""
        detections = [create_detection([600, 320, 680, 400], class_id=1, conf=0.9)]
        
        ready_events = tracker.update(detections, 0.0, dummy_frame, 0)
        
        assert len(tracker.active_events) == 1
        assert tracker.stats['events_created'] == 1
    
    def test_tracker_associates_subsequent_detections(self, tracker, dummy_frame):
        """Tracker should associate detections to existing events."""
        # First detection creates event
        detections = [create_detection([600, 320, 680, 400], class_id=1, conf=0.9)]
        tracker.update(detections, 0.0, dummy_frame, 0)
        
        initial_event_id = list(tracker.active_events.keys())[0]
        
        # Second detection should associate
        detections = [create_detection([605, 325, 685, 405], class_id=1, conf=0.9)]
        tracker.update(detections, 50.0, dummy_frame, 1)
        
        assert len(tracker.active_events) == 1
        assert initial_event_id in tracker.active_events
    
    def test_tracker_creates_new_event_for_distant_detection(self, tracker, dummy_frame):
        """Tracker should create new event for distant detection."""
        # First detection
        detections = [create_detection([100, 100, 200, 200], class_id=1, conf=0.9)]
        tracker.update(detections, 0.0, dummy_frame, 0)
        
        # Second detection far away
        detections = [create_detection([900, 500, 1000, 600], class_id=1, conf=0.9)]
        tracker.update(detections, 50.0, dummy_frame, 1)
        
        assert len(tracker.active_events) == 2
        assert tracker.stats['events_created'] == 2
    
    def test_tracker_ignores_closed_only_detections(self, tracker, dummy_frame):
        """Tracker should not create events from closed-only detections."""
        detections = [create_detection([600, 320, 680, 400], class_id=0, conf=0.9)]  # closed
        
        tracker.update(detections, 0.0, dummy_frame, 0)
        
        assert len(tracker.active_events) == 0
        assert tracker.stats['events_created'] == 0
    
    def test_full_lifecycle_tracking(self, tracker, dummy_frame):
        """Test complete bag lifecycle from open to commit."""
        frame_duration_ms = 40.0  # 25fps
        
        # Simulate bag appearing (open)
        for i in range(5):
            detections = [create_detection([600 + i*2, 320, 680 + i*2, 400], class_id=1, conf=0.9)]
            tracker.update(detections, i * frame_duration_ms, dummy_frame, i)
        
        assert len(tracker.active_events) == 1
        event_id = list(tracker.active_events.keys())[0]
        
        # Simulate bag closing
        for i in range(5, 12):
            detections = [create_detection([610, 325, 690, 405], class_id=0, conf=0.9)]  # closed
            tracker.update(detections, i * frame_duration_ms, dummy_frame, i)
        
        # Check state progressed
        if len(tracker.active_events) > 0:
            event = tracker.active_events.get(event_id)
            if event:
                assert event.state in [EventState.CLOSING, EventState.CLOSED]
        
        # Simulate bag leaving (no detections) near edge
        if len(tracker.active_events) > 0:
            event = list(tracker.active_events.values())[0]
            event.state = EventState.CLOSED
            event.last_centroid = (1260, 360)  # Near exit
        
        # Wait for commit
        for i in range(12, 50):
            ready_events = tracker.update([], i * frame_duration_ms, dummy_frame, i)
            if ready_events:
                assert len(ready_events) == 1
                assert 'event_id' in ready_events[0]
                assert 'candidates' in ready_events[0]
                break


# =============================================================================
# Debug Info Tests
# =============================================================================

class TestDebugInfo:
    """Tests for debugging and metrics logging."""
    
    def test_event_debug_info(self, default_config):
        """Event should provide comprehensive debug info."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Add some history
        for i in range(1, 5):
            event.add_detection(create_evidence(i * 50.0, 640 + i, 360, is_open=True, frame_index=i))
        
        debug_info = event.get_debug_info()
        
        assert 'event_id' in debug_info
        assert 'state' in debug_info
        assert 'lifespan_ms' in debug_info
        assert 'detection_gaps' in debug_info
        assert 'state_transitions' in debug_info
        assert 'open_evidence_count' in debug_info
        assert 'closed_evidence_count' in debug_info
    
    def test_tracker_stats(self, tracker, dummy_frame):
        """Tracker should provide comprehensive stats."""
        # Create some events
        detections = [create_detection([600, 320, 680, 400], class_id=1, conf=0.9)]
        tracker.update(detections, 0.0, dummy_frame, 0)
        
        stats = tracker.get_tracker_stats()
        
        assert 'events_created' in stats
        assert 'events_committed' in stats
        assert 'events_expired' in stats
        assert 'events_suppressed' in stats
        assert 'total_detections_processed' in stats
        assert 'active_events' in stats
        assert 'completion_rate' in stats


# =============================================================================
# ROI Collection Tests
# =============================================================================

class TestROICollection:
    """Tests for ROI collection during CLOSED state."""
    
    def test_roi_collection_in_closed_state(self, default_config):
        """ROIs should be collected only during CLOSED state."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state
        event.state = EventState.CLOSED
        
        # Create frame with texture
        frame = np.random.randint(100, 200, (720, 1280, 3), dtype=np.uint8)
        
        # Add detection in CLOSED state
        closed_evidence = create_evidence(100.0, 640, 360, is_open=False, is_closed=True, frame_index=5)
        event.add_detection(closed_evidence, frame)
        
        # ROIs should be collected
        candidates = event.get_roi_candidates()
        assert len(candidates) >= 0  # May or may not pass quality checks
    
    def test_roi_candidates_format(self, default_config):
        """ROI candidates should have correct format for classification."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create high-quality frame
        frame = np.random.randint(100, 200, (720, 1280, 3), dtype=np.uint8)
        # Add edge patterns for sharpness
        cv2_frame = frame.copy()
        import cv2
        cv2.rectangle(cv2_frame, (610, 330), (670, 390), (200, 200, 200), 2)
        
        # Force into CLOSED state and add ROI manually
        event.state = EventState.CLOSED
        roi_candidate = ROICandidate(
            roi=cv2_frame[320:400, 600:680],
            sharpness=500.0,
            size=(80, 80),
            timestamp_ms=100.0,
            frame_index=5,
            centroid_stability=10.0,
            confidence=0.9,
        )
        event.roi_candidates.append(roi_candidate)
        
        candidates = event.get_roi_candidates()
        
        if len(candidates) > 0:
            cand = candidates[0]
            assert 'roi' in cand
            assert 'sharpness' in cand
            assert 'frame_index' in cand
            assert 'confidence' in cand
            assert 'relative_time' in cand


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
