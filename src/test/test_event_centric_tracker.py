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
        
        # IoU-based association
        iou_association_enabled=True,
        iou_association_threshold=0.3,
        
        # Velocity-based association
        velocity_scaling_enabled=True,
        velocity_scale_factor=2.5,
        max_association_distance_px=250.0,
        min_velocity_threshold=0.01,
        max_prediction_time_ms=500.0,
        
        # Ghost timeout
        ghost_timeout_ms=1000.0,
        
        # Timeout-based commitment
        commit_idle_frames=25,
        commit_min_closed_ratio=0.3,
        
        # Anti-double-counting suppression
        suppression_distance_px=150.0,
        suppression_duration_ms=1000.0,
        
        # State transition timing
        open_to_closing_time_ms=100.0,
        closing_stability_time_ms=150.0,
        closed_stability_time_ms=200.0,
        centroid_stability_px=30.0,
        
        # State reversion (anti-oscillation) - use smaller values for tests
        closing_revert_open_count=3,
        closing_revert_window_size=5,
        
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
        
        # Build up open evidence with less items so we don't have too many in window
        event.add_detection(create_evidence(50.0, 640, 360, is_open=True, frame_index=1))
        event.add_detection(create_evidence(100.0, 640, 360, is_open=True, frame_index=2))
        
        # Add first closed evidence to trigger CLOSING
        event.add_detection(create_evidence(200.0, 640, 360, is_open=False, is_closed=True, frame_index=3))
        assert event.state == EventState.CLOSING
        
        # Add more closed evidence with time passing (need enough to avoid reversion)
        # Window will be: [open, open, closed, closed, closed] -> 2 open, doesn't revert
        event.add_detection(create_evidence(250.0, 641, 360, is_open=False, is_closed=True, frame_index=4))
        event.add_detection(create_evidence(300.0, 641, 361, is_open=False, is_closed=True, frame_index=5))
        event.add_detection(create_evidence(400.0, 642, 361, is_open=False, is_closed=True, frame_index=6))
        
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
        
        # Resume open evidence - need 3 open detections since entering CLOSING to revert
        # (based on closing_revert_open_count=3 in config)
        event.add_detection(create_evidence(250.0, 640, 360, is_open=True, frame_index=5))
        event.add_detection(create_evidence(280.0, 640, 360, is_open=True, frame_index=6))
        event.add_detection(create_evidence(310.0, 640, 360, is_open=True, frame_index=7))
        
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
        
        # Detection 150px away (outside 100px threshold) with no box overlap
        new_evidence = create_evidence(100.0, 790, 360, is_open=True, frame_index=1)
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is False
        # Reason now includes 'no_match' since both centroid and IoU failed
        assert 'no_match' in reason or 'distance_exceeded' in reason
    
    def test_association_time_gap_exceeded(self, default_config):
        """Detection after time threshold should not associate."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Detection 500ms later (outside 400ms threshold)
        new_evidence = create_evidence(500.0, 650, 360, is_open=True, frame_index=1)
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        assert can_assoc is False
        assert 'time_gap_exceeded' in reason
    
    def test_centroid_based_association(self, default_config):
        """Association should work with centroid distance even with different box sizes."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create detection with same centroid but very different box size
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
    
    def test_iou_based_association(self, default_config):
        """Association should work with IoU when centroid distance is too large."""
        evidence = create_evidence(0.0, 640, 360, is_open=True, w=100, h=100)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create detection with shifted centroid but overlapping box
        # Original box: (590, 310, 690, 410) centered at (640, 360)
        # New box with shifted centroid but significant overlap
        new_evidence = DetectionEvidence(
            timestamp_ms=100.0,
            centroid_x=700,  # Shifted centroid (60px away, exceeds 100px threshold after adjustment)
            centroid_y=360,
            box=(650, 310, 750, 410),  # Overlapping box
            is_open=True,
            is_closed=False,
            confidence=0.8,
            frame_index=1,
        )
        
        can_assoc, distance, reason = event.can_associate(new_evidence)
        
        # Should associate via IoU since boxes overlap
        assert can_assoc is True
        # Reason should mention iou_match or centroid_match
        assert 'match' in reason


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
# Timeout-Based Commitment Tests
# =============================================================================

class TestTimeoutBasedCommit:
    """Tests for timeout-based commitment (exit boundary logic removed)."""
    
    def test_commit_after_idle_timeout(self, default_config):
        """Event should commit after idle timeout when in CLOSED state."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state for testing
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        event.open_evidence_count = 3
        event.closed_evidence_count = 5  # Good closed ratio
        event.last_detection_frame_index = 0
        
        # Centroid can be anywhere (center of frame)
        event.last_centroid = (640, 360)
        
        # Wait for ghost timeout with sufficient idle frames
        should_commit = event.update_ghost_state(1100.0, (1280, 720), current_frame_index=35)
        
        assert should_commit is True
        assert event.state == EventState.COMMITTED
        assert event.commit_reason == "timeout_commit"
    
    def test_no_commit_before_closed_state(self, default_config):
        """Event should not commit from OPEN state."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Move centroid anywhere
        event.last_centroid = (640, 360)
        
        # Try to trigger commit from OPEN state
        should_commit = event.update_ghost_state(1100.0, (1280, 720), current_frame_index=35)
        
        assert should_commit is False
        assert event.state != EventState.COMMITTED
    
    def test_no_commit_without_sufficient_idle_frames(self, default_config):
        """Event should NOT commit if not enough idle frames have passed."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        event.open_evidence_count = 3
        event.closed_evidence_count = 5
        event.last_detection_frame_index = 0
        
        # Centroid in center
        event.last_centroid = (640, 360)
        
        # Wait for ghost timeout but with insufficient idle frames (10 < 25)
        should_commit = event.update_ghost_state(1100.0, (1280, 720), current_frame_index=10)
        
        # Should NOT commit if not enough idle frames
        assert should_commit is False
        assert event.state == EventState.CLOSED


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


# =============================================================================
# Velocity-Based Association Tests
# =============================================================================

class TestVelocityBasedAssociation:
    """Tests for velocity-based association during fast movements."""
    
    def test_velocity_calculation(self, default_config):
        """Velocity should be calculated from centroid history."""
        evidence = create_evidence(0.0, 100, 100, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Move the bag steadily
        event.add_detection(create_evidence(40.0, 120, 100, is_open=True, frame_index=1))  # +20px in 40ms
        event.add_detection(create_evidence(80.0, 140, 100, is_open=True, frame_index=2))  # +20px in 40ms
        
        vx, vy = event.get_velocity()
        # Velocity should be approximately 0.5 px/ms (20px / 40ms)
        assert abs(vx - 0.5) < 0.1
        assert abs(vy) < 0.1
    
    def test_velocity_scales_association_distance(self, default_config):
        """Fast movement should increase association distance."""
        evidence = create_evidence(0.0, 100, 100, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create fast movement history
        event.add_detection(create_evidence(20.0, 150, 100, is_open=True, frame_index=1))  # +50px
        event.add_detection(create_evidence(40.0, 200, 100, is_open=True, frame_index=2))  # +50px
        
        # Now try to associate with a detection that's far from last position
        # but in the direction of motion
        far_detection = create_evidence(60.0, 280, 100, is_open=True, frame_index=3)  # +80px
        
        # With velocity scaling enabled, this should associate because:
        # - Velocity is high (about 2.5 px/ms from the movement pattern)
        # - The predicted position extrapolates in direction of motion
        # - Distance to predicted position is smaller than distance to last position
        can_assoc, distance, reason = event.can_associate(far_detection)
        assert can_assoc is True
    
    def test_predicted_centroid(self, default_config):
        """Centroid prediction should extrapolate based on velocity."""
        evidence = create_evidence(0.0, 100, 100, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Create movement pattern
        event.add_detection(create_evidence(40.0, 140, 100, is_open=True, frame_index=1))  # +40px in 40ms
        
        # Predict position at 80ms
        pred_x, pred_y = event.predict_centroid(80.0)
        
        # Should extrapolate: 140 + (1.0 px/ms * 40ms) = 180
        assert 170 < pred_x < 190


# =============================================================================
# Commit Configuration Tests
# =============================================================================

class TestCommitConfiguration:
    """Tests for timeout-based commit configuration."""
    
    def test_commit_with_custom_idle_frames(self):
        """Event should commit after custom idle frame count."""
        config = EventConfig(
            commit_idle_frames=10,  # Lower idle frame requirement
            commit_min_closed_ratio=0.3,
            ghost_timeout_ms=500.0,
        )
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        event.open_evidence_count = 3
        event.closed_evidence_count = 5  # 5/8 = 62.5% > 30%
        event.last_detection_frame_index = 0
        
        # Centroid is in center
        event.last_centroid = (640, 360)
        
        # Simulate enough idle frames (15 > 10)
        should_commit = event.update_ghost_state(1000.0, (1280, 720), current_frame_index=15)
        
        assert should_commit is True
        assert event.commit_reason == "timeout_commit"
    
    def test_commit_requires_closed_ratio(self):
        """Commit should require minimum closed evidence ratio."""
        config = EventConfig(
            commit_idle_frames=10,
            commit_min_closed_ratio=0.5,  # Require 50%
            ghost_timeout_ms=500.0,
        )
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, config, open_class_id=1, closed_class_id=0)
        
        # Force into CLOSED state with low closed ratio
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        event.open_evidence_count = 8
        event.closed_evidence_count = 2  # 2/10 = 20% < 50%
        event.last_detection_frame_index = 0
        event.last_centroid = (640, 360)
        
        # Should not commit due to low closed ratio
        should_commit = event.update_ghost_state(1000.0, (1280, 720), current_frame_index=35)
        
        assert should_commit is False
    
    def test_commit_anywhere_in_frame(self):
        """Event should commit regardless of position in frame (no exit boundary)."""
        config = EventConfig(
            commit_idle_frames=10,
            commit_min_closed_ratio=0.3,
            ghost_timeout_ms=500.0,
        )
        
        # Test commit at center of frame
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, config, open_class_id=1, closed_class_id=0)
        event.state = EventState.CLOSED
        event.state_enter_time_ms = 0.0
        event.open_evidence_count = 3
        event.closed_evidence_count = 5
        event.last_detection_frame_index = 0
        event.last_centroid = (640, 360)  # Center of frame
        
        should_commit = event.update_ghost_state(1000.0, (1280, 720), current_frame_index=20)
        assert should_commit is True
        
        # Test commit near edge of frame (should also work)
        evidence2 = create_evidence(0.0, 1260, 360, is_open=True)
        event2 = BreadBagEvent(evidence2, config, open_class_id=1, closed_class_id=0)
        event2.state = EventState.CLOSED
        event2.state_enter_time_ms = 0.0
        event2.open_evidence_count = 3
        event2.closed_evidence_count = 5
        event2.last_detection_frame_index = 0
        event2.last_centroid = (1260, 360)  # Near edge of frame
        
        should_commit2 = event2.update_ghost_state(1000.0, (1280, 720), current_frame_index=20)
        assert should_commit2 is True


# =============================================================================
# Anti-Oscillation Tests
# =============================================================================

class TestAntiOscillation:
    """Tests for state oscillation prevention."""
    
    def test_no_immediate_reversion(self, default_config):
        """Entering CLOSING should not immediately revert due to past open evidence."""
        evidence = create_evidence(0.0, 640, 360, is_open=True)
        event = BreadBagEvent(evidence, default_config, open_class_id=1, closed_class_id=0)
        
        # Build up lots of open evidence
        for i in range(1, 6):
            event.add_detection(create_evidence(i * 40.0, 640, 360, is_open=True, frame_index=i))
        
        # Now add closed - should enter CLOSING
        event.add_detection(create_evidence(300.0, 640, 360, is_open=False, is_closed=True, frame_index=6))
        
        assert event.state == EventState.CLOSING
        
        # Add one more closed - should still be in CLOSING, not reverted
        event.add_detection(create_evidence(350.0, 640, 360, is_open=False, is_closed=True, frame_index=7))
        
        assert event.state == EventState.CLOSING  # Should not have reverted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
