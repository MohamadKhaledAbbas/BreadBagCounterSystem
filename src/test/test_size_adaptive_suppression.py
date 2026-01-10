"""
Unit Tests for Size-Adaptive Suppression (Issue #4 Fix).

Tests cover:
1. Adaptive suppression distance calculation based on bag diagonal
2. Min/max bounds for adaptive distance
3. Large bags get larger suppression zones (prevent overcounting)
4. Small bags get smaller suppression zones (prevent undercounting)
5. Fallback to fixed distance when adaptive is disabled
6. Integration with suppression logic

Run with: python -m pytest src/test/test_size_adaptive_suppression.py -v
"""

import pytest
import math
import numpy as np

from src.tracking.EventCentricTracker import (
    EventCentricTracker,
    EventConfig,
    BreadBagEvent,
    EventState,
    DetectionEvidence,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def adaptive_config():
    """Create EventConfig with size-adaptive suppression enabled."""
    return EventConfig(
        # Association parameters
        association_distance_px=100.0,
        association_time_ms=400.0,
        
        # Ghost timeout
        ghost_timeout_ms=1000.0,
        
        # Timeout-based commitment
        commit_idle_frames=5,  # Smaller for faster tests
        commit_min_closed_ratio=0.3,
        
        # Size-adaptive suppression enabled
        suppression_use_adaptive_distance=True,
        suppression_diagonal_multiplier=1.5,
        suppression_min_distance_px=60.0,
        suppression_max_distance_px=250.0,
        suppression_distance_px=120.0,  # Fallback if adaptive disabled
        suppression_duration_ms=1500.0,
        suppression_require_box_overlap=False,  # Simplify for tests
        
        # Work zone disabled for simplicity
        work_zone_enabled=False,
        
        # Evidence thresholds (relaxed for testing)
        min_open_evidence_count=1,
        min_closed_evidence_count=1,
        min_detection_confidence=0.4,
        
        # Resource limits
        max_active_events=10,
    )


@pytest.fixture
def fixed_config():
    """Create EventConfig with size-adaptive suppression disabled."""
    return EventConfig(
        association_distance_px=100.0,
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        commit_idle_frames=5,
        commit_min_closed_ratio=0.3,
        
        # Size-adaptive suppression DISABLED
        suppression_use_adaptive_distance=False,
        suppression_distance_px=120.0,  # Fixed distance
        suppression_duration_ms=1500.0,
        suppression_require_box_overlap=False,
        
        work_zone_enabled=False,
        min_open_evidence_count=1,
        min_closed_evidence_count=1,
        min_detection_confidence=0.4,
        max_active_events=10,
    )


@pytest.fixture
def adaptive_tracker(adaptive_config):
    """Create tracker with adaptive suppression."""
    return EventCentricTracker(
        config=adaptive_config,
        open_class_id=1,
        closed_class_id=0
    )


@pytest.fixture
def fixed_tracker(fixed_config):
    """Create tracker with fixed suppression."""
    return EventCentricTracker(
        config=fixed_config,
        open_class_id=1,
        closed_class_id=0
    )


@pytest.fixture
def dummy_frame():
    """Create a dummy frame for testing."""
    return np.random.randint(100, 200, (720, 1280, 3), dtype=np.uint8)


def create_detection(cx, cy, w, h, class_id, conf=0.8):
    """Helper to create detection dict from center and size."""
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return {
        'box': [x1, y1, x2, y2],
        'class_id': class_id,
        'conf': conf,
    }


# =============================================================================
# Test: Suppression Distance Calculation
# =============================================================================

class TestSuppressionDistanceCalculation:
    """Tests for _calculate_suppression_distance method."""
    
    def test_small_bag_gets_small_suppression_zone(self, adaptive_tracker):
        """Small bags should have smaller suppression zones.
        
        A small bag (100x100) has diagonal ~141px.
        Expected suppression: 141 * 1.5 = 212px (within bounds)
        """
        # Small bag: 100x100 => diagonal = sqrt(100^2 + 100^2) = 141.4px
        small_box = (590, 310, 690, 410)  # 100x100
        width = small_box[2] - small_box[0]
        height = small_box[3] - small_box[1]
        expected_diagonal = math.sqrt(width**2 + height**2)
        
        recently_committed = {
            'box': small_box,
            'diagonal': expected_diagonal,
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = adaptive_tracker._calculate_suppression_distance(recently_committed)
        expected = expected_diagonal * 1.5  # ~212px
        
        assert abs(distance - expected) < 0.1, f"Expected {expected:.1f}px, got {distance:.1f}px"
        assert 60.0 <= distance <= 250.0, "Should be within bounds"
    
    def test_large_bag_gets_large_suppression_zone(self, adaptive_tracker):
        """Large bags should have larger suppression zones.
        
        A large bag (200x180) has diagonal ~269px.
        Expected suppression: 269 * 1.5 = 403px => clamped to 250px max
        """
        # Large bag: 200x180 => diagonal = sqrt(200^2 + 180^2) = 269px
        large_box = (540, 270, 740, 450)  # 200x180
        width = large_box[2] - large_box[0]
        height = large_box[3] - large_box[1]
        expected_diagonal = math.sqrt(width**2 + height**2)
        
        recently_committed = {
            'box': large_box,
            'diagonal': expected_diagonal,
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = adaptive_tracker._calculate_suppression_distance(recently_committed)
        
        # Diagonal is ~269, so 269 * 1.5 = 403 => clamped to 250
        assert distance == 250.0, f"Should be clamped to max, got {distance:.1f}px"
    
    def test_tiny_bag_hits_minimum_distance(self, adaptive_tracker):
        """Very small bags should have suppression clamped to minimum.
        
        A tiny bag (30x30) has diagonal ~42px.
        Expected: 42 * 1.5 = 63px > min 60px => 63px
        But if bag is 25x25, diagonal ~35px => 35 * 1.5 = 53px => clamped to 60px
        """
        # Tiny bag: 25x25 => diagonal = sqrt(25^2 + 25^2) = 35.4px
        tiny_box = (627, 347, 652, 372)  # 25x25
        width = tiny_box[2] - tiny_box[0]
        height = tiny_box[3] - tiny_box[1]
        diagonal = math.sqrt(width**2 + height**2)
        
        recently_committed = {
            'box': tiny_box,
            'diagonal': diagonal,
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = adaptive_tracker._calculate_suppression_distance(recently_committed)
        
        # Diagonal ~35.4, so 35.4 * 1.5 = 53 => clamped to 60
        assert distance == 60.0, f"Should be clamped to min, got {distance:.1f}px"
    
    def test_fixed_distance_when_adaptive_disabled(self, fixed_tracker):
        """Should use fixed distance when adaptive suppression is disabled."""
        recently_committed = {
            'box': (540, 270, 740, 450),  # Large bag
            'diagonal': 269.0,
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = fixed_tracker._calculate_suppression_distance(recently_committed)
        
        # Should use fixed suppression_distance_px
        assert distance == 120.0, f"Should use fixed distance, got {distance:.1f}px"
    
    def test_fallback_when_no_diagonal(self, adaptive_tracker):
        """Should fall back to fixed distance if diagonal not available."""
        recently_committed = {
            'box': (540, 270, 740, 450),
            # No 'diagonal' key
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = adaptive_tracker._calculate_suppression_distance(recently_committed)
        
        # Should fallback to fixed suppression_distance_px
        assert distance == 120.0, f"Should fallback to fixed distance, got {distance:.1f}px"


# =============================================================================
# Test: Diagonal Calculation During Commit
# =============================================================================

class TestDiagonalCalculationOnCommit:
    """Tests for diagonal calculation when events are committed."""
    
    def test_diagonal_stored_on_commit(self, dummy_frame):
        """Verify diagonal is calculated and stored when event is committed."""
        # Create config with proper state machine thresholds
        config = EventConfig(
            association_distance_px=100.0,
            association_time_ms=400.0,
            ghost_timeout_ms=1000.0,
            commit_idle_frames=5,  # Shorter for test
            commit_min_closed_ratio=0.2,  # Lower threshold
            suppression_use_adaptive_distance=True,
            suppression_diagonal_multiplier=1.5,
            suppression_min_distance_px=60.0,
            suppression_max_distance_px=250.0,
            suppression_distance_px=120.0,
            suppression_duration_ms=3000.0,
            suppression_require_box_overlap=False,
            work_zone_enabled=False,
            # State machine thresholds
            min_open_evidence_count=2,
            min_closed_evidence_count=1,
            min_detection_confidence=0.4,
            open_to_closing_frames=2,
            closing_stability_frames=2,
            closed_stability_frames=2,
            max_active_events=10,
        )
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Create event with multiple open detections to satisfy min_open_evidence_count
        detection1 = create_detection(640, 360, 150, 120, class_id=1)  # Open, 150x120
        tracker.update([detection1], 0.0, dummy_frame, 0)
        tracker.update([detection1], 40.0, dummy_frame, 1)
        tracker.update([detection1], 80.0, dummy_frame, 2)
        
        # Add closed detections for state transition
        detection_closed = create_detection(640, 360, 150, 120, class_id=0)  # Closed
        tracker.update([detection_closed], 120.0, dummy_frame, 3)
        tracker.update([detection_closed], 160.0, dummy_frame, 4)
        tracker.update([detection_closed], 200.0, dummy_frame, 5)
        
        # Verify event is in CLOSED state
        assert len(tracker.active_events) == 1, "Should have one active event"
        event = list(tracker.active_events.values())[0]
        # Event may be in CLOSING or CLOSED depending on state machine
        
        # Let idle timeout trigger commit
        for i in range(20):
            ready = tracker.update([], 200.0 + (i+1)*40.0, dummy_frame, 6+i)
            if len(tracker.recently_committed) > 0:
                break
        
        # Check recently_committed for diagonal
        assert len(tracker.recently_committed) > 0, "Should have committed events"
        committed = tracker.recently_committed[0]
        
        expected_diagonal = math.sqrt(150**2 + 120**2)  # ~192px
        assert 'diagonal' in committed, "Should have diagonal stored"
        assert abs(committed['diagonal'] - expected_diagonal) < 1.0, \
            f"Diagonal should be ~{expected_diagonal:.1f}, got {committed['diagonal']:.1f}"


# =============================================================================
# Test: Large vs Small Bag Suppression Behavior
# =============================================================================

def create_proper_commit_config(
    suppression_use_adaptive_distance: bool = True,
    suppression_diagonal_multiplier: float = 1.5,
    suppression_min_distance_px: float = 60.0,
    suppression_max_distance_px: float = 250.0,
    suppression_require_box_overlap: bool = False,
):
    """Create EventConfig with proper state machine settings for reliable commit."""
    return EventConfig(
        association_distance_px=100.0,
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        commit_idle_frames=5,
        commit_min_closed_ratio=0.2,
        suppression_use_adaptive_distance=suppression_use_adaptive_distance,
        suppression_diagonal_multiplier=suppression_diagonal_multiplier,
        suppression_min_distance_px=suppression_min_distance_px,
        suppression_max_distance_px=suppression_max_distance_px,
        suppression_distance_px=120.0,
        suppression_duration_ms=5000.0,  # Longer suppression for tests
        suppression_require_box_overlap=suppression_require_box_overlap,
        suppression_iou_threshold=0.15,
        work_zone_enabled=False,
        min_open_evidence_count=2,
        min_closed_evidence_count=1,
        min_detection_confidence=0.4,
        open_to_closing_frames=2,
        closing_stability_frames=2,
        closed_stability_frames=2,
        max_active_events=10,
    )


def commit_event(tracker, cx, cy, w, h, dummy_frame, start_time=0.0, start_frame=0):
    """Helper to reliably commit an event and return the end time/frame."""
    # Multiple open detections
    open_det = create_detection(cx, cy, w, h, class_id=1)
    tracker.update([open_det], start_time, dummy_frame, start_frame)
    tracker.update([open_det], start_time + 40.0, dummy_frame, start_frame + 1)
    tracker.update([open_det], start_time + 80.0, dummy_frame, start_frame + 2)
    
    # Closed detections
    closed_det = create_detection(cx, cy, w, h, class_id=0)
    tracker.update([closed_det], start_time + 120.0, dummy_frame, start_frame + 3)
    tracker.update([closed_det], start_time + 160.0, dummy_frame, start_frame + 4)
    tracker.update([closed_det], start_time + 200.0, dummy_frame, start_frame + 5)
    
    # Idle frames to trigger commit
    end_time = start_time + 200.0
    end_frame = start_frame + 5
    for i in range(20):
        end_time += 40.0
        end_frame += 1
        tracker.update([], end_time, dummy_frame, end_frame)
        if len(tracker.recently_committed) > 0:
            break
    
    return end_time, end_frame


class TestLargeVsSmallBagSuppression:
    """Tests for different suppression behavior based on bag size."""
    
    def test_large_bag_suppresses_further(self, dummy_frame):
        """Large bag should suppress detections at further distance.
        
        Large bag (200x180) diagonal ~269px => suppression 250px (max)
        Detection 200px away should be suppressed
        """
        config = create_proper_commit_config()
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Create and commit a large bag
        end_time, end_frame = commit_event(tracker, 640, 360, 200, 180, dummy_frame)
        
        assert len(tracker.recently_committed) > 0, "Event should be committed"
        
        # Verify the suppression zone
        committed = tracker.recently_committed[0]
        expected_diagonal = math.sqrt(200**2 + 180**2)  # ~269px
        assert 'diagonal' in committed
        assert committed['diagonal'] > 260  # ~269px
        
        # The adaptive suppression should be 269 * 1.5 = 403 => clamped to 250px
        adaptive_dist = tracker._calculate_suppression_distance(committed)
        assert adaptive_dist == 250.0, f"Large bag should use max distance, got {adaptive_dist}"
        
        # Try to create new event 200px away (should be suppressed - within 250px zone)
        new_det = create_detection(840, 360, 100, 100, class_id=1)  # 200px to the right
        next_time = end_time + 100.0  # Still within suppression duration
        tracker.update([new_det], next_time, dummy_frame, end_frame + 1)
        
        # Should be suppressed (no new event)
        assert len(tracker.active_events) == 0, \
            "Detection within suppression zone should be suppressed"
    
    def test_small_bag_allows_closer_new_event(self, dummy_frame):
        """Small bag should allow new events at closer distance.
        
        Small bag (80x80) diagonal ~113px => suppression ~170px
        Detection 180px away should NOT be suppressed
        """
        config = create_proper_commit_config()
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Create and commit a small bag
        end_time, end_frame = commit_event(tracker, 640, 360, 80, 80, dummy_frame)
        
        assert len(tracker.recently_committed) > 0, "Event should be committed"
        
        # Verify the suppression zone
        committed = tracker.recently_committed[0]
        expected_diagonal = math.sqrt(80**2 + 80**2)  # ~113px
        expected_suppression = expected_diagonal * 1.5  # ~170px
        
        adaptive_dist = tracker._calculate_suppression_distance(committed)
        assert abs(adaptive_dist - expected_suppression) < 1.0, \
            f"Expected ~{expected_suppression:.1f}px suppression, got {adaptive_dist:.1f}px"
        
        # Try to create new event 180px away (should NOT be suppressed - outside ~170px zone)
        new_det = create_detection(820, 360, 100, 100, class_id=1)  # 180px to the right
        next_time = end_time + 100.0  # Still within suppression duration
        tracker.update([new_det], next_time, dummy_frame, end_frame + 1)
        
        # Should NOT be suppressed - outside suppression zone
        assert len(tracker.active_events) == 1, \
            f"Detection outside suppression zone ({expected_suppression:.1f}px) should create new event"


# =============================================================================
# Test: Configuration Options
# =============================================================================

class TestConfigurationOptions:
    """Tests for configuration options."""
    
    def test_custom_multiplier(self, dummy_frame):
        """Test custom diagonal multiplier."""
        config = EventConfig(
            suppression_use_adaptive_distance=True,
            suppression_diagonal_multiplier=2.0,  # Custom 2.0x instead of 1.5x
            suppression_min_distance_px=60.0,
            suppression_max_distance_px=400.0,  # Increased max
            suppression_distance_px=120.0,
            suppression_duration_ms=1500.0,
            suppression_require_box_overlap=False,
            work_zone_enabled=False,
            min_open_evidence_count=1,
            min_closed_evidence_count=1,
            ghost_timeout_ms=1000.0,
            commit_idle_frames=5,
        )
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # 100x100 bag => diagonal 141px => suppression 141 * 2.0 = 282px
        recently_committed = {
            'box': (590, 310, 690, 410),
            'diagonal': math.sqrt(100**2 + 100**2),
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        
        distance = tracker._calculate_suppression_distance(recently_committed)
        expected = math.sqrt(100**2 + 100**2) * 2.0  # ~282px
        
        assert abs(distance - expected) < 0.1, f"Expected {expected:.1f}px with 2.0x multiplier"
    
    def test_custom_min_max_bounds(self, dummy_frame):
        """Test custom min/max bounds."""
        config = EventConfig(
            suppression_use_adaptive_distance=True,
            suppression_diagonal_multiplier=1.5,
            suppression_min_distance_px=100.0,  # Custom min
            suppression_max_distance_px=150.0,  # Custom max
            suppression_distance_px=120.0,
            suppression_duration_ms=1500.0,
            suppression_require_box_overlap=False,
            work_zone_enabled=False,
            min_open_evidence_count=1,
            min_closed_evidence_count=1,
            ghost_timeout_ms=1000.0,
            commit_idle_frames=5,
        )
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Tiny bag should hit min (100px)
        tiny_committed = {
            'diagonal': 30.0,  # 30 * 1.5 = 45 < 100 => clamp to 100
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 1
        }
        assert tracker._calculate_suppression_distance(tiny_committed) == 100.0
        
        # Large bag should hit max (150px)
        large_committed = {
            'diagonal': 200.0,  # 200 * 1.5 = 300 > 150 => clamp to 150
            'centroid': (640, 360),
            'timestamp_ms': 1000.0,
            'event_id': 2
        }
        assert tracker._calculate_suppression_distance(large_committed) == 150.0


# =============================================================================
# Test: Integration with Existing Suppression Logic
# =============================================================================

class TestSuppressionIntegration:
    """Tests for integration with existing suppression features."""
    
    def test_adaptive_works_with_box_overlap_check(self, dummy_frame):
        """Adaptive suppression should work alongside box overlap checking."""
        config = create_proper_commit_config(
            suppression_require_box_overlap=True,
        )
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Create and commit an event
        end_time, end_frame = commit_event(tracker, 640, 360, 100, 100, dummy_frame)
        
        assert len(tracker.recently_committed) > 0
        
        # Wait for temporal cooldown to expire (400ms) then create new detection
        # within adaptive suppression distance but no box overlap should NOT be suppressed
        # (because suppression_require_box_overlap=True and IoU is < threshold)
        # 100x100 bag => diagonal ~141px => suppression ~212px
        # 140px away is within suppression zone but has no box overlap
        new_det = create_detection(780, 360, 100, 100, class_id=1)  # 140px away, no overlap
        # Wait past temporal cooldown (500ms > 400ms)
        tracker.update([new_det], end_time + 500.0, dummy_frame, end_frame + 13)
        
        # Should create new event (box overlap check saves it)
        assert len(tracker.active_events) == 1, \
            "Should create new event when box overlap check fails"
    
    def test_stats_tracking(self, dummy_frame):
        """Verify suppression stats are tracked correctly."""
        config = create_proper_commit_config()
        tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
        
        # Create and commit an event
        end_time, end_frame = commit_event(tracker, 640, 360, 100, 100, dummy_frame)
        
        initial_suppressed = tracker.stats['events_suppressed']
        
        # Try to create event within suppression zone
        # 100x100 bag => diagonal ~141px => suppression ~212px
        # 10px away should definitely be suppressed
        suppressed_det = create_detection(650, 360, 100, 100, class_id=1)  # Very close
        tracker.update([suppressed_det], end_time + 100.0, dummy_frame, end_frame + 1)
        
        # Stats should show suppression
        assert tracker.stats['events_suppressed'] > initial_suppressed, \
            "Suppression should be tracked in stats"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
