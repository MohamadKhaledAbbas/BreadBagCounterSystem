import cv2
import uuid
from typing import List, Tuple

from src.utils.AppLogging import logger


class BagEvent:
    def __init__(self, box, frame_img, open_id, closed_id):
        self.id = int(uuid.uuid4().int >> 96)
        self.box = box

        # State Machine: detecting_open -> detecting_closed -> counted
        self.state = 'detecting_open'

        # Threshold Counters
        self.open_hits = 1
        self.closed_hits = 0

        self.frames_since_update = 0

        # Buffer settings
        self.max_open_samples = 8  # Max ROIs during open phase
        self.max_closed_samples = 4  # Max ROIs during closed phase

        self.open_id = open_id
        self.closed_id = closed_id

        # Separate buffers for open and closed ROIs
        self.open_rois: List[Tuple[float, any]] = []  # (sharpness, roi)
        self.closed_rois: List[Tuple[float, any]] = []  # (sharpness, roi)

        # Add first frame
        self._add_roi(box, frame_img, is_open=True)

        logger.debug(f"[BagEvent] Created event ID={self.id}")

    def _add_roi(self, box, frame_img, is_open: bool):
        """Extract ROI and add to appropriate buffer."""
        h, w = frame_img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.debug(f"[BagEvent:{self.id}] Invalid ROI dimensions")
            return False

        roi = frame_img[y1:y2, x1:x2].copy()

        # Quality check
        sharpness = self._is_valid_roi(roi)
        if not sharpness >= 50:
            return False

        if is_open:
            self.open_rois.append((sharpness, roi))
            # Keep top N sharpest
            self.open_rois.sort(key=lambda x: x[0], reverse=True)
            if len(self.open_rois) > self.max_open_samples:
                self.open_rois = self.open_rois[:self.max_open_samples]
            logger.debug(
                f"[BagEvent:{self.id}] Added OPEN ROI "
                f"(sharpness={sharpness:.1f}, total={len(self.open_rois)})"
            )
        else:
            self.closed_rois.append((sharpness, roi))
            # Keep top N sharpest
            self.closed_rois.sort(key=lambda x: x[0], reverse=True)
            if len(self.closed_rois) > self.max_closed_samples:
                self.closed_rois = self.closed_rois[:self.max_closed_samples]
            logger.debug(
                f"[BagEvent:{self.id}] Added CLOSED ROI "
                f"(sharpness={sharpness:.1f}, total={len(self.closed_rois)})"
            )

        return True

    def add_open_frame(self, box, frame_img):
        """Add ROI from open detection."""
        self.box = box
        self.frames_since_update = 0
        self._add_roi(box, frame_img, is_open=True)

    def add_closed_frame(self, box, frame_img):
        """Add ROI from closed detection."""
        self.box = box
        self.frames_since_update = 0
        self._add_roi(box, frame_img, is_open=False)

    def _is_valid_roi(self, roi, min_size=80, min_sharpness=50):
        """Basic quality gate."""
        h, w = roi.shape[:2]
        if h < min_size or w < min_size:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness

    def get_all_candidates(self) -> List:
        """
        Return all collected ROIs (both open and closed),
        sorted by sharpness (best first).
        """
        # Combine both buffers
        all_rois = self.open_rois + self.closed_rois

        # Sort by sharpness (highest first)
        all_rois.sort(key=lambda x: x[0], reverse=True)

        # Return just the images (not the sharpness scores)
        candidates = [roi for _, roi in all_rois]

        logger.debug(
            f"[BagEvent:{self.id}] Returning {len(candidates)} candidates "
            f"({len(self.open_rois)} open, {len(self.closed_rois)} closed)"
        )
        return candidates

    def get_stats(self) -> dict:
        """Return stats about collected ROIs."""
        return {
            "open_count": len(self.open_rois),
            "closed_count": len(self.closed_rois),
            "total": len(self.open_rois) + len(self.closed_rois)
        }


class BagStateMonitor:
    def __init__(self, open_cls_id, closed_cls_id,
                 iou_threshold=0.45,
                 min_open_frames=5,
                 min_closed_frames=2):

        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        self.iou_threshold = iou_threshold

        self.min_open_frames = min_open_frames
        self.min_closed_frames = min_closed_frames

        self.active_events = []

        logger.info(
            f"[BagStateMonitor] Initialized: open_id={open_cls_id}, "
            f"closed_id={closed_cls_id}, iou={iou_threshold}, "
            f"min_open={min_open_frames}, min_closed={min_closed_frames}"
        )

    def compute_iou(self, boxA, boxB):
        # Sanity checks for box coordinates
        if (boxA[2] <= boxA[0] or boxA[3] <= boxA[1] or
            boxB[2] <= boxB[0] or boxB[3] <= boxB[1]):
            logger.debug("[BagStateMonitor] Invalid box coordinates in IoU computation")
            return 0.0
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections, frame_img):
        ready_to_classify = []

        open_dets = [d for d in detections if d['class_id'] == self.open_id]
        closed_dets = [d for d in detections if d['class_id'] == self.closed_id]

        logger.debug(
            f"[BagStateMonitor] Frame: {len(open_dets)} open, "
            f"{len(closed_dets)} closed, {len(self.active_events)} active"
        )

        used_open_indices = set()
        used_closed_indices = set()
        matched_event_ids = set()  # Prevent same event matching twice

        # ---------------------------------------------------
        # 1. Match OPEN detections to existing events
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                # Skip if event already matched in this frame
                if event.id in matched_event_ids:
                    continue
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                # Collect OPEN ROI
                best_event.add_open_frame(det['box'], frame_img)
                used_open_indices.add(i)
                matched_event_ids.add(best_event.id)  # Mark as matched

                if best_event.state != 'counted':
                    best_event.open_hits += 1
                    logger.debug(
                        f"[BagStateMonitor] Event {best_event.id}: "
                        f"open_hits={best_event.open_hits} (IoU={best_iou:.2f})"
                    )

                    # If was detecting closed but reopened, reset
                    if best_event.state == 'detecting_closed':
                        logger.debug(
                            f"[BagStateMonitor] Event {best_event.id}: "
                            f"Reopened, resetting closed_hits"
                        )
                        best_event.closed_hits = 0
                        best_event.state = 'detecting_open'

        # ---------------------------------------------------
        # 2.  Match CLOSED detections to existing events
        # ---------------------------------------------------
        for j, det in enumerate(closed_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                # Skip if event already matched in this frame
                if event.id in matched_event_ids:
                    continue
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                # Collect CLOSED ROI
                best_event.add_closed_frame(det['box'], frame_img)
                used_closed_indices.add(j)
                matched_event_ids.add(best_event.id)  # Mark as matched

                if best_event.state != 'counted':
                    if best_event.open_hits >= self.min_open_frames:
                        best_event.closed_hits += 1
                        best_event.state = 'detecting_closed'
                        logger.debug(
                            f"[BagStateMonitor] Event {best_event.id}: "
                            f"closed_hits={best_event.closed_hits} (IoU={best_iou:.2f})"
                        )

        # ---------------------------------------------------
        # 3.  Create NEW events for unmatched open detections
        # ---------------------------------------------------
        MAX_ACTIVE_EVENTS = 50  # Prevent memory issues
        for i, det in enumerate(open_dets):
            if i not in used_open_indices:
                # Add minimum confidence threshold for creating new events
                min_conf_threshold = 0.3
                if det.get('conf', 1.0) < min_conf_threshold:
                    logger.debug(
                        f"[BagStateMonitor] Skipping low confidence detection: "
                        f"conf={det.get('conf', 1.0):.3f} < {min_conf_threshold}"
                    )
                    continue
                
                # Prevent memory issues with too many events
                if len(self.active_events) >= MAX_ACTIVE_EVENTS:
                    logger.warning(
                        f"[BagStateMonitor] Max active events reached ({MAX_ACTIVE_EVENTS}), "
                        f"skipping new event creation"
                    )
                    break
                
                new_event = BagEvent(det['box'], frame_img, self.open_id, self.closed_id)
                self.active_events.append(new_event)
                logger.info(
                    f"[BagStateMonitor] New event: ID={new_event.id}, "
                    f"conf={det.get('conf', 1.0):.3f}"
                )

        # ---------------------------------------------------
        # 4. Check triggers & cleanup
        # ---------------------------------------------------
        active_next_frame = []
        expired_count = 0

        for event in self.active_events:
            event.frames_since_update += 1

            # Trigger classification when closed threshold reached
            if (event.state == 'detecting_closed' and
                    event.closed_hits >= self.min_closed_frames and
                    event.state != 'counted'):

                candidates = event.get_all_candidates()
                stats = event.get_stats()

                if candidates:
                    ready_to_classify.append((event.id, candidates))
                    logger.info(
                        f"[BagStateMonitor] Event {event.id} READY: "
                        f"{stats['total']} candidates "
                        f"({stats['open_count']} open, {stats['closed_count']} closed)"
                    )
                else:
                    logger.warning(
                        f"[BagStateMonitor] Event {event.id} triggered but no candidates!"
                    )

                event.state = 'counted'
                logger.debug(f"[BagStateMonitor] Event {event.id} state -> counted")

            # State-aware expiry: different timeouts based on state
            if event.state == 'detecting_open':
                expiry_threshold = 8
            elif event.state == 'detecting_closed':
                expiry_threshold = 15
            else:  # 'counted'
                expiry_threshold = 5

            # Keep event alive if recently updated
            if event.frames_since_update < expiry_threshold:
                active_next_frame.append(event)
            else:
                expired_count += 1
                logger.debug(
                    f"[BagStateMonitor] Event {event.id} expired "
                    f"(state={event.state}, frames_since_update={event.frames_since_update})"
                )

        if expired_count > 0:
            logger.debug(f"[BagStateMonitor] Expired {expired_count} events")

        self.active_events = active_next_frame

        return ready_to_classify