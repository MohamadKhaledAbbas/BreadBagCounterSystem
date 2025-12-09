import cv2
import uuid
from typing import List, Tuple

from src.utils.AppLogging import logger
from src.config.tracking_config import tracking_config


class BagEvent:
    # Quality score weights for ROI selection
    SHARPNESS_WEIGHT = 0.6  # Weight for sharpness in quality score
    IOU_WEIGHT = 0.4        # Weight for IoU in quality score
    IOU_SCALE = 100.0       # Scale IoU to match sharpness magnitude
    
    def __init__(self, box, frame_img, open_id, closed_id):
        self.id = int(uuid.uuid4().int >> 96)
        self.box = box

        # State Machine: detecting_open -> detecting_closed -> counted
        self.state = 'detecting_open'

        # Threshold Counters
        self.open_hits = 1
        self.closed_hits = 0

        self.frames_since_update = 0

        # Buffer settings (from centralized config)
        self.max_open_samples = tracking_config.max_open_samples
        self.max_closed_samples = tracking_config.max_closed_samples

        self.open_id = open_id
        self.closed_id = closed_id

        # Separate buffers for open and closed ROIs with IoU tracking
        # Each entry is (sharpness, iou_score, roi)
        self.open_rois: List[Tuple[float, float, any]] = []
        self.closed_rois: List[Tuple[float, float, any]] = []
        
        # Track anchor box for IoU computation (updated with each detection)
        self.anchor_box = box

        # Add first frame (IoU is 1.0 for first frame against itself)
        self._add_roi(box, frame_img, is_open=True, iou_score=1.0)

        logger.debug(f"[BagEvent] Created event ID={self.id}")

    def _compute_iou(self, boxA, boxB):
        """Compute IoU between two boxes."""
        # Sanity checks for box coordinates
        if (boxA[2] <= boxA[0] or boxA[3] <= boxA[1] or
            boxB[2] <= boxB[0] or boxB[3] <= boxB[1]):
            return 0.0
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _add_roi(self, box, frame_img, is_open: bool, iou_score: float = None):
        """Extract ROI and add to appropriate buffer with IoU tracking."""
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
        if not sharpness >= tracking_config.min_roi_sharpness:
            return False

        # Compute IoU against anchor box if not provided
        if iou_score is None:
            iou_score = self._compute_iou(box, self.anchor_box)

        if is_open:
            self.open_rois.append((sharpness, iou_score, roi))
            # Sort by combined quality score: weight sharpness and IoU
            # Higher sharpness and higher IoU are both better
            quality_score = lambda x: (x[0] * self.SHARPNESS_WEIGHT + x[1] * self.IOU_SCALE * self.IOU_WEIGHT)
            self.open_rois.sort(key=quality_score, reverse=True)
            if len(self.open_rois) > self.max_open_samples:
                self.open_rois = self.open_rois[:self.max_open_samples]
            logger.debug(
                f"[BagEvent:{self.id}] Added OPEN ROI "
                f"(sharpness={sharpness:.1f}, IoU={iou_score:.3f}, total={len(self.open_rois)})"
            )
        else:
            self.closed_rois.append((sharpness, iou_score, roi))
            # Sort by combined quality score
            quality_score = lambda x: (x[0] * self.SHARPNESS_WEIGHT + x[1] * self.IOU_SCALE * self.IOU_WEIGHT)
            self.closed_rois.sort(key=quality_score, reverse=True)
            if len(self.closed_rois) > self.max_closed_samples:
                self.closed_rois = self.closed_rois[:self.max_closed_samples]
            logger.debug(
                f"[BagEvent:{self.id}] Added CLOSED ROI "
                f"(sharpness={sharpness:.1f}, IoU={iou_score:.3f}, total={len(self.closed_rois)})"
            )

        return True

    def add_open_frame(self, box, frame_img):
        """Add ROI from open detection."""
        self.box = box
        # Update anchor box for better IoU tracking
        self.anchor_box = box
        self.frames_since_update = 0
        self._add_roi(box, frame_img, is_open=True)

    def add_closed_frame(self, box, frame_img):
        """Add ROI from closed detection."""
        self.box = box
        # Update anchor box for better IoU tracking
        self.anchor_box = box
        self.frames_since_update = 0
        self._add_roi(box, frame_img, is_open=False)

    def _is_valid_roi(self, roi, min_size=None, min_sharpness=None):
        """Basic quality gate."""
        # Use config values if not explicitly provided
        if min_size is None:
            min_size = tracking_config.min_roi_size
        if min_sharpness is None:
            min_sharpness = tracking_config.min_roi_sharpness
        
        h, w = roi.shape[:2]
        if h < min_size or w < min_size:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness

    def get_all_candidates(self) -> List:
        """
        Return all collected ROIs (both open and closed),
        sorted by combined quality score (sharpness * 0.6 + IoU * 100 * 0.4).
        This prioritizes ROIs with high consistency (IoU) and good image quality (sharpness).
        """
        # Combine both buffers
        all_rois = self.open_rois + self.closed_rois

        # Sort by combined quality score (sharpness + IoU-weighted)
        quality_score = lambda x: (x[0] * self.SHARPNESS_WEIGHT + x[1] * self.IOU_SCALE * self.IOU_WEIGHT)
        all_rois.sort(key=quality_score, reverse=True)

        # Return just the images (not the quality scores)
        candidates = [roi for _, _, roi in all_rois]
        
        # Calculate average IoU for logging
        if all_rois:
            avg_iou = sum(iou for _, iou, _ in all_rois) / len(all_rois)
            avg_sharpness = sum(sharp for sharp, _, _ in all_rois) / len(all_rois)
            logger.debug(
                f"[BagEvent:{self.id}] Returning {len(candidates)} candidates "
                f"({len(self.open_rois)} open, {len(self.closed_rois)} closed) - "
                f"avg_sharpness={avg_sharpness:.1f}, avg_IoU={avg_iou:.3f}"
            )
        else:
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
                 iou_threshold=None,
                 min_open_frames=None,
                 min_closed_frames=None):

        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        
        # Use config values if not explicitly provided
        self.iou_threshold = iou_threshold if iou_threshold is not None else tracking_config.iou_threshold
        self.min_open_frames = min_open_frames if min_open_frames is not None else tracking_config.min_open_frames
        self.min_closed_frames = min_closed_frames if min_closed_frames is not None else tracking_config.min_closed_frames

        self.active_events = []

        logger.info(
            f"[BagStateMonitor] Initialized: open_id={open_cls_id}, "
            f"closed_id={closed_cls_id}, iou={self.iou_threshold}, "
            f"min_open={self.min_open_frames}, min_closed={self.min_closed_frames}"
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
        for i, det in enumerate(open_dets):
            if i not in used_open_indices:
                # Add minimum confidence threshold for creating new events
                if det.get('conf', 1.0) < tracking_config.min_conf_threshold:
                    logger.debug(
                        f"[BagStateMonitor] Skipping low confidence detection: "
                        f"conf={det.get('conf', 1.0):.3f} < {tracking_config.min_conf_threshold}"
                    )
                    continue
                
                # Prevent memory issues with too many events
                if len(self.active_events) >= tracking_config.max_active_events:
                    logger.warning(
                        f"[BagStateMonitor] Max active events reached ({tracking_config.max_active_events}), "
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

            # State-aware expiry: different timeouts based on state (from centralized config)
            if event.state == 'detecting_open':
                expiry_threshold = tracking_config.expiry_detecting_open
            elif event.state == 'detecting_closed':
                expiry_threshold = tracking_config.expiry_detecting_closed
            else:  # 'counted'
                expiry_threshold = tracking_config.expiry_counted

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