import cv2
import numpy as np
import uuid

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
        self.max_buffer_size = 15

        self.open_id = open_id
        self.closed_id = closed_id

        # Buffer: Stores all candidate ROI images
        self.candidate_rois = []
        self.add_frame(box, frame_img)

        logger.debug(f"[BagEvent] Created new event ID={self.id}, box={box}")

    def add_frame(self, box, frame_img):
        """Extract ROI and add to candidates buffer."""
        self.box = box
        self.frames_since_update = 0

        if self.state == 'detecting_open':
            h, w = frame_img.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = frame_img[y1:y2, x1:x2].copy()

                if self._is_valid_roi(roi):
                    self.candidate_rois.append(roi)
                    logger.debug(
                        f"[BagEvent:{self.id}] Added candidate ROI "
                        f"(total: {len(self.candidate_rois)})"
                    )

                    if len(self.candidate_rois) > self.max_buffer_size:
                        self.candidate_rois.pop(0)
                        logger.debug(
                            f"[BagEvent:{self.id}] Buffer full, removed oldest candidate"
                        )
                else:
                    logger.debug(
                        f"[BagEvent:{self.id}] Rejected ROI (failed quality check)"
                    )
            else:
                logger.debug(
                    f"[BagEvent:{self.id}] Invalid ROI dimensions: "
                    f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
                )

    def _is_valid_roi(self, roi, min_size=30, min_sharpness=50):
        """
        Basic quality gate - reject obviously bad frames.
        """
        h, w = roi.shape[:2]
        if h < min_size or w < min_size:
            logger.debug(
                f"[BagEvent:{self.id}] ROI too small: {w}x{h} (min: {min_size})"
            )
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < min_sharpness:
            logger.debug(
                f"[BagEvent:{self.id}] ROI too blurry: {sharpness:.1f} (min: {min_sharpness})"
            )
            return False

        return True

    def get_all_candidates(self):
        """Return all collected candidate ROIs for classifier evaluation."""
        logger.debug(
            f"[BagEvent:{self.id}] Returning {len(self.candidate_rois)} candidates"
        )
        return self.candidate_rois.copy()

    def get_best_roi(self):
        """Fallback: Return first candidate if needed."""
        if not self.candidate_rois:
            logger.warning(f"[BagEvent:{self.id}] No candidates available")
            return None
        return self.candidate_rois[0]


class BagStateMonitor:
    def __init__(self, open_cls_id, closed_cls_id,
                 iou_threshold=0.25,
                 min_open_frames=3,
                 min_closed_frames=3):

        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        self.iou_threshold = iou_threshold

        self.min_open_frames = min_open_frames
        self.min_closed_frames = min_closed_frames

        self.active_events = []

        logger.info(
            f"[BagStateMonitor] Initialized with open_id={open_cls_id}, "
            f"closed_id={closed_cls_id}, iou_threshold={iou_threshold}, "
            f"min_open={min_open_frames}, min_closed={min_closed_frames}"
        )

    def compute_iou(self, boxA, boxB):
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
            f"[BagStateMonitor] Frame update: {len(open_dets)} open, "
            f"{len(closed_dets)} closed, {len(self.active_events)} active events"
        )

        used_open_indices = set()

        # ---------------------------------------------------
        # 1. Update existing events (Match Open detections)
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                best_event.add_frame(det['box'], frame_img)
                used_open_indices.add(i)
                if best_event.state != 'counted':
                    best_event.open_hits += 1
                    logger.debug(
                        f"[BagStateMonitor] Event {best_event.id}: "
                        f"open_hits={best_event.open_hits} (IoU={best_iou:.2f})"
                    )
                    if best_event.state == 'detecting_closed':
                        logger.debug(
                            f"[BagStateMonitor] Event {best_event.id}: "
                            f"Reopened, resetting closed_hits"
                        )
                        best_event.closed_hits = 0
                        best_event.state = 'detecting_open'

        # ---------------------------------------------------
        # 2.  Update existing events (Match Closed detections)
        # ---------------------------------------------------
        for det in closed_dets:
            best_iou = 0
            best_event = None
            for event in self.active_events:
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                best_event.add_frame(det['box'], frame_img)

                if best_event.state != 'counted':
                    if best_event.open_hits >= self.min_open_frames:
                        best_event.closed_hits += 1
                        best_event.state = 'detecting_closed'
                        logger.debug(
                            f"[BagStateMonitor] Event {best_event.id}: "
                            f"closed_hits={best_event.closed_hits} (IoU={best_iou:.2f})"
                        )

        # ---------------------------------------------------
        # 3.  Create NEW events
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            if i not in used_open_indices:
                new_event = BagEvent(det['box'], frame_img, self.open_id, self.closed_id)
                self.active_events.append(new_event)
                logger.info(
                    f"[BagStateMonitor] New event created: ID={new_event.id}"
                )

        # ---------------------------------------------------
        # 4. Check Triggers & Cleanup
        # ---------------------------------------------------
        active_next_frame = []
        expired_count = 0

        for event in self.active_events:
            event.frames_since_update += 1

            if (event.state == 'detecting_closed' and
                    event.closed_hits >= self.min_closed_frames and
                    event.state != 'counted'):

                candidates = event.get_all_candidates()
                if candidates:
                    ready_to_classify.append((event.id, candidates))
                    logger.info(
                        f"[BagStateMonitor] Event {event.id} READY for classification "
                        f"({len(candidates)} candidates, "
                        f"open_hits={event.open_hits}, closed_hits={event.closed_hits})"
                    )
                else:
                    logger.warning(
                        f"[BagStateMonitor] Event {event.id} triggered but no candidates"
                    )

                event.state = 'counted'

            if event.frames_since_update < 10:
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