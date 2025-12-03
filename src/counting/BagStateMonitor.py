import cv2
import numpy as np
import uuid


class BagEvent:
    def __init__(self, box, frame_img, open_id, closed_id):
        self.id = int(uuid.uuid4().int >> 96)
        self.box = box

        # State Machine: detecting_open -> detecting_closed -> counted
        self.state = 'detecting_open'

        # Threshold Counters
        self.open_hits = 1  # How many times seen as Open
        self.closed_hits = 0  # How many times seen as Closed

        self.frames_since_update = 0
        self.max_buffer_size = 10

        self.open_id = open_id
        self.closed_id = closed_id

        # Buffer: Stores tuples of (sharpness_score, image_roi)
        self.image_buffer = []
        self.add_frame(box, frame_img)

    def add_frame(self, box, frame_img):
        """Extract ROI, calculate sharpness, add to buffer."""
        self.box = box  # Update latest position
        self.frames_since_update = 0

        # Only process buffer if we haven't finished counting yet
        if self.state != 'counted':
            h, w = frame_img.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = frame_img[y1:y2, x1:x2].copy()
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()

                self.image_buffer.append((score, roi))
                self.image_buffer.sort(key=lambda x: x[0], reverse=True)
                if len(self.image_buffer) > self.max_buffer_size:
                    self.image_buffer = self.image_buffer[:self.max_buffer_size]

    def get_best_roi(self):
        if not self.image_buffer: return None
        return self.image_buffer[0][1]


class BagStateMonitor:
    def __init__(self, open_cls_id, closed_cls_id,
                 iou_threshold=0.25,
                 min_open_frames=3,  # Must see 'open' 3 times to start tracking
                 min_closed_frames=3):  # Must see 'closed' 3 times to trigger count

        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        self.iou_threshold = iou_threshold

        # Configurable Thresholds
        self.min_open_frames = min_open_frames
        self.min_closed_frames = min_closed_frames

        self.active_events = []

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

        used_open_indices = set()

        # ---------------------------------------------------
        # 1. Update existing events (Match Open detections)
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                # We match if IoU is good.
                # Note: We continue tracking even if state is 'counted' (Zombie mode)
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                best_event.add_frame(det['box'], frame_img)
                used_open_indices.add(i)
                # If we see it as Open again, we reset the closed counter
                # unless it's already fully counted
                if best_event.state != 'counted':
                    best_event.open_hits += 1
                    # If we were detecting closed, but it opened up again (flicker), reset closed
                    if best_event.state == 'detecting_closed':
                        best_event.closed_hits = 0
                        best_event.state = 'detecting_open'

        # ---------------------------------------------------
        # 2. Update existing events (Match Closed detections)
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

                # LOGIC: Only increment closed_hits if we established it was open first
                if best_event.state != 'counted':
                    # Only start counting closed hits if we have enough open hits
                    if best_event.open_hits >= self.min_open_frames:
                        best_event.closed_hits += 1
                        best_event.state = 'detecting_closed'  # Tentative state

        # ---------------------------------------------------
        # 3. Create NEW events
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            if i not in used_open_indices:
                new_event = BagEvent(det['box'], frame_img, self.open_id, self.closed_id)
                self.active_events.append(new_event)

        # ---------------------------------------------------
        # 4. Check Triggers & Cleanup
        # ---------------------------------------------------
        active_next_frame = []

        for event in self.active_events:
            event.frames_since_update += 1

            # TRIGGER CONDITION:
            # 1. Must be in 'detecting_closed' state
            # 2. Must have seen enough confirmed CLOSED frames (reduce flicker)
            # 3. Must NOT have been counted yet
            if (event.state == 'detecting_closed' and
                    event.closed_hits >= self.min_closed_frames and
                    event.state != 'counted'):

                best_roi = event.get_best_roi()
                if best_roi is not None:
                    ready_to_classify.append((event.id, best_roi))

                # KEY FIX: Mark as counted but KEEP tracking it
                event.state = 'counted'

                # Cleanup:
            # If it's lost for > 10 frames, remove it.
            # This applies to 'counted' bags too (remove them once they leave screen)
            if event.frames_since_update < 10:
                active_next_frame.append(event)

        self.active_events = active_next_frame

        return ready_to_classify