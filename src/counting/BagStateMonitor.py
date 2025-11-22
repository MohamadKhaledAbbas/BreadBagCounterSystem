class BagStateMonitor:
    """
    Manages the lifecycle of a bag (Open -> Closed -> Counted).
    Separated from visualizer and model logic.
    """
    def __init__(self, open_cls_id: int, closed_cls_id: int, n_open: int = 3, m_closed: int = 4):
        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        self.n_open = n_open
        self.m_closed = m_closed

        # Format: {track_id: {'state': str, 'open_c': int, 'closed_c': int}}
        self.states = {}

    def update(self, track_id: int, class_id: int) -> str:
        """
        Updates state for a track ID.
        Returns 'READY_TO_CLASSIFY' if the cycle completes, else None.
        """
        if track_id not in self.states:
            self.states[track_id] = {'state': 'detecting_open', 'open_c': 0, 'closed_c': 0}

        info = self.states[track_id]
        event = None

        if info['state'] == 'detecting_open':
            if class_id == self.open_id:
                info['open_c'] += 1
            else:
                info['open_c'] = 0

            if info['open_c'] >= self.n_open:
                info['state'] = 'detecting_closed'
                info['open_c'] = 0 # reset

        elif info['state'] == 'detecting_closed':
            if class_id == self.closed_id:
                info['closed_c'] += 1
            else:
                info['closed_c'] = 0

            if info['closed_c'] >= self.m_closed:
                info['state'] = 'counted'
                event = 'READY_TO_CLASSIFY'

        return event

    def cleanup(self, active_track_ids: set):
        """Removes lost tracks to prevent memory leaks."""
        # In this logic, we just reset counts for lost tracks, or you can delete them
        for tid in list(self.states.keys()):
            if tid not in active_track_ids:
                # Resetting is safer than deleting if tracking is jittery
                self.states[tid]['open_c'] = 0
                self.states[tid]['closed_c'] = 0
