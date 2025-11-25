from dataclasses import dataclass
from typing import List


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    box: List[float]