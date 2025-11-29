from abc import ABC, abstractmethod


class FrameSource(ABC):
    @abstractmethod
    def frames(self):
        """
        Yields tuples (frame, latency_ms)
        """
        pass

    def cleanup(self):
        pass
