"""
ROS2 Spool Module for Accuracy Mode.

Provides ROS2 nodes for H.264 frame recording and pull-based replay
to enable accurate frame-by-frame processing without drops.

Note: Import the node classes explicitly when needed to avoid
RuntimeWarning about module execution order:

    from src.ros2_spool.spool_recorder_node import SpoolRecorderNode
    from src.ros2_spool.spool_processor_node import SpoolProcessorNode
"""

# Avoid importing node modules at package level to prevent RuntimeWarning
# when running as `python -m src.ros2_spool.spool_*_node`
__all__ = [
    'SpoolRecorderNode',
    'SpoolProcessorNode',
]


def get_recorder_node():
    """Lazy import of SpoolRecorderNode."""
    from src.ros2_spool.spool_recorder_node import SpoolRecorderNode
    return SpoolRecorderNode


def get_processor_node():
    """Lazy import of SpoolProcessorNode."""
    from src.ros2_spool.spool_processor_node import SpoolProcessorNode
    return SpoolProcessorNode
