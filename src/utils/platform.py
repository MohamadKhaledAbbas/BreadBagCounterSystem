"""Platform detection utilities for RDK vs Windows/other platforms."""
import sys


def is_rdk_platform():
    """Detect if running on RDK board vs Windows/other platforms.
    
    RDK runs Linux ARM and has hobot_dnn libraries available.
    """
    try:
        import hobot_dnn
        return True
    except ImportError:
        try:
            import hobot_dnn_rdkx5
            return True
        except ImportError:
            return False


def is_windows():
    """Check if running on Windows platform."""
    return sys.platform == 'win32'


# Platform detection flags
IS_RDK = is_rdk_platform()
IS_WINDOWS = is_windows()
