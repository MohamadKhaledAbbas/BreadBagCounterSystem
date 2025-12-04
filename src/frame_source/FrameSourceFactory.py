
from src.frame_source.FrameSource import FrameSource
from src.frame_source.OpenCvFrameSource import OpenCVFrameSource
from src.utils.platform import IS_RDK


class FrameSourceFactory:
    @staticmethod
    def create(source_type, **kwargs) -> FrameSource:
        """
        source_type: 'ros2' or 'opencv'
        kwargs for ROS2: topic, target_fps
        kwargs for OpenCV: source
        """
        if source_type.lower() == 'ros2':
            if not IS_RDK:
                raise ValueError(
                    "ROS2 frame source only available on RDK platform. "
                    "Use 'opencv' source type on Windows/other platforms."
                )
            # Import ROS2 frame server only when needed (RDK platform)
            from src.frame_source.Ros2FrameServer import FrameServer
            topic = kwargs.get('topic', '/nv12_images')
            target_fps = kwargs.get('target_fps', 10.0)
            node = FrameServer(topic=topic, target_fps=target_fps)
            return node
        elif source_type.lower() == 'opencv':
            source = kwargs.get('source', 0)  # 0 for webcam, or path/string for file/camera URL
            return OpenCVFrameSource(source)
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

# Example usage:
# factory = FrameSourceFactory()
# frame_source = factory.create('ros2', topic='/nv12_images')
# for frame, latency_ms in frame_source.frames():
#     print("ROS2 Frame:", frame.shape, latency_ms)
#
# frame_source = factory.create('opencv', source=0)
# for frame, latency_ms in frame_source.frames():
#     print("OpenCV Frame:", frame.shape, latency_ms)