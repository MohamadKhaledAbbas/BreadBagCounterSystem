import os

from src.frame_source.FrameSource import FrameSource
from src.frame_source.OpenCvFrameSource import OpenCVFrameSource
from src.utils.platform import IS_RDK


class FrameSourceFactory:
    @staticmethod
    def create(source_type, **kwargs) -> FrameSource:
        """
        Create a frame source based on the specified type.
        
        Args:
            source_type: 'ros2' or 'opencv'
            
        Kwargs for ROS2:
            topic: ROS2 topic name (default: '/nv12_images')
            target_fps: Target frames per second (default: 30.0)
            
        Kwargs for OpenCV:
            source: Video source - file path, camera index, or RTSP URL (default: 0)
            target_fps: Target FPS for frame pacing (default: None = source FPS)
            testing_mode: If True, enables synchronous on-demand frame reading
                         for testing on slower machines without frame drops.
                         Ideal for development/testing where processing every
                         frame is more important than real-time playback.
                         (default: False)
        
        Returns:
            FrameSource instance
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
            target_fps = kwargs.get('target_fps', 30.0)

            node = FrameServer(topic=topic, target_fps=target_fps)
            return node
        elif source_type.lower() == 'opencv':
            source = kwargs.get('source', 0)  # 0 for webcam, or path/string for file/camera URL
            target_fps = kwargs.get('target_fps', None)  # None = use source FPS
            testing_mode = kwargs.get('testing_mode', False)  # Testing mode for slower machines
            return OpenCVFrameSource(source, target_fps=target_fps, testing_mode=testing_mode)
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