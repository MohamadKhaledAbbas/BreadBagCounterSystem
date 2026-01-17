import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

class FrameSubscriber(Node):
    def __init__(self, topic_name="/nv12_images", save_video=False, video_filename="output.mp4", fps=30):
        super().__init__('frame_subscriber_node')
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            qos
        )
        self.latest_frame = None
        self.frame_counter = 0
        self.winname = "ROS2 FrameSubscriber"
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Subscribed to topic '{topic_name}' as sensor_msgs/msg/Image.")
        # Saving options
        self.save_video = save_video
        self.video_filename = video_filename
        self.fps = fps
        self.video_writer = None
        self.video_size = None

    def listener_callback(self, msg):
        h = msg.height
        w = msg.width
        # Convert NV12 data to BGR
        # img_data = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        expected = msg.height * msg.width * 3 // 2
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        if img_data.size != expected:
            self.get_logger().error(f"Bad NV12 size: got {img_data.size}, expected {expected}")
            return
        nv12_img = img_data.reshape((msg.height * 3 // 2, msg.width))

        bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)

        # Resize for display, keep original for saving
        resized_bgr = cv2.resize(bgr, (1024, 1024))
        cv2.imshow(self.winname, resized_bgr)

        # Optional: Save frame to video
        if self.save_video:
            # Lazy initialization of VideoWriter
            if self.video_writer is None:
                self.video_size = (bgr.shape[1], bgr.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.video_filename, fourcc, self.fps, self.video_size
                )
                self.get_logger().info(f"Saving video to {self.video_filename} at {self.fps} FPS, size={self.video_size}")
            self.video_writer.write(bgr)

        self.frame_counter += 1
        # Optionally save latest frame as image (add flag if desired)
        # if self.save_frames:
        #     cv2.imwrite(f"frame_{self.frame_counter:04d}.png", bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def get_latest_frame(self):
        if self.latest_frame is None:
            return None
        return self.latest_frame.copy()

    def close_node(self):
        # Release video writer if in use
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Video saved to {self.video_filename}")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    # Set save_video=True to enable MP4 saving
    node = FrameSubscriber(topic_name="/nv12_images", save_video=False, video_filename="/home/sunrise/BreadCounting/data/nv12_output.mp4", fps=20)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()