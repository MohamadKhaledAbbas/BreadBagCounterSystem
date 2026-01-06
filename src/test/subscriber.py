import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hbm_img_msgs.msg import HbmMsg1080P

class FrameSubscriber(Node):
    def __init__(self, topic_name="/nv12_images"):
        super().__init__('frame_subscriber_node')
        self.subscription = self.create_subscription(
            HbmMsg1080P,
            topic_name,
            self.listener_callback,
            qos_profile_sensor_data
        )
        self.latest_frame = None
        self.frame_counter = 0
        self.winname = "ROS2 FrameSubscriber"
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Subscribed to topic '{topic_name}' as HbmMsg1080P.")

    def listener_callback(self, msg):
        h = msg.height
        w = msg.width
        frame_len = int(h * w * 1.5)
        img_data = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        nv12_img = img_data.reshape((msg.height * 3 // 2, msg.width))
        bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
        resized_bgr = cv2.resize(bgr, (1024, 1025))
        cv2.imshow(self.winname, resized_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            import rclpy
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def get_latest_frame(self):
        if self.latest_frame is None:
            return None
        return self.latest_frame.copy()

    def close_node(self):
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FrameSubscriber("/nv12_images")
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