from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable


def generate_launch_description():

    env_setup = [
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', '/opt/tros/humble/lib/hobot_shm/config/shm_fastdds.xml'),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '1'),
        SetEnvironmentVariable('ROS_DISABLE_LOANED_MESSAGES', '0'),
    ]

    image_pub = Node(
        package='hobot_image_publisher',
        executable='hobot_image_pub',
        output='screen',
        parameters=[
            {
                # 1. FIX PATH: Double check this file exists!
                'image_source': '/home/sunrise/BreadCounting/output4.h264',
                'image_format': 'h264',

                # 2. DISABLE SHARED MEM: Use standard ROS messages for easier debugging
                'is_shared_mem': False,

                # 3. SET TOPIC: Depending on version, it might be 'msg_pub_topic_name'
                # or 'ros_pub_topic_name'. We set the common one.
                'msg_pub_topic_name': '/test_h264_images',
                'pub_topic': '/test_h264_images',  # keeping this just in case
                'pub_qos_reliability': 'reliable',
            }
        ]
    )

    hw_decode_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        output='screen',
        parameters=[
            {
                'in_mode': 'ros',  # Match publisher (not shared_mem)
                'in_format': 'h264',
                'out_mode': 'ros',  # Output can still be shared_mem for your next node
                'out_format': 'nv12',
                'sub_topic': '/test_h264_images',  # Must match publisher
                'dump_output': False,
                'pub_topic': '/nv12_images',
                'sub_qos_reliability': 'best_effort',
            }
        ]
    )

    return LaunchDescription( env_setup +
        [image_pub, hw_decode_node]
    )