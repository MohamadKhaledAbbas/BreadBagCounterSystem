import os
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    # Environment setup actions
    env_setup = [
        # These work for processes started in this launch file
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', '/opt/tros/humble/lib/hobot_shm/config/shm_fastdds.xml'),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '1'),
        SetEnvironmentVariable('ROS_DISABLE_LOANED_MESSAGES', '0'),
        SetEnvironmentVariable('HOME', '/home/sunrise')
        # If you source setup.bash manually before running launch, it's fine. Otherwise, see below for shell script notes.
    ]

    PRODUCTION_RTSP = 'rtsp://admin:a12345678@192.168.2.108:554/cam/realmonitor?channel=1&subtype=0'
    TEST_RTSP = 'rtsp://192.168.1.188:8554/test_cam'

    CURRENT_RTSP = PRODUCTION_RTSP

    rtsp_node = Node(
        package='hobot_rtsp_client',
        executable='hobot_rtsp_client',
        output='screen',
        parameters=[
            {
                'rtsp_url_num': 1,
                'rtsp_url_0': CURRENT_RTSP
            }
        ]
    )

    hw_decode_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        output='screen',
        parameters=[
            {
                 'in_format': 'h264',
                 'out_mode': 'shared_mem',
                 'out_format': 'nv12',
                 'sub_topic': '/rtsp_image_ch_0',
                 'dump_output': False,
                 'pub_topic': '/nv12_images'
            }
        ],
        arguments=['--ros-args', '--log-level', 'ERROR']
    )

    return LaunchDescription(env_setup + [
        rtsp_node,
        hw_decode_node
    ])