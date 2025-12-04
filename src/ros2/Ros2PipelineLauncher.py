import sys
sys.path.append("/home/sunrise/BreadCounting")

import os
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


from src.logging.Database import DatabaseManager
import src.constants as constants
from src.utils.AppLogging import logger

def generate_launch_description():
    import sys
    logger.debug("[Ros2PipelineLauncher] System paths:\n" + "\n".join(sys.path))

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

    db = DatabaseManager("/home/sunrise/BreadCounting/data/db/bag_events.db")
    rtsp_username = db.get_config_value(constants.rtsp_username)
    rtsp_password = db.get_config_value(constants.rtsp_password)
    rtsp_host = db.get_config_value(constants.rtsp_host)
    rtsp_port = db.get_config_value(constants.rtsp_port)

    PRODUCTION_RTSP = "rtsp://"+rtsp_username+":"+rtsp_password+"@"+rtsp_host+":"+rtsp_port+"/cam/realmonitor?channel=1&subtype=0"

    # PRODUCTION_RTSP = "rtsp://" + rtsp_username + ":a12345678@192.168.2.108:554/cam/realmonitor?channel=1&subtype=0"

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