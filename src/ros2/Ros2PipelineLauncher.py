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
    ]

    db = DatabaseManager("/home/sunrise/BreadCounting/data/db/bag_events.db")
    rtsp_username = db.get_config_value(constants.rtsp_username)
    rtsp_password = db.get_config_value(constants.rtsp_password)
    rtsp_host = db.get_config_value(constants.rtsp_host)
    rtsp_port = db.get_config_value(constants.rtsp_port)

    # NOTE: use subtype=1 (substream) if you want lower bitrate / smaller frames,
    # or use subtype=0 for main stream. Adjust as needed.
    PRODUCTION_RTSP = (
        f"rtsp://{rtsp_username}:{rtsp_password}@{rtsp_host}:{rtsp_port}"
        "/cam/realmonitor?channel=1&subtype=0"
    )

    CURRENT_RTSP = PRODUCTION_RTSP

    # hobot_rtsp_client parameters:
    # - some clients expose flags to force TCP or tune reassembly buffer sizes.
    # - If hobot_rtsp_client supports them, set 'rtsp_transport': 'tcp' to avoid UDP truncation.
    # - If it doesn't, see the suggested patch to increase LIVE555 buffer (below).
    rtsp_node = Node(
        package='hobot_rtsp_client',
        executable='hobot_rtsp_client',
        output='screen',
        parameters=[
            {
                'rtsp_url_num': 1,
                'rtsp_url_0': CURRENT_RTSP,
                # Optional/experimental parameters — only effective if the node reads them:
                'rtsp_transport': 'tcp',     # try TCP interleaved to avoid UDP reassembly limits
                'rtsp_subtype': 0,           # keep examples explicit
                'rtp_reassembly_buffer_bytes': 1048576  # request larger buffer if supported
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
                 'sub_topic': '/spool_image_ch_0',
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