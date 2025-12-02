#!/bin/sh

echo "[INFO] Checking process [Ros2PipelineLauncher.py] ..."
if pgrep -f "Ros2PipelineLauncher.py" > /dev/null; then
    pkill -f Ros2PipelineLauncher.py
    echo "[INFO] Ros2PipelineLauncher.py stop signal sent."
else
    echo "[INFO] Ros2PipelineLauncher.py is NOT running."
fi

echo "\n"

echo "[INFO] Checking child process [hobot_codec_republish] of Ros2PipelineLauncher.py ..."
if pgrep -f hobot_codec_republish > /dev/null; then
    pkill -f hobot_codec_republish
    echo "[INFO] hobot_codec_republish stop signal sent.."
else
    echo "[INFO] hobot_codec_republish is NOT running."
fi

echo "\n"

echo "[INFO] Checking child process [hobot_rtsp_client] of Ros2PipelineLauncher.py ..."
if pgrep -f hobot_rtsp_client > /dev/null; then
    pkill -f hobot_rtsp_client
    echo "[INFO] hobot_rtsp_client stop signal sent.."
else
    echo "[INFO] hobot_rtsp_client is NOT running."
fi

echo "\n"

echo "[INFO] Checking process [main.py] ..."
if pgrep -f "main.py" > /dev/null; then
    pkill -f main.py
    echo "[INFO] main.py stop signal sent."
else
    echo "[INFO] main.py is NOT running."
fi

echo "\n"

echo "[INFO] Checking process [uvicorn] ..."
if pgrep -f "uvicorn" > /dev/null; then
    pkill -f "uvicorn"
    echo "[INFO] Uvicorn server stop signal sent."
else
    echo "[INFO] Uvicorn server is NOT running."
fi

echo "\n"

echo "[INFO] Stop sequence complete. You can check with
'ps aux | grep Ros2PipelineLauncher.py'
'ps aux | grep hobot_rtsp_client.py'
'ps aux | grep hobot_codec_republish.py'
'ps aux | grep main.py'
'ps aux | grep uvicorn'
for any remaining processes."