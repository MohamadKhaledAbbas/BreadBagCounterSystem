#!/bin/sh

echo "[INFO] Checking and attempting to stop Ros2PipelineLauncher.py ..."
if pgrep -f "Ros2PipelineLauncher.py" > /dev/null; then
    pkill -f Ros2PipelineLauncher.py
    echo "[INFO] Ros2PipelineLauncher.py stop signal sent."
else
    echo "[INFO] Ros2PipelineLauncher.py is NOT running."
fi

echo "[INFO] Checking and attempting to stop main.py ..."
if pgrep -f "main.py" > /dev/null; then
    pkill -f main.py
    echo "[INFO] main.py stop signal sent."
else
    echo "[INFO] main.py is NOT running."
fi

echo "[INFO] Checking and attempting to stop uvicorn server ..."
if pgrep -f "uvicorn src.endpoint.server:app --host 192.168.1.206" > /dev/null; then
    pkill -f "uvicorn src.endpoint.server:app --host 192.168.1.206"
    echo "[INFO] Uvicorn server stop signal sent."
else
    echo "[INFO] Uvicorn server is NOT running."
fi

echo "[INFO] Stop sequence complete. You can check with 'ps aux | grep Ros2PipelineLauncher.py', 'ps aux | grep main.py', and 'ps aux | grep uvicorn' for any remaining processes."