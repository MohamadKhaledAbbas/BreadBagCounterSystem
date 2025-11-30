#!/bin/sh

# Usage:
#   ./run_app.sh --is_production true --show_ui_screen true

IS_PRODUCTION=""
SHOW_UI_SCREEN=""

while [ $# -gt 0 ]; do
    case "$1" in
        --is_production) IS_PRODUCTION="$2"; shift 2;;
        --show_ui_screen) SHOW_UI_SCREEN="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ "$IS_PRODUCTION" = "" ] || [ "$SHOW_UI_SCREEN" = "" ]; then
    echo "Usage: $0 --is_production [true|false] --show_ui_screen [true|false]"
    exit 1
fi

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

PROCESS1_CMD="source /opt/tros/humble/setup.bash && ros2 launch /home/sunrise/BreadCounting/src/ros2/Ros2PipelineLauncher.py"
PROCESS2_CMD="source /opt/tros/humble/setup.bash && python /home/sunrise/BreadCounting/main.py"
MAIN_PY_ARGS="--is_production=$IS_PRODUCTION --show_ui_screen=$SHOW_UI_SCREEN"
UVICORN_CMD="uvicorn src.endpoint.Server:app --host 192.168.1.206"

if [ "$IS_PRODUCTION" = "true" ]; then
    # Check for Ros2PipelineLauncher.py
    if pgrep -f "Ros2PipelineLauncher.py" > /dev/null; then
        echo "[WARN] Ros2PipelineLauncher.py is already running! Skipping launch."
    else
        nohup bash -c "$PROCESS1_CMD" > "$LOG_DIR/ros2_$RUN_TIMESTAMP.log" 2>&1 &
        echo "[INFO] Process 1 started (ros2.launch), log: $LOG_DIR/ros2_$RUN_TIMESTAMP.log"
    fi

    # Check for main.py
    if pgrep -f "main.py" > /dev/null; then
        echo "[WARN] main.py is already running! Skipping launch."
    else
        nohup bash -c "$PROCESS2_CMD $MAIN_PY_ARGS" > "$LOG_DIR/mainpy_$RUN_TIMESTAMP.log" 2>&1 &
        echo "[INFO] Process 2 started (main.py), log: $LOG_DIR/mainpy_$RUN_TIMESTAMP.log"
    fi

    # Check for uvicorn
    if pgrep -f "uvicorn src.endpoint.Server:app --host 192.168.1.206" > /dev/null; then
        echo "[WARN] Uvicorn is already running! Skipping launch."
    else
        nohup bash -c "$UVICORN_CMD" > "$LOG_DIR/uvicorn_$RUN_TIMESTAMP.log" 2>&1 &
        echo "[INFO] Uvicorn server started, log: $LOG_DIR/uvicorn_$RUN_TIMESTAMP.log"
    fi

    echo "[INFO] Process check/launch done."
else
    # Check for main.py
    if pgrep -f "main.py" > /dev/null; then
        echo "[WARN] main.py is already running! Skipping launch."
    else
        nohup bash -c "$PROCESS2_CMD $MAIN_PY_ARGS" > "$LOG_DIR/mainpy_dev_$RUN_TIMESTAMP.log" 2>&1 &
        echo "[INFO] Process 2 started (main.py development mode), log: $LOG_DIR/mainpy_dev_$RUN_TIMESTAMP.log"
    fi

    # Check for uvicorn
    if pgrep -f "uvicorn src.endpoint.Server:app --host 192.168.1.206" > /dev/null; then
        echo "[WARN] Uvicorn is already running! Skipping launch."
    else
        nohup bash -c "$UVICORN_CMD" > "$LOG_DIR/uvicorn_dev_$RUN_TIMESTAMP.log" 2>&1 &
        echo "[INFO] Uvicorn server started (development), log: $LOG_DIR/uvicorn_dev_$RUN_TIMESTAMP.log"
    fi

    echo "[INFO] Process check/launch done."
fi