#!/bin/bash

# Configuration file path
CONFIG_PY="./config.py"

# --- Configuration Retrieval ---
echo "=== Current Config ==="
python3 "$CONFIG_PY" --get_all
echo "====================="

# Get flags, ensuring we handle spaces/output correctly
# Using tail -n 1 to specifically get the last line of output if --get outputs multiple lines
IS_PRODUCTION=$(python3 "$CONFIG_PY" --get --key is_production | tail -n 1 | awk -F' = ' '{print $2}' | tr -d '[:space:]')
SHOW_UI_SCREEN=$(python3 "$CONFIG_PY" --get --key show_ui_screen | tail -n 1 | awk -F' = ' '{print $2}' | tr -d '[:space:]')

# --- Set Defaults if Not Set ---
# Note: When checking for defaults, we should check against an empty string or null.
if [ -z "$IS_PRODUCTION" ]; then
    python3 "$CONFIG_PY" --key is_production --value 0
    IS_PRODUCTION=0
fi

if [ -z "$SHOW_UI_SCREEN" ]; then
    python3 "$CONFIG_PY" --key show_ui_screen --value 1
    SHOW_UI_SCREEN=1
fi

echo "[INFO] is_production=$IS_PRODUCTION, show_ui_screen=$SHOW_UI_SCREEN"

# --- Stop all services first for a clean restart/selection ---
echo "[INFO] Stopping all breadcount services for controlled startup..."
sudo supervisorctl stop breadcount-ros2 breadcount-main breadcount-uvicorn > /dev/null

# --- Start services based on is_production ---
if [ "$IS_PRODUCTION" = "1" ]; then
    echo "[INFO] Starting PRODUCTION services (ROS2, MAIN, UVICORN)..."
    # In production, we assume UI is off (show_ui_screen should be set to 0 in your config utility)
    sudo supervisorctl start breadcount-ros2 breadcount-main breadcount-uvicorn

    # You might want to explicitly set the UI flag to 0 in production mode if you rely on the script for configuration
    python3 "$CONFIG_PY" --key show_ui_screen --value 0

else
    echo "[INFO] Starting DEVELOPMENT services (MAIN, UVICORN)..."
    # Only start the core services needed for testing/API access
    sudo supervisorctl start breadcount-main breadcount-uvicorn

    # ROS2 is excluded, allowing main/uvicorn to run simpler if ROS isn't needed.
fi

echo "=== Current Service Status ==="
sudo supervisorctl status breadcount-ros2 breadcount-main breadcount-uvicorn
echo "=============================="