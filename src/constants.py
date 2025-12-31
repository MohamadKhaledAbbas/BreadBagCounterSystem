show_ui_screen_key = 'show_ui_screen'
is_development_key = 'is_development'
is_recording_key = 'is_recording'
recording_dir = 'recording_dir'
recording_seconds = 'recording_seconds'
recording_fps = 'recording_fps'
rtsp_username = "rtsp_username"
rtsp_password = "rtsp_password"
rtsp_host = "rtsp_host"
rtsp_port = "rtsp_port"
is_profiler_enabled = "is_profiler_enabled"

# Accuracy Mode / Spool configuration keys
accuracy_mode_enabled = "accuracy_mode_enabled"
spool_dir = "spool_dir"
spool_segment_duration = "spool_segment_duration"
spool_retention_seconds = "spool_retention_seconds"
spool_ack_timeout = "spool_ack_timeout"
spool_retry_count = "spool_retry_count"
spool_ack_free_mode = "spool_ack_free_mode"  # V6: Disable ACK-based flow control
spool_target_fps = "spool_target_fps"  # V6: Target FPS for ACK-free mode
# V7: Reliable pacing and smart skipping configuration
spool_max_inflight_frames = "spool_max_inflight_frames"  # Max frames in-flight before backpressure
spool_lag_skip_threshold_seconds = "spool_lag_skip_threshold_seconds"  # Lag threshold to trigger smart skipping
spool_prefer_idr_skip = "spool_prefer_idr_skip"  # Prefer skipping non-IDR frames when lagged

CONFIG_KEYS = [
    show_ui_screen_key,
    is_development_key,
    is_recording_key,
    rtsp_username,
    rtsp_password,
    rtsp_host,
    rtsp_port,
    recording_dir,
    recording_seconds,
    recording_fps,
    is_profiler_enabled,
    # Accuracy Mode / Spool config
    accuracy_mode_enabled,
    spool_dir,
    spool_segment_duration,
    spool_retention_seconds,
    spool_ack_timeout,
    spool_retry_count,
    spool_ack_free_mode,
    spool_target_fps,
    # V7: Reliable pacing config
    spool_max_inflight_frames,
    spool_lag_skip_threshold_seconds,
    spool_prefer_idr_skip,
]