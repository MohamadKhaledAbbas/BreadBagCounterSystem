# Log Analyzer HTML Report - Visual Example

This document shows what the generated HTML report looks like.

## Report Structure

### 1. Header Section
```
🍞 BreadBag Counter - Log Analysis Report
==========================================
```

### 2. Summary KPI Cards (6 cards with color-coded status)

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Total Log Entries   │  │ Errors              │  │ Warnings            │
│      500            │  │   3   [GREEN]       │  │   24  [GREEN]       │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Unknown Rate        │  │ Avg FPS             │  │ Bags Counted        │
│   0.0%  [GREEN]     │  │  25.4  [GREEN]      │  │   33                │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

Color coding:
- **GREEN**: Good (gradient: teal to green)
- **YELLOW**: Warning (gradient: orange to yellow)
- **RED**: Error (gradient: red to orange)

### 3. Time Range Table

| Start Time (UTC)          | End Time (UTC)            | Duration     |
|---------------------------|---------------------------|--------------|
| 2025-12-16T00:00:00+00:00 | 2025-12-16T23:59:59.999999+00:00 | 24.00 hours |

### 4. Issue Findings (Expandable Cards)

```
🚨 Issue Findings
=================

[ERROR BADGE] Frame Drops Due to Backpressure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Description: Total frames dropped: 161
Likely Cause: System cannot keep up with input frame rate (CPU/GPU overload)
Where to Look: Check frame processing times, consider reducing input FPS or optimizing models

[WARNING BADGE] High Frame Processing Time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Description: P95 frame time: 54.8ms (avg: 48.7ms)
Likely Cause: Detection or monitoring bottleneck, hardware limitations
Where to Look: Compare detection_time_ms vs monitor_time_ms to identify bottleneck component

[ERROR BADGE] High Event Expiration Rate (Under-counting Risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Description: Expiration rate: 50.0% (17/34)
Likely Cause: Bags not tracked long enough (too fast), insufficient hits for state transitions
Where to Look: Check event_expired_by_state breakdown, consider lowering thresholds
```

### 5. Frame Performance Chart

```
📈 Frame Performance
====================

Bar Chart showing:
- X-axis: Avg | P50 | P95 | Max
- Y-axis: Time (ms)
- Two series:
  * Blue bars: Total Frame Time (42.8, 42.8, 54.8, 54.8 ms)
  * Red bars: Detection Time (32.5, 32.5, 41.5, 41.5 ms)
```

**Performance Table:**

| Metric            | Count | Avg    | P50    | P95    | Max    |
|-------------------|-------|--------|--------|--------|--------|
| Total Frame Time  | 167   | 48.7ms | 48.8ms | 54.8ms | 54.8ms |
| Detection Time    | 167   | 34.4ms | 34.5ms | 41.5ms | 41.5ms |
| Monitor Time      | 167   | 6.6ms  | 6.3ms  | 8.3ms  | 8.3ms  |
| FPS               | 167   | 25.4   | 25.4   | 27.4   | 27.4   |

### 6. Counting Metrics Table

| Metric               | Count |
|----------------------|-------|
| Events Created       | 34    |
| Events Expired       | 17    |
| Events Suppressed    | 34    |
| Count Updates        | 33    |

### 7. Bag Type Distribution (Pie Chart)

```
🎯 Counting Metrics
===================

Pie Chart showing:
- White: 17 bags (51.5%)
- Bran: 16 bags (48.5%)

Colors: Each type gets a different color from a palette
```

### 8. Event Expiration by State Table

| State              | Count |
|--------------------|-------|
| detecting_closed   | 17    |

### 9. Classification Quality Section

| Metric                    | Value           |
|---------------------------|-----------------|
| Total Classifications     | 33              |
| Unknown Classifications   | 0 (0.0%)        |
| Avg Confidence            | 0.920           |

**Rejection Reasons Table:**

| Reason          | Count |
|-----------------|-------|
| (none shown)    | 0     |

### 10. Top Errors Table

| Component   | Message                                                          | Count |
|-------------|------------------------------------------------------------------|-------|
| LogicThread | [ERROR] component=LogicThread, op=frame_processing, type=ValueError... | 3     |

### 11. Top Warnings Table

| Component        | Message                                                      | Count |
|------------------|--------------------------------------------------------------|-------|
| BagStateMonitor  | [EVENT_EXPIRED] id=..., state=detecting_closed...           | 17    |
| BagCounterApp    | [BACKPRESSURE] queue=input_queue, util=85.0%...             | 7     |

### 12. Time Series Chart (Multi-axis Line Chart)

```
⏱️ Time Series
==============

Line Chart showing per-minute trends:
- Left Y-axis (count): Errors, Warnings, Unknown Classifications
- Right Y-axis (fps): Avg FPS
- X-axis: Time (HH:MM format)

Example data points:
00:00 - Errors: 1, Warnings: 6, Backpressure: 46 drops, FPS: 25.4
00:01 - Errors: 1, Warnings: 6, Backpressure: 46 drops, FPS: 25.4
00:02 - Errors: 0, Warnings: 6, Backpressure: 46 drops, FPS: 25.4
00:03 - Errors: 1, Warnings: 5, Backpressure: 23 drops, FPS: 25.4
00:04 - Errors: 0, Warnings: 1, Backpressure: 0 drops, FPS: 25.3
```

### 13. Footer

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report Generated: 2025-12-16T10:44:32.123456Z
Total Entries Parsed: 500 | Skipped: 0
Analyzer Version: 1.0
```

## Visual Design Features

### Color Scheme
- **Primary**: Blue (#3498db) for headers and primary actions
- **Success**: Green gradient for good metrics
- **Warning**: Yellow gradient for concerning metrics
- **Error**: Red gradient for critical issues
- **Background**: Light gray (#f5f5f5)
- **Cards**: White with subtle shadow

### Typography
- **Font**: System fonts (Segoe UI on Windows, San Francisco on macOS, Roboto on Linux)
- **Headers**: Bold, color-coded by section
- **Tables**: Zebra-striping on hover
- **KPI Cards**: Large numbers (32px) with small labels (14px)

### Layout
- **Container**: Max-width 1400px, centered
- **Grid**: Responsive KPI cards (auto-fit, min 250px)
- **Charts**: Fixed height 400px, responsive width
- **Tables**: Full width, collapsible on mobile

### Interactive Elements
- **Charts**: Hover to see exact values (Chart.js tooltips)
- **Tables**: Hover to highlight rows
- **Cards**: Subtle shadow that increases on hover
- **Links**: Smooth transitions

## Browser Compatibility

The report works in:
- ✅ Chrome 90+ (Windows, macOS, Linux)
- ✅ Firefox 88+ (Windows, macOS, Linux)
- ✅ Safari 14+ (macOS)
- ✅ Edge 90+ (Windows)

## Accessibility

- Semantic HTML5 structure
- ARIA labels for charts
- Color contrast meets WCAG AA standards
- Keyboard navigation support
- Print-friendly styles

## File Size

- HTML: ~19 KB (uncompressed)
- Chart.js: ~200 KB (loaded from CDN, cached by browser)
- Total first load: ~220 KB
- Subsequent loads: ~19 KB (Chart.js cached)

## Opening the Report

### Windows
```powershell
# Option 1: Open in default browser
Start-Process reports\2025-12-16\report.html

# Option 2: Double-click the file in Explorer
explorer reports\2025-12-16\
```

### Linux/macOS
```bash
# Option 1: Open in default browser
xdg-open reports/2025-12-16/report.html  # Linux
open reports/2025-12-16/report.html      # macOS

# Option 2: Open in specific browser
google-chrome reports/2025-12-16/report.html
firefox reports/2025-12-16/report.html
```

## Customization

The HTML report can be customized by:
1. Modifying the CSS in `generate_html_report()` function
2. Adding/removing sections
3. Changing Chart.js configuration
4. Adjusting color schemes
5. Modifying threshold values for issue detection

See `tools/log_analyzer.py` starting at line ~600 for the HTML generation code.
