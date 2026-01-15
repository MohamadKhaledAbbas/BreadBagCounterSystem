from typing import Optional, List, Dict

from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Request, HTTPException, Query

from datetime import datetime, timedelta
from src.endpoint.Shared import templates
from src.endpoint.Shared import db
from src.utils.AppLogging import logger

router = APIRouter()


def parse_datetime(val: Optional[str]):
    if not val:
        return None
    val = val.replace(' ', 'T')  # Normalizes both formats
    val = val.strip()
    if len(val) == 16:
        val += ":00"
    elif len(val) == 19:
        pass
    else:
        raise HTTPException(status_code=500, detail="Invalid time format")
    try:
        date_time = datetime.fromisoformat(val)
        return date_time
    except Exception:
        raise HTTPException(status_code=500, detail=f"parsing datetime {val} failed")


def build_consecutive_runs_with_noise(ordered_events: List[Dict]) -> List[Dict]:
    """
    Collapse ordered events into consecutive runs of the same bag_type_id.
    Each run tracks start/end times and total count.
    Runs with count < 3 are grouped and summed as a single 'noise' run.
    """
    runs = []
    noise_count = 0
    noise_threshold = 10
    noise_start = None
    noise_end = None
    current = None

    # Step 1: Initial consecutive run grouping and noise tracking
    for ev in ordered_events:
        if current is None or ev["bag_type_id"] != current["bag_type_id"]:
            if current:
                if current["count"] >= noise_threshold:
                    runs.append(current)
                else:
                    noise_count += current["count"]
                    if noise_start is None or current["start"] < noise_start:
                        noise_start = current["start"]
                    if noise_end is None or current["end"] > noise_end:
                        noise_end = current["end"]
            current = {
                "bag_type_id": ev["bag_type_id"],
                "class_name": ev["class_name"],
                "arabic_name": ev.get("arabic_name"),
                "thumb": ev["thumb"],
                "weight": ev.get("weight") or 0,
                "start": ev["timestamp"],
                "end": ev["timestamp"],
                "count": 1,
            }
        else:
            current["end"] = ev["timestamp"]
            current["count"] += 1

    # Handle last run
    if current:
        if current["count"] >= noise_threshold:
            runs.append(current)
        else:
            noise_count += current["count"]
            if noise_start is None or current["start"] < noise_start:
                noise_start = current["start"]
            if noise_end is None or current["end"] > noise_end:
                noise_end = current["end"]

    # Step 2: Merge adjacent/consecutive runs with same bag_type_id
    merged = []
    last = None
    for run in runs:
        if last is None:
            last = run
        elif run["bag_type_id"] == last["bag_type_id"]:
            # Merge into last
            last["end"] = run["end"]
            last["count"] += run["count"]
        else:
            merged.append(last)
            last = run
    if last:
        merged.append(last)

    # Step 3: Add noise at the end if it exists
    if noise_count > 0:
        merged.append({
            "bag_type_id": "NOISE",
            "class_name": "Noise",
            "arabic_name": "تصنيفات مفلترة غير دقيقة",
            "thumb": "",
            "weight": 0,
            "start": noise_start,
            "end": noise_end,
            "count": noise_count,
        })

    return merged


def get_stats(start_time: datetime, end_time: datetime):
    stats = db.get_aggregated_stats(start_time, end_time)

    # Get timeline data
    ordered_events = db.get_ordered_bag_events(start_time, end_time)
    per_class_windows = db.get_per_class_time_windows(start_time, end_time)
    runs = build_consecutive_runs_with_noise(ordered_events)

    return {
        "meta": {
            "start": start_time,
            "end": end_time
        },
        "data": stats,
        "timeline": {
            "ordered_events": ordered_events,
            "per_class_windows": per_class_windows,
            "runs": runs
        }
    }


@router.get("/analytics", response_class=HTMLResponse)
async def analytics(
        request: Request,
        start_time: Optional[str] = Query(None, description="Start Time (ISO Format), e.g. 2025-11-24T08:00:00"),
        end_time: Optional[str] = Query(None, description="End Time (ISO Format), e.g. 2025-11-24T18:00:00")
):
    if start_time is None or end_time is None:
        return templates.TemplateResponse("analytics_form.html", {"request": request})
    """
    Get accumulated counts of bags per class within a specific time range.
    """
    logger.debug(f"[Analytics] Request: start_time={start_time}, end_time={end_time}")

    start_dt = parse_datetime(start_time)
    end_dt = parse_datetime(end_time)

    logger.debug(f"[Analytics] Parsed: start_dt={start_dt}, end_dt={end_dt}")

    if start_dt > end_dt:
        raise HTTPException(status_code=422, detail="Start time must be before end time")
    try:
        start_dt = start_dt - timedelta(hours=3)
        end_dt = end_dt - timedelta(hours=3)
        stats = get_stats(start_dt, end_dt)
        logger.debug(f"[Analytics] Stats retrieved: {stats}")
        for c in stats["data"]["classifications"]:
            thumb = c.get("thumb")
            if isinstance(thumb, str):
                c["thumb"] = (thumb.replace("data/classes/","known_classes/")
                              .replace("data/unknown/","unknown_classes/"))
            else:
                c["thumb"] = ""

        # Fix image paths for timeline events
        for event in stats["timeline"]["ordered_events"]:
            thumb = event.get("thumb")
            if isinstance(thumb, str):
                event["thumb"] = (thumb.replace("data/classes/", "known_classes/")
                                  .replace("data/unknown/","unknown_classes/"))
            else:
                event["thumb"] = ""

        # Fix image paths for per-class windows
        for class_id, class_data in stats["timeline"]["per_class_windows"].items():
            thumb = class_data.get("thumb")
            if isinstance(thumb, str):
                class_data["thumb"] = (thumb.replace("data/classes/", "known_classes/")
                                       .replace("data/unknown/","unknown_classes/"))
            else:
                class_data["thumb"] = ""

        # Fix image paths for consecutive runs
        for run in stats["timeline"].get("runs", []):
            thumb = run.get("thumb")
            if isinstance(thumb, str):
                run["thumb"] = (thumb.replace("data/classes/", "known_classes/")
                                .replace("data/unknown/","unknown_classes/"))
            else:
                run["thumb"] = ""

        # Adjusting timezone for preview +3
        stats["meta"]["start"] = start_dt + timedelta(hours=3)
        stats["meta"]["end"] = end_dt + timedelta(hours=3)
        stats["meta"]["request_time"] = datetime.now().strftime("%Y/%m/%d - %H:%M:%S")

        logger.info(
            f"[Analytics] Serving analytics: total={stats['data']['total']}, classes={len(stats['data']['classifications'])}")
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "meta": stats["meta"],
            "total": stats["data"]["total"],
            "classifications": stats["data"]["classifications"],
            "timeline": stats["timeline"],
        })
    except Exception as e:
        logger.error(f"[Analytics] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/daily", response_class=HTMLResponse)
async def get_daily_analytics(
        request: Request
):
    time_now = datetime.now() + timedelta(hours=3)

    if time_now.hour in [16, 17, 18, 19, 20, 21, 22, 23]:
        start_time = time_now
        end_time = (time_now + timedelta(days=1))
    else:  # time_now.hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        start_time = (time_now - timedelta(days=1))
        end_time = time_now

    # Time (ISO Format), e.g. 2025-11-24T08:00:00
    start_time = start_time.replace(hour=16, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time.replace(hour=14, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    return await analytics(request=request, start_time=start_time, end_time=end_time)

# To run: uvicorn server:app --reload