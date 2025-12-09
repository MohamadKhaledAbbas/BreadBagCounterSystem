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


def group_low_frequency_types(classifications: List[Dict], threshold: int = 10) -> List[Dict]:
    """
    Group bag types with count < threshold into a single "Other" category.
    
    Args:
        classifications: List of classification dictionaries with 'count', 'weight', etc.
        threshold: Minimum count to be considered a significant type (default 10)
    
    Returns:
        List of classifications with low-frequency types grouped as "Other"
    """
    if not classifications:
        return []
    
    significant = []
    other_items = []
    
    for item in classifications:
        if item["count"] >= threshold:
            significant.append(item)
        else:
            other_items.append(item)
    
    # If there are low-frequency items, create an "Other" category
    if other_items:
        other_count = sum(item["count"] for item in other_items)
        other_weight = sum(item.get("weight", 0) for item in other_items)
        
        other_category = {
            "id": -1,  # Special ID for "Other"
            "name": "Other",
            "arabic_name": "أنواع قليلة",
            "number_of_breads": None,
            "weight": other_weight,
            "thumb": None,  # Could use a generic icon
            "is_known": False,
            "count": other_count,
            "is_other": True,  # Flag to identify this as the "Other" category
            "grouped_types": [item["name"] for item in other_items]  # Track which types are grouped
        }
        significant.append(other_category)
    
    return significant


def build_consecutive_runs(ordered_events: List[Dict]) -> List[Dict]:
    """
    Collapse ordered events into consecutive runs of the same bag_type_id.
    Each run tracks start/end times and total count.
    """
    runs = []
    current = None

    for ev in ordered_events:
        if current is None or ev["bag_type_id"] != current["bag_type_id"]:
            if current:
                runs.append(current)
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

    if current:
        runs.append(current)
    return runs


def get_stats(start_time: datetime, end_time: datetime, group_low_freq: bool = True):
    stats = db.get_aggregated_stats(start_time, end_time)

    # Group low-frequency types as "Other" in classifications
    if group_low_freq and stats["data"]["classifications"]:
        stats["data"]["classifications"] = group_low_frequency_types(
            stats["data"]["classifications"], 
            threshold=10
        )
        logger.info(
            f"[Analytics] Grouped low-frequency types: "
            f"{len(stats['data']['classifications'])} types after grouping"
        )

    # Get timeline data
    ordered_events = db.get_ordered_bag_events(start_time, end_time)
    per_class_windows = db.get_per_class_time_windows(start_time, end_time)
    runs = build_consecutive_runs(ordered_events)

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
            # Skip thumb replacement for "Other" category (it has no thumb)
            if c.get("thumb"):
                c["thumb"] = c["thumb"].replace("data/classes/", "known_classes/").replace("data/unknown/",
                                                                                           "unknown_classes/")

        # Fix image paths for timeline events
        for event in stats["timeline"]["ordered_events"]:
            event["thumb"] = event["thumb"].replace("data/classes/", "known_classes/").replace("data/unknown/",
                                                                                               "unknown_classes/")

        # Fix image paths for per-class windows
        for class_id, class_data in stats["timeline"]["per_class_windows"].items():
            class_data["thumb"] = class_data["thumb"].replace("data/classes/", "known_classes/").replace(
                "data/unknown/", "unknown_classes/")

        # Fix image paths for consecutive runs
        for run in stats["timeline"].get("runs", []):
            run["thumb"] = run["thumb"].replace("data/classes/", "known_classes/").replace("data/unknown/",
                                                                                           "unknown_classes/")

        # Adjusting timezone for preview +3
        stats["meta"]["start"] = start_dt + timedelta(hours=3)
        stats["meta"]["end"] = end_dt + timedelta(hours=3)

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


@router.get("/analytics/low-confidence")
async def get_low_confidence_events(
        start_time: Optional[str] = Query(None, description="Start Time (ISO Format)"),
        end_time: Optional[str] = Query(None, description="End Time (ISO Format)"),
        limit: Optional[int] = Query(100, description="Maximum number of events to return")
):
    """
    Get low-confidence classification events for review and retraining.
    Returns events where is_low_confidence flag is true.
    """
    if start_time is None or end_time is None:
        # Default to last 24 hours
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(hours=24)
    else:
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
    
    # Adjust for timezone
    start_dt = start_dt - timedelta(hours=3)
    end_dt = end_dt - timedelta(hours=3)
    
    try:
        events = db.get_low_confidence_events(start_dt, end_dt, limit)
        
        # Fix image paths
        for event in events:
            if event.get("thumb"):
                event["thumb"] = event["thumb"].replace("data/classes/", "known_classes/").replace(
                    "data/unknown/", "unknown_classes/"
                )
        
        logger.info(
            f"[Analytics/LowConfidence] Retrieved {len(events)} low-confidence events "
            f"between {start_dt} and {end_dt}"
        )
        
        return {
            "meta": {
                "start": (start_dt + timedelta(hours=3)).isoformat(),
                "end": (end_dt + timedelta(hours=3)).isoformat(),
                "count": len(events),
                "limit": limit
            },
            "events": events
        }
    except Exception as e:
        logger.error(f"[Analytics/LowConfidence] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/daily", response_class=HTMLResponse)
async def get_daily_analytics(
        request: Request
):
    """
    Daily analytics endpoint that automatically sets the shift time window.
    End time is now based on the last event timestamp on or before 16:00,
    ensuring all events for the shift are included.
    """
    time_now = datetime.now() + timedelta(hours=3)

    if time_now.hour in [16, 17, 18, 19, 20, 21, 22, 23]:
        start_time = time_now
        end_time_target = (time_now + timedelta(days=1))
    else:  # time_now.hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        start_time = (time_now - timedelta(days=1))
        end_time_target = time_now

    # Set start to 16:00
    start_time = start_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Set target end to 16:00 next day
    end_time_cutoff = end_time_target.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Query for the last event on or before the cutoff time
    # Adjust for timezone offset (-3 hours for database storage)
    last_event_time = db.get_last_event_time_before(end_time_cutoff - timedelta(hours=3))
    
    if last_event_time:
        # Use the actual last event time (add back timezone offset for display)
        end_time_actual = last_event_time + timedelta(hours=3)
        logger.info(
            f"[Analytics/Daily] Using last event time as end: {end_time_actual.isoformat()} "
            f"(cutoff was {end_time_cutoff.isoformat()})"
        )
        end_time = end_time_actual
    else:
        # No events found, use the cutoff time
        logger.warning(
            f"[Analytics/Daily] No events found before cutoff {end_time_cutoff.isoformat()}, "
            f"using cutoff as end time"
        )
        end_time = end_time_cutoff

    # Format for analytics endpoint (ISO Format)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    logger.info(
        f"[Analytics/Daily] Shift window: {start_time_str} to {end_time_str}"
    )
    
    return await analytics(request=request, start_time=start_time_str, end_time=end_time_str)

# To run: uvicorn server:app --reload