from typing import Optional

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

def get_stats(start_time: datetime, end_time: datetime):
    stats = db.get_aggregated_stats(start_time, end_time)
    return {
        "meta": {
            "start": start_time,
            "end": end_time
        },
        "data": stats
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
            c["thumb"] = c["thumb"].replace("data/classes/", "known_classes/").replace("data/unknown/","unknown_classes/")

        # Adjusting timezone for preview +3
        stats["meta"]["start"] = start_dt + timedelta(hours=3)
        stats["meta"]["end"] = end_dt + timedelta(hours=3)

        logger.info(f"[Analytics] Serving analytics: total={stats['data']['total']}, classes={len(stats['data']['classifications'])}")
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "meta": stats["meta"],
            "total": stats["data"]["total"],
            "classifications":  stats["data"]["classifications"],
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
    else: # time_now.hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        start_time = (time_now - timedelta(days=1))
        end_time = time_now

    # Time (ISO Format), e.g. 2025-11-24T08:00:00
    start_time = start_time.replace(hour=16, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time.replace(hour=11, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%S")
    return await analytics(request=request, start_time=start_time, end_time=end_time)

# To run: uvicorn server:app --reload
