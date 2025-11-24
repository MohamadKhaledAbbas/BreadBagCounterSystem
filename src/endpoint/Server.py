from fastapi import FastAPI, HTTPException, Query
from datetime import datetime

from src.logging.Database import DatabaseManager

app = FastAPI(title="Bag Counting Analytics API")
db = DatabaseManager("bag_events.db")  # Connects to the same file

@app.get("/analytics/counts")
def get_bag_counts(
    start_time: datetime = Query(..., description="Start Time (ISO Format), e.g. 2025-11-24T08:00:00"),
    end_time: datetime = Query(..., description="End Time (ISO Format), e.g. 2025-11-24T18:00:00")
):
    """
    Get accumulated counts of bags per class within a specific time range.
    """
    if start_time > end_time:
        raise HTTPException(status_code=400, detail="Start time must be before end time")

    try:
        stats = db.get_aggregated_stats(start_time, end_time)
        return {
            "meta": {
                "start": start_time,
                "end": end_time
            },
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn server:app --reload