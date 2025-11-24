import sqlite3
from datetime import datetime
from typing import Dict, List


class DatabaseManager:
    def __init__(self, db_path="bag_events.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the table with an index for fast time-based queries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bag_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    class_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Index is crucial for performance when filtering by Start/End time
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON bag_logs(timestamp)")
            conn.commit()

    def log_event(self, track_id: int, class_name: str):
        """Insert a new counting event."""
        # We use UTC to ensure consistency across timezones
        now = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bag_logs (track_id, class_name, timestamp) VALUES (?, ?, ?)",
                (track_id, class_name, now)
            )
            conn.commit()

    def get_aggregated_stats(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """
        Returns counts per class within the specific time range.
        Example Return: {'bag_type_A': 10, 'bag_type_B': 5}
        """
        query = """
            SELECT class_name, COUNT(*) 
            FROM bag_logs 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY class_name
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # SQLite stores datetimes as strings, so we convert parameters to string
            cursor.execute(query, (start_time, end_time))
            rows = cursor.fetchall()

        return {row[0]: row[1] for row in rows}