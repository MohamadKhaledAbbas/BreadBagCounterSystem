import sqlite3
import imagehash
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
import threading

from src.constants import CONFIG_KEYS
from src.utils.AppLogging import logger


class DatabaseManager:
    def __init__(self, db_path="bag_data.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get a thread-local database connection, creating one if needed."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.execute("PRAGMA journal_mode=WAL;")
        try:
            yield self._local.connection
            self._local.connection.commit()
        except Exception:
            self._local.connection.rollback()
            raise

    def close(self):
        """Close the thread-local database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
            except Exception as e:
                logger.debug(f"[DatabaseManager] Error closing connection: {e}")
            finally:
                self._local.connection = None

    def _init_db(self):
        with self.get_connection() as conn:

            # Table 1: Defined Bag Types (Known and Unknowns)
            # phash is stored as a hex string
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bag_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    arabic_name TEXT,
                    is_known BOOLEAN,
                    phash TEXT,
                    image_path TEXT,
                    number_of_breads INTEGER,
                    weight REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table 2: Counting Events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bag_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bag_type_id INTEGER,
                    track_id INTEGER,
                    confidence FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(bag_type_id) REFERENCES bag_types(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            for config_key in CONFIG_KEYS:
                conn.execute(
                    "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                    (config_key, '0')
                )

    def get_or_create_bag_type(self, label: str, phash_obj, image_path: str = None) -> int:
        """
        If label is 'Unknown', tries to find a matching pHash.
        If found, returns that ID. If not, creates 'unknown_X'.
        If label is known (e.g. 'red_bag'), simply returns/creates it.
        """
        phash_str = str(phash_obj) if phash_obj else None

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Handle KNOWN classes
            if label != "Unknown":
                cursor.execute("SELECT id FROM bag_types WHERE name = ?", (label,))
                row = cursor.fetchone()
                if row:
                    return row[0]

                # Register new known class
                cursor.execute("""
                    INSERT INTO bag_types (name, is_known, image_path) 
                    VALUES (?, 1, ?)
                """, (label, image_path))
                return cursor.lastrowid

            # 2. Handle UNKNOWN classes (pHash logic)
            # Fetch all unknown bag types to compare hashes
            cursor.execute("SELECT id, name, phash FROM bag_types WHERE is_known = 0")
            unknowns = cursor.fetchall()

            current_hash = imagehash.hex_to_hash(phash_str)

            # Threshold for "similarity" (0-5 is usually the same object)
            best_match_id = None
            min_dist = 20

            for uid, uname, uhash in unknowns:
                if uhash:
                    stored_hash = imagehash.hex_to_hash(uhash)
                    dist = current_hash - stored_hash
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = uid

            if best_match_id:  # 8 is a safe threshold
                return best_match_id

            # 3. Create NEW Unknown Type
            # Generate name like unknown_1, unknown_2
            new_count = len(unknowns) + 1
            new_name = f"unknown_bag_{new_count}"

            cursor.execute("""
                INSERT INTO bag_types (name, is_known, phash, image_path) 
                VALUES (?, 0, ?, ?)
            """, (new_name, phash_str, image_path))

            return cursor.lastrowid

    def log_event(self, bag_type_id: int, track_id: int, confidence: float = 1.0):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO bag_events (bag_type_id, track_id, confidence)
                VALUES (?, ?, ?)
            """, (bag_type_id, track_id, confidence))

    def get_aggregated_stats(self, start_time, end_time):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row

            bag_type_stats = conn.execute("""
                SELECT
                    bt.id,
                    bt.name AS bag_type,
                    bt.image_path,
                    bt.is_known,
                    bt.arabic_name,
                    bt.number_of_breads,
                    bt.weight,
                    COUNT(be.id) AS count
                FROM bag_types bt
                LEFT JOIN bag_events be
                    ON bt.id = be.bag_type_id AND be.timestamp BETWEEN ? AND ?
                GROUP BY bt.id, bt.name, bt.image_path, bt.is_known
                ORDER BY count DESC
            """, (start_time, end_time)).fetchall()

            total_count = conn.execute("""
                SELECT COUNT(*) FROM bag_events
                WHERE timestamp BETWEEN ? AND ?;
            """, (start_time, end_time)).fetchone()[0]

            total_weight = conn.execute("""
                SELECT SUM(COALESCE(bt.weight, 0))
                FROM bag_events be
                JOIN bag_types bt ON be.bag_type_id = bt.id
                WHERE be.timestamp BETWEEN ? AND ?;
            """, (start_time, end_time)).fetchone()[0]

        stats = {
            "total": {
                "count": total_count,
                "weight": (total_weight or 0) / 1000
            },
            "classifications": [
                {
                    "id": row["id"],
                    "name": row["bag_type"],
                    "arabic_name": row["arabic_name"],
                    "number_of_breads": row["number_of_breads"],
                    "weight": ((row["weight"] or 0) / 1000),
                    "thumb": row["image_path"],
                    "is_known": bool(row["is_known"]),
                    "count": row["count"],
                } for row in bag_type_stats
            ]
        }

        return stats

    def get_ordered_bag_events(self, start_time, end_time):
        """
        Retrieve time-ordered bag events with bag type information.
        Returns list of events with timestamp, class name, arabic_name, thumb, weight, and count (1 per event).
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            events = conn.execute("""
                SELECT
                    be.id,
                    be.timestamp,
                    bt.id as bag_type_id,
                    bt.name AS class_name,
                    bt.arabic_name,
                    bt.image_path AS thumb,
                    bt.weight,
                    1 as count
                FROM bag_events be
                JOIN bag_types bt ON be.bag_type_id = bt.id
                WHERE be.timestamp BETWEEN ? AND ?
                ORDER BY be.timestamp ASC
            """, (start_time, end_time)).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "bag_type_id": row["bag_type_id"],
                    "class_name": row["class_name"],
                    "arabic_name": row["arabic_name"],
                    "thumb": row["thumb"],
                    "weight": (row["weight"] or 0) / 1000,  # Convert to kg
                    "count": row["count"]
                } for row in events
            ]

    def get_per_class_time_windows(self, start_time, end_time):
        """
        Get first_seen and last_seen timestamps for each bag class within the time window.
        Also returns per-class event lists.
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            class_windows = conn.execute("""
                SELECT
                    bt.id,
                    bt.name AS class_name,
                    bt.arabic_name,
                    bt.image_path AS thumb,
                    bt.weight,
                    MIN(be.timestamp) as first_seen,
                    MAX(be.timestamp) as last_seen,
                    COUNT(be.id) as event_count
                FROM bag_types bt
                JOIN bag_events be ON bt.id = be.bag_type_id
                WHERE be.timestamp BETWEEN ? AND ?
                GROUP BY bt.id, bt.name, bt.arabic_name, bt.image_path, bt.weight
            """, (start_time, end_time)).fetchall()
            
            # Get all events in a single query to avoid N+1 pattern
            all_events = conn.execute("""
                SELECT
                    be.id,
                    be.timestamp,
                    bt.id as bag_type_id,
                    bt.weight,
                    1 as count
                FROM bag_events be
                JOIN bag_types bt ON be.bag_type_id = bt.id
                WHERE be.timestamp BETWEEN ? AND ?
                ORDER BY bt.id, be.timestamp ASC
            """, (start_time, end_time)).fetchall()
            
            # Group events by bag_type_id
            events_by_type = {}
            for event in all_events:
                bag_type_id = event["bag_type_id"]
                if bag_type_id not in events_by_type:
                    events_by_type[bag_type_id] = []
                events_by_type[bag_type_id].append({
                    "id": event["id"],
                    "timestamp": event["timestamp"],
                    "weight": (event["weight"] or 0) / 1000,
                    "count": event["count"]
                })
            
            # Build result dictionary
            result = {}
            for row in class_windows:
                bag_type_id = row["id"]
                result[bag_type_id] = {
                    "class_name": row["class_name"],
                    "arabic_name": row["arabic_name"],
                    "thumb": row["thumb"],
                    "weight": (row["weight"] or 0) / 1000,  # Convert to kg
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "event_count": row["event_count"],
                    "events": events_by_type.get(bag_type_id, [])
                }
            
            return result

    def get_config_value(self, key):
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT value FROM config WHERE key=?", (key,))
                row = cur.fetchone()
                if row is not None:
                    return row[0]  # row = (value,), so row[0] is the value
        except Exception as e:
            logger.error(f"[DatabaseManager] get_config_value error: {e}")

        return None
