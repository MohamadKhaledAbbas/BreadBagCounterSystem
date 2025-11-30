import sqlite3
import imagehash
from datetime import datetime
from typing import Optional, Tuple, List, Dict


class DatabaseManager:
    def __init__(self, db_path="bag_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

            # Table 1: Defined Bag Types (Known and Unknowns)
            # phash is stored as a hex string
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bag_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    is_known BOOLEAN,
                    phash TEXT,
                    image_path TEXT,
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
            conn.commit()

    def get_or_create_bag_type(self, label: str, phash_obj, image_path: str = None) -> int:
        """
        If label is 'Unknown', tries to find a matching pHash.
        If found, returns that ID. If not, creates 'unknown_X'.
        If label is known (e.g. 'red_bag'), simply returns/creates it.
        """
        phash_str = str(phash_obj) if phash_obj else None

        with sqlite3.connect(self.db_path) as conn:
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
            min_dist = 10

            for uid, uname, uhash in unknowns:
                if uhash:
                    stored_hash = imagehash.hex_to_hash(uhash)
                    dist = current_hash - stored_hash
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = uid

            if best_match_id is not None and min_dist <= 8:  # 8 is a safe threshold
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO bag_events (bag_type_id, track_id, confidence)
                VALUES (?, ?, ?)
            """, (bag_type_id, track_id, confidence))

    def get_aggregated_stats(self, start_time, end_time):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            bag_type_stats = conn.execute("""
                SELECT
                    bt.id,
                    bt.name AS bag_type,
                    bt.image_path,
                    bt.is_known,
                    COUNT(be.id) AS count
                FROM bag_types bt
                LEFT JOIN bag_events be
                    ON bt.id = be.bag_type_id AND be.timestamp BETWEEN ? AND ?
                GROUP BY bt.id, bt.name, bt.image_path, bt.is_known
                ORDER BY count DESC
            """, (start_time, end_time)).fetchall()

            total_count = conn.execute("""
                SELECT COUNT(*) FROM bag_events
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time)).fetchone()[0]

        # Adapt this for your return structure (dict/list suitable for template)
        stats = {
            "total": {
                "count": total_count,
                # "weight": ... if you ever add that
            },
            "classifications": [
                {
                    "id": row["id"],
                    "name": row["bag_type"],
                    "thumb": row["image_path"],
                    "is_known": bool(row["is_known"]),
                    "count": row["count"],
                    # "weight": ... if/when you store one for each event or type
                } for row in bag_type_stats
            ]
        }
        return stats