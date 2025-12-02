import sqlite3
import argparse

DB_PATH = "data/db/bag_events.db"  # Change to your actual DB file path

def set_config(key, value):
    with sqlite3.connect(DB_PATH) as conn:
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Insert or update value
        conn.execute("""
            INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)
        """, (key, value))
        conn.commit()
    print(f"Set config: {key} = {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set a config key-value in SQLite.")
    parser.add_argument("--key", required=True, help="Config key")
    parser.add_argument("--value", required=True, help="Config value")
    args = parser.parse_args()
    set_config(args.key, args.value)