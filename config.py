import sqlite3
import argparse
import os

DB_PATH = "data/db/bag_events.db"  # Adjust your path

def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

def set_config(key, value):
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table(conn)
        conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
    print(f"Set config: {key} = {value}")

def get_config(key):
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table(conn)
        cursor = conn.execute("SELECT value FROM config WHERE key=?", (key,))
        row = cursor.fetchone()
        if row:
            return row[0]
        return None

def delete_config(key):
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table(conn)
        cursor = conn.execute("DELETE FROM config WHERE key=?", (key,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Deleted key: {key}")
        else:
            print(f"Key not found: {key}")

def get_all():
    with sqlite3.connect(DB_PATH) as conn:
        ensure_table(conn)
        cursor = conn.execute("SELECT key, value FROM config")
        return cursor.fetchall()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage config key-values in SQLite.")
    parser.add_argument("--key", help="Config key")
    parser.add_argument("--value", help="Config value")
    parser.add_argument("--get", action="store_true", help="Get the value for a key")
    parser.add_argument("--get_all", action="store_true", help="Get all key-values")
    parser.add_argument("--delete", action="store_true", help="Delete a key")
    args = parser.parse_args()

    if args.get_all:
        all_items = get_all()
        if all_items:
            for k, v in all_items:
                print(f"{k} = {v}")
        else:
            print("Config table is empty.")

    elif args.get:
        if not args.key:
            print("Error: --key is required with --get")
        else:
            value = get_config(args.key)
            if value is not None:
                print(f"{args.key} = {value}")
            else:
                print(f"{args.key} not set.")

    elif args.delete:
        if not args.key:
            print("Error: --key is required with --delete")
        else:
            delete_config(args.key)

    elif args.key and args.value:
        set_config(args.key, args.value)

    else:
        parser.print_help()
