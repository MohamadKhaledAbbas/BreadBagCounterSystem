
import sqlite3
from typing import Any, Iterable, List, Tuple, Optional

class DBCLI:
    def __init__(self, path: str):
        self.path = path

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def query(self, sql: str, params: Optional[Iterable[Any]] = None) -> List[sqlite3.Row]:
        params = params or ()
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        return rows

    def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> int:
        params = params or ()
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount

    def print_rows(self, rows: List[sqlite3.Row]):
        if not rows:
            print("No results.")
            return
        columns = rows[0].keys()
        print(" | ".join(columns))
        print("-" * (len(columns) * 8))
        for row in rows:
            print(" | ".join(str(row[col]) for col in columns))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python db_cli.py <db_path> '<SQL>' [params...]")
        sys.exit(1)

    db_path = sys.argv[1]
    sql = sys.argv[2]
    params = tuple(sys.argv[3:]) or ()

    cli = DBCLI(db_path)

    if sql.strip().lower().startswith(("select", "pragma")):
        rows = cli.query(sql, params)
        cli.print_rows(rows)
    else:
        count = cli.execute(sql, params)
        print(f"Rows affected: {count}")
