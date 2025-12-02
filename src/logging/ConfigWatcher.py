# ConfigWatcher.py
import threading
import time
import sqlite3

class ConfigWatcher:
    def __init__(self, db_path, poll_interval=0.5):
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None

        # store last-known values
        self.last_values = {}

        # callbacks: key -> function
        self.callbacks = {}

    def add_watch(self, key, callback):
        self.callbacks[key] = callback

    def _fetch_config(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM Config")
        rows = cur.fetchall()
        conn.close()
        return {k: v for (k, v) in rows}

    def _loop(self):
        while self.running:
            current = self._fetch_config()

            # look for changes
            for key, callback in self.callbacks.items():
                new_value = current.get(key)
                old_value = self.last_values.get(key)

                if old_value != new_value:
                    # update before callback to avoid race
                    self.last_values[key] = new_value
                    callback(new_value)

            # update snapshot
            self.last_values = current

            time.sleep(self.poll_interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
