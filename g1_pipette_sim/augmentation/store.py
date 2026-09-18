"""One locked coordinator, isolated workers, durable task state and retry counts."""
import json
import sqlite3
from pathlib import Path


class Ledger:
    def __init__(self, root):
        self.db = sqlite3.connect(Path(root) / 'tasks.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, result TEXT)')
        # Called only while the run's exclusive coordinator lock is held.
        self.db.execute("UPDATE tasks SET state='pending' WHERE state='running'")
        self.db.commit()

    def add(self, task):
        self.db.execute("INSERT OR IGNORE INTO tasks(id,state) VALUES (?, 'pending')", (task,))
        self.db.commit()

    def claim(self, task, retries):
        with self.db:
            changed = self.db.execute("UPDATE tasks SET state='running', attempts=attempts+1 WHERE id=? AND state IN ('pending','error') AND attempts < ?", (task, retries)).rowcount
        return bool(changed)

    def finish(self, task, state, result):
        with self.db:
            self.db.execute('UPDATE tasks SET state=?,result=? WHERE id=?', (state, json.dumps(result), task))

    def rows(self):
        return [dict(zip(['id', 'state', 'attempts', 'result'], row)) for row in self.db.execute('SELECT * FROM tasks ORDER BY id')]

    def close(self):
        self.db.close()
