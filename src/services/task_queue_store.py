"""SQLite-backed persistence for the in-memory analysis task queue.

The queue itself stays in memory for fast access; this store mirrors task
records to disk so that task ids stay queryable across service restarts
(the WebUI polls ``/api/v1/analysis/status/{task_id}`` and would otherwise
receive 404 for every task submitted before a restart).

The module deliberately has no dependency on ``task_queue`` to avoid a
circular import; callers pass and receive plain dictionaries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_queue_tasks (
    task_id          TEXT PRIMARY KEY,
    stock_code       TEXT NOT NULL,
    stock_name       TEXT,
    status           TEXT NOT NULL,
    progress         INTEGER NOT NULL DEFAULT 0,
    message          TEXT,
    result_json      TEXT,
    error            TEXT,
    report_type      TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    original_query   TEXT,
    selection_source TEXT,
    skills_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_queue_created
    ON task_queue_tasks(created_at);
"""

_COLUMNS = (
    "task_id", "stock_code", "stock_name", "status", "progress", "message",
    "result_json", "error", "report_type", "created_at", "started_at",
    "completed_at", "original_query", "selection_source", "skills_json",
)


class TaskQueueStore:
    """Thread-safe SQLite store for task queue records."""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create the schema if needed. Failures are logged, never fatal."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = self._connect()
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.executescript(_SCHEMA)
                    conn.commit()
                finally:
                    conn.close()
                self._initialized = True
                logger.info(f"[TaskQueueStore] 持久化存储就绪: {self._db_path}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"[TaskQueueStore] 初始化失败，任务将仅保存在内存中: {exc}")

    def upsert(self, payload: Dict[str, Any]) -> None:
        """Insert or replace one task record. Persistence failures never raise."""
        try:
            self.initialize()
        except Exception:
            pass
        values = tuple(payload.get(col) for col in _COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT OR REPLACE INTO task_queue_tasks ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(sql, values)
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[TaskQueueStore] 写入任务 {payload.get('task_id')} 失败: {exc}")

    def delete(self, task_id: str) -> None:
        try:
            self.initialize()
        except Exception:
            pass
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute("DELETE FROM task_queue_tasks WHERE task_id = ?", (task_id,))
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[TaskQueueStore] 删除任务 {task_id} 失败: {exc}")

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            self.initialize()
        except Exception:
            pass
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(
                        f"SELECT {', '.join(_COLUMNS)} FROM task_queue_tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                finally:
                    conn.close()
                return dict(row) if row else None
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[TaskQueueStore] 查询任务 {task_id} 失败: {exc}")
                return None

    def load_all(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Load persisted task records, newest first."""
        try:
            self.initialize()
        except Exception:
            pass
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        f"SELECT {', '.join(_COLUMNS)} FROM task_queue_tasks "
                        "ORDER BY created_at DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                finally:
                    conn.close()
                return [dict(r) for r in rows]
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[TaskQueueStore] 加载任务失败: {exc}")
                return []

    @staticmethod
    def encode_json(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return None

    @staticmethod
    def decode_json(value: Optional[str]) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None
