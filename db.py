"""
Minimal SQLite storage layer for detected plates.
Swap this for asyncpg/SQLModel + Postgres later if you want it to match
your usual production stack (Kestrel-style) — the schema stays the same.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "alpr.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS plate_reads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT NOT NULL,
    ocr_confidence REAL,
    detector_confidence REAL,
    source_image TEXT,
    cropped_image_path TEXT,
    detected_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def save_plate_read(
    plate_text: str,
    ocr_confidence: float | None,
    detector_confidence: float | None,
    source_image: str,
    cropped_image_path: str | None = None,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO plate_reads
               (plate_text, ocr_confidence, detector_confidence, source_image, cropped_image_path, detected_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                plate_text,
                ocr_confidence,
                detector_confidence,
                source_image,
                cropped_image_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_recent_reads(limit: int = 20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM plate_reads ORDER BY detected_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
