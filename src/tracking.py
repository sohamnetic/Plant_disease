"""Phase 6 (optional part): per-plant history and spread alerts.

A caller may send a `plant_id` with a photo. Every reliable result for that plant
is stored in SQLite (outputs/tracking.db, git-ignored). The response then gets a
`trend` block comparing with the previous check:
  * affected % change and change per day
  * stage change
  * alerts: new disease, stage got worse, affected % up by >= ALERT_PCT_POINTS,
    or a healthy plant that now shows disease

  GET /plants/{plant_id}/history   -> all stored checks
"""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "tracking.db"
STAGE_INDEX = {"none": 0, "early": 1, "moderate": 2, "severe": 3, "critical": 4}
ALERT_PCT_POINTS = 5.0
PLANT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_LOCK = threading.Lock()


def valid_plant_id(plant_id: str) -> bool:
    return bool(PLANT_ID.match(plant_id))


def _conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, plant_id TEXT NOT NULL, ts TEXT NOT NULL,
        prediction TEXT, stage TEXT, affected_pct REAL, lesion_count INTEGER,
        confidence REAL, status TEXT, filename TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_plant ON checks(plant_id, ts)")
    return c


def history(plant_id: str, path: Path = DB_PATH) -> list[dict]:
    with _LOCK, _conn(path) as c:
        rows = c.execute("SELECT ts, prediction, stage, affected_pct, lesion_count, confidence, status, filename "
                         "FROM checks WHERE plant_id=? ORDER BY ts, id", (plant_id,)).fetchall()
    keys = ["ts", "prediction", "stage", "affected_pct", "lesion_count", "confidence", "status", "filename"]
    return [dict(zip(keys, r)) for r in rows]


def record_and_trend(plant_id: str, result: dict, filename: str | None, path: Path = DB_PATH,
                     now: datetime | None = None) -> dict:
    """Store a reliable result and compare it with the plant's previous check."""
    plant = result.get("plant") or {}
    cur = {
        "ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "prediction": result["prediction"],
        "stage": plant.get("stage", "none"),
        "affected_pct": plant.get("affected_pct"),
        "lesion_count": plant.get("lesion_count", 0),
    }
    prev_rows = history(plant_id, path)
    prev = prev_rows[-1] if prev_rows else None
    with _LOCK, _conn(path) as c:
        c.execute("INSERT INTO checks (plant_id, ts, prediction, stage, affected_pct, lesion_count, confidence, "
                  "status, filename) VALUES (?,?,?,?,?,?,?,?,?)",
                  (plant_id, cur["ts"], cur["prediction"], cur["stage"], cur["affected_pct"], cur["lesion_count"],
                   result["confidence"], result["status"], filename))
    trend = {"plant_id": plant_id, "checks_stored": len(prev_rows) + 1, "previous": prev, "alerts": []}
    if prev is None:
        trend["summary"] = "First check stored for this plant; send another photo in a few days to see the trend."
        return trend

    days = max((datetime.fromisoformat(cur["ts"]) - datetime.fromisoformat(prev["ts"])).total_seconds() / 86400, 1e-6)
    alerts = trend["alerts"]
    if prev["prediction"] == "no_disease" and cur["prediction"] != "no_disease":
        alerts.append(f"New disease on a plant that was healthy: {cur['prediction'].replace('_', ' ')}.")
    elif cur["prediction"] not in ("no_disease", prev["prediction"]):
        alerts.append(f"Diagnosis changed from {prev['prediction'].replace('_', ' ')} to {cur['prediction'].replace('_', ' ')}.")
    if STAGE_INDEX.get(cur["stage"], 0) > STAGE_INDEX.get(prev["stage"], 0):
        alerts.append(f"Stage got worse: {prev['stage']} -> {cur['stage']}.")
    if cur["affected_pct"] is not None and prev["affected_pct"] is not None:
        delta = cur["affected_pct"] - prev["affected_pct"]
        trend["affected_pct_change"] = round(delta, 2)
        trend["affected_pct_change_per_day"] = round(delta / days, 2)
        if delta >= ALERT_PCT_POINTS:
            alerts.append(f"Affected area grew by {delta:.0f} percentage points in {days:.1f} days.")
    trend["days_since_previous"] = round(days, 2)
    if (cur["lesion_count"] or 0) > (prev["lesion_count"] or 0) and cur["prediction"] != "no_disease":
        trend["new_spots"] = int(cur["lesion_count"] - (prev["lesion_count"] or 0))
    trend["summary"] = " ".join(alerts) if alerts else "No worsening since the previous check."
    return trend
