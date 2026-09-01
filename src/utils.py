"""
utils.py
--------
Shared helpers used across the pipeline:
  - lightweight data containers (plain dicts, kept JSON-serializable on purpose
    so the exact same objects can be cached to disk and reloaded for demo events)
  - disk I/O for demo events
  - small text/number formatting helpers used by the UI

Design note: we deliberately avoid heavy classes/ORMs. Every pipeline stage
passes around plain dicts / lists of dicts. This keeps the "live pipeline" and
the "load cached JSON" paths identical from the UI's point of view.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "data" / "demo_events"


# ---------------------------------------------------------------------------
# Article schema (documented here, enforced loosely at runtime)
# ---------------------------------------------------------------------------
# An "article" dict looks like:
# {
#   "id": "src_abc123",
#   "title": str,
#   "source": str,               # publication name, e.g. "Reuters"
#   "url": str,
#   "published_at": "2026-08-29T09:20:00",   # ISO 8601, naive UTC
#   "text": str,                  # extracted article body
#   "image_url": Optional[str],
#   "image_path": Optional[str],  # local cached path, if downloaded
# }


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_dir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: os.PathLike, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Demo event catalogue
# ---------------------------------------------------------------------------

def list_demo_events() -> List[Dict[str, Any]]:
    """Return lightweight metadata (id, title, summary, source count) for every
    cached demo event so the landing page can render event cards without
    loading full article bodies / embeddings."""
    events = []
    if not DEMO_DIR.exists():
        return events
    for event_dir in sorted(DEMO_DIR.iterdir()):
        event_json = event_dir / "event.json"
        if event_json.exists():
            try:
                meta = load_json(event_json)
                meta["_dir"] = event_dir.name
                events.append(meta)
            except Exception:
                continue
    return events


def load_demo_event(event_dir_name: str) -> Dict[str, Any]:
    """Load a full cached event: event.json (metadata + all analysis results)
    and articles.json (raw article records). This is exactly the same shape
    produced by the live pipeline (see pipeline.run_pipeline)."""
    event_dir = DEMO_DIR / event_dir_name
    event = load_json(event_dir / "event.json")
    articles = load_json(event_dir / "articles.json")
    event["articles"] = articles
    event["_dir"] = event_dir_name
    return event


def save_demo_event(event_dir_name: str, event: Dict[str, Any], articles: List[Dict[str, Any]]) -> None:
    event_dir = DEMO_DIR / event_dir_name
    ensure_dir(event_dir / "images")
    event_to_save = {k: v for k, v in event.items() if k != "articles"}
    save_json(event_dir / "event.json", event_to_save)
    save_json(event_dir / "articles.json", articles)


# ---------------------------------------------------------------------------
# Small formatting helpers used by the UI
# ---------------------------------------------------------------------------

def pct(x: float) -> str:
    return f"{round(x * 100)}%"


def fmt_time_range(articles: List[Dict[str, Any]]) -> str:
    times = []
    for a in articles:
        try:
            times.append(datetime.fromisoformat(a["published_at"]))
        except Exception:
            continue
    if not times:
        return "unknown time range"
    lo, hi = min(times), max(times)
    if lo.date() == hi.date():
        return f"{lo.strftime('%H:%M')}\u2013{hi.strftime('%H:%M')} on {lo.strftime('%d %b %Y')}"
    return f"{lo.strftime('%d %b %Y %H:%M')} \u2013 {hi.strftime('%d %b %Y %H:%M')}"


def unique_sources(articles: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for a in articles:
        if a["source"] not in seen:
            seen.append(a["source"])
    return seen
