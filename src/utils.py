from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def detect_project_root(start_path: Path | None = None) -> Path:
    """Detect the project root by walking up until expected folders are found."""

    start = (start_path or Path.cwd()).resolve()
    candidates = (start, *start.parents)

    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not detect the project root from the provided starting path."
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured module logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def timestamp_utc() -> str:
    """Return an ISO 8601 UTC timestamp for report metadata."""

    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert common Python and pandas-adjacent objects into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(data: Any, output_path: Path) -> None:
    """Write structured data as UTF-8 encoded JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
