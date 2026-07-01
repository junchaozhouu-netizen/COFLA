from __future__ import annotations

from datetime import datetime


def now_str() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def format_elapsed_hms(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(round(float(elapsed_seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_elapsed_record(elapsed_seconds: float) -> dict[str, float | str]:
    elapsed_seconds = round(max(0.0, float(elapsed_seconds)), 3)
    return {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": format_elapsed_hms(elapsed_seconds),
    }
