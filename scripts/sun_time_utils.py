from __future__ import annotations

from datetime import datetime


def parse_utc_ms(value: str) -> int:
    if value.endswith("Z"):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(value + "+00:00")
    return int(dt.timestamp() * 1000)
