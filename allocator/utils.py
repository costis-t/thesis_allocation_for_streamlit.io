from __future__ import annotations
import csv
from typing import Dict, List, Iterable, Any


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def norm_header(h: str) -> str:
    # normalize header: lowercase, strip, replace non-alnum with underscore
    import re
    if h is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", h.strip().lower()).strip("_")


def normalize_headers(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows
    key_map = {k: norm_header(k) for k in rows[0].keys()}
    out = []
    for r in rows:
        out.append({key_map[k]: v for k, v in r.items()})
    return out


def split_pipe(cell: str) -> List[str]:
    if not cell:
        return []
    return [x.strip() for x in str(cell).split("|") if str(x).strip()]


def to_int_or_zero(s: Any) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0
