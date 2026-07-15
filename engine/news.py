"""Fresh headlines for a market question — free, no API key.

Google News RSS gives the crowd something real to react to. Failures
return an empty list: a news outage should never stop the cycle.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CACHE_DIR = config.CACHE / "news"


def parse_rss(xml_text: str, limit: int = 3) -> list[str]:
    """Pull item titles out of an RSS feed. Bad XML -> empty list."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    titles = [item.findtext("title") or "" for item in root.iter("item")]
    return [t for t in titles if t][:limit]


def headlines_for(query: str, limit: int = 3) -> list[str]:
    """Today's headlines for a query, cached per day."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{date.today().isoformat()}_{quote(query)[:80]}"
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    try:
        import requests
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US"
        heads = parse_rss(requests.get(url, timeout=15).text, limit)
    except Exception:
        heads = []
    cache_file.write_text(json.dumps(heads))
    return heads
