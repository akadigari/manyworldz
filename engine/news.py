"""Fresh headlines for a market question — free, no API key.

Google News RSS gives the crowd something real to react to. Failures
return an empty list: a news outage should never stop the cycle.
"""
from __future__ import annotations

import hashlib
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
    """Pull headline titles out of an RSS feed.

    RSS is just a simple XML format news sites use to list their recent
    articles. Returns up to `limit` headline strings, or an empty list if
    the XML is broken or empty.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    titles = [item.findtext("title") or "" for item in root.iter("item")]
    return [t for t in titles if t][:limit]


def _fetch_xml(url: str) -> str:
    """Grab raw RSS text from the network. Split out so tests can fake it."""
    import requests

    return requests.get(url, timeout=15).text


_STOPWORDS = {"will", "the", "a", "an", "in", "on", "at", "by", "of", "to",
              "be", "is", "are", "was", "this", "that", "it", "its", "before",
              "after", "than", "more", "and", "or", "for", "with", "does",
              "do", "get", "have", "has", "above", "below", "during"}


def key_terms(question: str, max_terms: int = 8) -> str:
    """Boil a question down to its search-worthy words.

    "Will the album drop before July 31?" -> "album drop july 31".
    Dropping filler words gives the news search a second, wider angle.
    """
    words = [w.strip("?,.()!\"'").lower() for w in question.split()]
    kept = [w for w in words if w and w not in _STOPWORDS]
    return " ".join(kept[:max_terms])


def research(question: str, limit: int = 8) -> list[str]:
    """Do the crowd's homework: search the news from two angles.

    Angle 1: the question as asked. Angle 2: just its key words (which
    catches articles that phrase things differently). Results are merged
    with duplicates removed, newest-angle first. Like headlines_for, this
    never raises — worst case is an empty list.
    """
    seen, merged = set(), []
    for query in (question, key_terms(question)):
        if not query:
            continue
        for headline in headlines_for(query, limit=limit // 2 + 1):
            if headline not in seen:
                seen.add(headline)
                merged.append(headline)
    return merged[:limit]


def headlines_for(query: str, limit: int = 3) -> list[str]:
    """Today's headlines for a query, cached per day.

    Never raises: any problem (network, disk, bad cache) returns [].
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Hash the query so no user text ends up in the filename.
        digest = hashlib.sha256(query.encode()).hexdigest()[:16]
        key = f"{date.today().isoformat()}_{digest}"
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        try:
            url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US"
            heads = parse_rss(_fetch_xml(url), limit)
        except Exception:
            heads = []
        cache_file.write_text(json.dumps(heads))
        return heads
    except Exception:
        return []
