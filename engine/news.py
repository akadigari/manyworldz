"""Fresh news for a market question.

Two sources, best first: AskNews (free for FutureEval entrants, what
the ranked bots use; on only when ASKNEWS_CLIENT_ID and
ASKNEWS_CLIENT_SECRET are set) and Google News RSS (free, keyless, the
default). Any AskNews failure falls straight back to RSS, and any RSS
failure returns an empty list: a news outage never stops the cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
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


# AskNews endpoints, from docs.asknews.app. Verify both against the
# docs when the tournament signup lands; a wrong URL is harmless here
# because every failure falls back to RSS.
_ASKNEWS_TOKEN_URL = "https://auth.asknews.app/oauth2/token"
_ASKNEWS_SEARCH_URL = "https://api.asknews.app/v1/news/search"


def _asknews_enabled() -> bool:
    return bool(os.environ.get("ASKNEWS_CLIENT_ID")
                and os.environ.get("ASKNEWS_CLIENT_SECRET"))


def _asknews_search(query: str, limit: int) -> list[str]:
    """Live AskNews pull: OAuth2 client-credentials token, then one
    search. Split out so tests can replace it. Raises on any problem;
    research() catches and falls back to RSS."""
    import requests
    tok = requests.post(
        _ASKNEWS_TOKEN_URL,
        data={"grant_type": "client_credentials", "scope": "news"},
        auth=(os.environ["ASKNEWS_CLIENT_ID"],
              os.environ["ASKNEWS_CLIENT_SECRET"]),
        timeout=20)
    tok.raise_for_status()
    resp = requests.get(
        _ASKNEWS_SEARCH_URL,
        params={"query": query, "n_articles": limit,
                "return_type": "dicts", "method": "nl"},
        headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
        timeout=20)
    resp.raise_for_status()
    articles = resp.json().get("as_dicts") or []
    out = []
    for a in articles:
        line = a.get("summary") or a.get("eng_title") or a.get("title") or ""
        if line:
            out.append(line[:300])
    return out[:limit]


def research(question: str, limit: int = 8) -> list[str]:
    """Do the crowd's homework.

    AskNews first when its credentials exist: real article summaries
    beat bare headlines. Otherwise (or on any AskNews failure) the
    original two-angle Google News RSS search: the question as asked,
    then just its key words. Never raises; worst case an empty list.
    """
    if _asknews_enabled():
        try:
            found = _asknews_search(question, limit)
            if found:
                return found
        except Exception as exc:
            print(f"  asknews failed ({exc}); falling back to RSS")
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
