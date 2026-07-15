import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.news import parse_rss

FIXTURE = Path(__file__).parent / "fixtures" / "news_rss.xml"


def test_parses_headlines_and_respects_limit():
    heads = parse_rss(FIXTURE.read_text(), limit=3)
    assert len(heads) == 3
    assert heads[0] == "Album officially announced for July 25 release"


def test_garbage_xml_returns_empty_not_crash():
    assert parse_rss("<not really xml", limit=3) == []
