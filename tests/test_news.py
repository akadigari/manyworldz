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


def test_slash_in_query_does_not_crash(tmp_path, monkeypatch):
    from engine import news
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path / "news")
    monkeypatch.setattr(news, "_fetch_xml", lambda url: FIXTURE.read_text())
    heads = news.headlines_for("AC/DC reunion tour")
    assert len(heads) == 3


def test_network_failure_returns_empty(tmp_path, monkeypatch):
    from engine import news
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path / "news")

    def boom(url):
        raise OSError("no internet")

    monkeypatch.setattr(news, "_fetch_xml", boom)
    assert news.headlines_for("anything") == []


def test_corrupted_cache_returns_empty_not_crash(tmp_path, monkeypatch):
    from engine import news
    import hashlib
    from datetime import date
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path / "news")
    (tmp_path / "news").mkdir(parents=True)
    key = f"{date.today().isoformat()}_{hashlib.sha256(b'q').hexdigest()[:16]}"
    (tmp_path / "news" / f"{key}.json").write_text("{corrupted")
    monkeypatch.setattr(news, "_fetch_xml", lambda url: "irrelevant")
    assert news.headlines_for("q") == []
