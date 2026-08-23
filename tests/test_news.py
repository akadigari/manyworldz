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


# ---- asknews, when the tournament credentials exist ----

from engine import news

RSS = """<rss><channel>
<item><title>rss headline one</title></item>
<item><title>rss headline two</title></item>
</channel></rss>"""


def test_research_prefers_asknews_when_credentials_are_set(monkeypatch):
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "cid")
    monkeypatch.setenv("ASKNEWS_CLIENT_SECRET", "sec")
    monkeypatch.setattr(news, "_asknews_search",
                        lambda query, limit: ["asknews summary A",
                                              "asknews summary B"])
    got = news.research("Will the thing happen?")
    assert got[:2] == ["asknews summary A", "asknews summary B"]


def test_research_falls_back_to_rss_when_asknews_breaks(monkeypatch, tmp_path):
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "cid")
    monkeypatch.setenv("ASKNEWS_CLIENT_SECRET", "sec")
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path)
    def broken(query, limit):
        raise RuntimeError("asknews down")
    monkeypatch.setattr(news, "_asknews_search", broken)
    monkeypatch.setattr(news, "_fetch_xml", lambda url: RSS)
    got = news.research("Will the thing happen?")
    assert "rss headline one" in got


def test_research_ignores_asknews_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(news, "CACHE_DIR", tmp_path)
    def never(query, limit):
        raise AssertionError("asknews must not be called without creds")
    monkeypatch.setattr(news, "_asknews_search", never)
    monkeypatch.setattr(news, "_fetch_xml", lambda url: RSS)
    got = news.research("Will the thing happen?")
    assert "rss headline one" in got
