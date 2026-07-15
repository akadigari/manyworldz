"""Hide who is playing, keep what the numbers say.

mask_statsheet() turns a stat-sheet into text where the only identities are
"Team A" (home) and "Team B" (away). The re-ID probe (below) then measures
how often models can un-hide the teams anyway. We publish that number.
"""
from __future__ import annotations

# Every NBA team as (abbreviation, city, nickname) — e.g. ("BOS", "Boston", "Celtics").
NBA_TEAMS: list[tuple[str, str, str]] = [
    ("ATL", "Atlanta", "Hawks"), ("BOS", "Boston", "Celtics"),
    ("BKN", "Brooklyn", "Nets"), ("CHA", "Charlotte", "Hornets"),
    ("CHI", "Chicago", "Bulls"), ("CLE", "Cleveland", "Cavaliers"),
    ("DAL", "Dallas", "Mavericks"), ("DEN", "Denver", "Nuggets"),
    ("DET", "Detroit", "Pistons"), ("GSW", "Golden State", "Warriors"),
    ("HOU", "Houston", "Rockets"), ("IND", "Indiana", "Pacers"),
    ("LAC", "LA", "Clippers"), ("LAL", "Los Angeles", "Lakers"),
    ("MEM", "Memphis", "Grizzlies"), ("MIA", "Miami", "Heat"),
    ("MIL", "Milwaukee", "Bucks"), ("MIN", "Minnesota", "Timberwolves"),
    ("NOP", "New Orleans", "Pelicans"), ("NYK", "New York", "Knicks"),
    ("OKC", "Oklahoma City", "Thunder"), ("ORL", "Orlando", "Magic"),
    ("PHI", "Philadelphia", "76ers"), ("PHX", "Phoenix", "Suns"),
    ("POR", "Portland", "Trail Blazers"), ("SAC", "Sacramento", "Kings"),
    ("SAS", "San Antonio", "Spurs"), ("TOR", "Toronto", "Raptors"),
    ("UTA", "Utah", "Jazz"), ("WAS", "Washington", "Wizards"),
]

# Casual nicknames fans use that aren't the official team name (like
# "Dubs" for the Warriors) — banned too, so they can't leak the team either.
_SHORTHANDS = ["Sixers", "Cavs", "Mavs", "Wolves", "Blazers", "Lakeshow",
               "Dubs", "Nola", "OKC"]

# Every word that could give away which team is playing: abbreviations
# ("BOS"), city-name words ("Boston"), nickname words ("Celtics"), and
# casual shorthands ("Dubs"). This is the full redaction list.
BANNED_TOKENS: list[str] = sorted(
    {t[0] for t in NBA_TEAMS}
    | {word for t in NBA_TEAMS for word in t[1].split()}
    | {word for t in NBA_TEAMS for word in t[2].split()}
    | set(_SHORTHANDS)
)


def month_label(date: str) -> str:
    """Turn an exact date (like "2025-01-14") into a vague label like
    "mid-season (January)", so nobody can look up that exact date and find
    out which real game — and which teams — it was.
    """
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    month_num = int(date.split("-")[1])
    phase = {10: "early-season", 11: "early-season", 12: "mid-season",
             1: "mid-season", 2: "mid-season", 3: "late-season",
             4: "late-season"}.get(month_num, "off-calendar")
    return f"{phase} ({months[month_num - 1]})"


def mask_statsheet(sheet: dict) -> str:
    """Turn a stat-sheet into the plain-text description an agent (or the
    re-ID probe below) actually reads — with real team names replaced by
    "Team A" (home) and "Team B" (away).
    """
    lines = [
        f"A professional basketball game, {month_label(sheet['date'])}.",
        "Team A is the home side. Team B is the visitor.",
        "",
    ]
    for label, side in (("Team A", "home"), ("Team B", "away")):
        f = sheet[f"{side}_form"]
        lines.append(
            f"{label}: won {f['last10_wins']} of its last 10; "
            f"scores {f['avg_pts_for']} and allows {f['avg_pts_against']} "
            f"per game over that stretch; {sheet[f'{side}_rest_days']} day(s) rest."
        )
    return "\n".join(lines)


import json as _json
import os as _os
import random as _random
import re as _re
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import config as _config

_NICK_TO_ABBREV = {t[2].lower(): t[0] for t in NBA_TEAMS}
_ABBREVS = {t[0] for t in NBA_TEAMS}

# City name -> every team abbreviation that city could mean. Most cities
# only have one team, so this ends up being a size-1 set almost everywhere.
_CITY_TO_ABBREVS: dict[str, set[str]] = {}
for _abbrev, _city, _nick in NBA_TEAMS:
    _CITY_TO_ABBREVS.setdefault(_city.lower(), set()).add(_abbrev)
# "LA" and "Los Angeles" both mean "one of the two LA teams" in normal
# speech, even though our team list only spells one of them per team. Mark
# both spellings ambiguous on purpose so a bare "Los Angeles" never gets
# credited as either the Lakers or the Clippers.
_CITY_TO_ABBREVS.setdefault("la", set()).update({"LAC", "LAL"})
_CITY_TO_ABBREVS.setdefault("los angeles", set()).update({"LAC", "LAL"})

# Check longer nicknames before shorter ones, so a two-word nickname like
# "Trail Blazers" is tried before any shorter word that might be part of it.
_NICKNAMES_LONGEST_FIRST = sorted(_NICK_TO_ABBREV, key=len, reverse=True)

_PROBE_PROMPT = (
    "Below is an anonymized description of a real NBA game from the 2024-25 "
    "season. Using only these clues, name the two teams.\n\n{masked}\n\n"
    'Reply with ONLY JSON like {{"home": "City Nickname", "away": "City Nickname"}}.'
)


def _word_in(word: str, low_text: str) -> bool:
    """True if `word` appears in `low_text` as a whole word, not just a substring.

    This is what stops "nets" from matching inside "hornets" — plain
    substring checks can't tell the difference, but a word-boundary regex can.
    """
    return _re.search(rf"\b{_re.escape(word)}\b", low_text) is not None


def _abbrev_from_answer(text: str) -> str | None:
    """Pull a team out of free text: nickname first, then abbreviation, then city.

    A nickname (like "Celtics") always wins if one is there. A 3-letter
    abbreviation (like "BOS") or a plain city name (like "Boston") also
    counts, but only when that city belongs to exactly one team — "Los
    Angeles" or "LA" alone could mean the Lakers or the Clippers, so those
    never count on their own.
    """
    low = text.lower()

    for nick in _NICKNAMES_LONGEST_FIRST:
        if _word_in(nick, low):
            return _NICK_TO_ABBREV[nick]

    for abbrev in _ABBREVS:
        if _word_in(abbrev.lower(), low):
            return abbrev

    for city, abbrevs in _CITY_TO_ABBREVS.items():
        if len(abbrevs) == 1 and _word_in(city, low):
            return next(iter(abbrevs))

    return None


def score_probe_answer(answer_text: str, truth: tuple[str, str]) -> bool:
    """Check a model's un-masking guess against the real teams.

    `answer_text` is the model's raw reply (expected to hold JSON with
    "home"/"away" guesses); `truth` is the real (home, away) abbreviation
    pair. Returns True only when BOTH guesses are right — no partial credit.
    """
    try:
        guess = _json.loads(answer_text[answer_text.find("{"):answer_text.rfind("}") + 1])
        home_guess = _abbrev_from_answer(str(guess.get("home", "")))
        away_guess = _abbrev_from_answer(str(guess.get("away", "")))
    except (ValueError, AttributeError):
        return False
    return home_guess == truth[0] and away_guess == truth[1]


def run_reid_probe(games: list[dict], n: int, models: list[str]) -> dict:
    """Ask each model to un-mask n games. Publish the rate, whatever it is.

    Needs ANTHROPIC_API_KEY. Every call is cached, so re-runs are free.
    """
    from adapters.nba import build_statsheet
    import anthropic

    client = anthropic.Anthropic()
    cache_dir = _config.CACHE / "probe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = _random.Random(_config.SEED)
    # Only probe games late enough to have real form numbers behind them.
    candidates = [i for i in range(len(games)) if i > 200]
    picks = rng.sample(candidates, n)

    spent_calls = 0
    max_calls = int(_config.PROBE_BUDGET_USD / 0.002)  # ~$0.002/call is generous
    per_model: dict[str, float] = {}
    for model in models:
        hits = 0
        for i in picks:
            game = games[i]
            cache_file = cache_dir / f"{game['game_id']}_{model}.json"
            if cache_file.exists():
                answer = _json.loads(cache_file.read_text())["answer"]
            else:
                if spent_calls >= max_calls:
                    raise RuntimeError("probe budget cap hit — raise PROBE_BUDGET_USD to continue")
                masked = mask_statsheet(build_statsheet(games, i))
                msg = client.messages.create(
                    model=model, max_tokens=100,
                    messages=[{"role": "user",
                               "content": _PROBE_PROMPT.format(masked=masked)}])
                answer = msg.content[0].text
                cache_file.write_text(_json.dumps({"answer": answer}))
                spent_calls += 1
            if score_probe_answer(answer, (game["home"], game["away"])):
                hits += 1
        per_model[model] = round(hits / n, 3)

    result = {"per_model": per_model, "n": n}
    (_config.DATA / "probe_results.json").write_text(_json.dumps(result, indent=1))
    # Grade by the WORST model's rate, not the average — we'd rather be too
    # cautious about leaks than too generous.
    worst = max(per_model.values())
    lines = ["# Re-identification probe — published either way", "",
             f"Masked games shown: {n} per model", ""]
    for m, r in per_model.items():
        lines.append(f"- `{m}`: named both teams on **{r:.1%}** of games")
    lines += ["", f"G-leak gate (>= {_config.REID_DEMOTION_RATE:.0%} demotes the "
              f"pre-cutoff backtest): **{'TRIGGERED' if worst >= _config.REID_DEMOTION_RATE else 'not triggered'}**"]
    (_Path(__file__).parent / "PROBE.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    # CLI: venv/bin/python masker.py --probe
    if "--probe" in _sys.argv:
        if not _os.environ.get("ANTHROPIC_API_KEY"):
            _sys.exit("set ANTHROPIC_API_KEY first")
        from adapters.nba import fetch_season_results
        games = fetch_season_results("2024-25")
        out = run_reid_probe(games, _config.PROBE_N, _config.PROBE_MODELS)
        print(_json.dumps(out, indent=1))
