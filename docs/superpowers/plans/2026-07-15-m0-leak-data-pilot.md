# Agamotto M0 — Leak & Data Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove (or kill) Agamotto's two foundations before any swarm code exists: (1) measure how badly LLMs re-identify "anonymized" NBA games, and (2) build the post-cutoff scoring table of games joined to verified market closing prices — then issue the pre-registered GO/NO-GO.

**Architecture:** Pure-Python pipeline, no framework. `adapters/nba.py` pulls and caches game results; `masker.py` anonymizes stat-sheets and runs the re-ID probe against Claude models; `markets/closes.py` loads the MGM Kaggle closing-odds CSV (and optionally Kalshi candlesticks); `jointable.py` joins results×closes into the scoring table with a quarantine file; `verdict_m0.py` applies the pre-registered gates. Everything cached to disk; tests run on fixtures with zero network and zero API keys.

**Tech Stack:** Python 3.11+, `nba_api`, `anthropic`, `pandas`, `pytest`. Project venv at `agamotto/venv/`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-agamotto-design.md` — M0 section governs.
- **No swarm code in M0** (no personas, no ensemble, no voting) — spec's explicit rule.
- Comments/docstrings/output in plain, simple English — high-school level, no jargon (owner rule).
- Commits: never add AI co-author trailers. Commit as the repo's existing identity (`akadigari` / `arkadigari@gmail.com`).
- All dates are plain `"YYYY-MM-DD"` strings end-to-end — never pandas datetime columns (dodges pandas-3 ns/µs pitfalls that bit MechLab).
- Determinism: every random choice uses `SEED = 14000605` from `config.py`.
- API spend: the re-ID probe is the ONLY step that calls a paid API in M0; hard-capped at `PROBE_BUDGET_USD = 5.00`.
- Public framing: README text never mentions MiroFish and never frames the project as a betting product (owner rule; spec "Public framing").
- Data files live in `data/` (gitignored). Tests never touch `data/` — fixtures only.
- Pre-registered gate values (copied from spec, locked): re-ID demotion at **≥10%**, GO needs **≥350** post-cutoff games with verified closes and **≤1%** errors in a 50-row hand audit.

---

### Task 1: Scaffold, GATES.md, config

**Files:**
- Create: `.gitignore`, `requirements.txt`, `README.md`, `GATES.md`, `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.py` constants used by every later task:
  `ROOT: Path`, `DATA: Path`, `CACHE: Path`, `SEED: int = 14000605`,
  `MODEL_CUTOFF_DATE: str = "2025-03-01"`, `PROBE_MODELS: list[str]`,
  `PROBE_N: int = 100`, `PROBE_BUDGET_USD: float = 5.00`,
  `REID_DEMOTION_RATE: float = 0.10`, `GO_MIN_GAMES: int = 350`,
  `GO_MAX_JOIN_ERROR: float = 0.01`

- [ ] **Step 1: Create venv and install deps**

```bash
cd /Users/kadigari/Documents/ARKPrograms/agamotto
python3 -m venv venv
venv/bin/pip install --upgrade pip
```

Create `requirements.txt`:

```
nba_api>=1.4
anthropic>=0.40
pandas>=2.2
pytest>=8.0
```

```bash
venv/bin/pip install -r requirements.txt
```

Expected: all four install clean.

- [ ] **Step 2: Write `.gitignore`**

```
venv/
data/
__pycache__/
*.pyc
.env
.superpowers/
```

- [ ] **Step 3: Write `config.py`**

```python
"""One place for every knob in Agamotto M0.

Change numbers here, never inside the pipeline files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"

# The same random seed everywhere, so every run can be repeated exactly.
# (Yes, it's the number of futures Strange checked.)
SEED = 14000605

# The probe asks these models to un-mask games. Haiku is also the model
# whose training-data cutoff defines "post-cutoff" games below.
PROBE_MODELS = ["claude-haiku-4-5", "claude-sonnet-5"]
PROBE_N = 100            # how many masked games each model sees
PROBE_BUDGET_USD = 5.00  # hard stop for probe spend

# claude-haiku-4-5's documented training cutoff is February 2025.
# We add a one-month buffer: games on/after this date count as
# "post-cutoff" (the model cannot have seen their results in training).
# Execution step in Task 9 re-checks this date against Anthropic's
# current model docs before the probe runs.
MODEL_CUTOFF_DATE = "2025-03-01"

# Pre-registered gate numbers. Locked in GATES.md before results exist.
REID_DEMOTION_RATE = 0.10   # both-teams-named rate that demotes the backtest
GO_MIN_GAMES = 350          # post-cutoff games with a verified close
GO_MAX_JOIN_ERROR = 0.01    # allowed error rate in the 50-row hand audit
```

- [ ] **Step 4: Write `GATES.md`** (pre-registration — this exists BEFORE any result)

```markdown
# Agamotto — the gates (locked before any results exist)

Same idea as kayfabe/MechLab/TrendLab: we write the pass/fail rules first,
so we can't fool ourselves later. Verdicts get published either way.

## M0 gates (this milestone)

- **GO:** the post-cutoff scoring table has >= 350 games where each game
  carries a verified market close, AND a 50-row hand audit of the join
  finds <= 1% errors.
- **NO-GO:** either fails -> fix the closing-price source before writing
  any engine code.
- **G-leak:** if any probed model correctly names BOTH teams of a masked
  game on >= 10% of probes (chance is far under 1%), the pre-cutoff
  backtest is demoted to calibration-training only. The re-ID rate is
  published either way.

## Project gates (M1+, restated from the spec so they're in one place)

- **G0 (power):** >= 300 graded games in the held-out set before judging.
- **G1 (the Dr Strange test):** crowd Brier beats the market close.
- **G2 (the theater test):** crowd beats the boring logistic baseline.
- **G3 (luck test):** sign-randomization on edge picks beats shuffled labels.
- **G4 (cost test):** edge survives Kalshi fees + spread with a buffer.
- **G5 (worth-it test):** enough depth that the edge means real money.

Passing everything = a human may place small real bets. Failing any = an
honest "no edge," published as a portfolio result.
```

- [ ] **Step 5: Write `README.md`** (stub, public-framing compliant)

```markdown
# Agamotto

A crowd of AI forecasters that simulates futures. Each agent imagines how
an event could play out; the crowd's futures become a probability; a
what-if mode re-runs every future with a fact forced true. The crowd
learns from every settled event — agents that keep being wrong lose their
voice, get benched, and can be retired.

Everything is measured against real-world outcomes and market closing
prices, with pass/fail gates written down before any results exist
(see GATES.md). Paper-only research. Verdicts get published either way.

**Status: Milestone 0 — data + leak audit. No engine yet.**
```

- [ ] **Step 6: Write the failing test** — `tests/test_config.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def test_gate_numbers_match_the_spec():
    assert config.GO_MIN_GAMES == 350
    assert config.GO_MAX_JOIN_ERROR == 0.01
    assert config.REID_DEMOTION_RATE == 0.10
    assert config.SEED == 14000605


def test_cutoff_date_is_a_plain_string():
    assert isinstance(config.MODEL_CUTOFF_DATE, str)
    assert len(config.MODEL_CUTOFF_DATE) == 10  # YYYY-MM-DD
```

- [ ] **Step 7: Run tests**

Run: `venv/bin/pytest tests/test_config.py -v`
Expected: 2 PASS (config.py already written in Step 3 — this test locks the numbers against drift).

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt README.md GATES.md config.py tests/test_config.py
git commit -m "m0: scaffold, pre-registered gates, config"
```

---

### Task 2: NBA results adapter (fetch + parse + cache)

**Files:**
- Create: `adapters/__init__.py` (empty), `adapters/nba.py`
- Test: `tests/test_nba_adapter.py`, `tests/fixtures/gamefinder_rows.json`

**Interfaces:**
- Consumes: `config.CACHE`, `config.SEED`
- Produces:
  `parse_gamefinder_rows(rows: list[dict]) -> list[dict]` — returns game dicts
  sorted by date, each: `{"game_id": str, "date": "YYYY-MM-DD", "home": str,
  "away": str, "home_pts": int, "away_pts": int, "home_won": bool}`
  (home/away are 3-letter abbreviations like "BOS").
  `fetch_season_results(season: str) -> list[dict]` — same shape, cached to
  `data/cache/results_{season}.json`; `season` looks like `"2024-25"`.

- [ ] **Step 1: Build the fixture** — `tests/fixtures/gamefinder_rows.json`

nba_api's LeagueGameFinder returns one row per team per game; the home team's
`MATCHUP` says `"BOS vs. LAL"`, the away team's says `"LAL @ BOS"`. Fixture with
one complete game (two rows) plus one orphan row (game missing its pair):

```json
[
  {"GAME_ID": "0022400001", "GAME_DATE": "2024-10-22", "TEAM_ABBREVIATION": "BOS",
   "MATCHUP": "BOS vs. NYK", "PTS": 132, "WL": "W"},
  {"GAME_ID": "0022400001", "GAME_DATE": "2024-10-22", "TEAM_ABBREVIATION": "NYK",
   "MATCHUP": "NYK @ BOS", "PTS": 109, "WL": "L"},
  {"GAME_ID": "0022400999", "GAME_DATE": "2024-10-23", "TEAM_ABBREVIATION": "LAL",
   "MATCHUP": "LAL vs. MIN", "PTS": 110, "WL": "W"}
]
```

- [ ] **Step 2: Write the failing test** — `tests/test_nba_adapter.py`

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.nba import parse_gamefinder_rows

FIXTURE = Path(__file__).parent / "fixtures" / "gamefinder_rows.json"


def rows():
    return json.loads(FIXTURE.read_text())


def test_pairs_two_rows_into_one_game():
    games = parse_gamefinder_rows(rows())
    complete = [g for g in games if g["game_id"] == "0022400001"]
    assert len(complete) == 1
    g = complete[0]
    assert g["home"] == "BOS" and g["away"] == "NYK"
    assert g["home_pts"] == 132 and g["away_pts"] == 109
    assert g["home_won"] is True
    assert g["date"] == "2024-10-22"


def test_orphan_rows_are_dropped_not_guessed():
    games = parse_gamefinder_rows(rows())
    assert all(g["game_id"] != "0022400999" for g in games)


def test_output_sorted_by_date():
    games = parse_gamefinder_rows(rows() + rows())  # dupes exercise stability
    dates = [g["date"] for g in games]
    assert dates == sorted(dates)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_nba_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: adapters`

- [ ] **Step 4: Write `adapters/nba.py`**

```python
"""Pull NBA game results and turn them into simple game records.

The only network call lives in fetch_season_results(); everything else is
pure parsing so tests never need the internet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def parse_gamefinder_rows(rows: list[dict]) -> list[dict]:
    """Combine per-team rows into one record per game.

    A game only counts when we see BOTH of its rows (home and away).
    Orphans are dropped — we never guess a missing side.
    """
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["GAME_ID"], []).append(r)

    games = []
    for gid, pair in by_id.items():
        if len(pair) != 2:
            continue
        home = next((r for r in pair if "vs." in r["MATCHUP"]), None)
        away = next((r for r in pair if "@" in r["MATCHUP"]), None)
        if home is None or away is None:
            continue
        games.append({
            "game_id": gid,
            "date": home["GAME_DATE"][:10],
            "home": home["TEAM_ABBREVIATION"],
            "away": away["TEAM_ABBREVIATION"],
            "home_pts": int(home["PTS"]),
            "away_pts": int(away["PTS"]),
            "home_won": int(home["PTS"]) > int(away["PTS"]),
        })
    games.sort(key=lambda g: (g["date"], g["game_id"]))
    return games


def fetch_season_results(season: str) -> list[dict]:
    """Fetch one regular season's results, with a disk cache.

    season looks like "2024-25". Cached forever — settled games don't change.
    """
    config.CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = config.CACHE / f"results_{season}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    from nba_api.stats.endpoints import leaguegamefinder
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    rows = finder.get_normalized_dict()["LeagueGameFinderResults"]
    time.sleep(1)  # be polite to the free endpoint
    games = parse_gamefinder_rows(rows)
    cache_file.write_text(json.dumps(games, indent=1))
    return games


if __name__ == "__main__":
    # CLI: venv/bin/python adapters/nba.py 2024-25
    season = sys.argv[1]
    games = fetch_season_results(season)
    print(f"{season}: {len(games)} games cached")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_nba_adapter.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add adapters/ tests/test_nba_adapter.py tests/fixtures/gamefinder_rows.json
git commit -m "m0: nba results adapter with offline-tested parser"
```

---

### Task 3: Stat-sheet builder (the features a persona would read)

**Files:**
- Modify: `adapters/nba.py` (append functions)
- Test: `tests/test_statsheet.py`

**Interfaces:**
- Consumes: game dicts from Task 2
- Produces:
  `team_history(games: list[dict], team: str, before_date: str) -> list[dict]`
  — that team's games strictly before the date, oldest first.
  `build_statsheet(games: list[dict], idx: int) -> dict` — for game `games[idx]`:
  `{"date": str, "home": str, "away": str,
    "home_form": {"last10_wins": int, "avg_pts_for": float, "avg_pts_against": float},
    "away_form": {...same keys...},
    "home_rest_days": int, "away_rest_days": int}`
  (rest days capped at 9 so season openers don't leak "this is October game 1").

- [ ] **Step 1: Write the failing test** — `tests/test_statsheet.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.nba import build_statsheet, team_history


def little_season():
    """Three games: BOS beats NYK, then LAL beats BOS two days later."""
    return [
        {"game_id": "1", "date": "2025-01-01", "home": "BOS", "away": "NYK",
         "home_pts": 120, "away_pts": 100, "home_won": True},
        {"game_id": "2", "date": "2025-01-03", "home": "LAL", "away": "BOS",
         "home_pts": 111, "away_pts": 99, "home_won": True},
    ]


def test_history_only_looks_backward():
    games = little_season()
    h = team_history(games, "BOS", before_date="2025-01-03")
    assert len(h) == 1 and h[0]["game_id"] == "1"


def test_statsheet_form_and_rest():
    games = little_season()
    sheet = build_statsheet(games, 1)  # the Jan 3 game
    assert sheet["home"] == "LAL" and sheet["away"] == "BOS"
    assert sheet["away_form"]["last10_wins"] == 1
    assert sheet["away_form"]["avg_pts_for"] == 120.0
    assert sheet["away_rest_days"] == 2
    assert sheet["home_rest_days"] == 9  # no prior game -> capped, not a leak
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_statsheet.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_statsheet'`

- [ ] **Step 3: Append to `adapters/nba.py`**

```python
def team_history(games: list[dict], team: str, before_date: str) -> list[dict]:
    """Every game this team played strictly before the date, oldest first."""
    return [g for g in games
            if g["date"] < before_date and team in (g["home"], g["away"])]


def _days_between(d1: str, d2: str) -> int:
    from datetime import date
    a = date(*map(int, d1.split("-")))
    b = date(*map(int, d2.split("-")))
    return (b - a).days


def _form(history: list[dict], team: str) -> dict:
    """Last-10 record and scoring averages, from that team's point of view."""
    last10 = history[-10:]
    wins, pts_for, pts_against = 0, [], []
    for g in last10:
        we_are_home = g["home"] == team
        our_pts = g["home_pts"] if we_are_home else g["away_pts"]
        their_pts = g["away_pts"] if we_are_home else g["home_pts"]
        pts_for.append(our_pts)
        pts_against.append(their_pts)
        if (our_pts > their_pts):
            wins += 1
    n = max(len(last10), 1)
    return {"last10_wins": wins,
            "avg_pts_for": round(sum(pts_for) / n, 1) if pts_for else 0.0,
            "avg_pts_against": round(sum(pts_against) / n, 1) if pts_against else 0.0}


def build_statsheet(games: list[dict], idx: int) -> dict:
    """Everything an agent may know about a game — from BEFORE tipoff only."""
    game = games[idx]
    sheet = {"date": game["date"], "home": game["home"], "away": game["away"]}
    for side in ("home", "away"):
        team = game[side]
        hist = team_history(games, team, game["date"])
        sheet[f"{side}_form"] = _form(hist, team)
        # Rest days capped at 9: "well rested" reads the same for a season
        # opener and a long break, so the number can't fingerprint the date.
        rest = _days_between(hist[-1]["date"], game["date"]) if hist else 9
        sheet[f"{side}_rest_days"] = min(rest, 9)
    return sheet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_statsheet.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/nba.py tests/test_statsheet.py
git commit -m "m0: stat-sheet builder (backward-looking form + capped rest)"
```

---

### Task 4: The masker

**Files:**
- Create: `masker.py`
- Test: `tests/test_masker.py`

**Interfaces:**
- Consumes: stat-sheet dicts from Task 3
- Produces:
  `NBA_TEAMS: list[tuple[str, str, str]]` — 30 rows of `(abbrev, city, nickname)`.
  `BANNED_TOKENS: list[str]` — every abbrev, city word, nickname, plus common
  shorthands ("Sixers", "Cavs", "Mavs", "Wolves", "Blazers", "Lakeshow").
  `month_label(date: str) -> str` — `"2025-01-03"` → `"mid-season (January)"`.
  `mask_statsheet(sheet: dict) -> str` — the anonymized text block; the ONLY
  identity words are "Team A" (home) and "Team B" (away).

- [ ] **Step 1: Write the failing test** — `tests/test_masker.py`

```python
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import BANNED_TOKENS, NBA_TEAMS, mask_statsheet


def sheet_for(home, away):
    return {"date": "2025-01-15", "home": home, "away": away,
            "home_form": {"last10_wins": 7, "avg_pts_for": 118.2, "avg_pts_against": 110.0},
            "away_form": {"last10_wins": 3, "avg_pts_for": 105.4, "avg_pts_against": 114.9},
            "home_rest_days": 2, "away_rest_days": 1}


def test_thirty_teams_and_no_identity_survives():
    assert len(NBA_TEAMS) == 30
    abbrevs = [t[0] for t in NBA_TEAMS]
    for i in range(0, 30, 2):  # 15 masked sheets covers all 30 teams
        text = mask_statsheet(sheet_for(abbrevs[i], abbrevs[i + 1])).lower()
        for token in BANNED_TOKENS:
            assert not re.search(rf"\b{re.escape(token.lower())}\b", text), \
                f"banned token '{token}' leaked for {abbrevs[i]} vs {abbrevs[i+1]}"


def test_exact_date_is_hidden():
    text = mask_statsheet(sheet_for("BOS", "NYK"))
    assert "2025-01-15" not in text and "January)" in text


def test_teams_become_a_and_b():
    text = mask_statsheet(sheet_for("BOS", "NYK"))
    assert "Team A" in text and "Team B" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_masker.py -v`
Expected: FAIL — `ModuleNotFoundError: masker`

- [ ] **Step 3: Write `masker.py`** (mask half; probe added in Task 5)

```python
"""Hide who is playing, keep what the numbers say.

mask_statsheet() turns a stat-sheet into text where the only identities are
"Team A" (home) and "Team B" (away). The re-ID probe (below) then measures
how often models can un-hide the teams anyway. We publish that number.
"""
from __future__ import annotations

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

_SHORTHANDS = ["Sixers", "Cavs", "Mavs", "Wolves", "Blazers", "Lakeshow",
               "Dubs", "Nola", "OKC"]

BANNED_TOKENS: list[str] = sorted(
    {t[0] for t in NBA_TEAMS}
    | {word for t in NBA_TEAMS for word in t[1].split()}
    | {word for t in NBA_TEAMS for word in t[2].split()}
    | set(_SHORTHANDS)
)


def month_label(date: str) -> str:
    """Coarsen an exact date to a month word, so the date can't be looked up."""
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    m = int(date.split("-")[1])
    phase = {10: "early-season", 11: "early-season", 12: "mid-season",
             1: "mid-season", 2: "mid-season", 3: "late-season",
             4: "late-season"}.get(m, "off-calendar")
    return f"{phase} ({months[m - 1]})"


def mask_statsheet(sheet: dict) -> str:
    """The anonymized text an agent (or the probe) gets to read."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_masker.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add masker.py tests/test_masker.py
git commit -m "m0: masker — Team A/B stat-sheets, banned-token tested"
```

---

### Task 5: The re-ID probe

**Files:**
- Modify: `masker.py` (append probe functions)
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `mask_statsheet`, `NBA_TEAMS`, stat-sheets (Task 3), `config.PROBE_*`
- Produces:
  `score_probe_answer(answer_text: str, truth: tuple[str, str]) -> bool` —
  True only if the model named BOTH teams correctly (nickname match counts;
  truth is `(home_abbrev, away_abbrev)`).
  `run_reid_probe(games: list[dict], n: int, models: list[str]) -> dict` —
  returns `{"per_model": {model: rate}, "n": n}`, writes
  `data/probe_results.json` and `PROBE.md`; caches each call to
  `data/cache/probe/{game_id}_{model}.json`; stops if estimated spend
  exceeds `config.PROBE_BUDGET_USD`.

- [ ] **Step 1: Write the failing test** — `tests/test_probe.py` (no API calls)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import score_probe_answer


def test_both_right_by_nickname_counts():
    ans = '{"home": "Boston Celtics", "away": "New York Knicks"}'
    assert score_probe_answer(ans, ("BOS", "NYK")) is True


def test_one_wrong_does_not_count():
    ans = '{"home": "Boston Celtics", "away": "Brooklyn Nets"}'
    assert score_probe_answer(ans, ("BOS", "NYK")) is False


def test_garbage_answer_scores_false_not_crash():
    assert score_probe_answer("no idea, sorry", ("BOS", "NYK")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_probe.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_probe_answer'`

- [ ] **Step 3: Append to `masker.py`**

```python
import json as _json
import os as _os
import random as _random
import re as _re
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import config as _config

_NICK_TO_ABBREV = {t[2].lower(): t[0] for t in NBA_TEAMS}

_PROBE_PROMPT = (
    "Below is an anonymized description of a real NBA game from the 2024-25 "
    "season. Using only these clues, name the two teams.\n\n{masked}\n\n"
    'Reply with ONLY JSON like {{"home": "City Nickname", "away": "City Nickname"}}.'
)


def _abbrev_from_answer(text: str) -> str | None:
    """Pull a team out of free text by its nickname, if any."""
    low = text.lower()
    for nick, ab in _NICK_TO_ABBREV.items():
        if nick in low:
            return ab
    return None


def score_probe_answer(answer_text: str, truth: tuple[str, str]) -> bool:
    """True only when BOTH the home and away guesses are right."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_probe.py tests/test_masker.py -v`
Expected: 6 PASS (probe scoring + masker still green)

- [ ] **Step 5: Commit**

```bash
git add masker.py tests/test_probe.py
git commit -m "m0: re-ID probe — cached, budget-capped, offline-tested scoring"
```

---

### Task 6: Closing-odds loader (MGM Kaggle CSV)

**Files:**
- Create: `markets/__init__.py` (empty), `markets/closes.py`
- Test: `tests/test_closes.py`, `tests/fixtures/closes_sample.csv`

**Interfaces:**
- Consumes: a CSV manually downloaded to `data/nba_closes.csv` (execution task
  covers finding it; loader validates rather than assumes)
- Produces:
  `american_to_prob(ml: int) -> float` — raw implied probability.
  `devig(p_home_raw: float, p_away_raw: float) -> float` — home prob, vig removed.
  `COLUMN_MAP: dict[str, str]` — our name → expected CSV column name (one dict
  to edit if the real file differs).
  `validate_schema(df) -> None` — raises with the full found-column list if
  any mapped column is missing.
  `load_kaggle_closes(csv_path: Path) -> pd.DataFrame` — columns exactly:
  `date` (str YYYY-MM-DD), `home` (abbrev), `away` (abbrev),
  `home_close_prob` (float 0-1), `provenance` (str, `"kaggle"`).

- [ ] **Step 1: Build the fixture** — `tests/fixtures/closes_sample.csv`

```csv
game_date,home_team,away_team,home_ml_close,away_ml_close
2025-03-05,Boston Celtics,New York Knicks,-220,+180
2025-03-06,Los Angeles Lakers,Denver Nuggets,+150,-175
2025-03-07,Utah Jazz,,-300,+240
```

- [ ] **Step 2: Write the failing test** — `tests/test_closes.py`

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from markets.closes import american_to_prob, devig, load_kaggle_closes

FIXTURE = Path(__file__).parent / "fixtures" / "closes_sample.csv"


def test_american_odds_to_probability():
    assert american_to_prob(-220) == pytest.approx(0.6875, abs=1e-4)
    assert american_to_prob(+180) == pytest.approx(0.3571, abs=1e-4)


def test_devig_makes_the_pair_sum_to_one():
    p = devig(american_to_prob(-220), american_to_prob(+180))
    assert 0.65 < p < 0.67  # fair home prob, vig stripped


def test_loader_normalizes_names_and_drops_bad_rows():
    df = load_kaggle_closes(FIXTURE)
    assert list(df.columns) == ["date", "home", "away", "home_close_prob", "provenance"]
    assert len(df) == 2                      # the row with a blank team is dropped
    assert set(df["home"]) == {"BOS", "LAL"}  # full names became abbrevs
    assert ((df["home_close_prob"] > 0) & (df["home_close_prob"] < 1)).all()
    assert (df["provenance"] == "kaggle").all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_closes.py -v`
Expected: FAIL — `ModuleNotFoundError: markets`

- [ ] **Step 4: Write `markets/closes.py`**

```python
"""Load verified market closing prices for NBA games.

The Kaggle CSV is downloaded by hand into data/ (see the plan's execution
task). We validate its columns loudly instead of assuming — if the real
file names differ, edit COLUMN_MAP and nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import NBA_TEAMS

# our name -> the column we expect in the raw CSV. One dict to edit.
COLUMN_MAP = {
    "date": "game_date",
    "home": "home_team",
    "away": "away_team",
    "home_ml": "home_ml_close",
    "away_ml": "away_ml_close",
}

_FULLNAME_TO_ABBREV = {f"{t[1]} {t[2]}".lower(): t[0] for t in NBA_TEAMS}


def american_to_prob(ml: int) -> float:
    """American moneyline -> raw implied probability (still has vig)."""
    ml = int(ml)
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def devig(p_home_raw: float, p_away_raw: float) -> float:
    """Strip the bookmaker's margin: scale the pair to sum to 1."""
    return p_home_raw / (p_home_raw + p_away_raw)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}. Fix COLUMN_MAP in closes.py.")


def _to_abbrev(name: str) -> str | None:
    return _FULLNAME_TO_ABBREV.get(str(name).strip().lower())


def load_kaggle_closes(csv_path: Path) -> pd.DataFrame:
    """Return one clean row per game: date, home, away, home_close_prob."""
    raw = pd.read_csv(csv_path)
    validate_schema(raw)
    rows = []
    for _, r in raw.iterrows():
        home, away = _to_abbrev(r[COLUMN_MAP["home"]]), _to_abbrev(r[COLUMN_MAP["away"]])
        try:
            p_home = devig(american_to_prob(r[COLUMN_MAP["home_ml"]]),
                           american_to_prob(r[COLUMN_MAP["away_ml"]]))
        except (ValueError, TypeError):
            home = None  # bad odds -> drop below
        if home is None or away is None:
            continue
        rows.append({"date": str(r[COLUMN_MAP["date"]])[:10], "home": home,
                     "away": away, "home_close_prob": round(p_home, 4),
                     "provenance": "kaggle"})
    return pd.DataFrame(rows, columns=["date", "home", "away",
                                       "home_close_prob", "provenance"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_closes.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add markets/ tests/test_closes.py tests/fixtures/closes_sample.csv
git commit -m "m0: kaggle closes loader — devig, name normalization, loud schema check"
```

---

### Task 7: Join table + quarantine + audit sample

**Files:**
- Create: `jointable.py`
- Test: `tests/test_jointable.py`

**Interfaces:**
- Consumes: game dicts (Task 2), closes DataFrame (Task 6),
  `config.MODEL_CUTOFF_DATE`, `config.SEED`
- Produces:
  `build_table(games: list[dict], closes: pd.DataFrame, cutoff: str)
  -> tuple[pd.DataFrame, pd.DataFrame]` — `(table, quarantine)`.
  `table` columns: `date, home, away, home_won, home_close_prob, provenance`
  — one row per post-cutoff game that matched exactly one close.
  `quarantine` columns: `date, home, away, reason` — post-cutoff games with
  no close (`"no_close"`) or more than one (`"duplicate_close"`).
  `write_outputs(table, quarantine) -> None` — writes
  `data/scoring_table.csv`, `data/quarantine.csv`, and
  `data/audit_sample.csv` (50 seeded random rows of the table).

- [ ] **Step 1: Write the failing test** — `tests/test_jointable.py`

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointable import build_table


def games():
    return [
        {"game_id": "1", "date": "2025-03-05", "home": "BOS", "away": "NYK",
         "home_pts": 120, "away_pts": 100, "home_won": True},
        {"game_id": "2", "date": "2025-03-06", "home": "LAL", "away": "DEN",
         "home_pts": 100, "away_pts": 110, "home_won": False},
        {"game_id": "3", "date": "2025-03-07", "home": "MIA", "away": "ORL",
         "home_pts": 99, "away_pts": 98, "home_won": True},   # no close -> quarantine
        {"game_id": "4", "date": "2025-01-01", "home": "BOS", "away": "MIA",
         "home_pts": 100, "away_pts": 90, "home_won": True},  # pre-cutoff -> excluded silently
    ]


def closes():
    return pd.DataFrame([
        {"date": "2025-03-05", "home": "BOS", "away": "NYK", "home_close_prob": 0.66, "provenance": "kaggle"},
        {"date": "2025-03-06", "home": "LAL", "away": "DEN", "home_close_prob": 0.45, "provenance": "kaggle"},
        {"date": "2025-03-06", "home": "LAL", "away": "DEN", "home_close_prob": 0.46, "provenance": "kaggle"},
    ])


def test_join_quarantines_instead_of_guessing():
    table, quarantine = build_table(games(), closes(), cutoff="2025-03-01")
    assert len(table) == 1                       # only the clean BOS game
    assert table.iloc[0]["home_close_prob"] == 0.66
    assert bool(table.iloc[0]["home_won"]) is True
    reasons = dict(zip(quarantine["home"], quarantine["reason"]))
    assert reasons == {"LAL": "duplicate_close", "MIA": "no_close"}


def test_pre_cutoff_games_are_excluded_not_quarantined():
    table, quarantine = build_table(games(), closes(), cutoff="2025-03-01")
    assert "2025-01-01" not in set(table["date"]) | set(quarantine["date"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_jointable.py -v`
Expected: FAIL — `ModuleNotFoundError: jointable`

- [ ] **Step 3: Write `jointable.py`**

```python
"""Join game results to verified market closes — the table M0 stands on.

Anything odd goes to a quarantine file with a reason. We never guess and
never silently drop a post-cutoff game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def build_table(games: list[dict], closes: pd.DataFrame,
                cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = [g for g in games if g["date"] >= cutoff]
    key = lambda d, h, a: f"{d}|{h}|{a}"
    by_key: dict[str, list[dict]] = {}
    for _, r in closes.iterrows():
        by_key.setdefault(key(r["date"], r["home"], r["away"]), []).append(dict(r))

    rows, bad = [], []
    for g in post:
        matches = by_key.get(key(g["date"], g["home"], g["away"]), [])
        if len(matches) == 1:
            rows.append({"date": g["date"], "home": g["home"], "away": g["away"],
                         "home_won": g["home_won"],
                         "home_close_prob": matches[0]["home_close_prob"],
                         "provenance": matches[0]["provenance"]})
        else:
            bad.append({"date": g["date"], "home": g["home"], "away": g["away"],
                        "reason": "no_close" if not matches else "duplicate_close"})
    cols = ["date", "home", "away", "home_won", "home_close_prob", "provenance"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(bad, columns=["date", "home", "away", "reason"]))


def write_outputs(table: pd.DataFrame, quarantine: pd.DataFrame) -> None:
    config.DATA.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DATA / "scoring_table.csv", index=False)
    quarantine.to_csv(config.DATA / "quarantine.csv", index=False)
    n = min(50, len(table))
    table.sample(n, random_state=config.SEED).to_csv(
        config.DATA / "audit_sample.csv", index=False)


if __name__ == "__main__":
    # CLI: venv/bin/python jointable.py
    from adapters.nba import fetch_season_results
    from markets.closes import load_kaggle_closes
    games = []
    for season in ("2024-25", "2025-26"):
        games += fetch_season_results(season)
    closes = load_kaggle_closes(config.DATA / "nba_closes.csv")
    table, quarantine = build_table(games, closes, config.MODEL_CUTOFF_DATE)
    write_outputs(table, quarantine)
    print(f"scoring table: {len(table)} games | quarantine: {len(quarantine)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_jointable.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add jointable.py tests/test_jointable.py
git commit -m "m0: results-x-closes join with quarantine and seeded audit sample"
```

---

### Task 8: M0 verdict

**Files:**
- Create: `verdict_m0.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Consumes: `config` gate constants; at runtime: `data/scoring_table.csv`,
  `data/probe_results.json`, `AUDIT.md` (human-written count)
- Produces:
  `evaluate(n_games: int, audit_errors: int, audit_n: int,
  worst_reid_rate: float) -> dict` — keys: `go: bool`, `demote_precutoff: bool`,
  `reasons: list[str]`. CLI writes `M0_VERDICT.md`.

- [ ] **Step 1: Write the failing test** — `tests/test_verdict.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verdict_m0 import evaluate


def test_clean_pass_is_go():
    v = evaluate(n_games=400, audit_errors=0, audit_n=50, worst_reid_rate=0.31)
    assert v["go"] is True
    assert v["demote_precutoff"] is True  # 31% leak still demotes — separate axis


def test_too_few_games_is_no_go():
    v = evaluate(n_games=349, audit_errors=0, audit_n=50, worst_reid_rate=0.0)
    assert v["go"] is False
    assert any("350" in r for r in v["reasons"])


def test_dirty_join_is_no_go():
    v = evaluate(n_games=400, audit_errors=2, audit_n=50, worst_reid_rate=0.0)
    assert v["go"] is False  # 4% error rate > 1% gate


def test_low_leak_does_not_demote():
    v = evaluate(n_games=400, audit_errors=0, audit_n=50, worst_reid_rate=0.05)
    assert v["demote_precutoff"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_verdict.py -v`
Expected: FAIL — `ModuleNotFoundError: verdict_m0`

- [ ] **Step 3: Write `verdict_m0.py`**

```python
"""Apply the pre-registered M0 gates. The rules live in GATES.md and
config.py — this file only checks them, it never bends them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def evaluate(n_games: int, audit_errors: int, audit_n: int,
             worst_reid_rate: float) -> dict:
    reasons = []
    if n_games >= config.GO_MIN_GAMES:
        reasons.append(f"games with verified closes: {n_games} (need {config.GO_MIN_GAMES}) — pass")
        games_ok = True
    else:
        reasons.append(f"games with verified closes: {n_games} (need {config.GO_MIN_GAMES}) — FAIL")
        games_ok = False

    err_rate = audit_errors / max(audit_n, 1)
    if err_rate <= config.GO_MAX_JOIN_ERROR:
        reasons.append(f"hand-audit errors: {audit_errors}/{audit_n} ({err_rate:.1%}) — pass")
        audit_ok = True
    else:
        reasons.append(f"hand-audit errors: {audit_errors}/{audit_n} ({err_rate:.1%}) — FAIL")
        audit_ok = False

    demote = worst_reid_rate >= config.REID_DEMOTION_RATE
    reasons.append(
        f"worst re-ID rate: {worst_reid_rate:.1%} — "
        + ("pre-cutoff backtest DEMOTED to calibration-only" if demote
           else "mask holds; pre-cutoff backtest stays scoreable"))

    return {"go": games_ok and audit_ok, "demote_precutoff": demote,
            "reasons": reasons}


if __name__ == "__main__":
    import pandas as pd
    table = pd.read_csv(config.DATA / "scoring_table.csv")
    probe = json.loads((config.DATA / "probe_results.json").read_text())
    # AUDIT.md line format (human writes it): "errors: 0 of 50"
    audit_line = next(l for l in Path("AUDIT.md").read_text().splitlines()
                      if l.startswith("errors:"))
    parts = audit_line.replace("errors:", "").split("of")
    audit_errors, audit_n = int(parts[0]), int(parts[1])

    v = evaluate(len(table), audit_errors, audit_n,
                 max(probe["per_model"].values()))
    lines = ["# M0 verdict", "",
             f"**{'GO' if v['go'] else 'NO-GO'}** — " +
             ("engine work may start." if v["go"]
              else "fix the data before any engine code."), ""]
    lines += [f"- {r}" for r in v["reasons"]]
    Path("M0_VERDICT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_verdict.py -v`
Expected: 4 PASS

- [ ] **Step 5: Run the whole suite**

Run: `venv/bin/pytest -v`
Expected: all tests green (config 2, adapter 3, statsheet 2, masker 3, probe 3, closes 3, jointable 2, verdict 4 = 22 PASS)

- [ ] **Step 6: Commit**

```bash
git add verdict_m0.py tests/test_verdict.py
git commit -m "m0: verdict engine applying the pre-registered gates"
```

---

### Task 9: Execute the pilot (real data, real probe — human steps marked)

**Files:**
- Create (by running, not writing): `data/cache/results_*.json`,
  `data/scoring_table.csv`, `data/quarantine.csv`, `data/audit_sample.csv`,
  `data/probe_results.json`, `PROBE.md`, `AUDIT.md`, `M0_VERDICT.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the M0 verdict — the input to planning M1.

- [ ] **Step 1: Confirm the model cutoff date** — open
  https://docs.anthropic.com/en/docs/about-claude/models and check
  claude-haiku-4-5's "training data cutoff". If it is NOT February 2025,
  update `MODEL_CUTOFF_DATE` in `config.py` to (cutoff month + 1) and commit
  before proceeding.

- [ ] **Step 2: Fetch results** —
  `venv/bin/python adapters/nba.py 2024-25` then
  `venv/bin/python adapters/nba.py 2025-26`.
  Expected: `2024-25: ~1230 games cached`, `2025-26: ~900+ games cached`.

- [ ] **Step 3 (HUMAN): Download the closes CSV** — search Kaggle for the NBA
  betting-odds dataset covering 2021-22 through the 2026 All-Star break with
  closing moneylines (the scout board's "MGM Kaggle dataset";
  wiki/resources/scout-trading-multiagent-simulation-for-prediction-and-forecasting.md
  has the trail). Download the CSV to `data/nba_closes.csv`. If its column
  names differ from `COLUMN_MAP` in `markets/closes.py`, edit that one dict —
  `validate_schema` will print exactly what it found.

- [ ] **Step 4: Build the table** — `venv/bin/python jointable.py`.
  Expected: `scoring table: N games | quarantine: M` with N well over 350.
  If N < 350: check `data/quarantine.csv` reasons before anything else —
  name normalization gaps are the usual suspect.

- [ ] **Step 5 (HUMAN): Hand-audit 50 rows** — open `data/audit_sample.csv`;
  for each row check the winner and the close against an independent source
  (Basketball-Reference box score + the raw CSV row). Write `AUDIT.md`:

```markdown
# M0 hand audit — 2026-MM-DD
Checked data/audit_sample.csv (seeded sample, SEED=14000605) against
Basketball-Reference and the raw closes CSV.
errors: 0 of 50
notes: (anything odd goes here)
```

- [ ] **Step 6: Run the probe** (~$1-2 of API spend, capped at $5) —
  `export ANTHROPIC_API_KEY=... && venv/bin/python masker.py --probe`.
  Expected: `data/probe_results.json` + `PROBE.md` with a rate per model.
  The number is the number — 0% or 60%, it gets published.

- [ ] **Step 7: The verdict** — `venv/bin/python verdict_m0.py`.
  Expected: `M0_VERDICT.md` printed and written: GO/NO-GO + demotion status.

- [ ] **Step 8: Commit the paper trail**

```bash
git add PROBE.md AUDIT.md M0_VERDICT.md
git commit -m "m0: pilot executed — probe rate, audit, and verdict on the record"
```

- [ ] **Step 9: Report** — read `M0_VERDICT.md` back to the owner with the three
  headline numbers (games, audit errors, worst re-ID rate) and what they mean
  for M1 planning. If NO-GO: the failing gate's fix is the next plan, not the
  engine.

---

## Self-review notes (done at write time)

- **Spec coverage:** M0 section fully covered (masker ✓ probe ✓ closes table ✓
  join+audit ✓ GO/NO-GO ✓). Engine/dashboard/money-arm intentionally out —
  they are M1+/M-viz/M-money plans, written after this verdict.
- **Placeholders:** none; every step has runnable content. The one unknown
  (exact Kaggle CSV columns) is handled by a loud `validate_schema` + a single
  `COLUMN_MAP` edit point, not a TODO.
- **Type consistency:** game dict keys (`game_id/date/home/away/home_pts/
  away_pts/home_won`) and closes columns (`date/home/away/home_close_prob/
  provenance`) are identical across Tasks 2→7; gate constants come only from
  `config.py`.
