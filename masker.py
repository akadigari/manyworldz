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
