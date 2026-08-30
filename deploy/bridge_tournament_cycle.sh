#!/usr/bin/env bash
# One tournament beat on may-bridge, every 5 minutes from
# manyworldz-tournament.timer.
#
# GitHub throttles this repo's Actions cron by hours at a time (measured
# gaps of 56 to 456 minutes, 2026-08-26/29) while MiniBench questions are
# only open about 3 hours, so the schedule alone cannot hold the 80%
# coverage gate. This script is the un-throttled clock.
#
# Two paths, in order of preference:
#
#   1. HEARTBEAT (default). The bridge holds no secrets: it force-pushes
#      a fresh one-commit `heartbeat` branch, and that push event starts
#      .github/workflows/tournament-heartbeat.yml, which checks out main
#      and runs the cycle with the tokens already stored in GitHub. Push
#      events are not throttled, so the 5-minute cadence is real.
#
#   2. LOCAL. If METACULUS_TOKEN is present in the env file, the bridge
#      runs the cycle itself instead and commits the log directly. Same
#      work, one less hop; it just requires living tokens on the box.
#
# Either way the bot dedupes through the Metaculus my_forecasts flag and
# the committed log, so overlapping with the cron watchdog is safe.
set -u
cd "$(dirname "$0")/.."

# Fresh view of the shared log and spend meter before doing anything.
git pull --rebase --quiet origin main || true

if [ -n "${METACULUS_TOKEN:-}" ]; then
  # Its own spend meter, matching the workflows: the Kalshi loop must not
  # be able to eat the tournament's budget.
  export MANYWORLDZ_SPEND_FILE="${MANYWORLDZ_SPEND_FILE:-$PWD/data/spend_tournament.json}"
  .venv/bin/python tournament.py
  code=$?

  # Persist the submission log and status receipt like the workflow does.
  git add data/tournament_log.csv data/tournament_status.json data/spend_tournament.json 2>/dev/null || true
  if ! git diff --staged --quiet; then
    git -c user.name="manyworldz-bridge" -c user.email="bridge@users.noreply.github.com" \
      commit --quiet -m "tournament: bridge cycle log update"
    # One rebase retry covers a race with a concurrent Actions push.
    git push --quiet origin main || { git pull --rebase --quiet origin main && git push --quiet origin main; }
  fi
  exit $code
fi

# Heartbeat path. The branch is rewritten, never extended: a parentless
# commit over the same tree, force-pushed, so its history stays at one
# commit no matter how many beats it has carried.
if ! git fetch --quiet origin heartbeat; then
  echo "heartbeat branch missing on origin: nothing can trigger the cycle" >&2
  exit 1
fi

tree=$(git rev-parse FETCH_HEAD^{tree}) || exit 1
beat=$(git -c user.name="manyworldz-bridge" -c user.email="bridge@users.noreply.github.com" \
  commit-tree "$tree" -m "heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ)") || exit 1

if ! git push --force --quiet origin "$beat:refs/heads/heartbeat"; then
  echo "heartbeat push failed: the tournament cycle did not start" >&2
  exit 1
fi
