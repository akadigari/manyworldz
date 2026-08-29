#!/usr/bin/env bash
# One tournament cycle on may-bridge. Runs every 5 minutes from
# manyworldz-tournament.timer, replacing GitHub Actions cron as the
# primary scheduler (Actions stays as fallback; the bot dedupes via the
# Metaculus my_forecasts flag, so double-running is safe).
#
# Fail-closed: with METACULUS_TOKEN unset in the env file, tournament.py
# prints a warning and exits 0. Nothing is fetched or submitted.
set -u
cd "$(dirname "$0")/.."

# Fresh view of the shared log and spend meter before doing anything.
git pull --rebase --quiet origin main || true

.venv/bin/python tournament.py
code=$?

# Persist the submission log and status receipt like the workflow does.
git add data/tournament_log.csv data/tournament_status.json data/spend.json 2>/dev/null || true
if ! git diff --staged --quiet; then
  git -c user.name="manyworldz-bridge" -c user.email="bridge@users.noreply.github.com" \
    commit --quiet -m "tournament: bridge cycle log update"
  # One rebase retry covers a race with a concurrent Actions push.
  git push --quiet origin main || { git pull --rebase --quiet origin main && git push --quiet origin main; }
fi
exit $code
