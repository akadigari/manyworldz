# heartbeat

Not code. This branch is a doorbell.

GitHub throttles this repo's `schedule:` cron by hours at a time, which
would lose MiniBench questions that are only open for three. Push events
are not throttled, so may-bridge force-pushes a fresh one-commit version
of this branch every 5 minutes and that push starts
`.github/workflows/tournament-heartbeat.yml`, which checks out `main` and
runs the tournament cycle there.

The branch is intentionally orphan and rewritten on every beat, so its
history never grows. Nothing here is ever merged into `main`.
