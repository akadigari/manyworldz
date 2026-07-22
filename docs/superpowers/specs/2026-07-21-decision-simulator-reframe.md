# manyworldz: decision-simulator reframe (spec)

**Date:** 2026-07-21
**Status:** approved by owner ("ok build it"); built from this spec, not from chat
**Owner:** Ayush (akadigari)

## Goal

A stranger who lands on the README or the site should get it in five
seconds: manyworldz is a decision simulator. You test the decision before
you make it. Right now both lead with "predictor," which is the proof, not
the point. This is a packaging build: words and layout only, zero engine
changes.

## The one-line pitch (use it, both places)

"Test the decision before you make it. manyworldz runs your question
through a crowd of different minds, imagines every way it could go, and
keeps score against reality in public."

## The five questions frame

The whole product is five plain questions a person already asks before a
decision. Map each to the mode that answers it, in this order:

1. "What are the odds?" : the default ask (stories + one number)
2. "What happens if I do X?" : --whatif (force a fact, watch the shift)
3. "What are all the ways this could go?" : --deep (the map of worlds)
4. "Is there a path to the outcome I want?" : --path (Dr Strange mode)
5. "How sure are we, really?" : --carlo (a million rolled futures + the
   doubt band)

This frame is the spine of both rewrites. Do not invent a sixth.

## Design

### M1: README reframe

- Keep the banner, the badges (tests badge says 207, keep it true), the
  Dr Strange opening, and the seed easter egg at the bottom.
- Rewrite the intro so the second paragraph is the one-line pitch and the
  "decision" word appears before the "predict" word.
- Replace the current "What it does" bullets with the five questions
  frame: each question bolded, one plain sentence each, the flag named.
- Everything below (See the stories, the --deep/--path/--carlo sections,
  How it works, setup, rules, tournament, structure) stays, lightly
  reworded only where it says "predictor" as the identity. The honest
  limits stay word for word wherever they already exist.
- Zero em dashes anywhere. High-school English. The owner's voice, not
  corporate voice: short sentences, no buzzwords, no "leverage".

### M2: site reframe (web/index.html)

- Hero: headline becomes the decision line (something like "test the
  decision before you make it"), subline keeps the many-worlds flavor.
  Keep the Realistic Space palette, THE SPLIT animation, the replay
  button, and all dashboard plumbing exactly as they are.
- Add one thin strip under the hero: the five questions, small cards or
  one row, each question + the command that answers it. Pure static HTML
  and CSS in the same file, same palette, no new dependencies, no new
  network calls.
- Do not touch data.json plumbing, report.py, ledger code, run.py,
  tournament.py, or anything under .github/.

## Non-goals

- No repo split (the separate site repo is its own future spec).
- No new engine features, no new flags, no test changes beyond keeping
  the suite green.
- No screenshots or fake numbers presented as real output anywhere new;
  the README's existing "numbers are invented" disclaimer pattern is the
  ceiling for any example.

## Gates

- All 207 tests still pass untouched (nothing in engine/ or the CLIs
  changes at all).
- web/index.html opens locally with zero console errors, the animation
  still runs, and the page reads right in both light and dark ambient
  (it is a dark page; just do not break it).
- grep finds zero em dashes in every touched file.
- The word "decision" appears in the first two sentences of both the
  README and the site hero.

## Milestones

- M1: README reframe, committed alone.
- M2: site hero + five-questions strip, committed alone, verified in a
  real browser before the commit message claims it works.
