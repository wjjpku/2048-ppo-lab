# Publication checks

Verified on 2026-10-12, without extending the completed experiment.

- A new Python 3.12 virtual environment installed the built wheel with PyTorch
  2.11.0 and NumPy 2.4.3. Tests ran outside the source directory against that
  installed package: **57 passed, 1 CUDA-only check skipped**.
- Both packaged models passed hash, parameter-count and 64-board forward-reference
  checks. Each has 977,605 parameters and a 3,913,613-byte inference export.
- Installed-package CPU demos reached natural terminal states for both models:
  base4-5840 scored 58,020 (max tile 4096); exp4to8-7640 scored 27,164 (max tile
  2048). These reuse seed 1 / bank16 / environment 0 / greedy and are installation
  smoke checks, not new independent performance evidence.
- The one-command demo script completed using the new test environment via its
  optional `--venv` argument. The default creates `.venv` in the project.
- All six archived replays passed rule, RNG-spawn, score and terminal-state
  verification: **8,346 steps**, including **7,511 steps** across the five cases.
- A real Chromium browser rendered every step of the five cases. Board tiles,
  accessible board labels, scores, actions, maximum tiles and timeline positions
  matched the archived data. Case selection, first/last, single-step, timeline,
  playback/pause, 8x speed, keyboard controls and milestones were exercised.
- Responsive checks passed at 320, 360, 390 and 1280 pixels for the replay; the
  project overview passed at 320, 390 and 1280 pixels. Screenshots were reviewed.
- The ten archived numerical-core files match the hashes in `provenance.json`.
  The publication uses fresh Git history and excludes operational archives,
  local environments, generated demos, browser logs and private infrastructure.

The CPU workflow repeats automated checks on GitHub. Its current outcome is
available under the repository's Actions tab. It does not run a training campaign
or deploy a website. Browser interaction checks above were performed locally.
