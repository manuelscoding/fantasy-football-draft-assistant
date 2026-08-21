# Fantasy Football Draft Assistant

Ranks every fantasy-relevant NFL player on one cross-position draft board
using Value-Based Drafting, projects individual stat lines for the
upcoming season, and tracks who's still available as you draft.

Built as a companion project to the
[Premier League Predictor](https://github.com/manuelscoding/premier-league-predictor),
reusing its live-computation dashboard architecture and dark visual theme —
separate codebase and data, no shared dependencies.

**Live app:** _add your Streamlit Community Cloud URL here after deploying_

## Data sources

- **[nflverse-data](https://github.com/nflverse/nflverse-data)** — player
  regular-season stat totals (`stats_player_reg_{season}.csv`, 2020-2025)
  and current rosters (`rosters/roster_{season}.parquet`), fetched directly
  from GitHub releases rather than through the `nfl_data_py` Python package.
  That package pins `pandas<2.0`, which has no Python 3.12 wheels — a
  hard blocker for Streamlit Cloud's plain `pip install -r requirements.txt`.
  `src/fetch_player_stats.py` and `src/fetch_rosters.py` replicate the
  package's own internal fetch logic without the dependency.

## Live architecture

Same pattern as the soccer project: the dashboard recomputes everything on
a rolling cache (every 6 hours) instead of reading pre-baked predictions.

- **Fetched live, every cache cycle:** the current-season roster (team,
  position, jersey number) via `fetch_current_roster()` in
  `dashboard/app.py` — keeps player team/position current through
  trades, cuts, and depth-chart moves without waiting on a data refresh.
- **Committed to the repo, refreshed periodically:** `player_season_history.csv`
  (six seasons of per-player stats with lag features) and
  `player_projections.csv` — the slow-to-build historical corpus, rebuilt
  weekly by `.github/workflows/refresh-data.yml` rather than per-request.

## Models

1. **Per-position projection models** (`src/projections_model.py`) — one
   `RandomForestRegressor` per stat per position (QB/RB/WR/TE/K), trained
   on each player's own trailing 1-2 seasons (`prev_X`, `avg2_X` lag
   features — same approach as the soccer project's player-stats model).
   Players need at least one full prior season with 4+ games played to
   train on; rookies and players with no qualifying history get `NaN`
   projections, which the draft board simply excludes rather than guessing.
   Held-out R² on `fantasy_points_ppr`: RB 0.49, WR 0.57, TE 0.54.
   Kicker point totals are close to unpredictable year-over-year even with
   good features (direct model R² ≈ 0), so kicker fantasy points are
   instead *derived* from the model's (much more stable) FG-made and
   PAT-made predictions: `3 × pred_fg_made + pred_pat_made`.
2. **Value-Based Drafting board** (`src/draft_board.py`) — converts
   per-position projections into one ranked board by subtracting each
   position's *replacement-level* score (the Nth-best player at that
   position, N scaled to league size) from every player's projection.
   This is what makes a QB and a WR comparable on the same board — raw
   projected points aren't, since position scarcity differs wildly.

## Known limitations

- **No Defense/Special Teams position.** DST scoring is team-level
  (sacks, turnovers, points allowed), not derivable from the player-level
  stat files this project uses. Standard-league drafters will need to
  handle DST/K-adjacent bench decisions separately.
- **Kicker projections are noisy.** See above — treat the K tier as
  directional, not precise.
- **Rookies have no projection** until they accumulate one full
  qualifying season; the draft board excludes them entirely rather than
  guessing from draft capital or college stats.

## Running locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# one-time / periodic: build the historical corpus the dashboard reads at startup
python src/fetch_player_stats.py
python src/build_dataset.py
python src/projections_model.py

streamlit run dashboard/app.py
```

## Deploying

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), connect the repo
   and set the main file path to `dashboard/app.py`.
3. The weekly GitHub Action keeps the committed historical data fresh;
   the dashboard's own cache keeps the current roster fresh on top of that.
