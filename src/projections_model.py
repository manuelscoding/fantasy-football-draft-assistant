"""Per-position regression models predicting next-season fantasy stats from
trailing performance — same lag-feature RandomForest approach as the
Premier League Predictor's player_stats_model.py.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MIN_GAMES = 4  # drop tiny-sample seasons from training (injuries, late call-ups)

POSITION_TARGETS = {
    "QB": ["passing_yards", "passing_tds", "passing_interceptions",
           "rushing_yards", "rushing_tds", "fantasy_points_ppr"],
    "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards",
           "receiving_tds", "fantasy_points_ppr"],
    "WR": ["receptions", "targets", "receiving_yards", "receiving_tds", "fantasy_points_ppr"],
    "TE": ["receptions", "targets", "receiving_yards", "receiving_tds", "fantasy_points_ppr"],
    "K": ["fg_made", "fg_att", "pat_made", "fantasy_points_ppr"],
}

LAG_STATS = ["games", "passing_yards", "passing_tds", "passing_interceptions",
             "rushing_yards", "rushing_tds", "receptions", "targets",
             "receiving_yards", "receiving_tds", "fantasy_points", "fantasy_points_ppr",
             "fg_made", "fg_att", "pat_made", "pat_att"]
FEATURE_COLS = [f"prev_{c}" for c in LAG_STATS] + [f"avg2_{c}" for c in LAG_STATS] + ["seasons_played"]


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    return df[(df["seasons_played"] > 0) & (df["games"] >= MIN_GAMES)].copy()


def train_position_models(df: pd.DataFrame, with_cv: bool = True, verbose: bool = True) -> dict:
    models = {}
    for position, targets in POSITION_TARGETS.items():
        pdf = df[df["position"] == position].copy()
        pdf[FEATURE_COLS] = pdf[FEATURE_COLS].fillna(0)
        models[position] = {}
        if verbose:
            print(f"\n{position} (n={len(pdf)}):")
        for target in targets:
            valid = pdf.dropna(subset=[target])
            if len(valid) < 20:
                continue
            X, y = valid[FEATURE_COLS], valid[target]
            model = RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42,
            )
            if with_cv:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                if verbose:
                    print(f"  {target:22s} R2 = {scores.mean():.3f} (+/- {scores.std():.3f})")
            model.fit(X, y)
            models[position][target] = model
    return models


def predict_next_season(models: dict, full_df: pd.DataFrame) -> pd.DataFrame:
    """Uses each player's most recent season as 'prev' features to predict
    the upcoming season. Rookies / players with no history get NaN
    predictions here — the dashboard merges in the current roster
    separately and flags them rather than guessing."""
    latest = full_df.sort_values("season").groupby("player_id").tail(1).copy()

    # blend this season's actual value with its own avg2 (already an avg of
    # the *prior* two seasons) to approximate a rolling avg2 for next
    # season, computed before we overwrite/drop the old lag columns below
    new_avg2 = {
        f"avg2_{col}": latest[[col, f"avg2_{col}"]].mean(axis=1, skipna=True)
        for col in LAG_STATS
    }

    old_lag_cols = [c for c in latest.columns if c.startswith("prev_") or c.startswith("avg2_")]
    rename_map = {c: f"prev_{c}" for c in LAG_STATS}
    feat = latest.drop(columns=old_lag_cols).rename(columns=rename_map)
    for col, series in new_avg2.items():
        feat[col] = series
    feat["seasons_played"] = feat["seasons_played"] + 1
    feat[FEATURE_COLS] = feat[FEATURE_COLS].fillna(0)

    results = []
    for position, targets in POSITION_TARGETS.items():
        pdf = feat[feat["position"] == position].copy()
        if pdf.empty:
            continue
        out = pdf[["player_id", "player_display_name", "position", "recent_team", "season"]].copy()
        for target in targets:
            if target in models.get(position, {}):
                out[f"pred_{target}"] = models[position][target].predict(pdf[FEATURE_COLS]).round(1)
        if position == "K" and {"pred_fg_made", "pred_pat_made"} <= set(out.columns):
            # kicker fantasy points are famously close to unpredictable
            # year-over-year (driven by FG distance/game-script noise on top
            # of the count stats) — the direct fantasy_points_ppr model's R2
            # is ~0, so derive an estimate from the better-behaved raw counts
            # instead (standard scoring: 3 pts/FG made, 1 pt/PAT made; no
            # distance-based bonus tiers, so this slightly underrates
            # long-range specialists)
            out["pred_fantasy_points_ppr"] = (3 * out["pred_fg_made"] + out["pred_pat_made"]).round(1)
        results.append(out)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    df = load_training_data()
    print(f"Training rows (>=1 prior season, >= {MIN_GAMES} games): {len(df)}")
    models = train_position_models(df)

    full_df = pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    predictions = predict_next_season(models, full_df)
    predictions.to_csv(PROCESSED_DIR / "player_projections.csv", index=False)
    print(f"\nSaved projections for {len(predictions)} players.")

    for position in POSITION_TARGETS:
        top = predictions[predictions["position"] == position]
        if "pred_fantasy_points_ppr" in top.columns:
            top = top.sort_values("pred_fantasy_points_ppr", ascending=False).head(5)
            print(f"\nTop 5 predicted {position}s by PPR points:")
            print(top.drop(columns=["player_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
