"""Consolidate per-season player stat files into one long table with
lag features (each season's row also carries the player's prior-season
totals), for the position-level projection models — same pattern as the
Premier League Predictor's build_player_dataset.py.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K"]

# raw counting stats we care about, per position — kept narrow on purpose;
# nflverse's file has 148 columns (incl. defensive/punting stats that don't
# apply to any fantasy-relevant position here)
STAT_COLS = [
    "games", "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    "fg_made", "fg_att", "fg_made_50_59", "fg_made_60_", "pat_made", "pat_att",
    "fantasy_points", "fantasy_points_ppr",
]


def load_all_seasons() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("stats_player_reg_*.csv")):
        df = pd.read_csv(path)
        df = df[df["position"].isin(FANTASY_POSITIONS)].copy()
        keep = ["player_id", "player_display_name", "position", "recent_team", "season"] + \
            [c for c in STAT_COLS if c in df.columns]
        frames.append(df[keep])
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["player_id", "season"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    targets = ["games", "passing_yards", "passing_tds", "passing_interceptions",
               "rushing_yards", "rushing_tds", "receptions", "targets",
               "receiving_yards", "receiving_tds", "fantasy_points", "fantasy_points_ppr",
               "fg_made", "fg_att", "pat_made", "pat_att"]
    grp = df.groupby("player_id")
    for col in targets:
        if col not in df.columns:
            continue
        df[f"prev_{col}"] = grp[col].shift(1)
        df[f"avg2_{col}"] = grp[col].shift(1).rolling(2, min_periods=1).mean().reset_index(level=0, drop=True)
    df["seasons_played"] = grp.cumcount()
    return df


def main() -> None:
    print("Loading season files...")
    df = load_all_seasons()
    print(f"  {len(df)} player-season rows across {df['season'].nunique()} seasons")

    print("Adding lag features...")
    df = add_lag_features(df)

    out_path = PROCESSED_DIR / "player_season_history.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print("\nPosition breakdown:")
    print(df["position"].value_counts())


if __name__ == "__main__":
    main()
