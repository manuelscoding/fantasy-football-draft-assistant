"""Value-Based Drafting: turns position-by-position projections into one
cross-position draft board, ranked by how far each player's projected
points sit above a replacement-level player at their position — the
standard framework fantasy analysts use to compare, say, a QB against a TE
on the same board (raw points don't compare across positions; points
*above replacement* do).
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# rank of the "replacement level" player at each position in a 12-team
# league (roughly: startable slots + a little bench depth) — scaled by
# league size in the dashboard
BASE_REPLACEMENT_RANK = {"QB": 12, "RB": 30, "WR": 36, "TE": 12, "K": 12}
BASE_LEAGUE_SIZE = 12


def load_current_rosters() -> pd.DataFrame:
    path = PROCESSED_DIR.parent / "raw"
    roster_files = sorted(path.glob("rosters_*.csv"))
    return pd.read_csv(roster_files[-1]) if roster_files else pd.DataFrame()


def build_draft_board(projections: pd.DataFrame, league_size: int = 12,
                       rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    """projections: output of projections_model.predict_next_season (must
    have pred_fantasy_points_ppr). rosters: current player_id/team/position
    (pass a live-fetched one from the dashboard; falls back to the last
    committed snapshot on disk for standalone/CLI use). Returns one row per
    player with current team/position (more current than the stats file's
    last-known team) and a VBD score, sorted best-to-worst."""
    df = projections.dropna(subset=["pred_fantasy_points_ppr"]).copy()

    if rosters is None:
        rosters = load_current_rosters()
    if not rosters.empty:
        current = rosters[["player_id", "team", "position"]].rename(
            columns={"team": "current_team", "position": "current_position"}
        )
        df = df.merge(current, on="player_id", how="left")
        df["team"] = df["current_team"].fillna(df["recent_team"])
        df["position"] = df["current_position"].fillna(df["position"])
        df = df.drop(columns=["current_team", "current_position", "recent_team"])
    else:
        df = df.rename(columns={"recent_team": "team"})

    scale = league_size / BASE_LEAGUE_SIZE
    for pos, base_rank in BASE_REPLACEMENT_RANK.items():
        rank = max(1, round(base_rank * scale))
        pos_df = df[df["position"] == pos].sort_values("pred_fantasy_points_ppr", ascending=False)
        if pos_df.empty:
            continue
        idx = min(rank, len(pos_df)) - 1
        replacement_value = pos_df.iloc[idx]["pred_fantasy_points_ppr"]
        df.loc[df["position"] == pos, "replacement_value"] = replacement_value

    df["vbd"] = df["pred_fantasy_points_ppr"] - df["replacement_value"]
    return df.sort_values("vbd", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    projections = pd.read_csv(PROCESSED_DIR / "player_projections.csv")
    board = build_draft_board(projections, league_size=12)
    print(board[["player_display_name", "position", "team", "pred_fantasy_points_ppr", "vbd"]].head(25).to_string(index=False))
