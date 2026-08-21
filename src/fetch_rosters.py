"""Current-season roster (player -> team/position), fetched directly from
nflverse-data's GitHub release rather than through the `nfl_data_py`
package: that package pins pandas<2.0 (no Python 3.12 wheels, forces a
source build that fails on modern setuptools) — a real problem for
Streamlit Cloud's plain `pip install -r requirements.txt`. This is the
exact fetch nfl_data_py's own `import_seasonal_rosters` does internally
(github.com/nflverse/nfl_data_py, `__import_rosters`), just without the
dependency.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = 2026
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.parquet"


def fetch_current_roster(season: int = CURRENT_SEASON) -> pd.DataFrame:
    df = pd.read_parquet(ROSTER_URL.format(season=season))
    df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    cols = ["player_id", "player_name", "position", "team", "jersey_number", "years_exp"]
    return df[[c for c in cols if c in df.columns]].dropna(subset=["player_id"])


def main() -> None:
    rosters = fetch_current_roster()
    out_path = RAW_DIR / f"rosters_{CURRENT_SEASON}.csv"
    rosters.to_csv(out_path, index=False)
    print(f"Saved {len(rosters)} roster rows to {out_path}")
    print(rosters["position"].value_counts())


if __name__ == "__main__":
    main()
