"""Download per-player season stats from nflverse-data, direct from GitHub
releases rather than through the `nfl_data_py` package's own fetch
functions — that package pins pandas<2.0 (no Python 3.12 wheels, so we
install it with --no-deps) and its bundled URL logic 404s on the current
season's file (nflverse reorganized release assets since the package's
last release). Hitting the release assets directly sidesteps both issues,
and gets the newer, richer schema (148 cols incl. kicking/defense/special
teams stats) that nflverse now publishes.

fantasy_points / fantasy_points_ppr are nflverse's own computed columns —
standard scoring (4-pt passing TD, -2 INT, 6-pt rush/rec TD, no PPR) and
PPR (same + 1 pt/reception). Most leagues match this; if yours doesn't,
the raw component stats (yards, TDs, receptions, etc.) are all here too.
"""
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = list(range(2020, 2026))  # last several complete seasons


def download_season_stats(season: int) -> Path | None:
    out_path = RAW_DIR / f"stats_player_reg_{season}.csv"
    if out_path.exists():
        return out_path
    url = f"{RELEASE_BASE}/stats_player/stats_player_reg_{season}.csv"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  skip {season}: not available ({resp.status_code})")
        return None
    out_path.write_bytes(resp.content)
    return out_path


def main() -> None:
    downloaded = []
    for season in SEASONS:
        path = download_season_stats(season)
        if path:
            downloaded.append(path)
            print(f"  ok: {path.name}")
    print(f"\nDownloaded {len(downloaded)} season files to {RAW_DIR}")


if __name__ == "__main__":
    main()
