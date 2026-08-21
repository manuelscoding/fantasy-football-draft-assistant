"""Fantasy Football Draft Assistant: a Value-Based-Drafting board, player
projections by position, and a live "who's left" tracker for draft day.

Same live-computation-with-caching architecture and visual design as the
Premier League Predictor (dark theme, JetBrains Mono) — a separate project
reusing those patterns, not sharing any code or data with it.

Run with: streamlit run dashboard/app.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from draft_board import BASE_REPLACEMENT_RANK, build_draft_board  # noqa: E402
from fetch_rosters import fetch_current_roster  # noqa: E402
from projections_model import (  # noqa: E402
    load_training_data, predict_next_season, train_position_models,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CURRENT_SEASON = 2026
REFRESH_TTL_SECONDS = 6 * 3600

# same design tokens as the Premier League Predictor
INK = "#0D141D"
INK_2 = "#161F2A"
LINE = "#2B3A4C"
BONE = "#ECE7DB"
BONE_DIM = "#9FA8B3"
MARIGOLD = "#E9A13B"
SAGE = "#79AE99"
RUST = "#C4614A"

POSITION_COLORS = {"QB": RUST, "RB": MARIGOLD, "WR": SAGE, "TE": "#7A9CC6", "K": BONE_DIM}

st.set_page_config(page_title="Fantasy Football Draft Assistant", layout="wide", page_icon="🏈")

st.markdown(f"""
<style>
.stApp {{ background-color: {INK}; color: {BONE}; }}
section[data-testid="stSidebar"] {{ background-color: {INK_2}; }}
h1, h2, h3, h4 {{ color: {BONE}; font-family: 'JetBrains Mono', monospace; }}
[data-testid="stMetricValue"] {{ color: {MARIGOLD}; }}
[data-testid="stMetricLabel"] {{ color: {BONE_DIM}; }}
.stDataFrame {{ border: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner="Fetching current roster...")
def fetch_live_roster():
    return fetch_current_roster(CURRENT_SEASON), datetime.now(timezone.utc)


@st.cache_data(show_spinner=False)
def load_player_history():
    return pd.read_csv(PROCESSED_DIR / "player_season_history.csv")


@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner="Training projection models...")
def compute_projections():
    player_hist = load_player_history()
    train_df = load_training_data()
    models = train_position_models(train_df, with_cv=False, verbose=False)
    projections = predict_next_season(models, player_hist)
    return projections


st.title("Fantasy Football Draft Assistant")
st.caption("Value-Based Drafting board, position-by-position projections, and a live "
           "best-available tracker — built on nflverse play-by-play/season stats.")

roster, fetched_at = fetch_live_roster()
projections = compute_projections()

st.caption(f"🔄 Roster data as of {fetched_at.strftime('%Y-%m-%d %H:%M UTC')} "
           f"— refreshes automatically every {REFRESH_TTL_SECONDS // 3600} hours.")

if "drafted" not in st.session_state:
    st.session_state.drafted = set()

tab1, tab2, tab3 = st.tabs(["Draft Board", "Player Projections", "My Team"])

with tab1:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        league_size = st.selectbox("League size", [8, 10, 12, 14, 16], index=2)
    with col2:
        pos_filter = st.multiselect("Position", list(BASE_REPLACEMENT_RANK.keys()))
    with col3:
        hide_drafted = st.checkbox("Hide drafted players", value=True)

    # board_all (all positions, drafted players included) is the source of
    # truth for the "mark players drafted" widget below — it must never
    # shrink just because the display filters (pos_filter/hide_drafted) hide
    # a player, or Streamlit treats the multiselect's changed `options` as a
    # new widget instance, drops the now-invalid default, and silently wipes
    # st.session_state.drafted back to empty on the very next rerun.
    board_all = build_draft_board(projections, league_size=league_size, rosters=roster)
    board = board_all
    if pos_filter:
        board = board[board["position"].isin(pos_filter)]
    if hide_drafted:
        board = board[~board["player_id"].isin(st.session_state.drafted)]

    st.subheader(f"Draft board — top available ({league_size}-team league)")
    st.caption("Ranked by Value Above Replacement (VBD): projected PPR points minus what a "
               "readily-available replacement at that position would score. Lets you compare "
               "a QB against a WR on the same board.")

    top = board.head(40)
    fig = go.Figure()
    for pos in top["position"].unique():
        pdf = top[top["position"] == pos]
        fig.add_trace(go.Bar(
            x=pdf["vbd"], y=pdf["player_display_name"], orientation="h",
            name=pos, marker_color=POSITION_COLORS.get(pos, BONE_DIM),
        ))
    fig.update_layout(
        plot_bgcolor=INK, paper_bgcolor=INK, font_color=BONE,
        xaxis_title="Value above replacement (PPR pts)",
        yaxis=dict(autorange="reversed"), barmode="stack",
        height=650, margin=dict(l=10, r=10, t=10, b=10), legend_title="Position",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full draft board")
    show_cols = ["player_display_name", "position", "team", "pred_fantasy_points_ppr", "vbd"]
    rename = {"player_display_name": "Player", "position": "Pos", "team": "Team",
              "pred_fantasy_points_ppr": "Proj. PPR Pts", "vbd": "VBD"}
    st.dataframe(
        board[show_cols].rename(columns=rename).style.format(
            {"Proj. PPR Pts": "{:.1f}", "VBD": "{:.1f}"}
        ).background_gradient(subset=["VBD"], cmap="YlOrBr"),
        hide_index=True, use_container_width=True, height=500,
    )

    st.subheader("Mark players drafted")
    # build the label from same-indexed columns *before* reindexing by
    # player_id — adding series with mismatched indexes (player_id vs the
    # default range index) silently aligns to all-NaN instead of erroring
    label = (board_all["player_display_name"] + " (" + board_all["position"] + ", "
              + board_all["team"].fillna("FA") + ")")
    draft_names = label.set_axis(board_all["player_id"])
    picked = st.multiselect(
        "Select players as they're drafted (by you or anyone else)",
        options=draft_names.index.tolist(),
        format_func=lambda pid: draft_names.get(pid, pid),
        default=list(st.session_state.drafted & set(draft_names.index)),
        key="drafted_multiselect",
    )
    st.session_state.drafted = set(picked)

with tab2:
    position = st.selectbox("Position", list(BASE_REPLACEMENT_RANK.keys()), key="proj_position")
    pdf = projections[projections["position"] == position].copy()
    pdf = pdf.dropna(subset=["pred_fantasy_points_ppr"]).sort_values(
        "pred_fantasy_points_ppr", ascending=False
    ).head(40)

    st.subheader(f"Top projected {position}s for {CURRENT_SEASON}")
    pred_cols = [c for c in pdf.columns if c.startswith("pred_") and pdf[c].notna().any()
                 and c != "pred_fantasy_points_ppr"]
    show_cols = ["player_display_name", "recent_team", "pred_fantasy_points_ppr"] + pred_cols
    rename = {c: c.replace("pred_", "").replace("_", " ").title() for c in pred_cols}
    rename.update({"player_display_name": "Player", "recent_team": "Team",
                    "pred_fantasy_points_ppr": "Proj. PPR Pts"})
    st.dataframe(
        pdf[show_cols].rename(columns=rename),
        hide_index=True, use_container_width=True, height=700,
    )
    st.caption("Projections are a random-forest model per position, trained on each player's "
               "trailing 1-2 seasons of stats (nflverse). Kicker fantasy points are derived from "
               "projected FG/PAT makes rather than a direct model — kicker scoring is famously "
               "close to unpredictable year-over-year, more so than the underlying count stats.")

with tab3:
    st.subheader("My team")
    if not st.session_state.drafted:
        st.info("No players marked drafted yet — use the Draft Board tab to build your team as you draft.")
    else:
        mine = projections[projections["player_id"].isin(st.session_state.drafted)]
        mine = mine.dropna(subset=["pred_fantasy_points_ppr"]).sort_values(
            "pred_fantasy_points_ppr", ascending=False
        )
        counts = mine["position"].value_counts()
        cols = st.columns(len(BASE_REPLACEMENT_RANK))
        for col, pos in zip(cols, BASE_REPLACEMENT_RANK):
            with col:
                st.metric(pos, int(counts.get(pos, 0)))

        st.dataframe(
            mine[["player_display_name", "position", "recent_team", "pred_fantasy_points_ppr"]]
            .rename(columns={"player_display_name": "Player", "position": "Pos",
                              "recent_team": "Team", "pred_fantasy_points_ppr": "Proj. PPR Pts"}),
            hide_index=True, use_container_width=True,
        )
        st.metric("Total projected PPR points", f"{mine['pred_fantasy_points_ppr'].sum():.1f}")
