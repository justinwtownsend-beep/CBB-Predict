import datetime as dt
import re
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Data pull + standardization
# ----------------------------
@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def load_torvik_team_results(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    df = pd.read_csv(url)

    # Normalize TEAM column
    team_col = None
    for c in df.columns:
        if c.strip().lower() in {"team", "teams", "teamname"} or c.strip().upper() == "TEAM":
            team_col = c
            break
    if team_col is None:
        team_col = df.columns[0]

    df = df.rename(columns={team_col: "TEAM"})
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df


def _find_col(df: pd.DataFrame, patterns: list[str]) -> str:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.fullmatch(c) or rx.search(c):
                return c
    raise KeyError(f"Missing column for {patterns}. Columns: {cols}")


def standardize_torvik_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col_adj_oe = _find_col(out, [r"AdjOE", r"ADJ.*OE", r"OE.*Adj"])
    col_adj_de = _find_col(out, [r"AdjDE", r"ADJ.*DE", r"DE.*Adj"])

    try:
        col_tempo = _find_col(out, [r"AdjT", r"Tempo", r"Pace", r"Poss"])
    except KeyError:
        col_tempo = None

    rename_map = {col_adj_oe: "ADJ_OE", col_adj_de: "ADJ_DE"}
    if col_tempo:
        rename_map[col_tempo] = "TEMPO"

    out = out.rename(columns=rename_map)

    for c in ["ADJ_OE", "ADJ_DE", "TEMPO"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_team_lookup(df: pd.DataFrame) -> dict[str, str]:
    def simp(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    lookup = {}
    for t in df["TEAM"].dropna().unique():
        lookup[simp(t)] = t
    return lookup


def resolve_team_name(user_input: str, lookup: dict[str, str]) -> str:
    def simp(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    key = simp(user_input)
    if key in lookup:
        return lookup[key]

    # fallback: contains match
    candidates = [(k, v) for k, v in lookup.items() if key in k or k in key]
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        candidates.sort(key=lambda kv: len(kv[0]))
        return candidates[0][1]

    raise ValueError(f"Could not resolve team: '{user_input}'")


# ----------------------------
# Slate pull (FIXED)
# ----------------------------
@st.cache_data(ttl=10 * 60)  # cache slate for 10 minutes
def get_torvik_daily_slate(date_yyyymmdd: str) -> tuple[pd.DataFrame, str]:
    """
    Pull Torvik daily schedule table for a given date.
    If Torvik returns a page with no tables (common on no-game days), return empty DF instead of crashing.
    """
    url = f"https://barttorvik.com/schedule.php?conlimit=&date={date_yyyymmdd}&sort=time"

    # IMPORTANT: Torvik sometimes returns a page with no <table> when there are no games.
    try:
        tables = pd.read_html(url, flavor="bs4")
    except Exception:
        return pd.DataFrame(), url

    if not tables:
        return pd.DataFrame(), url

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Find matchup column
    matchup_col = None
    for c in df.columns:
        if "matchup" in c.lower():
            matchup_col = c
            break

    if matchup_col is None:
        return pd.DataFrame(), url

    df = df.rename(columns={matchup_col: "Matchup"})

    # Keep only rows that look like real games
    df = df[df["Matchup"].astype(str).str.contains(r"\s(at|vs)\s", case=False, regex=True)].reset_index(drop=True)

    return df, url


def parse_matchup(matchup: str):
    m = str(matchup).strip()
    if re.search(r"\s+vs\s+", m, flags=re.IGNORECASE):
        a, h = re.split(r"\s+vs\s+", m, flags=re.IGNORECASE)
        return
