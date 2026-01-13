import datetime as dt
import re
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Data pull + standardization
# ----------------------------
@st.cache_data(ttl=60*60)  # cache for 1 hour
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

    candidates = [(k, v) for k, v in lookup.items() if key in k or k in key]
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        candidates.sort(key=lambda kv: len(kv[0]))
        return candidates[0][1]

    raise ValueError(f"Could not resolve team: '{user_input}'")

@st.cache_data(ttl=10*60)  # cache slate for 10 minutes
def get_torvik_daily_slate(date_yyyymmdd: str) -> tuple[pd.DataFrame, str]:
    url = f"https://barttorvik.com/schedule.php?conlimit=&date={date_yyyymmdd}&sort=time"
    tables = pd.read_html(url, flavor="bs4")
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    matchup_col = None
    for c in df.columns:
        if "matchup" in c.lower():
            matchup_col = c
            break
    if matchup_col is None:
        raise ValueError(f"Could not find Matchup column. Columns: {df.columns.tolist()}")

    df = df.rename(columns={matchup_col: "Matchup"})
    df = df[df["Matchup"].astype(str).str.contains(r"\s(at|vs)\s", case=False, regex=True)].reset_index(drop=True)
    return df, url

def parse_matchup(matchup: str):
    m = str(matchup).strip()
    if re.search(r"\s+vs\s+", m, flags=re.IGNORECASE):
        a, h = re.split(r"\s+vs\s+", m, flags=re.IGNORECASE)
        return a.strip(), h.strip(), True
    if re.search(r"\s+at\s+", m, flags=re.IGNORECASE):
        a, h = re.split(r"\s+at\s+", m, flags=re.IGNORECASE)
        return a.strip(), h.strip(), False
    return None, None, None

def predict_cbb_game(df_teams: pd.DataFrame, away: str, home: str, neutral: bool,
                     hca_points: float, n_sims: int, margin_sd: float):
    lookup = build_team_lookup(df_teams)
    away_team = resolve_team_name(away, lookup)
    home_team = resolve_team_name(home, lookup)

    a = df_teams.loc[df_teams["TEAM"] == away_team].iloc[0]
    h = df_teams.loc[df_teams["TEAM"] == home_team].iloc[0]

    poss = float(np.nanmean([a.get("TEMPO", np.nan), h.get("TEMPO", np.nan)]))
    if np.isnan(poss):
        poss = 68.0

    away_pp100 = (float(a["ADJ_OE"]) + float(h["ADJ_DE"])) / 2.0
    home_pp100 = (float(h["ADJ_OE"]) + float(a["ADJ_DE"])) / 2.0

    away_pts = (away_pp100 / 100.0) * poss
    home_pts = (home_pp100 / 100.0) * poss

    if not neutral:
        home_pts += hca_points / 2.0
        away_pts -= hca_points / 2.0

    mean_margin = home_pts - away_pts

    rng = np.random.default_rng(7)
    sims_margin = rng.normal(loc=mean_margin, scale=margin_sd, size=n_sims)
    home_win_prob = float(np.mean(sims_margin > 0))

    return {
        "Away": away_team,
        "Home": home_team,
        "Neutral": neutral,
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": home_pts + away_pts,
        "Proj_Margin_Home": mean_margin,
        "Home_Win_%": home_win_prob * 100
    }

def run_slate(year: int, date_yyyymmdd: str, n_sims: int, hca_points: float, margin_sd: float) -> tuple[pd.DataFrame, str]:
    raw = load_torvik_team_results(year)
    df_teams = standardize_torvik_columns(raw)
    slate, slate_url = get_torvik_daily_slate(date_yyyymmdd)

    rows = []
    for _, r in slate.iterrows():
        away, home, neutral = parse_matchup(r["Matchup"])
        if away is None:
            continue
        try:
            res = predict_cbb_game(df_teams, away, home, neutral, hca_points, n_sims, margin_sd)
            res["Time"] = r.get("Time", "")
            res["Matchup"] = r["Matchup"]
            rows.append(res)
        except Exception as e:
            rows.append({"Matchup": r["Matchup"], "Error": str(e)})

    out = pd.DataFrame(rows)
    if "Proj_Margin_Home" in out.columns:
        out["Abs_Margin"] = out["Proj_Margin_Home"].abs()
        out["Close_Game_Score"] = (out["Home_Win_%"] - 50).abs()
    return out, slate_url

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CBB Torvik Slate Predictor", layout="wide")
st.title("CBB Torvik Slate Predictor")

with st.sidebar:
    st.header("Settings")
    season_year = st.number_input("Season year (2026 = 2025–26)", value=2026, step=1)
    date_val = st.date_input("Slate date", value=dt.date.today())
    n_sims = st.slider("Simulations", 1000, 20000, 10000, step=1000)
    hca_points = st.slider("Home-court advantage (pts)", 0.0, 5.0, 2.5, step=0.5)
    margin_sd = st.slider("Margin SD (tunable)", 6.0, 18.0, 11.0, step=0.5)
    run_btn = st.button("Run slate")

date_yyyymmdd = date_val.strftime("%Y%m%d")

if run_btn:
    preds, slate_url = run_slate(int(season_year), date_yyyymmdd, int(n_sims), float(hca_points), float(margin_sd))
    st.caption(f"Slate source: {slate_url}")

    if "Error" in preds.columns and preds.shape[1] <= 2:
        st.error("Most matchups failed to resolve. Try running again or check Torvik table format.")
        st.dataframe(preds)
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Biggest projected margins")
            st.dataframe(preds.sort_values("Abs_Margin", ascending=False).head(25), use_container_width=True)

        with c2:
            st.subheader("Closest games")
            st.dataframe(preds.sort_values("Close_Game_Score", ascending=True).head(25), use_container_width=True)

        with c3:
            st.subheader("Highest projected totals")
            st.dataframe(preds.sort_values("Proj_Total", ascending=False).head(25), use_container_width=True)

        st.subheader("All games")
        st.dataframe(preds.sort_values(["Time","Matchup"]), use_container_width=True)
else:
    st.info("Set your options on the left, then click **Run slate**.")
