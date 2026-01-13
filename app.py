import os
import json
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import gspread
from kiteconnect import KiteConnect


# ============================
# Constants
# ============================
IST = pytz.timezone("Asia/Kolkata")

# Index option underlyings we want (liquid NSE index derivatives)
INDEX_UNDERLYINGS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

HORIZON_BARS = {
    "30 min": 6,   # 6 x 5m
    "60 min": 12,  # 12 x 5m
}

DEFAULTS = {
    "lookback_days": 60,
    "impulse_bars": 2,
    "min_n": 40,
    "min_abs_corr": 0.06,
    "min_t": 2.0,
    "refresh_sec": 30,
    "prefer_itm": True,
}


# ============================
# Utilities
# ============================
def ist_now() -> dt.datetime:
    return dt.datetime.now(tz=IST)

def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S")

def safe_sign(x: float) -> int:
    if pd.isna(x) or abs(x) < 1e-12:
        return 0
    return 1 if x > 0 else -1

def corr_t_stat(r: float, n: int) -> float:
    if n <= 2 or pd.isna(r) or abs(r) >= 1:
        return np.nan
    return float(r * math.sqrt((n - 2) / (1 - r * r)))

def get_cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.environ.get(key, default)

def get_query_params() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

def market_is_open_ist(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt.time(9, 15) <= t <= dt.time(15, 30)

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)


# ============================
# Google Sheets
# ============================
def get_gspread_client():
    sa_json = get_cfg("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in Streamlit secrets.")
    sa_dict = json.loads(sa_json)
    return gspread.service_account_from_dict(sa_dict)

def upsert_worksheet(sh, title: str, rows: int = 2000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def ensure_headers(ws, headers: List[str]):
    existing = ws.row_values(1)
    if existing[: len(headers)] != headers:
        ws.update("A1", [headers])

def write_token_to_sheets(access_token: str, user_id: str = "") -> str:
    """
    Writes token to:
      - ZerodhaTokenStore: C1 (token) and D1 (timestamp)
      - AccessTokenLog: append a row (TimestampIST, AccessToken, UserID, App)
    Controlled by secrets:
      GSHEET_ID, TOKENSTORE_TAB, TOKENLOG_TAB
    """
    gs_id = get_cfg("GSHEET_ID")
    if not gs_id:
        return "GSHEET_ID not set → skipped Sheets write."

    tokenstore_tab = get_cfg("TOKENSTORE_TAB", "ZerodhaTokenStore")
    tokenlog_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

    gc = get_gspread_client()
    sh = gc.open_by_key(gs_id)

    # Token store updates (batch for speed)
    ws_store = upsert_worksheet(sh, tokenstore_tab)
    ws_store.update("C1:D1", [[access_token, ist_now_str()]])

    # Token log
    ws_log = upsert_worksheet(sh, tokenlog_tab)
    headers = ["TimestampIST", "AccessToken", "UserID", "App"]
    ensure_headers(ws_log, headers)
    ws_log.append_row([ist_now_str(), access_token, user_id, "nse-index-autocorrelation"], value_input_option="RAW")

    return f"Updated {tokenstore_tab} (C1/D1) and appended to {tokenlog_tab}."


# ============================
# Kite: instruments + data
# ============================
def get_kite(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_instruments(api_key: str, access_token: str) -> pd.DataFrame:
    kite = get_kite(api_key, access_token)
    inst = kite.instruments()
    df = pd.DataFrame(inst)

    # normalize types
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df

def find_index_token(df_inst: pd.DataFrame, underlying: str) -> int:
    """
    Find NSE index instrument_token for underlying.
    Uses heuristic matching on NSE INDICES segment.
    """
    idx = df_inst[
        (df_inst["exchange"] == "NSE")
        & (df_inst["segment"].fillna("").str.contains("INDICES", case=False))
    ].copy()

    if idx.empty:
        raise RuntimeError("Could not find NSE INDICES segment in instruments.")

    target = underlying.upper()

    # keyword heuristics
    patterns = {
        "NIFTY": ["NIFTY 50", "NIFTY50", "NIFTY"],
        "BANKNIFTY": ["NIFTY BANK", "BANK"],
        "FINNIFTY": ["FIN", "FIN SERVICE", "FINANCIAL"],
        "MIDCPNIFTY": ["MID", "MIDCAP", "MID SELECT", "MIDCAP SELECT"],
        "NIFTYNXT50": ["NEXT 50", "NIFTY NEXT 50", "NXT 50"],
    }.get(target, [target])

    def score_row(r) -> int:
        ts = str(r.get("tradingsymbol", "")).upper()
        nm = str(r.get("name", "")).upper()
        s = 0
        for p in patterns:
            p = p.upper()
            if ts == p:
                s += 100
            if p in ts:
                s += 30
            if p in nm:
                s += 10
        # slight preference for shorter symbols
        s -= min(len(ts), 50) // 10
        return s

    idx["score"] = idx.apply(score_row, axis=1)
    best = idx.sort_values(["score"], ascending=False).head(1)
    if best.empty or int(best.iloc[0]["score"]) <= 0:
        raise RuntimeError(f"Could not confidently map {underlying} to an NSE index symbol. Check instrument list.")
    return int(best.iloc[0]["instrument_token"])

@st.cache_data(ttl=60, show_spinner=False)
def fetch_5m_candles(api_key: str, access_token: str, instrument_token: int, lookback_days: int) -> pd.DataFrame:
    kite = get_kite(api_key, access_token)
    now = ist_now()
    from_dt = (now - dt.timedelta(days=lookback_days)).replace(tzinfo=None)
    to_dt = now.replace(tzinfo=None)

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval="5minute",
        continuous=False,
        oi=False,
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def infer_strike_step(df_inst: pd.DataFrame, opt_name: str, expiry: dt.date) -> int:
    opt = df_inst[
        (df_inst["exchange"] == "NFO")
        & (df_inst["name"] == opt_name)
        & (df_inst["expiry"] == expiry)
        & (df_inst["instrument_type"].isin(["CE", "PE"]))
        & (df_inst["strike"].notna())
    ]
    strikes = np.sort(opt["strike"].unique())
    if len(strikes) < 5:
        # fallback guesses
        fallback = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25, "NIFTYNXT50": 50}
        return int(fallback.get(opt_name, 50))

    diffs = np.diff(strikes)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 50
    step = int(np.median(diffs))
    # snap to common steps
    common = np.array([25, 50, 100])
    step = int(common[np.argmin(np.abs(common - step))])
    return step

def pick_option_contract(
    df_inst: pd.DataFrame,
    opt_name: str,
    direction: int,
    spot: float,
    prefer_itm: bool,
) -> Optional[Dict]:
    """
    direction: +1 bullish -> CE, -1 bearish -> PE
    Picks nearest expiry ATM or 1-step ITM.
    """
    opt_type = "CE" if direction > 0 else "PE"

    opt = df_inst[
        (df_inst["exchange"] == "NFO")
        & (df_inst["name"] == opt_name)
        & (df_inst["instrument_type"] == opt_type)
        & (df_inst["expiry"].notna())
        & (df_inst["strike"].notna())
    ].copy()

    if opt.empty:
        return None

    today = ist_now().date()
    opt = opt[opt["expiry"] >= today]
    if opt.empty:
        return None

    expiry = opt["expiry"].min()
    step = infer_strike_step(df_inst, opt_name, expiry)
    atm = round_to_step(spot, step)

    if prefer_itm:
        strike = atm - step if opt_type == "CE" else atm + step
    else:
        strike = atm

    pick = opt[(opt["expiry"] == expiry) & (opt["strike"] == float(strike))]
    if pick.empty:
        # fallback to closest strike
        pick2 = opt[opt["expiry"] == expiry].copy()
        pick2["dist"] = (pick2["strike"] - strike).abs()
        pick2 = pick2.sort_values(["dist"]).head(1)
        if pick2.empty:
            return None
        return pick2.iloc[0].to_dict()

    return pick.sort_values("tradingsymbol").iloc[0].to_dict()


# ============================
# Strategy: seasonal bucket edge (5m) with 30/60m horizon
# ============================
@dataclass
class LiveSignal:
    bucket: str
    mode: str
    direction: int
    confidence: float
    corr: float
    t_stat: float
    n: int
    cont_prob: float
    reason: str

def compute_bucket_stats(df: pd.DataFrame, horizon: int, impulse_bars: int) -> pd.DataFrame:
    if df.empty or len(df) < (horizon + 50):
        return pd.DataFrame()

    d = df.copy()
    d["ret"] = np.log(d["close"]).diff()
    d["impulse"] = d["ret"].rolling(impulse_bars).sum()
    d["fwd_ret"] = np.log(d["close"].shift(-horizon) / d["close"])
    d["bucket"] = d["date"].dt.strftime("%H:%M")

    d = d.dropna(subset=["ret", "impulse", "fwd_ret", "bucket"])
    if d.empty:
        return pd.DataFrame()

    rows = []
    for b, g in d.groupby("bucket", sort=True):
        n = len(g)
        if n < 10:
            rows.append({"bucket": b, "n": n, "corr": np.nan, "t_stat": np.nan,
                         "mean_fwd": np.nan, "std_fwd": np.nan, "cont_prob": np.nan})
            continue

        corr = g["impulse"].corr(g["fwd_ret"])
        t = corr_t_stat(corr, n)

        s_imp = g["impulse"].apply(safe_sign)
        s_fwd = g["fwd_ret"].apply(safe_sign)
        valid = (s_imp != 0) & (s_fwd != 0)
        cont_prob = float((s_imp[valid] == s_fwd[valid]).mean()) if valid.sum() > 0 else np.nan

        rows.append({
            "bucket": b,
            "n": int(n),
            "corr": float(corr) if pd.notna(corr) else np.nan,
            "t_stat": float(t) if pd.notna(t) else np.nan,
            "mean_fwd": float(g["fwd_ret"].mean()),
            "std_fwd": float(g["fwd_ret"].std(ddof=1)),
            "cont_prob": cont_prob,
        })

    s = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    s["edge_score"] = (s["corr"].abs() * np.sqrt(s["n"].clip(lower=1))).replace([np.inf, -np.inf], np.nan)
    return s

def classify_row(row, min_n: int, min_abs_corr: float, min_t: float) -> str:
    if row["n"] < min_n or pd.isna(row["corr"]) or pd.isna(row["t_stat"]):
        return "NEUTRAL"
    if abs(row["corr"]) < min_abs_corr or abs(row["t_stat"]) < min_t:
        return "NEUTRAL"
    return "MOMENTUM" if row["corr"] > 0 else "MEAN_REVERSION"

def make_live_signal(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    horizon: int,
    impulse_bars: int,
    min_n: int,
    min_abs_corr: float,
    min_t: float,
) -> Optional[LiveSignal]:
    if df.empty or stats.empty or len(df) < (impulse_bars + 5):
        return None

    last_time = df["date"].iloc[-1].strftime("%H:%M")
    row = stats[stats["bucket"] == last_time]
    if row.empty:
        return LiveSignal(last_time, "NEUTRAL", 0, 0.0, np.nan, np.nan, 0, np.nan,
                          "Bucket not found in stats (data gap).")
    row = row.iloc[0]
    mode = classify_row(row, min_n, min_abs_corr, min_t)

    # impulse from most recent completed candle
    rets = np.log(df["close"]).diff()
    impulse_val = float(rets.tail(impulse_bars).sum())
    s_imp = safe_sign(impulse_val)

    if mode == "NEUTRAL" or s_imp == 0:
        reason = "Neutral bucket / weak evidence / no impulse."
        return LiveSignal(
            bucket=last_time,
            mode="NEUTRAL",
            direction=0,
            confidence=0.0,
            corr=float(row["corr"]) if pd.notna(row["corr"]) else np.nan,
            t_stat=float(row["t_stat"]) if pd.notna(row["t_stat"]) else np.nan,
            n=int(row["n"]),
            cont_prob=float(row["cont_prob"]) if pd.notna(row["cont_prob"]) else np.nan,
            reason=reason,
        )

    direction = s_imp if mode == "MOMENTUM" else -s_imp

    # confidence mapping: start at min_t, saturate near 4.0
    t = abs(float(row["t_stat"]))
    confidence = float(np.clip((t - min_t) / max(0.5, (4.0 - min_t)), 0.0, 1.0))

    reason = f"{mode}: corr(impulse, fwd_{horizon}bars)={row['corr']:.3f}, t={row['t_stat']:.2f}, n={int(row['n'])}, contP={row['cont_prob']:.2f if pd.notna(row['cont_prob']) else float('nan')}"

    return LiveSignal(
        bucket=last_time,
        mode=mode,
        direction=direction,
        confidence=confidence,
        corr=float(row["corr"]),
        t_stat=float(row["t_stat"]),
        n=int(row["n"]),
        cont_prob=float(row["cont_prob"]) if pd.notna(row["cont_prob"]) else np.nan,
        reason=reason,
    )


# ============================
# UI
# ============================
st.set_page_config(page_title="nse-index-autocorrelation", page_icon="📈", layout="wide")
st.title("📈 nse-index-autocorrelation")
st.caption("5m intraday seasonality engine for NSE index derivatives — signal-only, audit-first.")

# ---- Sidebar
with st.sidebar:
    st.header("🔐 Zerodha Login")

    api_key = get_cfg("KITE_API_KEY")
    api_secret = get_cfg("KITE_API_SECRET")

    if not api_key:
        st.error("Missing KITE_API_KEY in Streamlit secrets.")
        st.stop()

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    st.markdown(f"👉 **Login link:** [{login_url}]({login_url})")

    st.caption("Approve → redirected back with request_token → token auto-generated → written to Google Sheets.")

    st.divider()
    st.header("⚙️ Signal Settings")

    underlying = st.selectbox("Underlying", INDEX_UNDERLYINGS, index=0)
    horizon_label = st.selectbox("Holding horizon", list(HORIZON_BARS.keys()), index=0)
    horizon = HORIZON_BARS[horizon_label]

    lookback_days = st.slider("Lookback days", 20, 140, DEFAULTS["lookback_days"], 5)
    impulse_bars = st.slider("Impulse bars (recent momentum)", 1, 4, DEFAULTS["impulse_bars"], 1)

    st.divider()
    st.header("🧪 Evidence Filters")
    min_n = st.slider("Min samples per bucket (n)", 20, 120, DEFAULTS["min_n"], 5)
    min_abs_corr = st.slider("Min |corr|", 0.02, 0.20, float(DEFAULTS["min_abs_corr"]), 0.01)
    min_t = st.slider("Min |t-stat|", 1.0, 4.0, float(DEFAULTS["min_t"]), 0.25)

    st.divider()
    st.header("🧾 Option Suggestion")
    prefer_itm = st.checkbox("Prefer 1-step ITM", value=DEFAULTS["prefer_itm"])
    st.caption("Suggestion is informational: nearest expiry + ATM/ITM strike from instrument list.")

    st.divider()
    st.header("🔄 Refresh")
    refresh_sec = st.slider("Auto refresh (seconds)", 10, 120, DEFAULTS["refresh_sec"], 5)

# auto refresh (safe fallback)
try:
    st.autorefresh(interval=refresh_sec * 1000, key="__refresh")
except Exception:
    pass

# ---- Handle request_token redirect
qp = get_query_params()
request_token = None
if "request_token" in qp:
    v = qp["request_token"]
    request_token = v[0] if isinstance(v, (list, tuple)) else v

if request_token and st.session_state.get("last_request_token") != request_token:
    if not api_secret:
        st.error("Missing KITE_API_SECRET in Streamlit secrets (needed to generate session).")
        st.stop()

    st.info("✅ Detected request_token. Generating access_token…")
    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
        st.session_state["kite_access_token"] = access_token
        st.session_state["last_request_token"] = request_token
        st.session_state["token_generated_at"] = ist_now_str()
        clear_query_params()
        st.success("🎯 Access token generated.")
    except Exception as e:
        st.error(f"Token generation failed: {e}")
        st.stop()

# ---- Manual override (optional)
with st.sidebar:
    st.divider()
    st.subheader("Manual token (optional)")
    manual_token = st.text_input("Access Token", value=st.session_state.get("kite_access_token", ""), type="password")
    if manual_token:
        st.session_state["kite_access_token"] = manual_token.strip()

access_token = st.session_state.get("kite_access_token")
if not access_token:
    st.warning("Login using the sidebar link to generate today’s access token.")
    st.stop()

# ---- Kite client + profile check
kite = get_kite(api_key, access_token)

c1, c2, c3 = st.columns([1.2, 1.2, 1.6], gap="large")

with c1:
    st.subheader("Session")
    now = ist_now()
    st.write(f"**Now (IST):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Market:** {'OPEN ✅' if market_is_open_ist(now) else 'CLOSED ⛔'}")

with c2:
    st.subheader("Auth")
    try:
        prof = kite.profile()
        user_name = prof.get("user_name", "—")
        user_id = prof.get("user_id", "—")
        st.write(f"**User:** {user_name} ({user_id})")
        st.write("**Status:** Active ✅")
        st.write(f"**Token at:** {st.session_state.get('token_generated_at', 'unknown')}")
    except Exception as e:
        st.error(f"Auth failed (token invalid/expired?): {e}")
        st.stop()

with c3:
    st.subheader("Token → Google Sheets")
    # Write token to Sheets only once per token
    if st.session_state.get("last_logged_access_token") != access_token:
        try:
            msg = write_token_to_sheets(access_token, user_id=str(user_id))
            st.session_state["last_logged_access_token"] = access_token
            st.success(msg)
        except Exception as e:
            st.error(f"Sheets update failed: {e}")
    else:
        st.info("Token already logged for this session (no duplicate write).")

st.divider()

# ---- Load instruments (cached)
with st.spinner("Loading instruments (cached)…"):
    try:
        df_inst = load_instruments(api_key, access_token)
    except Exception as e:
        st.error(f"Failed to load instruments: {e}")
        st.stop()

# ---- Resolve index token and candles
try:
    idx_token = find_index_token(df_inst, underlying)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.spinner("Fetching 5m candles (cached)…"):
    try:
        df = fetch_5m_candles(api_key, access_token, idx_token, lookback_days)
    except Exception as e:
        st.error(f"Failed to fetch candles: {e}")
        st.stop()

if df.empty or len(df) < (horizon + 60):
    st.warning("Not enough candle data returned yet. Increase lookback or try again later.")
    st.stop()

spot = float(df["close"].iloc[-1])
last_ts = df["date"].iloc[-1]

# ---- Compute stats + live signal
stats = compute_bucket_stats(df, horizon=horizon, impulse_bars=impulse_bars)
if stats.empty:
    st.warning("Could not compute bucket stats (insufficient clean data).")
    st.stop()

stats["mode"] = stats.apply(lambda r: classify_row(r, min_n, min_abs_corr, min_t), axis=1)

signal = make_live_signal(
    df, stats, horizon=horizon, impulse_bars=impulse_bars,
    min_n=min_n, min_abs_corr=min_abs_corr, min_t=min_t
)

# ---- Header KPIs
k1, k2, k3, k4, k5 = st.columns(5, gap="large")
k1.metric("Underlying", underlying)
k2.metric("Last Close", f"{spot:.2f}")
k3.metric("Last Candle (IST)", last_ts.strftime("%H:%M"))
if signal is None:
    k4.metric("Signal", "—")
    k5.metric("Confidence", "—")
else:
    sig_txt = "NO TRADE" if signal.direction == 0 else ("BULLISH (CE)" if signal.direction > 0 else "BEARISH (PE)")
    k4.metric("Signal", sig_txt)
    k5.metric("Confidence", f"{signal.confidence:.2f}")

st.caption(f"Using completed 5m candles. Horizon = {horizon_label} ({horizon} bars).")

# ---- Main layout
left, right = st.columns([1.35, 1.0], gap="large")

with left:
    st.subheader("🎯 Live Decision (signal-only)")
    if signal is None:
        st.info("No signal available.")
    else:
        if signal.direction == 0:
            st.warning("NO TRADE — weak/neutral evidence for this bucket.")
        else:
            st.success("TRADE BIAS ACTIVE (signal-only) ✅")

        st.write(f"**Bucket:** `{signal.bucket}`")
        st.write(f"**Mode:** `{signal.mode}`")
        st.write(f"**corr(impulse, forward_return):** `{signal.corr:.3f}`  |  **t-stat:** `{signal.t_stat:.2f}`  |  **n:** `{signal.n}`")
        if pd.notna(signal.cont_prob):
            st.write(f"**Continuation probability:** `{signal.cont_prob:.2f}`")
        st.caption(signal.reason)

        # Option suggestion
        if signal.direction != 0:
            opt = pick_option_contract(df_inst, underlying, signal.direction, spot, prefer_itm=prefer_itm)
            st.divider()
            st.subheader("🧾 Suggested Contract")
            if opt is None:
                st.error("Could not resolve an option contract from instrument list.")
            else:
                st.write(f"**Tradingsymbol:** `{opt.get('tradingsymbol','—')}`")
                st.write(f"**Expiry:** `{opt.get('expiry','—')}`  |  **Strike:** `{opt.get('strike','—')}`  |  **Type:** `{opt.get('instrument_type','—')}`")
                st.write(f"**Token:** `{opt.get('instrument_token','—')}`  |  **Lot size:** `{opt.get('lot_size','—')}`")
                st.caption("This is informational only. You decide entries/exits; no orders are placed.")

    st.divider()
    st.subheader("📉 Price Context (last 2 trading days approx)")
    tail = df.tail(160)[["date", "close"]].set_index("date")
    st.line_chart(tail)

with right:
    st.subheader("🏆 Best Buckets (by edge_score)")
    top = stats.copy().sort_values("edge_score", ascending=False).head(12)
    show = top[["bucket", "mode", "n", "corr", "t_stat", "cont_prob", "edge_score"]].copy()
    show["corr"] = show["corr"].round(3)
    show["t_stat"] = show["t_stat"].round(2)
    show["cont_prob"] = show["cont_prob"].round(3)
    show["edge_score"] = show["edge_score"].round(3)
    st.dataframe(show, use_container_width=True, height=420)

    st.divider()
    st.subheader("🧾 Full Bucket Table")
    full = stats.copy()
    full["corr"] = full["corr"].round(3)
    full["t_stat"] = full["t_stat"].round(2)
    full["cont_prob"] = full["cont_prob"].round(3)
    full["mean_fwd_bps"] = (full["mean_fwd"] * 10000).round(2)
    full["std_fwd_bps"] = (full["std_fwd"] * 10000).round(2)
    st.dataframe(
        full[["bucket", "mode", "n", "corr", "t_stat", "cont_prob", "mean_fwd_bps", "std_fwd_bps", "edge_score"]],
        use_container_width=True,
        height=520,
    )

st.divider()
st.caption("Education-only tool. Autocorrelation edges are small & regime-sensitive. Expect many NO TRADE buckets. ✅")
