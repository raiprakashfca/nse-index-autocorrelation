```python
# app.py
# nse-index-autocorrelation — professional auth + sheet-sync + 5m autocorr signals + logging
# Replace your existing app.py with this file and commit/push.

import os
import re
import json
import math
import sys
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import gspread
from kiteconnect import KiteConnect


# ============================
# CONFIG
# ============================
IST = pytz.timezone("Asia/Kolkata")

# Liquid NSE index-derivative underlyings (NFO index options universe)
INDEX_UNDERLYINGS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

HORIZON_BARS = {
    "30 min": 6,    # 6 x 5m
    "60 min": 12,   # 12 x 5m
}

DEFAULTS = {
    "lookback_days": 60,
    "impulse_bars": 2,          # how many recent 5m bars define "impulse"
    "min_n": 40,                # min samples per bucket
    "min_abs_corr": 0.06,       # min absolute correlation
    "min_t": 2.0,               # min absolute t-stat
    "refresh_sec": 30,
    "prefer_itm": True,         # option suggestion: 1-step ITM vs ATM
}


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="nse-index-autocorrelation", page_icon="📈", layout="wide")


# ============================
# HELPERS
# ============================
def ist_now() -> dt.datetime:
    return dt.datetime.now(tz=IST)

def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S")

def get_cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and key in st.secrets:
        v = str(st.secrets[key])
        return v.strip()
    v = os.environ.get(key, default)
    return v.strip() if isinstance(v, str) else v

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

def normalize_qp(v):
    return v[0] if isinstance(v, (list, tuple)) else v

def mask(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= keep * 2:
        return s[:keep] + "…"
    return s[:keep] + "…" + s[-keep:]

def safe_sign(x: float) -> int:
    if pd.isna(x) or abs(x) < 1e-12:
        return 0
    return 1 if x > 0 else -1

def corr_t_stat(r: float, n: int) -> float:
    if n <= 2 or pd.isna(r) or abs(r) >= 1:
        return np.nan
    return float(r * math.sqrt((n - 2) / (1 - r * r)))

def market_is_open_ist(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt.time(9, 15) <= t <= dt.time(15, 30)

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def to_ist_datetime_series(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is None:
        return d.dt.tz_localize(IST)
    return d.dt.tz_convert(IST)

def extract_request_token(text: str) -> Optional[str]:
    """
    Accepts:
      - full redirect URL (preferred)
      - just the query string
      - or the raw request_token itself

    Returns request_token or None.
    """
    if not text:
        return None

    t = text.strip()

    # If they pasted only the token
    if "://" not in t and "request_token=" not in t and "&" not in t and "?" not in t:
        return t if len(t) >= 8 else None

    # If they pasted only query part
    if "request_token=" in t and "://" not in t:
        if not t.startswith("?"):
            t = "?" + t
        qs = parse_qs(t[1:])
        return qs.get("request_token", [None])[0]

    # Full URL
    try:
        u = urlparse(t)
        qs = parse_qs(u.query)
        return qs.get("request_token", [None])[0]
    except Exception:
        return None

def ts_is_today_ist(ts: str) -> bool:
    """
    Returns True if ts parses to today's date in IST.
    Accepts ISO or 'YYYY-mm-dd HH:MM:SS' style.
    """
    try:
        d = pd.to_datetime(ts, errors="coerce")
        if pd.isna(d):
            return False
        if getattr(d, "tzinfo", None) is None:
            d = d.tz_localize("Asia/Kolkata")
        else:
            d = d.tz_convert("Asia/Kolkata")
        return d.date() == ist_now().date()
    except Exception:
        return False


# ============================
# SERVICE ACCOUNT JSON HARDENING
# ============================
def load_service_account_json(raw: str) -> dict:
    """
    Robust parser:
    - Try json.loads
    - If TOML multiline accidentally introduced literal newlines inside private_key string,
      repair by replacing literal newlines with \\n inside that JSON value.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(
            r'("private_key"\s*:\s*")(.+?)(")',
            lambda m: m.group(1) + m.group(2).replace("\n", "\\n") + m.group(3),
            raw,
            flags=re.S,
        )
        return json.loads(fixed)


# ============================
# GOOGLE SHEETS
# ============================
def get_gspread_client() -> Tuple[gspread.Client, dict]:
    sa_json = get_cfg("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in Streamlit secrets.")
    sa_dict = load_service_account_json(sa_json)
    return gspread.service_account_from_dict(sa_dict), sa_dict

def upsert_worksheet(sh, title: str, rows: int = 2000, cols: int = 30):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def ensure_headers(ws, headers: List[str]):
    existing = ws.row_values(1)
    if existing[: len(headers)] != headers:
        ws.update("A1", [headers])

@st.cache_data(ttl=10, show_spinner=False)
def read_tokenstore_from_sheet() -> dict:
    """
    Reads TOKENSTORE_TAB (default: Sheet1) A1:D1:
      A1 = API Key
      B1 = API Secret
      C1 = Access Token
      D1 = Timestamp (optional)
    """
    out = {"api_key": None, "api_secret": None, "access_token": None, "ts": None, "status": "skipped", "error": "", "sheet_title": None}

    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")

    if not gs_id:
        out["error"] = "Missing GSHEET_ID in secrets."
        return out

    try:
        gc, _ = get_gspread_client()
        sh = gc.open_by_key(gs_id)
        out["sheet_title"] = sh.title
        ws = sh.worksheet(store_tab)

        row = ws.get("A1:D1")
        if not row or not row[0]:
            out["status"] = "empty"
            out["error"] = f"{store_tab} A1:D1 is empty."
            return out

        vals = row[0] + ["", "", "", ""]
        out["api_key"] = (vals[0] or "").strip() or None
        out["api_secret"] = (vals[1] or "").strip() or None
        out["access_token"] = (vals[2] or "").strip() or None
        out["ts"] = (vals[3] or "").strip() or None
        out["status"] = "ok"
        return out

    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out

def write_token_to_sheets_and_verify(access_token: str, user_id: str = "") -> Dict[str, str]:
    """
    Updates:
      - TOKENSTORE_TAB: C1 token, D1 timestamp
      - TOKENLOG_TAB: append row
    Then verifies C1 by reading it back.
    """
    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")
    log_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

    if not gs_id:
        raise RuntimeError("GSHEET_ID missing.")

    if store_tab == log_tab:
        raise RuntimeError("Misconfig: TOKENSTORE_TAB and TOKENLOG_TAB cannot be the same tab.")

    gc, sa_dict = get_gspread_client()
    sh = gc.open_by_key(gs_id)

    ws_store = upsert_worksheet(sh, store_tab)
    ts = ist_now_str()
    ws_store.update("C1:D1", [[access_token, ts]])

    back = ws_store.get("C1:D1")
    back_c = back[0][0] if back and back[0] and len(back[0]) > 0 else ""
    back_d = back[0][1] if back and back[0] and len(back[0]) > 1 else ""

    if str(back_c).strip() != str(access_token).strip():
        raise RuntimeError(f"Sheet write verify failed: C1 mismatch (read {mask(back_c)}).")

    ws_log = upsert_worksheet(sh, log_tab)
    ensure_headers(ws_log, ["TimestampIST", "AccessToken", "UserID", "App"])
    ws_log.append_row([ts, access_token, user_id, "nse-index-autocorrelation"], value_input_option="RAW")

    read_tokenstore_from_sheet.clear()

    return {
        "sheet_title": sh.title,
        "store_tab": store_tab,
        "log_tab": log_tab,
        "sa_email": sa_dict.get("client_email", ""),
        "written_ts": ts,
        "readback_ts": str(back_d),
    }

def test_sheet_write() -> Dict[str, str]:
    """
    Writes a ping to Z1 to confirm edit access. Safe (far column).
    """
    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")
    if not gs_id:
        raise RuntimeError("GSHEET_ID missing.")

    gc, sa_dict = get_gspread_client()
    sh = gc.open_by_key(gs_id)
    ws = sh.worksheet(store_tab)

    ping = f"PING {ist_now_str()}"
    ws.update("Z1", [[ping]])

    back = ws.get("Z1")
    showed = back[0][0] if back and back[0] and len(back[0]) > 0 else ""

    if str(showed).strip() != ping.strip():
        raise RuntimeError("Write test failed: Z1 readback mismatch (range protected / no edit rights?).")

    return {"sheet_title": sh.title, "tab": store_tab, "sa_email": sa_dict.get("client_email", ""), "z1": showed}

def append_signal_log(row: List, headers: List[str]) -> str:
    """
    Appends a signal snapshot row to SIGNALLOG_TAB (default: SignalLog)
    """
    gs_id = get_cfg("GSHEET_ID")
    sig_tab = get_cfg("SIGNALLOG_TAB", "SignalLog")
    if not gs_id:
        return "Signal log skipped (GSHEET_ID missing)."

    gc, _ = get_gspread_client()
    sh = gc.open_by_key(gs_id)
    ws = upsert_worksheet(sh, sig_tab)
    ensure_headers(ws, headers)
    ws.append_row(row, value_input_option="RAW")
    return f"Signal logged to {sig_tab}."


# ============================
# KITE HELPERS
# ============================
def kite_client(api_key: str, access_token: str) -> KiteConnect:
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k

def validate_token(api_key: str, access_token: str) -> Tuple[bool, Optional[dict], str]:
    try:
        k = kite_client(api_key, access_token)
        prof = k.profile()
        return True, prof, ""
    except Exception as e:
        return False, None, str(e)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_instruments(api_key: str, access_token: str) -> pd.DataFrame:
    k = kite_client(api_key, access_token)
    inst = k.instruments()
    df = pd.DataFrame(inst)

    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df

def resolve_index_token(df_inst: pd.DataFrame, underlying: str) -> int:
    mapping_candidates = {
        "NIFTY": ["NIFTY 50", "NIFTY50", "NIFTY"],
        "BANKNIFTY": ["NIFTY BANK", "BANKNIFTY", "BANK NIFTY"],
        "FINNIFTY": ["NIFTY FIN SERVICE", "FINNIFTY", "NIFTY FIN"],
        "MIDCPNIFTY": ["NIFTY MID SELECT", "MIDCPNIFTY", "MID SELECT", "MIDCAP"],
        "NIFTYNXT50": ["NIFTY NEXT 50", "NIFTYNXT50", "NEXT 50"],
    }.get(underlying.upper(), [underlying.upper()])

    idx = df_inst[df_inst["exchange"] == "NSE"].copy()
    idx["tradingsymbol_u"] = idx["tradingsymbol"].astype(str).str.upper()
    idx["segment_u"] = idx["segment"].astype(str).str.upper()

    # direct equals first
    for c in mapping_candidates:
        hit = idx[idx["tradingsymbol_u"] == c.upper()]
        if not hit.empty:
            return int(hit.iloc[0]["instrument_token"])

    # then contains; prefer INDICES-ish segments
    search_space = idx[idx["segment_u"].str.contains("IND", na=False)] if (idx["segment_u"].str.contains("IND", na=False).any()) else idx

    best_score = -1
    best_token = None
    for _, r in search_space.iterrows():
        ts = str(r["tradingsymbol"]).upper()
        score = 0
        for c in mapping_candidates:
            cu = c.upper()
            if cu in ts:
                score += 10
            if ts == cu:
                score += 50
        score -= min(len(ts), 60) // 10
        if score > best_score:
            best_score = score
            best_token = int(r["instrument_token"])

    if best_token is None or best_score <= 0:
        raise RuntimeError(f"Unable to map {underlying} to an NSE index instrument token.")
    return best_token

@st.cache_data(ttl=60, show_spinner=False)
def fetch_5m_candles(api_key: str, access_token: str, instrument_token: int, lookback_days: int) -> pd.DataFrame:
    k = kite_client(api_key, access_token)
    now = ist_now()
    from_dt = (now - dt.timedelta(days=lookback_days)).replace(tzinfo=None)
    to_dt = now.replace(tzinfo=None)

    data = k.historical_data(
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

    df["date"] = to_ist_datetime_series(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def infer_strike_step(df_inst: pd.DataFrame, name: str, expiry: dt.date) -> int:
    opt = df_inst[
        (df_inst["exchange"] == "NFO")
        & (df_inst["name"] == name)
        & (df_inst["expiry"] == expiry)
        & (df_inst["instrument_type"].isin(["CE", "PE"]))
        & (df_inst["strike"].notna())
    ]
    strikes = np.sort(opt["strike"].unique())
    if len(strikes) < 5:
        fallback = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25, "NIFTYNXT50": 50}
        return int(fallback.get(name, 50))

    diffs = np.diff(strikes)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 50
    step = int(np.median(diffs))
    common = np.array([25, 50, 100])
    return int(common[np.argmin(np.abs(common - step))])

def pick_option_contract(
    df_inst: pd.DataFrame,
    name: str,
    direction: int,
    spot: float,
    prefer_itm: bool,
) -> Optional[Dict]:
    opt_type = "CE" if direction > 0 else "PE"

    opt = df_inst[
        (df_inst["exchange"] == "NFO")
        & (df_inst["name"] == name)
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
    step = infer_strike_step(df_inst, name, expiry)
    atm = round_to_step(spot, step)

    if not prefer_itm:
        strike = atm
    else:
        strike = (atm - step) if opt_type == "CE" else (atm + step)

    pick = opt[(opt["expiry"] == expiry) & (opt["strike"] == float(strike))]
    if pick.empty:
        pick2 = opt[opt["expiry"] == expiry].copy()
        pick2["dist"] = (pick2["strike"] - strike).abs()
        pick2 = pick2.sort_values(["dist"]).head(1)
        if pick2.empty:
            return None
        return pick2.iloc[0].to_dict()

    return pick.sort_values("tradingsymbol").iloc[0].to_dict()


# ============================
# STRATEGY (seasonal autocorr by time bucket)
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
    if df.empty or len(df) < (horizon + 60):
        return pd.DataFrame()

    d = df.copy()
    d["ret"] = np.log(d["close"]).diff()
    d["impulse"] = d["ret"].rolling(impulse_bars).sum()
    d["fwd_ret"] = np.log(d["close"].shift(-horizon) / d["close"])
    d["bucket"] = d["date"].dt.strftime("%H:%M")

    d = d.dropna(subset=["impulse", "fwd_ret", "bucket"])
    if d.empty:
        return pd.DataFrame()

    rows = []
    for b, g in d.groupby("bucket", sort=True):
        n = len(g)
        if n < 10:
            rows.append({"bucket": b, "n": n, "corr": np.nan, "t_stat": np.nan, "mean_fwd": np.nan, "std_fwd": np.nan, "cont_prob": np.nan})
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

    last_bucket = df["date"].iloc[-1].strftime("%H:%M")
    row = stats[stats["bucket"] == last_bucket]
    if row.empty:
        return LiveSignal(last_bucket, "NEUTRAL", 0, 0.0, np.nan, np.nan, 0, np.nan, "Bucket not found in stats (data gap).")

    row = row.iloc[0]
    mode = classify_row(row, min_n, min_abs_corr, min_t)

    rets = np.log(df["close"]).diff()
    impulse_val = float(rets.tail(impulse_bars).sum())
    s_imp = safe_sign(impulse_val)

    if mode == "NEUTRAL" or s_imp == 0:
        return LiveSignal(
            bucket=last_bucket,
            mode="NEUTRAL",
            direction=0,
            confidence=0.0,
            corr=float(row["corr"]) if pd.notna(row["corr"]) else np.nan,
            t_stat=float(row["t_stat"]) if pd.notna(row["t_stat"]) else np.nan,
            n=int(row["n"]),
            cont_prob=float(row["cont_prob"]) if pd.notna(row["cont_prob"]) else np.nan,
            reason="Neutral bucket / weak evidence / no impulse.",
        )

    direction = s_imp if mode == "MOMENTUM" else -s_imp
    t = abs(float(row["t_stat"]))
    confidence = float(np.clip((t - min_t) / max(0.5, (4.0 - min_t)), 0.0, 1.0))

    reason = f"{mode}: corr={row['corr']:.3f}, t={row['t_stat']:.2f}, n={int(row['n'])}"
    if pd.notna(row.get("cont_prob")):
        reason += f", contP={float(row['cont_prob']):.2f}"

    return LiveSignal(
        bucket=last_bucket,
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
# UI: HEADER
# ============================
st.title("📈 nse-index-autocorrelation")
st.caption("Signal-only intraday seasonality engine (5m) for NSE index options — audit-first, NO-TRADE by default ✅")


# ============================
# SIDEBAR: AUTH + SETTINGS + DIAGNOSTICS
# ============================
sheet = read_tokenstore_from_sheet()

api_key_sheet = (sheet.get("api_key") or "").strip()
api_secret_sheet = (sheet.get("api_secret") or "").strip()
token_sheet = (sheet.get("access_token") or "").strip()
ts_sheet = (sheet.get("ts") or "").strip()

api_key_secret = (get_cfg("KITE_API_KEY") or "").strip()
api_secret_secret = (get_cfg("KITE_API_SECRET") or "").strip()

api_key = api_key_sheet or api_key_secret
api_secret = api_secret_sheet or api_secret_secret

store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")
log_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

# Hard-stop mismatch: generating token with one api_key and validating with another = guaranteed failure.
if api_key_sheet and api_key_secret and api_key_sheet != api_key_secret:
    st.error("API Key mismatch between Sheet (A1) and Streamlit Secrets (KITE_API_KEY). Fix it — no guessing.")
    st.write(f"Sheet A1: `{mask(api_key_sheet)}`")
    st.write(f"Secrets : `{mask(api_key_secret)}`")
    st.stop()

if store_tab == log_tab:
    st.error("Misconfig: TOKENSTORE_TAB and TOKENLOG_TAB cannot be the same tab.")
    st.stop()

with st.sidebar:
    st.header("🔐 Zerodha Login")

    # TokenStore status
    if sheet["status"] == "ok":
        st.success("TokenStore read OK ✅")
        st.write(f"**Spreadsheet:** `{sheet.get('sheet_title') or '—'}`")
        st.write(f"**Tab:** `{store_tab}`")
        st.write(f"**Last ts:** `{ts_sheet or '—'}`")
        if token_sheet:
            st.write(f"**Token present:** ✅")
            if ts_sheet and not ts_is_today_ist(ts_sheet):
                st.warning("Sheet token is from a previous day → treated as expired.")
        else:
            st.write("**Token present:** ❌")
    elif sheet["status"] == "empty":
        st.warning("TokenStore empty ⚠️")
        st.caption(sheet["error"])
    else:
        st.error("TokenStore read failed ❌")
        st.caption(sheet.get("error", "Unknown error"))

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔄 Reload Sheet"):
            read_tokenstore_from_sheet.clear()
            st.rerun()
    with colB:
        if st.button("🧨 Clear Session"):
            for k in ["kite_access_token", "token_source", "last_request_token", "last_logged_access_token", "manual_request_token", "last_signal_logged_key"]:
                st.session_state.pop(k, None)
            st.rerun()

    if not api_key:
        st.error("Missing KITE_API_KEY (Sheet A1 or Streamlit Secrets).")
        st.stop()

    # Step 1: login link
    kite_login = KiteConnect(api_key=api_key)
    login_url = kite_login.login_url()
    st.markdown(f"**Step 1:** Open Zerodha login\n\n👉 [{login_url}]({login_url})")

    st.divider()

    # Step 2: paste redirect URL UI (human-proof)
    st.markdown("### ✅ Step 2: Paste Zerodha Redirect URL")
    st.caption(
        "After approval, Zerodha shows a final URL containing `request_token`. "
        "Copy that full URL and paste it here."
    )
    st.text_area(
        "Paste redirect URL",
        key="pasted_redirect_url",
        height=120,
        placeholder="Example:\nhttps://nse-index-autocorrelation.streamlit.app/?request_token=XXXXX&status=success",
    )

    if st.button("🔑 Generate token from pasted URL", use_container_width=True):
        rt = extract_request_token(st.session_state.get("pasted_redirect_url", ""))
        if not rt:
            st.error("No request_token found. Paste the final URL after approval.")
        else:
            st.session_state["manual_request_token"] = rt
            st.success("Captured request_token ✅ Generating access_token…")
            st.rerun()

    st.divider()

    # Safe manual access token override (button applied)
    with st.expander("Manual access_token override (optional)", expanded=False):
        st.text_input("Access Token", key="manual_access_token", type="password")
        if st.button("Use manual access_token"):
            tok = (st.session_state.get("manual_access_token") or "").strip()
            if tok:
                st.session_state["kite_access_token"] = tok
                st.session_state["token_source"] = "Manual"
                st.rerun()
            else:
                st.warning("Empty token.")

    st.divider()

    # Diagnostics
    with st.expander("🧪 Diagnostics", expanded=False):
        st.write(f"Python: `{sys.version.split()[0]}`")
        try:
            import kiteconnect  # noqa
            st.write(f"kiteconnect: `{getattr(kiteconnect, '__version__', '—')}`")
        except Exception:
            st.write("kiteconnect: `not importable`")
        try:
            import gspread  # noqa
            st.write(f"gspread: `{getattr(gspread, '__version__', '—')}`")
        except Exception:
            st.write("gspread: `not importable`")

        if st.button("🧪 Test Sheet Write (Z1)", use_container_width=True):
            try:
                res = test_sheet_write()
                st.success("Sheet write OK ✅")
                st.write(f"Spreadsheet: `{res['sheet_title']}`")
                st.write(f"Tab: `{res['tab']}`")
                st.write(f"Service acct: `{res['sa_email']}`")
            except Exception as e:
                st.error(f"Sheet write FAILED: {e}")

    st.divider()

    # Signal settings
    st.header("⚙️ Signal Settings")
    underlying = st.selectbox("Underlying", INDEX_UNDERLYINGS, index=0)
    horizon_label = st.selectbox("Holding horizon", list(HORIZON_BARS.keys()), index=0)
    horizon = HORIZON_BARS[horizon_label]

    lookback_days = st.slider("Lookback days", 20, 140, DEFAULTS["lookback_days"], 5)
    impulse_bars = st.slider("Impulse bars", 1, 4, DEFAULTS["impulse_bars"], 1)

    st.divider()
    st.header("🧪 Evidence Filters")
    min_n = st.slider("Min samples per bucket (n)", 20, 120, DEFAULTS["min_n"], 5)
    min_abs_corr = st.slider("Min |corr|", 0.02, 0.20, float(DEFAULTS["min_abs_corr"]), 0.01)
    min_t = st.slider("Min |t-stat|", 1.0, 4.0, float(DEFAULTS["min_t"]), 0.25)

    st.divider()
    st.header("🧾 Option Suggestion")
    prefer_itm = st.checkbox("Prefer 1-step ITM", value=DEFAULTS["prefer_itm"])

    st.divider()
    st.header("🔄 Refresh")
    refresh_sec = st.slider("Auto refresh (seconds)", 10, 120, DEFAULTS["refresh_sec"], 5)


# autorefresh (ignore if not available)
try:
    st.autorefresh(interval=refresh_sec * 1000, key="__refresh")
except Exception:
    pass


# ============================
# AUTH FLOW
# ============================
# Prefer: session token. If none, use sheet token ONLY if it looks like "today" OR timestamp missing.
if "kite_access_token" not in st.session_state:
    if token_sheet:
        if (ts_sheet and ts_is_today_ist(ts_sheet)) or (not ts_sheet):
            st.session_state["kite_access_token"] = token_sheet
            st.session_state["token_source"] = "Google Sheet"
        else:
            # stale token; ignore to prevent confusing auth errors
            st.session_state.pop("kite_access_token", None)

# request_token sources: query param OR manual pasted URL
qp = get_query_params()
request_token_qp = normalize_qp(qp["request_token"]) if "request_token" in qp else None
request_token_manual = st.session_state.pop("manual_request_token", None)
request_token = request_token_qp or request_token_manual

# Generate access_token if request_token present and new
if request_token and st.session_state.get("last_request_token") != request_token:
    if not api_secret:
        st.error("Missing KITE_API_SECRET (Sheet B1 or Streamlit Secrets). Needed to generate access_token.")
        st.stop()

    st.info("✅ request_token received. Generating access_token…")
    try:
        tmp = KiteConnect(api_key=api_key)
        sess = tmp.generate_session(request_token, api_secret=api_secret)
        new_token = (sess.get("access_token") or "").strip()
        if not new_token:
            st.error("generate_session returned empty access_token (unexpected).")
            st.stop()

        st.session_state["kite_access_token"] = new_token
        st.session_state["token_source"] = "Generated via Login"
        st.session_state["last_request_token"] = request_token

        # Prevent widget clobber
        st.session_state["manual_access_token"] = new_token

        # Remove request_token from URL to stop rerun loops
        clear_query_params()

        st.success("🎯 access_token generated. Validating…")
    except Exception as e:
        st.error(f"Token generation failed: {e}")
        st.stop()

# Validate token
access_token = (st.session_state.get("kite_access_token") or "").strip()
if not access_token:
    st.warning("No valid access_token available. Use the sidebar login link and paste the final URL here.")
    st.stop()

ok, prof, err = validate_token(api_key, access_token)
if not ok:
    st.error(f"Auth failed: {err}")
    st.warning("Login again → copy the final redirect URL → paste → Generate token.")
    # Clear bad token to avoid repeated failures
    st.session_state.pop("kite_access_token", None)
    st.session_state.pop("token_source", None)
    st.stop()

user_id = str(prof.get("user_id", ""))
user_name = prof.get("user_name", "—")

# Write token to sheets once per token (HARD fail if it can't write)
if st.session_state.get("last_logged_access_token") != access_token:
    try:
        res = write_token_to_sheets_and_verify(access_token, user_id=user_id)
        st.session_state["last_logged_access_token"] = access_token
        st.toast(f"🧾 Sheet sync OK → {res['store_tab']} + {res['log_tab']}")
    except Exception as e:
        st.error(f"🧾 Sheet sync FAILED: {e}")
        st.error("Fix this first. Until sheet-write works, your daily login flow will never be reliable.")
        st.stop()


# ============================
# MAIN DASHBOARD
# ============================
now = ist_now()
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="large")
with c1:
    st.subheader("Session")
    st.write(f"**Now (IST):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Market:** {'OPEN ✅' if market_is_open_ist(now) else 'CLOSED ⛔'}")
with c2:
    st.subheader("Auth")
    st.write(f"**User:** {user_name} ({user_id})")
    st.write(f"**Token source:** {st.session_state.get('token_source', '—')}")
with c3:
    st.subheader("Config")
    st.write(f"**Underlying:** {underlying}")
    st.write(f"**Horizon:** {horizon_label} ({horizon} bars)")
with c4:
    st.subheader("Filters")
    st.write(f"**min n:** {min_n}")
    st.write(f"**min |corr|:** {min_abs_corr:.2f}")
    st.write(f"**min |t|:** {min_t:.2f}")

st.divider()


# ============================
# LOAD DATA
# ============================
with st.spinner("Loading instruments (cached)…"):
    df_inst = load_instruments(api_key, access_token)

try:
    idx_token = resolve_index_token(df_inst, underlying)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.spinner("Fetching 5m candles (cached)…"):
    df = fetch_5m_candles(api_key, access_token, idx_token, lookback_days)

if df.empty or len(df) < (horizon + 60):
    st.warning("Not enough candle data. Increase lookback days or try later.")
    st.stop()

spot = float(df["close"].iloc[-1])
last_ts = df["date"].iloc[-1]


# ============================
# STRATEGY OUTPUT
# ============================
stats = compute_bucket_stats(df, horizon=horizon, impulse_bars=impulse_bars)
if stats.empty:
    st.warning("Could not compute bucket stats (insufficient clean data).")
    st.stop()

stats["mode"] = stats.apply(lambda r: classify_row(r, min_n, min_abs_corr, min_t), axis=1)

signal = make_live_signal(
    df=df,
    stats=stats,
    horizon=horizon,
    impulse_bars=impulse_bars,
    min_n=min_n,
    min_abs_corr=min_abs_corr,
    min_t=min_t,
)

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

st.caption("Signals are generated from completed 5m candles only. Weak buckets → NO TRADE ✅")

left, right = st.columns([1.35, 1.0], gap="large")

opt = None
with left:
    st.subheader("🎯 Live Decision (signal-only)")

    if signal is None:
        st.info("No signal available.")
    else:
        if signal.direction == 0:
            st.warning("NO TRADE — evidence filters not met for this time bucket.")
        else:
            st.success("TRADE BIAS ACTIVE ✅ (signal-only)")

        st.write(f"**Bucket:** `{signal.bucket}`")
        st.write(f"**Mode:** `{signal.mode}`")
        st.write(f"**corr:** `{signal.corr:.3f}`  |  **t-stat:** `{signal.t_stat:.2f}`  |  **n:** `{signal.n}`")
        if pd.notna(signal.cont_prob):
            st.write(f"**Continuation probability:** `{signal.cont_prob:.2f}`")
        st.caption(signal.reason)

        if signal.direction != 0:
            opt = pick_option_contract(df_inst, underlying, signal.direction, spot, prefer_itm=prefer_itm)
            st.divider()
            st.subheader("🧾 Suggested Contract (informational)")
            if opt is None:
                st.error("Could not resolve an option contract from instrument list.")
            else:
                st.write(f"**Tradingsymbol:** `{opt.get('tradingsymbol','—')}`")
                st.write(f"**Expiry:** `{opt.get('expiry','—')}`  |  **Strike:** `{opt.get('strike','—')}`  |  **Type:** `{opt.get('instrument_type','—')}`")
                st.write(f"**Token:** `{opt.get('instrument_token','—')}`  |  **Lot:** `{opt.get('lot_size','—')}`")
                st.caption("No orders are placed. You execute (or ignore).")

    st.divider()
    st.subheader("📉 Price Context (recent)")
    tail = df.tail(180)[["date", "close"]].set_index("date")
    st.line_chart(tail)

with right:
    st.subheader("🏆 Best Buckets (edge_score)")
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


# ============================
# SIGNAL LOGGING (once-per-bucket guard)
# ============================
if signal is not None:
    log_key = f"{underlying}|{horizon_label}|{signal.bucket}"
    if st.session_state.get("last_signal_logged_key") != log_key:
        headers = [
            "TimestampIST", "Underlying", "Horizon", "Bucket",
            "Signal", "Mode", "Confidence",
            "Corr", "TStat", "N", "ContinuationProb",
            "Spot",
            "SuggestedSymbol", "Expiry", "Strike", "OptionType"
        ]

        sig_txt = "NO_TRADE" if signal.direction == 0 else ("BULLISH_CE" if signal.direction > 0 else "BEARISH_PE")
        suggested_symbol = opt.get("tradingsymbol") if isinstance(opt, dict) else ""
        expiry = str(opt.get("expiry")) if isinstance(opt, dict) else ""
        strike = opt.get("strike") if isinstance(opt, dict) else ""
        opt_type = opt.get("instrument_type") if isinstance(opt, dict) else ""

        row = [
            ist_now_str(), underlying, horizon_label, signal.bucket,
            sig_txt, signal.mode, round(signal.confidence, 4),
            round(signal.corr, 6) if pd.notna(signal.corr) else "",
            round(signal.t_stat, 4) if pd.notna(signal.t_stat) else "",
            int(signal.n),
            round(signal.cont_prob, 4) if pd.notna(signal.cont_prob) else "",
            round(spot, 2),
            suggested_symbol, expiry, strike, opt_type
        ]

        try:
            msg = append_signal_log(row=row, headers=headers)
            st.session_state["last_signal_logged_key"] = log_key
            st.caption(f"🧾 {msg}")
        except Exception as e:
            st.caption(f"Signal log skipped: {e}")

st.caption("Education-only. Autocorrelation edges are small & regime-sensitive. Expect lots of NO TRADE buckets. ✅")
```
