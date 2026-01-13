import os
import json
import datetime as dt

import streamlit as st
import pytz
import gspread
from kiteconnect import KiteConnect

IST = pytz.timezone("Asia/Kolkata")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="nse-index-autocorrelation",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Helpers: secrets/env + query params compatibility
# ----------------------------
def get_cfg(key: str, default: str | None = None) -> str | None:
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.environ.get(key, default)

def ist_now_str() -> str:
    return dt.datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")

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

def normalize_qp_value(v):
    return v[0] if isinstance(v, (list, tuple)) else v

# ----------------------------
# Google Sheets helpers
# ----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def _read_tokenstore_from_sheet() -> dict:
    """
    Reads ZerodhaTokenStore A1:D1:
      A1 = API Key
      B1 = API Secret
      C1 = Access Token
      D1 = Timestamp (IST) (optional)
    Returns dict with keys: api_key, api_secret, access_token, ts, status, error
    """
    out = {"api_key": None, "api_secret": None, "access_token": None, "ts": None, "status": "skipped", "error": ""}

    sa_json = get_cfg("GCP_SERVICE_ACCOUNT_JSON")
    gs_id = get_cfg("GSHEET_ID")
    tab = get_cfg("TOKENSTORE_TAB", "ZerodhaTokenStore")

    if not sa_json or not gs_id:
        out["status"] = "skipped"
        out["error"] = "Missing GCP_SERVICE_ACCOUNT_JSON and/or GSHEET_ID in Streamlit Secrets."
        return out

    try:
        sa_dict = json.loads(sa_json)
        gc = gspread.service_account_from_dict(sa_dict)
        sh = gc.open_by_key(gs_id)
        ws = sh.worksheet(tab)

        # A1:D1
        row = ws.get("A1:D1")
        if not row or not row[0]:
            out["status"] = "empty"
            out["error"] = "ZerodhaTokenStore A1:D1 is empty."
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

def _upsert_worksheet(sh, title: str, rows: int = 2000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))

def _ensure_headers(ws, headers: list[str]):
    existing = ws.row_values(1)
    if existing[: len(headers)] != headers:
        ws.update("A1", [headers])

def write_token_to_sheet(access_token: str, user_id: str = "") -> str:
    """
    Updates:
      ZerodhaTokenStore: C1 token, D1 timestamp
      AccessTokenLog: append timestamp/token/user_id
    Requires secrets:
      GCP_SERVICE_ACCOUNT_JSON, GSHEET_ID, TOKENSTORE_TAB, TOKENLOG_TAB
    """
    sa_json = get_cfg("GCP_SERVICE_ACCOUNT_JSON")
    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "ZerodhaTokenStore")
    log_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

    if not sa_json or not gs_id:
        return "Sheets write skipped (missing GCP_SERVICE_ACCOUNT_JSON / GSHEET_ID)."

    sa_dict = json.loads(sa_json)
    gc = gspread.service_account_from_dict(sa_dict)
    sh = gc.open_by_key(gs_id)

    ws_store = _upsert_worksheet(sh, store_tab)
    ws_store.update("C1:D1", [[access_token, ist_now_str()]])

    ws_log = _upsert_worksheet(sh, log_tab)
    headers = ["TimestampIST", "AccessToken", "UserID", "App"]
    _ensure_headers(ws_log, headers)
    ws_log.append_row([ist_now_str(), access_token, user_id, "nse-index-autocorrelation"], value_input_option="RAW")

    # Bust cache so next rerun reads latest
    _read_tokenstore_from_sheet.clear()
    return f"Updated {store_tab} (C1/D1) and appended to {log_tab}."

# ----------------------------
# UI: Sidebar controls
# ----------------------------
st.sidebar.title("🔐 Zerodha Login")

# Read from sheet FIRST
sheet_data = _read_tokenstore_from_sheet()

# Prefer sheet values; fallback to secrets/env
api_key = sheet_data.get("api_key") or get_cfg("KITE_API_KEY")
api_secret = sheet_data.get("api_secret") or get_cfg("KITE_API_SECRET")

# Token preference:
# 1) session_state token (if already verified/generated)
# 2) sheet token (latest stored)
# 3) manual override (later)
if "kite_access_token" not in st.session_state and sheet_data.get("access_token"):
    st.session_state["kite_access_token"] = sheet_data["access_token"]
    st.session_state["token_source"] = "Google Sheet"

# Buttons to control token flow
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("🔄 Reload from Sheet"):
        # Force refresh read + overwrite token in session with sheet token
        _read_tokenstore_from_sheet.clear()
        sheet_data = _read_tokenstore_from_sheet()
        if sheet_data.get("access_token"):
            st.session_state["kite_access_token"] = sheet_data["access_token"]
            st.session_state["token_source"] = "Google Sheet"
            st.sidebar.success("Loaded latest token from sheet.")
        else:
            st.sidebar.warning("No access_token found in sheet.")
with col_b:
    if st.button("🧨 Force New Login"):
        # Clears token so app shows login flow
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("token_source", None)
        st.session_state.pop("last_request_token", None)
        st.session_state.pop("last_logged_access_token", None)
        st.sidebar.info("Cleared session token. Use login link to generate new one.")

st.sidebar.markdown("---")
st.sidebar.caption("Sheet status")
if sheet_data["status"] == "ok":
    st.sidebar.write("✅ Read token store successfully")
    if sheet_data.get("ts"):
        st.sidebar.write(f"Last updated: `{sheet_data['ts']}`")
elif sheet_data["status"] == "skipped":
    st.sidebar.warning("Sheets read skipped")
    st.sidebar.caption(sheet_data["error"])
elif sheet_data["status"] == "empty":
    st.sidebar.warning("Token store empty")
    st.sidebar.caption(sheet_data["error"])
else:
    st.sidebar.error("Sheets read failed")
    st.sidebar.caption(sheet_data["error"])

if not api_key:
    st.sidebar.error("Missing API Key. Put it in ZerodhaTokenStore!A1 or Streamlit Secrets (KITE_API_KEY).")
    st.stop()

# Kite instance (needs api_key always)
kite = KiteConnect(api_key=api_key)

# Login link shown, but we won't force it if token works
login_url = kite.login_url()
st.sidebar.markdown(f"👉 **Login to Zerodha:** [{login_url}]({login_url})")

st.sidebar.caption(
    "If the token in your sheet is valid, the app will use it automatically. "
    "Only login if it’s expired/invalid."
)

# Manual override (optional)
st.sidebar.markdown("---")
manual_token = st.sidebar.text_input(
    "Access Token (optional override)",
    value=st.session_state.get("kite_access_token", ""),
    type="password",
    help="Paste a token if you have one. Otherwise rely on Sheet token or login link.",
)
if manual_token.strip():
    st.session_state["kite_access_token"] = manual_token.strip()
    st.session_state["token_source"] = "Manual"

# ----------------------------
# Handle redirect back: request_token -> access_token
# ----------------------------
qp = get_query_params()
request_token = normalize_qp_value(qp["request_token"]) if "request_token" in qp else None

if request_token and st.session_state.get("last_request_token") != request_token:
    st.info("✅ Detected `request_token` in URL. Generating access token…")

    if not api_secret:
        st.error("Missing API Secret. Put it in ZerodhaTokenStore!B1 or Streamlit Secrets (KITE_API_SECRET).")
        st.stop()

    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
        st.session_state["kite_access_token"] = access_token
        st.session_state["token_source"] = "Generated via Login"
        st.session_state["last_request_token"] = request_token
        st.session_state["token_generated_at"] = ist_now_str()
        clear_query_params()  # prevent rerun loops
        st.success("🎯 Access token generated and loaded into session.")
    except Exception as e:
        st.error(f"Token generation failed: {e}")
        st.stop()

# ----------------------------
# Use token if available (and validate)
# ----------------------------
access_token = st.session_state.get("kite_access_token")
if not access_token:
    st.warning("No access token found yet. If ZerodhaTokenStore has one, press 'Reload from Sheet'. Otherwise login.")
    st.stop()

kite.set_access_token(access_token)

# ----------------------------
# Main: Validate token first, then decide if we need login
# ----------------------------
st.title("📈 nse-index-autocorrelation")
st.caption("Uses latest token from Google Sheet automatically. Generates a new token only when needed.")

col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.subheader("✅ Auth Status")
    try:
        prof = kite.profile()
        user_name = prof.get("user_name", "—")
        user_id = prof.get("user_id", "—")
        st.write(f"**User:** {user_name} ({user_id})")
        st.write("**Session:** Active ✅")
        st.write(f"**Token source:** {st.session_state.get('token_source', '—')}")
        if sheet_data.get("ts"):
            st.write(f"**Sheet timestamp:** {sheet_data['ts']}")
        if st.session_state.get("token_generated_at"):
            st.write(f"**Generated at:** {st.session_state['token_generated_at']}")

        # If token came from login/manual and differs from sheet, write it once
        if st.session_state.get("last_logged_access_token") != access_token:
            try:
                msg = write_token_to_sheet(access_token, user_id=str(user_id))
                st.session_state["last_logged_access_token"] = access_token
                st.success(f"🧾 Sheets sync: {msg}")
            except Exception as e:
                st.error(f"Sheets sync failed: {e}")

    except Exception as e:
        # Token invalid -> force user to login (but we don't block the login link)
        st.error(f"Auth check failed (token invalid/expired): {e}")
        st.warning("👉 Use the sidebar login link to generate a fresh token, then you’ll be redirected back here.")
        # Also clear session token so app doesn't keep trying it
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("token_source", None)
        st.stop()

with col2:
    st.subheader("🔑 Current Access Token")
    st.caption("Shown for convenience. Token is also stored in Google Sheet once validated/generated.")
    st.code(access_token, language="text")

st.markdown("---")
st.header("📊 Strategy Output")
st.info("Auth + token sheet-sync is ready. Next step: plug the 5m seasonality engine here.")
