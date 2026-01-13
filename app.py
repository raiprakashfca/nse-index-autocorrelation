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
st.set_page_config(page_title="nse-index-autocorrelation", page_icon="📈", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def get_cfg(key: str, default: str | None = None) -> str | None:
    if hasattr(st, "secrets") and key in st.secrets:
        v = str(st.secrets[key])
        return v.strip() if isinstance(v, str) else v
    v = os.environ.get(key, default)
    return v.strip() if isinstance(v, str) else v

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

def normalize_qp(v):
    return v[0] if isinstance(v, (list, tuple)) else v


# ----------------------------
# Google Sheets
# ----------------------------
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

def ensure_headers(ws, headers: list[str]):
    existing = ws.row_values(1)
    if existing[: len(headers)] != headers:
        ws.update("A1", [headers])

@st.cache_data(ttl=10, show_spinner=False)
def read_tokenstore() -> dict:
    """
    Reads ZerodhaTokenStore A1:D1:
      A1=API Key, B1=API Secret, C1=Access Token, D1=Timestamp
    """
    out = {"api_key": None, "api_secret": None, "access_token": None, "ts": None, "status": "skipped", "error": ""}

    gs_id = get_cfg("GSHEET_ID")
    tab = get_cfg("TOKENSTORE_TAB", "ZerodhaTokenStore")
    if not gs_id:
        out["error"] = "Missing GSHEET_ID in secrets."
        return out

    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(gs_id)
        ws = sh.worksheet(tab)
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

def write_token_to_sheets(access_token: str, user_id: str = "") -> str:
    """
    Updates:
      - ZerodhaTokenStore: C1 token, D1 timestamp
      - AccessTokenLog: append row
    """
    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "ZerodhaTokenStore")
    log_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

    if not gs_id:
        return "Sheets write skipped (GSHEET_ID missing)."

    gc = get_gspread_client()
    sh = gc.open_by_key(gs_id)

    ws_store = upsert_worksheet(sh, store_tab)
    ws_store.update("C1:D1", [[access_token, ist_now_str()]])

    ws_log = upsert_worksheet(sh, log_tab)
    ensure_headers(ws_log, ["TimestampIST", "AccessToken", "UserID", "App"])
    ws_log.append_row([ist_now_str(), access_token, user_id, "nse-index-autocorrelation"], value_input_option="RAW")

    # bust cache so the sidebar shows latest immediately
    read_tokenstore.clear()
    return f"Updated {store_tab} (C1/D1) + appended to {log_tab}."


# ----------------------------
# Token validation
# ----------------------------
def validate_token(api_key: str, access_token: str) -> tuple[bool, dict | None, str]:
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        prof = kite.profile()  # raises if invalid
        return True, prof, ""
    except Exception as e:
        return False, None, str(e)


# ============================
# UI
# ============================
st.title("📈 nse-index-autocorrelation")
st.caption("Auth-first flow: uses latest API details & token from Google Sheet before asking for fresh login.")

# ---- Read sheet first
sheet = read_tokenstore()

# ---- Resolve API key/secret priority: Sheet -> Secrets/env
api_key = (sheet.get("api_key") or get_cfg("KITE_API_KEY") or "").strip()
api_secret = (sheet.get("api_secret") or get_cfg("KITE_API_SECRET") or "").strip()

# ---- Sidebar
with st.sidebar:
    st.header("🔐 Zerodha Auth")

    # Status of sheet
    st.subheader("Google Sheet Status")
    if sheet["status"] == "ok":
        st.success("TokenStore read OK ✅")
        st.write(f"**Last updated:** `{sheet.get('ts') or '—'}`")
        st.write(f"**Has API key:** {'✅' if sheet.get('api_key') else '❌'}")
        st.write(f"**Has API secret:** {'✅' if sheet.get('api_secret') else '❌'}")
        st.write(f"**Has access token:** {'✅' if sheet.get('access_token') else '❌'}")
    elif sheet["status"] == "empty":
        st.warning("TokenStore empty ⚠️")
        st.caption(sheet["error"])
    elif sheet["status"] == "error":
        st.error("TokenStore read failed ❌")
        st.caption(sheet["error"])
    else:
        st.info("TokenStore read skipped ℹ️")
        st.caption(sheet["error"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reload Sheet"):
            read_tokenstore.clear()
            st.rerun()
    with c2:
        if st.button("🧨 Clear Session"):
            for k in ["kite_access_token", "token_source", "last_request_token", "last_logged_access_token"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()

    # Login link (always visible)
    if not api_key:
        st.error("API key missing. Put it in ZerodhaTokenStore!A1 or Streamlit Secrets (KITE_API_KEY).")
        st.stop()

    kite_for_login = KiteConnect(api_key=api_key)
    login_url = kite_for_login.login_url()
    st.markdown(f"👉 **Login to Zerodha:** [{login_url}]({login_url})")
    st.caption("Only login if the current token is invalid/expired.")

    st.divider()

    # Manual token is now SAFE (won't overwrite unless you click button)
    with st.expander("Manual token override (optional)", expanded=False):
        st.text_input(
            "Paste token here",
            key="manual_access_token",
            type="password",
            placeholder="Paste access_token if you have it",
        )
        if st.button("Use manual token"):
            tok = (st.session_state.get("manual_access_token") or "").strip()
            if tok:
                st.session_state["kite_access_token"] = tok
                st.session_state["token_source"] = "Manual"
                st.success("Manual token loaded into session.")
                st.rerun()
            else:
                st.warning("Manual token is empty.")


# ---- If sheet has an access token and we don't have one in session, load it
if "kite_access_token" not in st.session_state and sheet.get("access_token"):
    st.session_state["kite_access_token"] = sheet["access_token"]
    st.session_state["token_source"] = "Google Sheet"

# ---- Handle redirect back: request_token -> access_token
qp = get_query_params()
request_token = normalize_qp(qp["request_token"]) if "request_token" in qp else None

if request_token and st.session_state.get("last_request_token") != request_token:
    if not api_secret:
        st.error("API secret missing. Put it in ZerodhaTokenStore!B1 or Streamlit Secrets (KITE_API_SECRET).")
        st.stop()

    st.info("✅ request_token detected. Generating access_token…")
    try:
        kite_tmp = KiteConnect(api_key=api_key)
        session = kite_tmp.generate_session(request_token, api_secret=api_secret)
        new_token = (session.get("access_token") or "").strip()

        if not new_token:
            st.error("generate_session succeeded but no access_token returned (unexpected).")
            st.stop()

        # Put into session first
        st.session_state["kite_access_token"] = new_token
        st.session_state["token_source"] = "Generated via Login"
        st.session_state["last_request_token"] = request_token
        st.session_state["token_generated_at"] = ist_now_str()

        # IMPORTANT: sync widget state so it cannot overwrite the new token
        st.session_state["manual_access_token"] = new_token

        # Clear query params to avoid loops
        clear_query_params()

        st.success("🎯 New access_token generated and loaded.")
    except Exception as e:
        st.error(f"Token generation failed: {e}")
        st.stop()

# ---- Validate current token
access_token = (st.session_state.get("kite_access_token") or "").strip()
if not access_token:
    st.warning("No access token available. Use the login link in the sidebar (or load from sheet).")
    st.stop()

ok, prof, err = validate_token(api_key, access_token)

# ---- Main cards
left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("✅ Auth Status")
    st.write(f"**API key source:** {'Sheet' if sheet.get('api_key') else 'Secrets/Env'}")
    st.write(f"**Token source:** {st.session_state.get('token_source', '—')}")
    if sheet.get("ts"):
        st.write(f"**Sheet token timestamp:** `{sheet['ts']}`")
    if st.session_state.get("token_generated_at"):
        st.write(f"**Generated at:** `{st.session_state['token_generated_at']}`")

    if ok:
        st.success("Session Active ✅")
        st.write(f"**User:** {prof.get('user_name', '—')} ({prof.get('user_id', '—')})")

        # Write to Sheets only once per token
        if st.session_state.get("last_logged_access_token") != access_token:
            try:
                msg = write_token_to_sheets(access_token, user_id=str(prof.get("user_id", "")))
                st.session_state["last_logged_access_token"] = access_token
                st.success(f"🧾 Sheets sync: {msg}")
            except Exception as e:
                st.error(f"Sheets sync failed: {e}")
    else:
        st.error(f"Token invalid/expired: {err}")
        st.warning("Use the sidebar login link to generate a fresh token.")
        # Clear session token so we don't keep retrying a bad one
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("token_source", None)
        st.stop()

with right:
    st.subheader("🔑 Current Access Token")
    st.caption("Shown for convenience. Also synced to your Google Sheet.")
    st.code(access_token, language="text")

st.divider()
st.header("📊 Strategy Output")
st.info("Auth + sheet-first token flow is stable now. Next: plug in the 5m seasonality engine here.")
