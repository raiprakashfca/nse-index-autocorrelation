import os
import re
import json
import datetime as dt

import streamlit as st
import pytz
import gspread
from kiteconnect import KiteConnect

IST = pytz.timezone("Asia/Kolkata")

st.set_page_config(page_title="nse-index-autocorrelation", page_icon="📈", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def ist_now():
    return dt.datetime.now(tz=IST)

def ist_now_str():
    return ist_now().strftime("%Y-%m-%d %H:%M:%S")

def get_cfg(key: str, default=None):
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

def mask(s: str, keep=4):
    if not s:
        return ""
    s = str(s)
    if len(s) <= keep * 2:
        return s[0:keep] + "…"
    return s[:keep] + "…" + s[-keep:]


# ----------------------------
# Service account JSON hardening
# ----------------------------
def load_service_account_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        show = raw[:250].replace("\n", "\\n")
        raise RuntimeError(f"GCP_SERVICE_ACCOUNT_JSON is not valid JSON. Starts with: {show}")
    except Exception as e:
        raise RuntimeError(f"GCP_SERVICE_ACCOUNT_JSON parse error: {e}")


def get_gspread_client():
    sa_json = get_cfg("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in Streamlit secrets.")
    sa_dict = load_service_account_json(sa_json)
    return gspread.service_account_from_dict(sa_dict), sa_dict


def upsert_worksheet(sh, title: str, rows: int = 2000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def ensure_headers(ws, headers):
    existing = ws.row_values(1)
    if existing[: len(headers)] != headers:
        ws.update("A1", [headers])


@st.cache_data(ttl=10, show_spinner=False)
def read_tokenstore():
    """
    Reads TOKENSTORE_TAB A1:D1:
      A1 api_key, B1 api_secret, C1 access_token, D1 timestamp
    """
    gs_id = get_cfg("GSHEET_ID")
    store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")

    out = {"status": "skipped", "error": "", "api_key": None, "api_secret": None, "access_token": None, "ts": None, "sheet_title": None}

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


def write_token_to_sheet_and_verify(access_token: str, user_id: str = ""):
    """
    Writes:
      - TOKENSTORE_TAB: C1 token, D1 timestamp
      - TOKENLOG_TAB: append row
    Then reads back C1:D1 to verify update.
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

    # Verify readback
    back = ws_store.get("C1:D1")
    back_c = back[0][0] if back and back[0] else ""
    back_d = back[0][1] if back and back[0] and len(back[0]) > 1 else ""
    if str(back_c).strip() != str(access_token).strip():
        raise RuntimeError(f"Sheet write verify failed: C1 did not match written token. Read C1={mask(back_c)}")

    ws_log = upsert_worksheet(sh, log_tab)
    ensure_headers(ws_log, ["TimestampIST", "AccessToken", "UserID", "App"])
    ws_log.append_row([ts, access_token, user_id, "nse-index-autocorrelation"], value_input_option="RAW")

    # Bust cache so UI refresh shows newest
    read_tokenstore.clear()

    return {
        "sheet_title": sh.title,
        "store_tab": store_tab,
        "log_tab": log_tab,
        "sa_email": sa_dict.get("client_email", ""),
        "written_ts": ts,
        "readback_ts": back_d,
    }


def test_sheet_write():
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
        raise RuntimeError(
            "Write test failed: could not read back Z1 correctly "
            "(range protected / no edit rights / different sheet?)."
        )

    return {
        "sheet_title": sh.title,
        "tab": store_tab,
        "sa_email": sa_dict.get("client_email", ""),
        "z1": showed,
    }



# ----------------------------
# UI
# ----------------------------
st.title("📈 nse-index-autocorrelation")
st.caption("Auth + Google Sheet sync (hard-verified). If this layer is clean, the rest of the app becomes boring ✅")

sheet = read_tokenstore()

# Resolve credentials from BOTH sources
api_key_sheet = (sheet.get("api_key") or "").strip()
api_secret_sheet = (sheet.get("api_secret") or "").strip()
api_key_secret = (get_cfg("KITE_API_KEY") or "").strip()
api_secret_secret = (get_cfg("KITE_API_SECRET") or "").strip()

api_key = api_key_sheet or api_key_secret
api_secret = api_secret_sheet or api_secret_secret

store_tab = get_cfg("TOKENSTORE_TAB", "Sheet1")
log_tab = get_cfg("TOKENLOG_TAB", "AccessTokenLog")

with st.sidebar:
    st.header("🔐 Auth (Sheet-first)")

    if sheet["status"] == "ok":
        st.success("TokenStore read OK ✅")
        st.write(f"**Spreadsheet:** `{sheet.get('sheet_title') or '—'}`")
        st.write(f"**Tab:** `{store_tab}`")
        st.write(f"**Last ts:** `{sheet.get('ts') or '—'}`")
        st.write(f"**Has token:** {'✅' if sheet.get('access_token') else '❌'}")
    else:
        st.error(f"TokenStore read failed ❌: {sheet.get('error')}")

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
    st.subheader("Diagnostics")

    # Key mismatch guard (silent killer)
    if api_key_sheet and api_key_secret and api_key_sheet != api_key_secret:
        st.error("API Key mismatch between Sheet1!A1 and Streamlit Secrets KITE_API_KEY.")
        st.write(f"Sheet A1: `{mask(api_key_sheet)}`")
        st.write(f"Secrets : `{mask(api_key_secret)}`")
        st.stop()

    if not api_key:
        st.error("Missing API Key. Put it in Sheet1!A1 or Streamlit Secrets.")
        st.stop()

    kite_login = KiteConnect(api_key=api_key)
    login_url = kite_login.login_url()
    st.markdown(f"👉 **Login to Zerodha:** [{login_url}]({login_url})")
    st.caption("After approve, you must land back here with ?request_token=... in the URL.")

    if st.button("🧪 Test Sheet Write (Z1)"):
        try:
            res = test_sheet_write()
            st.success("Write OK ✅")
            st.write(f"Spreadsheet: `{res['sheet_title']}`")
            st.write(f"Tab: `{res['tab']}`")
            st.write(f"SA: `{res['sa_email']}`")
        except Exception as e:
            st.error(f"Write test FAILED: {e}")

    st.divider()
    st.text_input("Manual token (optional)", key="manual_access_token", type="password")
    if st.button("Use manual token"):
        tok = (st.session_state.get("manual_access_token") or "").strip()
        if tok:
            st.session_state["kite_access_token"] = tok
            st.session_state["token_source"] = "Manual"
            st.rerun()


# Load sheet token into session if none
if "kite_access_token" not in st.session_state and sheet.get("access_token"):
    st.session_state["kite_access_token"] = sheet["access_token"]
    st.session_state["token_source"] = "Google Sheet"

# Handle redirect: request_token -> access_token
qp = get_query_params()
request_token = normalize_qp(qp["request_token"]) if "request_token" in qp else None

if request_token and st.session_state.get("last_request_token") != request_token:
    if not api_secret:
        st.error("API Secret missing. Put it in Sheet1!B1 or Streamlit Secrets.")
        st.stop()

    st.info("✅ request_token detected. Generating access_token…")
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
        st.session_state["manual_access_token"] = new_token  # prevent widget clobber

        clear_query_params()
        st.success("🎯 New token generated. Validating…")
    except Exception as e:
        st.error(f"Token generation failed: {e}")
        st.stop()

# Validate token
access_token = (st.session_state.get("kite_access_token") or "").strip()
if not access_token:
    st.warning("No access token available. Use login link.")
    st.stop()

try:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    prof = kite.profile()
except Exception as e:
    st.error(f"Auth failed: {e}")
    st.warning("Login again using sidebar link. If you *did* login, your app may not be receiving request_token.")
    st.stop()

# If valid: FORCE sheet sync and show proof (no silent warnings)
user_id = str(prof.get("user_id", ""))
st.success(f"✅ Auth OK: {prof.get('user_name','—')} ({user_id})")
st.write(f"Token source: **{st.session_state.get('token_source','—')}**")

try:
    res = write_token_to_sheet_and_verify(access_token, user_id=user_id)
    st.success("🧾 Sheet sync OK ✅")
    st.write(f"Spreadsheet: `{res['sheet_title']}`")
    st.write(f"Store tab : `{res['store_tab']}` (C1/D1 updated)")
    st.write(f"Log tab   : `{res['log_tab']}` (row appended)")
    st.write(f"Service Acct: `{res['sa_email']}`")
    st.write(f"Timestamp written: `{res['written_ts']}`")
except Exception as e:
    st.error(f"🧾 Sheet sync FAILED: {e}")
    st.error("This is why your Sheet1 token isn't updating. Fix the exact error above.")
    st.stop()

st.divider()
st.header("📊 Strategy Output")
st.info("Auth + sheet sync is now deterministic. Paste your autocorr signal engine here next.")
