import os
import streamlit as st
from kiteconnect import KiteConnect

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
    # Prefer Streamlit secrets, fallback to env
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.environ.get(key, default)

def get_query_params() -> dict:
    # Streamlit >= 1.30 has st.query_params; older has experimental_get_query_params
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

# ----------------------------
# Sidebar: Authentication
# ----------------------------
st.sidebar.title("🔐 Zerodha Login")

api_key = get_cfg("KITE_API_KEY")
api_secret = get_cfg("KITE_API_SECRET")

if not api_key:
    st.sidebar.error("Missing KITE_API_KEY in Streamlit secrets.")
    st.stop()

kite = KiteConnect(api_key=api_key)

login_url = kite.login_url()
st.sidebar.markdown(
    f"**Step 1:** Click to login & approve\n\n👉 [{login_url}]({login_url})",
    unsafe_allow_html=False,
)

st.sidebar.caption(
    "After approval, Zerodha redirects back with `request_token`. "
    "This app will auto-generate your `access_token`."
)

# ----------------------------
# Handle redirect back: request_token -> access_token
# ----------------------------
qp = get_query_params()
# qp values can be list-like (older API) or strings (new API). Normalize:
request_token = None
if "request_token" in qp:
    v = qp["request_token"]
    request_token = v[0] if isinstance(v, (list, tuple)) else v

if request_token:
    st.info("✅ Detected `request_token` in URL. Generating access token…")

    if not api_secret:
        st.error("Missing KITE_API_SECRET in Streamlit secrets. Needed to generate access_token.")
        st.stop()

    # Avoid re-generating repeatedly on reruns
    if st.session_state.get("last_request_token") != request_token:
        try:
            session = kite.generate_session(request_token, api_secret=api_secret)
            access_token = session["access_token"]
            st.session_state["kite_access_token"] = access_token
            st.session_state["last_request_token"] = request_token

            st.success("🎯 Access token generated. Using it for this session now.")
            clear_query_params()  # removes request_token from URL to prevent re-run loops
        except Exception as e:
            st.error(f"Token generation failed: {e}")
            st.stop()

# ----------------------------
# Manual override (optional)
# ----------------------------
st.sidebar.markdown("---")
token_in_state = st.session_state.get("kite_access_token", "")
manual_token = st.sidebar.text_input(
    "Access Token (optional override)",
    value=token_in_state,
    type="password",
    help="If you already have a valid token, paste it here.",
)

if manual_token:
    st.session_state["kite_access_token"] = manual_token.strip()

# ----------------------------
# Use token if available
# ----------------------------
access_token = st.session_state.get("kite_access_token")
if not access_token:
    st.warning("Login first using the link in the sidebar to generate today’s access token.")
    st.stop()

kite.set_access_token(access_token)

# ----------------------------
# Sanity check: profile (optional but trustable)
# ----------------------------
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    st.subheader("✅ Auth Status")
    try:
        prof = kite.profile()
        st.write(f"**User:** {prof.get('user_name', '—')} ({prof.get('user_id', '—')})")
        st.write(f"**Broker:** Zerodha Kite Connect")
        st.write("**Session:** Active")
    except Exception as e:
        st.error(f"Auth check failed (token invalid/expired?): {e}")
        st.stop()

with col2:
    st.subheader("🔑 Today’s Access Token")
    st.caption("Copy it if you want to store it elsewhere. (This app keeps it only in session memory.)")
    st.code(access_token, language="text")

st.markdown("---")
st.header("📊 Strategy Output")
st.info("Authentication is working. Next: plug in your 5m autocorr/seasonality logic and render signals here.")
