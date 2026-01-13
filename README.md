# nse-index-autocorrelation 🧭📈

A **signal-only** Streamlit dashboard for **options buyers** that hunts for a very specific kind of edge: **intraday repeatability**.

Instead of treating the trading day as one continuous blob, the app breaks the session into **time-of-day buckets** (09:20, 09:25, …) and measures whether each bucket historically shows **follow-through (momentum)** or **snap-back (mean reversion)** using **5-minute candles**. That behavior is converted into a **30–60 minute directional bias** designed for intraday holds.

✅ Transparent by design.  
⛔ Neutral by default.  
🚫 No auto-orders. No paper trading (MVP). No black box.

---

## What this app does

- **Supports liquid NSE index-derivative underlyings** (index options universe)
- Uses **5-minute candles**
- Produces a **30–60 minute bias** (configurable horizon)
- Classifies each time bucket as:
  - **Momentum** (positive autocorrelation)
  - **Mean Reversion** (negative autocorrelation)
  - **Neutral** (weak / insufficient evidence)
- Shows audit stats alongside every signal:
  - sample size (**n**)
  - **acf1** (lag-1 autocorrelation)
  - **t-stat** (confidence proxy)
  - continuation probability
- Gives an **options-buyer-friendly suggestion** (CE/PE direction) — signal only

---

## Trust-first rules (important)

This project is intentionally conservative:
- **NO TRADE** when evidence is weak (min samples + min t-stat filters)
- Signals are generated from **completed candles only**
- You can see *why* a signal exists (or why it doesn’t)

If you can’t audit it, you can’t trust it.

---

## Tech stack

- **Python + Streamlit**
- **Zerodha Kite Connect** for historical candles
- `pandas`, `numpy` for calculations

---

## Project structure (recommended)

```text
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ tools/
   └─ get_access_token.py   # optional helper for daily token refresh
