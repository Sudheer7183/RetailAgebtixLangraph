import zoneinfo, locale, datetime, streamlit as st
from typing import Dict, Any
import pandas as pd

UICONF = {
    "currency": {
        "USD": {"symbol": "$", "code": "en_US.UTF-8"},
        "EUR": {"symbol": "€", "code": "de_DE.UTF-8"},
        "INR": {"symbol": "₹", "code": "en_IN.UTF-8"}
    },
    "datefmt": {
        "MM/DD/YYYY": "%m/%d/%Y",
        "DD/MM/YYYY": "%d/%m/%Y",
        "YYYY-MM-DD": "%Y-%m-%d"
    },
    "units": {
        "metric": {"temp": "°C", "speed": "km/h"},
        "imperial": {"temp": "°F", "speed": "mph"}
    }
}

def localize(df: pd.DataFrame, cols: list, fmt: str = "currency") -> pd.DataFrame:
    """Convert numeric cols to localized string (currency, date, etc.)."""
    if fmt == "currency":
        sym = st.session_state.get("currency_symbol", "₹")
        for c in cols:
            df[c] = df[c].apply(lambda x: f"{sym}{x:,.2f}")
    return df

def tz_now():
    tz = st.session_state.get("timezone", "UTC")
    return datetime.datetime.now(zoneinfo.ZoneInfo(tz))