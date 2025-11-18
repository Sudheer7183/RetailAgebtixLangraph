# import streamlit as st, json, os
# from i18n.formats import UICONF
# import zoneinfo  # This is still for handling specific timezones.
# import pytz  # Use pytz to get available timezones.

# LABELS = json.load(open("i18n/ui_text.json",encoding="utf-8"))

# def settings_dialog():
#     """Top-right modal (popup) for theme / palette / localization."""
#     if st.button("⚙️", key="open_settings", help="Settings"):
#         st.session_state["_show_settings"] = True

#     if st.session_state.get("_show_settings"):
#         with st.expander(LABELS["settings_title"], expanded=True):
#             cols = st.columns(2)

#             # --- Look & feel ---
#             with cols[0]:
#                 palette = st.selectbox(
#                     LABELS["settings_palette"],
#                     ["blue", "green", "orange", "slate"],
#                     index=["blue", "green", "orange", "slate"].index(
#                         st.session_state.get("palette", "blue"))
#                 )
#                 theme = st.radio(
#                     LABELS["settings_theme"],
#                     ["light", "dark"],
#                     horizontal=True,
#                     index=["light", "dark"].index(st.session_state.get("theme", "light"))
#                 )

#             # --- Localization ---
#             with cols[1]:
#                 tz = st.selectbox(
#                     LABELS["settings_timezone"],
#                     zoneinfo.available_timezones(),
#                     index=sorted(zoneinfo.available_timezones()).index(
#                         st.session_state.get("timezone", "UTC"))
#                 )
#                 curr = st.selectbox(
#                     LABELS["settings_currency"],
#                     list(UICONF["currency"].keys()),
#                     index=list(UICONF["currency"].keys()).index(
#                         st.session_state.get("currency", "INR"))
#                 )
#                 datefmt = st.selectbox(
#                     LABELS["settings_datefmt"],
#                     list(UICONF["datefmt"].keys()),
#                     index=list(UICONF["datefmt"].keys()).index(
#                         st.session_state.get("datefmt", "DD/MM/YYYY"))
#                 )
#                 units = st.radio(
#                     LABELS["settings_units"],
#                     ["metric", "imperial"],
#                     horizontal=True,
#                     index=["metric", "imperial"].index(st.session_state.get("units", "metric"))
#                 )

#             if st.button("Save & Close"):
#                 st.session_state.update({
#                     "palette": palette,
#                     "theme": theme,
#                     "timezone": tz,
#                     "currency": curr,
#                     "currency_symbol": UICONF["currency"][curr]["symbol"],
#                     "datefmt": datefmt,
#                     "units": units
#                 })
#                 st.session_state["_show_settings"] = False
#                 st.rerun()


import streamlit as st, json, zoneinfo
from i18n.formats import UICONF

LABELS = json.load(open("i18n/ui_text.json",encoding="utf-8"))

@st.dialog("⚙️ " + LABELS["settings_title"], width="large")
def _settings_modal():
    cols = st.columns(2)
    # --- Look & feel ---
    with cols[0]:
        palette = st.selectbox(
            LABELS["settings_palette"],
            ["blue", "green", "orange", "slate"],
            index=["blue", "green", "orange", "slate"].index(
                st.session_state.get("palette", "blue"))
        )
        theme = st.radio(
            LABELS["settings_theme"],
            ["light", "dark"],
            horizontal=True,
            index=["light", "dark"].index(st.session_state.get("theme", "light"))
        )
    # --- Localization ---
    with cols[1]:
        tz = st.selectbox(
            LABELS["settings_timezone"],
            sorted(zoneinfo.available_timezones()),
            index=sorted(zoneinfo.available_timezones()).index(
                st.session_state.get("timezone", "UTC"))
        )
        curr = st.selectbox(
            LABELS["settings_currency"],
            list(UICONF["currency"].keys()),
            index=list(UICONF["currency"].keys()).index(
                st.session_state.get("currency", "INR"))
        )
        datefmt = st.selectbox(
            LABELS["settings_datefmt"],
            list(UICONF["datefmt"].keys()),
            index=list(UICONF["datefmt"].keys()).index(
                st.session_state.get("datefmt", "DD/MM/YYYY"))
        )
        units = st.radio(
            LABELS["settings_units"],
            ["metric", "imperial"],
            horizontal=True,
            index=["metric", "imperial"].index(st.session_state.get("units", "metric"))
        )

    if st.button("Save & Close", type="primary"):
        st.session_state.update({
            "palette": palette,
            "theme": theme,
            "timezone": tz,
            "currency": curr,
            "currency_symbol": UICONF["currency"][curr]["symbol"],
            "datefmt": datefmt,
            "units": units
        })
        st.rerun()          # close dialog + repaint

def settings_dialog():
    """Top-right gear that opens the modal."""
    if st.button("⚙️", key="open_settings", help="Settings"):
        _settings_modal()