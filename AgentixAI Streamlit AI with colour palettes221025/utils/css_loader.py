import streamlit as st, time

def load_css() -> None:
    """Inject CSS once per session with cache-busting."""
    if "css_loaded" not in st.session_state:
        with open("styles/main.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.session_state.css_loaded = True