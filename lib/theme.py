"""Inject the dashboard's CSS tokens that .streamlit/config.toml can't cover.

Only used for layout tokens — card radius, divider grey, secondary text
colour, sidebar styling. No widget logic or content lives here.
"""

from pathlib import Path

import streamlit as st


_CSS_PATH = Path(__file__).parent.parent / "assets" / "styles.css"


def inject_theme() -> None:
    """Read assets/styles.css and inject into the page head."""
    css = _CSS_PATH.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
