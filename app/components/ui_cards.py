"""
VoltIQ Dashboard: UI Cards and Styling Components
Provides custom CSS injection and reusable visual KPI metric cards.
"""

import streamlit as st


def load_custom_css() -> None:
    """
    Inject professional automotive analytics dark theme CSS into Streamlit.
    """
    st.markdown(
        """
        <style>
        /* Base background and text colors */
        .stApp {
            background-color: #0B132B;
            color: #E0E6ED;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Top header accent bar */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
            border-bottom: 1px solid rgba(0, 240, 255, 0.2);
        }

        /* Main title styling */
        h1, h2, h3 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        .main-header {
            background: linear-gradient(135deg, #00F0FF 0%, #0077B6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
        }

        /* KPI Card Container */
        .metric-card {
            background: linear-gradient(145deg, #1C2541 0%, #151D36 100%);
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.25s ease-in-out;
            margin-bottom: 15px;
        }
        .metric-card:hover {
            border-color: rgba(0, 240, 255, 0.5);
            box-shadow: 0 6px 20px rgba(0, 240, 255, 0.15);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8D99AE;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #00F0FF;
            line-height: 1.2;
        }
        .metric-subtext {
            font-size: 0.8rem;
            color: #ADB5BD;
            margin-top: 5px;
        }

        /* Callout / Disclaimer Boxes */
        .disclaimer-box {
            background: rgba(28, 37, 65, 0.85);
            border-left: 4px solid #00F0FF;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 15px 0 20px 0;
            font-size: 0.9rem;
            color: #CFD8DC;
        }
        .warning-box {
            background: rgba(43, 20, 30, 0.85);
            border-left: 4px solid #FF4B4B;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 15px 0 20px 0;
            font-size: 0.9rem;
            color: #FFCDD2;
        }

        /* Streamlit Tabs custom look */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #111D4A;
            padding: 6px 8px;
            border-radius: 10px;
            border: 1px solid rgba(0, 240, 255, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            color: #8D99AE;
            border-radius: 6px;
            font-weight: 600;
            padding: 0 16px;
            background-color: transparent;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            color: #00F0FF !important;
            background-color: #1C2541 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        }

        /* DataFrames */
        .stDataFrame {
            border: 1px solid rgba(0, 240, 255, 0.15);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    accent_color: str = "#00F0FF",
) -> None:
    """
    Render a styled automotive KPI metric card.
    """
    subtext_html = f'<div class="metric-subtext">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {accent_color};">{value}</div>
            {subtext_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer(message: str, is_warning: bool = False) -> None:
    """
    Render a styled callout container for scientific or demonstration disclaimers.
    """
    box_class = "warning-box" if is_warning else "disclaimer-box"
    prefix = "⚠️ **Scientific Disclaimer:**" if is_warning else "ℹ️ **Operational Note:**"
    st.markdown(
        f"""
        <div class="{box_class}">
            {prefix} {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
