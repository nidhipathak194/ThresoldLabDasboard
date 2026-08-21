import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import io

from src.data_loader import (
    generate_synthetic_ulb_dataset,
    preprocess_data,
    load_custom_csv,
    detect_local_dataset,
    PCA_FEATURE_DESCRIPTIONS
)
from src.ml_engine import FraudMLPipeline
from src.sorites_engine import (
    compute_elkan_threshold,
    evaluate_threshold_cost,
    analyze_contested_zone,
    find_sorites_twin_transactions,
    run_cost_ratio_sweep,
    run_threshold_cost_curve
)

# ---------------------------------------------------------
# Page Configuration & University of Liverpool Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Threshold Lab | Sorites Paradox & Fraud Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Clean Light Executive Dashboard Design System
st.markdown("""
<style>
    /* Adaptive Theme Variables for Seamless Light & Dark Mode Compatibility */
    :root {
        --bg-main: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-card: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --heading-color: #002147;
        --briefing-bg: #F8FAFC;
        --briefing-border: #CBD5E1;
        --briefing-header: #0F172A;
        --briefing-text: #334155;
        --tx-card-a-bg: #FEF2F2;
        --tx-card-a-border: #FCA5A5;
        --tx-card-a-header: #991B1B;
        --tx-card-a-text: #7F1D1D;
        --tx-card-b-bg: #ECFDF5;
        --tx-card-b-border: #6EE7B7;
        --tx-card-b-header: #065F46;
        --tx-card-b-text: #064E3B;
    }

    /* Dark Mode Overrides (Triggered by Streamlit top-right menu dark toggle or system dark preference) */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-main: #0E1117;
            --bg-card: #1E293B;
            --border-card: #334155;
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --heading-color: #38BDF8;
            --briefing-bg: #0F172A;
            --briefing-border: #334155;
            --briefing-header: #38BDF8;
            --briefing-text: #E2E8F0;
            --tx-card-a-bg: #450A0A;
            --tx-card-a-border: #991B1B;
            --tx-card-a-header: #FCA5A5;
            --tx-card-a-text: #FECACA;
            --tx-card-b-bg: #064E3B;
            --tx-card-b-border: #065F46;
            --tx-card-b-header: #6EE7B7;
            --tx-card-b-text: #A7F3D0;
        }
    }

    [data-theme="dark"], body.dark, div[data-testid="stAppViewContainer"][data-theme="dark"], .stApp[data-theme="dark"] {
        --bg-main: #0E1117 !important;
        --bg-card: #1E293B !important;
        --border-card: #334155 !important;
        --text-primary: #F8FAFC !important;
        --text-secondary: #CBD5E1 !important;
        --text-muted: #94A3B8 !important;
        --heading-color: #38BDF8 !important;
        --briefing-bg: #0F172A !important;
        --briefing-border: #334155 !important;
        --briefing-header: #38BDF8 !important;
        --briefing-text: #E2E8F0 !important;
        --tx-card-a-bg: #450A0A !important;
        --tx-card-a-border: #991B1B !important;
        --tx-card-a-header: #FCA5A5 !important;
        --tx-card-a-text: #FECACA !important;
        --tx-card-b-bg: #064E3B !important;
        --tx-card-b-border: #065F46 !important;
        --tx-card-b-header: #6EE7B7 !important;
        --tx-card-b-text: #A7F3D0 !important;
    }

    .main {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Academic Header Banner */
    .uol-banner {
        background: linear-gradient(135deg, #00152e 0%, #002147 60%, #1E3A8A 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 33, 71, 0.15);
        border-left: 6px solid #F59E0B;
    }
    
    .uol-banner-title {
        color: #FFFFFF !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.4px;
    }

    .uol-banner-subtitle {
        color: #FCD34D !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 0 0 8px 0 !important;
    }

    .uol-banner-meta {
        color: #E2E8F0 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    /* Modern Dashboard Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-card) !important;
        border-top: 4px solid var(--heading-color) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    div[data-testid="stMetricLabel"] > label {
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: var(--heading-color) !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
    }

    /* Typography & Headings (Electric Blue #0284C7 / #0070F3) */
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #0284C7 !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px;
    }

    p, span, label, li {
        color: var(--text-primary);
    }

    /* Cards & Container Boxes */
    .sorites-card, .briefing-card {
        background-color: var(--briefing-bg) !important;
        border: 1.5px solid var(--briefing-border) !important;
        border-left: 5px solid #0284C7 !important;
        border-radius: 10px !important;
        padding: 22px 26px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .sorites-card-header, .briefing-card h4 {
        color: #0284C7 !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 12px !important;
    }

    .sorites-card p, .sorites-card li, .briefing-card p, .briefing-card li {
        color: var(--briefing-text) !important;
        font-size: 1.0rem !important;
        line-height: 1.65 !important;
    }

    /* Transaction Cards A & B */
    .tx-card-a {
        background-color: var(--tx-card-a-bg) !important;
        border: 1.5px solid var(--tx-card-a-border) !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
    }
    .tx-card-a h4 {
        color: var(--tx-card-a-header) !important;
    }
    .tx-card-a ul, .tx-card-a li, .tx-card-a p {
        color: var(--tx-card-a-text) !important;
    }

    .tx-card-b {
        background-color: var(--tx-card-b-bg) !important;
        border: 1.5px solid var(--tx-card-b-border) !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
    }
    .tx-card-b h4 {
        color: var(--tx-card-b-header) !important;
    }
    .tx-card-b ul, .tx-card-b li, .tx-card-b p {
        color: var(--tx-card-b-text) !important;
    }

    /* Status Pills */
    .tag-genuine {
        background-color: #10B981;
        color: #FFFFFF;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tag-fraud {
        background-color: #EF4444;
        color: #FFFFFF;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tag-contested {
        background-color: #F59E0B;
        color: #FFFFFF;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-card) !important;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-card) !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: var(--bg-card) !important;
        border-radius: 8px;
        color: var(--text-secondary) !important;
        font-weight: 700;
        font-size: 0.92rem;
        padding: 0px 20px;
        border: 1px solid var(--border-card) !important;
    }

    /* Active Selected Tab: Electric Blue Background & Dark Black Text */
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid #005691 !important;
        box-shadow: 0 3px 12px rgba(2, 132, 199, 0.4) !important;
    }

    /* Expander & Controls */
    .stExpander {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }

    /* Sidebar Positioned on Right-Hand Side with Rich Oxford Blue Background */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        right: 0 !important;
        left: auto !important;
        border-left: 1.5px solid #1E293B !important;
        border-right: none !important;
    }
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"] {
        right: 1rem !important;
        left: auto !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #38BDF8 !important;
    }

    /* All Sidebar Click Buttons, Radio Options, and Labels in Crisp Pure White */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label *,
    section[data-testid="stSidebar"] .stRadio label *,
    section[data-testid="stSidebar"] div[data-baseweb="radio"] *,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* All Sidebar Radio Buttons Outer & Inner Circles: Default White, Active RED (#DC2626) */
    section[data-testid="stSidebar"] div[data-baseweb="radio"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="radio"] div {
        border-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="radio"] input + div {
        border-color: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="radio"] input:checked + div,
    section[data-testid="stSidebar"] div[data-baseweb="radio"] [aria-checked="true"] div {
        border-color: #DC2626 !important;
        background-color: #DC2626 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="radio"] input:checked + div > div,
    section[data-testid="stSidebar"] div[data-baseweb="radio"] [aria-checked="true"] div > div {
        background-color: #FFFFFF !important;
    }

    /* Selectbox Dropdown Container: Default White, Active Focus & Menu Highlight RED (#DC2626) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #FFFFFF !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {
        border-color: #DC2626 !important;
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.4) !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    ul[data-baseweb="menu"] li[aria-selected="true"],
    ul[data-baseweb="menu"] li:hover {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] li[aria-selected="true"] *,
    ul[data-baseweb="menu"] li:hover * {
        color: #FFFFFF !important;
    }

    /* Slider Track, Handle & Toggle: Vibrant Bright Red (#FF0000 / #FF1E1E) Accents */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
    section[data-testid="stSidebar"] div[data-baseweb="slider"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"],
    section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background-color: #FF0000 !important;
        background: #FF0000 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.7) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] div,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"]::before,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"]::after {
        background-color: #FF0000 !important;
        background: #FF0000 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
        background: rgba(255, 0, 0, 0.25) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTrack"],
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTrack"] > div {
        background: linear-gradient(90deg, #FF0000 0%, #FF2A2A 100%) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #FF0000 !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"] > div {
        background-color: #FF0000 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stToggle"] input:checked + div,
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] input:checked + div {
        background-color: #FF0000 !important;
        border-color: #FF0000 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Question Mark Symbol BLACK with WHITE CIRCLE Background in Sidebar Tooltips */
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg {
        background-color: #FFFFFF !important;
        border-radius: 50% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] circle,
    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] circle {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] path,
    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] path {
        fill: #000000 !important;
        stroke: #000000 !important;
    }

    /* Specific Dark Mode Overrides for 100% High-Contrast Text Readability */
    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] .stMarkdown p,
    [data-theme="dark"] .stMarkdown li,
    [data-theme="dark"] .stMarkdown span,
    [data-theme="dark"] .stMarkdown div,
    [data-theme="dark"] .stMarkdown b,
    [data-theme="dark"] .stMarkdown strong,
    [data-theme="dark"] label,
    [data-theme="dark"] .stCaption,
    [data-theme="dark"] div[data-testid="stCaptionContainer"] p,
    [data-theme="dark"] .stRadio label,
    [data-theme="dark"] .stSelectbox label,
    [data-theme="dark"] .stSlider label {
        color: #F8FAFC !important;
    }

    [data-theme="dark"] h1, 
    [data-theme="dark"] h2, 
    [data-theme="dark"] h3, 
    [data-theme="dark"] h4,
    [data-theme="dark"] .stMarkdown h1,
    [data-theme="dark"] .stMarkdown h2,
    [data-theme="dark"] .stMarkdown h3,
    [data-theme="dark"] .stMarkdown h4 {
        color: #0284C7 !important;
    }

    /* Info / Warning alert boxes in Dark Mode */
    [data-theme="dark"] div[data-testid="stAlert"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
    }
    [data-theme="dark"] div[data-testid="stAlert"] * {
        color: #F1F5F9 !important;
    }

    /* Table styling in Dark Mode */
    [data-theme="dark"] table,
    [data-theme="dark"] th,
    [data-theme="dark"] td,
    [data-theme="dark"] div[data-testid="stTable"] * {
        color: #F8FAFC !important;
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }

    /* Expander details in Dark Mode */
    [data-theme="dark"] .stExpander {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    [data-theme="dark"] .stExpander * {
        color: #F8FAFC !important;
    }

    /* Tab labels in Dark Mode */
    [data-theme="dark"] .stTabs [data-baseweb="tab"] {
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        border-color: #334155 !important;
    }
    [data-theme="dark"] .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: 1.5px solid #005691 !important;
        box-shadow: 0 3px 12px rgba(2, 132, 199, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Top Academic Header (At Very Top of Screen)
# ---------------------------------------------------------
st.markdown("""
<div class="uol-banner">
    <div class="uol-banner-title">Threshold Lab: Calibrating the Sorites Paradox in Credit Card Fraud Detection</div>
    <div class="uol-banner-subtitle">University of Liverpool — Department of Computer Science | COMP702 M.Sc. Project</div>
    <div class="uol-banner-meta">Bridging Analytical Philosophy (Sorites Paradox) & Cost-Sensitive Probability Calibration in Severe Class Imbalance</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Data Loading & Model Fitting Engine
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading benchmark dataset...")
def get_synthetic_data(n_samples=50000, random_state=42):
    return generate_synthetic_ulb_dataset(n_samples=n_samples, fraud_rate=0.00172, random_state=random_state)

@st.cache_resource(show_spinner="Training ML pipelines & probability calibration models...")
def fit_models(df, imbalance_strategy='class_weight', random_state=42):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.2, random_state=random_state)
    
    pipeline_logistic = FraudMLPipeline(model_type='logistic', imbalance_strategy=imbalance_strategy)
    pipeline_logistic.fit(X_train, y_train)

    pipeline_ensemble = FraudMLPipeline(model_type='ensemble', imbalance_strategy=imbalance_strategy)
    pipeline_ensemble.fit(X_train, y_train)

    pipeline_xgboost = FraudMLPipeline(model_type='xgboost', imbalance_strategy=imbalance_strategy)
    pipeline_xgboost.fit(X_train, y_train)

    probas_logistic = pipeline_logistic.predict_probabilities(X_test)
    probas_ensemble = pipeline_ensemble.predict_probabilities(X_test)
    probas_xgboost = pipeline_xgboost.predict_probabilities(X_test)

    metrics_logistic = pipeline_logistic.evaluate_calibration(y_test, probas_logistic)
    metrics_ensemble = pipeline_ensemble.evaluate_calibration(y_test, probas_ensemble)
    metrics_xgboost = pipeline_xgboost.evaluate_calibration(y_test, probas_xgboost)

    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'probas': {
            'logistic': probas_logistic,
            'ensemble': probas_ensemble,
            'xgboost': probas_xgboost
        },
        'metrics': {
            'logistic': metrics_logistic,
            'ensemble': metrics_ensemble,
            'xgboost': metrics_xgboost
        },
        'pipelines': {
            'logistic': pipeline_logistic,
            'ensemble': pipeline_ensemble,
            'xgboost': pipeline_xgboost
        }
    }

# ---------------------------------------------------------
# Sidebar Controls & Configuration (Right Blue Panel)
# ---------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("""
<div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px;">
    <span style="color: #000000 !important; font-weight: 600; font-size: 13px;">🟢 Live Public Deployment</span><br/>
    <a href="https://uniprojectfinal.streamlit.app" target="_blank" style="color: #000000 !important; font-weight: 500; text-decoration: none; font-size: 12px;">
        🌐 uniprojectfinal.streamlit.app ↗
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
    /* Force the Live Public Deployment badge text to black, overriding any theme/link color */
    section[data-testid="stSidebar"] a[href="https://uniprojectfinal.streamlit.app"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Source Switcher
local_file_found = detect_local_dataset()
data_source = st.sidebar.radio(
    "Data Source Engine",
    options=["synthetic", "upload"] + (["local"] if local_file_found else []),
    format_func=lambda x: {
        "synthetic": "Synthetic Imbalanced Generator (0.172% Fraud)",
        "upload": "Upload Custom CSV (creditcard.csv)",
        "local": f"Auto-detected Local file ({os.path.basename(local_file_found)})" if local_file_found else "Auto-detected Local file"
    }[x],
    help="Toggle between generated benchmark, uploaded CSV, or detected local dataset."
)

uploaded_file = None
if data_source == "upload":
    uploaded_file = st.sidebar.file_uploader("Upload Kaggle creditcard.csv", type=["csv"])
    st.sidebar.markdown("""
    <style>
        /* "Browse files" button on the Upload Kaggle creditcard.csv uploader — blue, for visibility */
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section button {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border: 1px solid #0284C7 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section button:hover {
            background-color: #0369A1 !important;
            border: 1px solid #0369A1 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] section button p {
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

dataset_size = st.sidebar.select_slider(
    "Synthetic Sample Size",
    options=[10000, 25000, 50000, 100000],
    value=50000,
    disabled=(data_source != "synthetic"),
    help="Number of synthetic transactions generated matching ULB statistical properties."
)

st.sidebar.markdown("""
<style>
    /* Synthetic Sample Size Slider — red accent only on the active selection */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] {
        --primary-color: #FF0000 !important;
    }
    /* Toggle handle (the draggable circle) — solid red so it's clearly visible */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"],
    section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {
        background-color: #FF0000 !important;
        background: #FF0000 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.8) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] div {
        background-color: #FF0000 !important;
        background: #FF0000 !important;
    }
    /* Active value label above the handle (e.g. "50000") — red text */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #FF0000 !important;
        font-weight: 800 !important;
    }
    /* Track: neutral background, no red fill behind any tick option */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div > div,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTrack"] {
        background: #CBD5E1 !important;
    }
    /* Tick bar row: no background at all — 10000 / 25000 / 100000 sit on plain sidebar background */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"] > div {
        background-color: transparent !important;
        background: none !important;
    }
    /* Tick labels: dark, readable text now that there's no red background behind them */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"] span {
        color: #1B365D !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Pipeline Settings")

model_choice = st.sidebar.radio(
    "Classifier Architecture",
    options=["logistic", "ensemble", "xgboost"],
    format_func=lambda x: {
        "logistic": "Logistic Regression (Linear Baseline)",
        "ensemble": "Random Forest Ensemble (Tree)",
        "xgboost": "XGBoost Gradient Boosting (Extreme Tree)"
    }[x],
    help="Choose linear baseline, Random Forest, or XGBoost gradient boosting."
)

imbalance_choice = st.sidebar.radio(
    "Imbalance Mitigation Strategy",
    options=["class_weight", "smote", "none"],
    format_func=lambda x: {
        "class_weight": "Class-Weighting (class_weight='balanced')",
        "smote": "SMOTE Synthetic Oversampling",
        "none": "Raw Unweighted Baseline"
    }[x],
    help="Compare SMOTE synthetic oversampling vs cost-balanced class weighting vs raw unweighted data."
)

calibration_mode = st.sidebar.selectbox(
    "Active Calibration Method",
    options=["raw", "platt", "isotonic"],
    format_func=lambda x: {
        "raw": "Uncalibrated / Raw Model Scores",
        "platt": "Platt Scaling (Sigmoid)",
        "isotonic": "Isotonic Regression (Piecewise)"
    }[x],
    index=1,
    help="Select probability calibration mapping."
)

# Load dataset based on selection, with robust handling for uploaded/local files
upload_report = None
upload_error = None

if data_source == "upload" and uploaded_file is not None:
    try:
        df_raw, upload_report = load_custom_csv(uploaded_file)
    except Exception as e:
        upload_error = str(e)
        df_raw = get_synthetic_data(n_samples=dataset_size)
elif data_source == "local" and local_file_found:
    try:
        df_raw, upload_report = load_custom_csv(local_file_found)
    except Exception as e:
        upload_error = str(e)
        df_raw = get_synthetic_data(n_samples=dataset_size)
else:
    df_raw = get_synthetic_data(n_samples=dataset_size)

# Surface the data-processing step to the user so cleaning is transparent,
# not silent — matches the proposal's "robust error handling" requirement.
if upload_error is not None:
    st.sidebar.error(
        f"⚠️ Could not use the uploaded/local file: {upload_error}\n\n"
        "Falling back to the synthetic generator so the app still runs."
    )
elif upload_report is not None:
    rows_removed = upload_report['rows_read'] - upload_report['rows_final']
    with st.sidebar.expander("✅ Data Processing Report", expanded=False):
        st.markdown(f"""
        **Rows read:** {upload_report['rows_read']:,}
        **Rows after cleaning:** {upload_report['rows_final']:,} ({rows_removed:,} removed)
        - Duplicate rows removed: {upload_report['duplicates_removed']:,}
        - Invalid / incomplete rows removed: {upload_report['invalid_rows_removed']:,}
        - Extra columns dropped: {len(upload_report['dropped_extra_columns'])}
        {(', '.join(upload_report['dropped_extra_columns']) if upload_report['dropped_extra_columns'] else '_none_')}

        **Resulting class balance:** {upload_report['genuine_count']:,} genuine, {upload_report['fraud_count']:,} fraud ({upload_report['fraud_rate_pct']}% fraud rate)
        """)

cache_data = fit_models(df_raw, imbalance_strategy=imbalance_choice)

df = cache_data['df']
X_test = cache_data['X_test']
y_test = cache_data['y_test']
y_probas = cache_data['probas'][model_choice][calibration_mode]

# ---------------------------------------------------------
# Main Application Structure (5 Main Tabs Directly Below Header)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Tab 1: Sorites Paradox Primer",
    "📊 Tab 2: Data Explorer & Imbalance",
    "⚙️ Tab 3: Elkan's Threshold Lab",
    "📈 Tab 4: Probability Calibration",
    "🔍 Tab 5: The Contested Zone"
])

# =========================================================
# TAB 1: THE SORITES PARADOX (PHILOSOPHICAL PRIMER & DEMO)
# =========================================================
with tab1:
    st.markdown("## 🏛️ The Sorites Paradox: From Sand Grains to Binary Fraud Decisions")
    
    st.markdown(r"""
    <div class="sorites-card">
        <div class="sorites-card-header">💡 Theoretical Framework: Eubulides of Miletus & Crispin Wright (1975)</div>
        <p>
        The <b>Sorites Paradox</b> (from Greek <i>soros</i>, meaning 'heap') demonstrates the fundamental failure of classical logic when applied to vague predicates.
        Consider a heap of sand: removing a single grain of sand ($n \to n-1$) does not turn a heap into a non-heap (<b>Wright's Tolerance Principle</b>). 
        However, repeated application of this premise leads to the absurd conclusion that a single grain of sand is still a heap.
        </p>
        <p>
        <b>The ML Analogy:</b> In credit card fraud detection, machine learning models assign a continuous risk probability $p(x) \in [0, 1]$. 
        Assigning a rigid binary threshold $\tau$ creates an arbitrary operational split: two transactions possessing virtually identical risk profiles 
        ($\Delta p < 0.001$) receive opposite classifications (Block vs Approve).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Visual Sand Heap Interactive Sandbox Widget
    st.markdown("### ⏳ Interactive Sandbox: Visual Sand Heap & Tolerance Principle")
    col_sand_ctrl1, col_sand_ctrl2 = st.columns([2, 1])
    with col_sand_ctrl1:
        n_grains = st.slider("Remove Sand Grains (n)", min_value=10, max_value=2000, value=1000, step=20, help="Drag slider to simulate removing grains from the heap.")
    with col_sand_ctrl2:
        st.markdown("**Tolerance Indicator (Wright 1975):**")
        if n_grains >= 800:
            st.markdown("<span class='tag-genuine'>DEFINITE HEAP (Predicate F(x) = True)</span>", unsafe_allow_html=True)
            heap_msg = "Under Wright's Tolerance Principle, removing 1 grain leaves 999 grains. It remains a heap."
        elif n_grains >= 250:
            st.markdown("<span class='tag-contested'>CONTESTED ZONE (Vague Predicate Boundary)</span>", unsafe_allow_html=True)
            heap_msg = "⚠️ Indeterminacy Region! Observers disagree whether this constitutes a heap."
        else:
            st.markdown("<span class='tag-fraud'>NON-HEAP (Predicate F(x) = False)</span>", unsafe_allow_html=True)
            heap_msg = "The predicate has flipped to Non-Heap, yet no single grain removal was responsible."

    # Plot Sand Heap
    np.random.seed(42)
    max_h = np.sqrt(n_grains) * 2.0
    x_sand = np.random.uniform(-max_h, max_h, n_grains)
    y_sand = np.array([np.random.uniform(0, max(0.1, max_h - abs(x))) for x in x_sand])
    
    color_sand = '#10B981' if n_grains >= 800 else ('#F59E0B' if n_grains >= 250 else '#EF4444')

    fig_sand = go.Figure()
    fig_sand.add_trace(go.Scatter(
        x=x_sand, y=y_sand, mode='markers',
        marker=dict(size=5, color=color_sand, opacity=0.85, line=dict(width=0.5, color='#0F172A')),
        name=f"{n_grains:,} Grains"
    ))
    fig_sand.update_layout(
        title=dict(text=f"Sand Heap Simulation: {n_grains:,} Grains — {heap_msg}", font=dict(size=14, color='#002147')),
        xaxis=dict(range=[-100, 100], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_sand, use_container_width=True)

    st.markdown("---")

    # 2. Concept Definitions: Rigid Decision Threshold vs Tolerance Margin
    st.markdown("### 📖 Core Philosophical & Machine Learning Concepts")
    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.markdown(r"""
        <div class="briefing-card">
            <h4>🎯 Rigid Decision Threshold ($\tau$)</h4>
            <p>
            A fixed numerical cutoff $\tau \in (0, 1)$ (e.g. $\tau = 0.5000$) used by binary classification systems.
            Transactions with estimated fraud risk $p(x) \ge \tau$ are classified as <b>Fraud (Block)</b>, whereas $p(x) < \tau$ are classified as <b>Genuine (Approve)</b>.
            While payment systems mandate binary actions, rigid thresholds treat probability boundaries as crisp lines, ignoring inherent statistical uncertainty.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_def2:
        st.markdown(r"""
        <div class="briefing-card">
            <h4>📏 Tolerance Margin ($\epsilon$)</h4>
            <p>
            The infinitesimal risk divergence parameter ($\epsilon > 0$) in Crispin Wright's (1975) Tolerance Principle.
            If two transactions $x_A$ and $x_B$ differ by less than $2\epsilon$ ($\Delta p = |p(x_A) - p(x_B)| = 2\epsilon$), 
            they possess virtually identical risk profiles. Applying opposite operational decisions (Block vs Approve) solely due to an arbitrary cutoff $\tau$ demonstrates boundary paradox in credit card scoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. The Epsilon Dilemma (Side-by-Side Transaction Cards with Assigned Numerical Values)
    st.markdown("### ⚖️ The Epsilon (ε) Dilemma: Boundary Disparity Demo")
    st.markdown("Adjust the tolerance parameter $\\epsilon$ to see how rigid decision thresholds force opposite business actions on two near-identical transactions:")
    
    col_eps1, col_eps2 = st.columns([2, 1])
    with col_eps1:
        epsilon_val = st.slider("Tolerance Margin (ε)", min_value=0.0001, max_value=0.0500, value=0.0010, step=0.0005, format="%.4f")
    with col_eps2:
        threshold_fixed = st.number_input("Rigid Decision Threshold (τ)", value=0.5000, step=0.0100, format="%.4f")

    prob_tx_a = threshold_fixed + epsilon_val
    prob_tx_b = threshold_fixed - epsilon_val

    col_card1, col_card2 = st.columns(2)
    with col_card1:
        st.markdown(f"""
        <div class="tx-card-a">
            <h4>💳 Transaction A (ID: TX-89402)</h4>
            <ul>
                <li><b>Transaction Amount:</b> $842.50</li>
                <li><b>Merchant Category:</b> Online Electronics Retail</li>
                <li><b>Geolocation Distance Delta (V17):</b> 14.2 km</li>
                <li><b>Transaction Velocity Anomaly (V14):</b> 3.81</li>
                <li><b>Calibrated Risk Score p(X_A):</b> <span style="font-weight: 700;">{prob_tx_a:.4f}</span></li>
                <li><b>Distance to Threshold τ:</b> +{epsilon_val:.4f}</li>
                <li><b>Decision Action:</b> <span class="tag-fraud">DECLINED & BLOCKED (FRAUD)</span></li>
                <li><b>Customer Impact:</b> Card frozen instantly at checkout; SMS OTP verification triggered.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_card2:
        st.markdown(f"""
        <div class="tx-card-b">
            <h4>💳 Transaction B (ID: TX-89403)</h4>
            <ul>
                <li><b>Transaction Amount:</b> $841.90</li>
                <li><b>Merchant Category:</b> Online Electronics Retail</li>
                <li><b>Geolocation Distance Delta (V17):</b> 14.1 km</li>
                <li><b>Transaction Velocity Anomaly (V14):</b> 3.79</li>
                <li><b>Calibrated Risk Score p(X_B):</b> <span style="font-weight: 700;">{prob_tx_b:.4f}</span></li>
                <li><b>Distance to Threshold τ:</b> -{epsilon_val:.4f}</li>
                <li><b>Decision Action:</b> <span class="tag-genuine">APPROVED & PASSED (GENUINE)</span></li>
                <li><b>Customer Impact:</b> Payment approved seamlessly without any friction.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"💡 **Epsilon Paradox Note:** Total risk score divergence is $\\Delta p = {2*epsilon_val:.4f}$. Treating Transaction A as Fraud and Transaction B as Genuine is an operational artifact enforced by binary cutoffs, not a qualitative difference in transaction risk.")

    st.markdown("---")

    # 4. Formal Mapping Table
    st.markdown("### 🔗 Formal Mapping Table: Analytical Philosophy ⟷ Machine Learning")
    mapping_data = [
        {"Philosophical Concept (Crispin Wright, 1975)": "Heap of Sand (Sorites Predicate F)", "Machine Learning Counterpart": "Fraudulent Transaction Cohort (Class 1)"},
        {"Philosophical Concept (Crispin Wright, 1975)": "Single Grain of Sand (ε)", "Machine Learning Counterpart": "Infinitesimal Risk Probability Delta (Δp = 0.001)"},
        {"Philosophical Concept (Crispin Wright, 1975)": "Tolerance Principle: F(x) ∧ x ≈ y ⇒ F(y)", "Machine Learning Counterpart": "Smoothness Assumption: p(xA) ≈ p(xB) ⇒ Action(xA) = Action(xB)"},
        {"Philosophical Concept (Crispin Wright, 1975)": "Crisp Boundary Cutoff", "Machine Learning Counterpart": "Rigid Binary Decision Threshold τ (e.g. τ = 0.50)"},
        {"Philosophical Concept (Crispin Wright, 1975)": "Zone of Indeterminacy", "Machine Learning Counterpart": "The Contested Zone [τ_min, τ_max] between Stakeholder Costs"}
    ]
    st.table(pd.DataFrame(mapping_data))

    with st.expander("📚 Academic Underpinning & Citations (Wright, 1975; Eubulides)"):
        st.markdown(r"""
        **Academic Reference:**
        - Wright, C. (1975). *On the Coherence of Vague Predicates*. Synthese, 30(3/4), 325-365.
        - Eubulides of Miletus (4th Century BCE). *The Megarian School Paradoxes*.
        - Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning*. IJCAI-01.
        
        **Summary:** Classical first-order logic relies on crisp sets ($x \in S$ or $x \notin S$). Real-world risk management requires partitioning a continuous space $[0,1]$ into discrete business actions. Threshold selection is inherently a value judgement balancing competing financial utilities.
        """)

# =========================================================
# TAB 2: DATA EXPLORER & IMBALANCE ANALYSIS
# =========================================================
with tab2:
    st.markdown("## 📊 Credit Card Fraud Dataset Analysis & Model Decision Visualizer")
    st.markdown("Replicating the statistical structure of Worldline & Université Libre de Bruxelles dataset (Dal Pozzolo et al., 2015).")

    # Executive Summary Metrics
    total_tx = len(df)
    fraud_tx = int(df['Class'].sum())
    genuine_tx = total_tx - fraud_tx
    fraud_pct = (fraud_tx / total_tx) * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Transactions", f"{total_tx:,}")
    col_m2.metric("Genuine (Class 0)", f"{genuine_tx:,}")
    col_m3.metric("Fraudulent (Class 1)", f"{fraud_tx:,}")
    col_m4.metric("Fraud Imbalance Rate", f"{fraud_pct:.3f}%")

    st.markdown("---")

    # ---------------------------------------------------------
    # MODULE 1: HOW MODELS CLASSIFY DATA — 2D DECISION CONTOUR SURFACE
    # ---------------------------------------------------------
    st.markdown("### 🧠 How Models See & Classify Data: 2D Decision Boundary Contour")
    st.markdown(r"""
    The chart below visualizes the **2D Decision Surface** created by the active model (""" + model_choice.upper() + r""").
    The background heatmap shows the model's predicted probability of fraud $p(x) \in [0, 1]$ across feature combinations:
    - **Linear Models (Logistic Regression):** Construct a straight separating hyper-plane ($w_1 x_1 + w_2 x_2 + b = 0$).
    - **Tree Ensembles (Random Forest):** Construct non-linear, step-wise orthogonal decision partitions wrapping around fraud clusters.
    """)

    # Selectbox Controls placed ABOVE the Decision Boundary Plot (Full Width Container)
    st.markdown("#### 🎛️ Select Feature Axes for Decision Boundary")
    col_ax1, col_ax2 = st.columns(2)
    with col_ax1:
        bound_x = st.selectbox(
            "X-Axis Feature Component",
            options=[f"V{i}" for i in range(1, 29)],
            index=13,
            format_func=lambda x: f"{PCA_FEATURE_DESCRIPTIONS.get(x, x)} ({x})"
        )
    with col_ax2:
        bound_y = st.selectbox(
            "Y-Axis Feature Component",
            options=[f"V{i}" for i in range(1, 29)],
            index=16,
            format_func=lambda x: f"{PCA_FEATURE_DESCRIPTIONS.get(x, x)} ({x})"
        )

    st.info(f"💡 **Model Decision Mechanism:** Notice how {model_choice.upper()} evaluates {PCA_FEATURE_DESCRIPTIONS.get(bound_x, bound_x)} vs {PCA_FEATURE_DESCRIPTIONS.get(bound_y, bound_y)}. Green dots represent Genuine transactions, Red diamonds represent Fraud.")

    # Stratified Test Feature Bounds
    test_sub_all = df.iloc[len(df)-len(y_test):]
    x_min, x_max = test_sub_all[bound_x].min() - 0.5, test_sub_all[bound_x].max() + 0.5
    y_min, y_max = test_sub_all[bound_y].min() - 0.5, test_sub_all[bound_y].max() + 0.5
    
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 45), np.linspace(y_min, y_max, 45))
    
    # Build grid DataFrame with median values for other features
    grid_df = pd.DataFrame(np.tile(X_test.median().values, (len(gx.ravel()), 1)), columns=X_test.columns)
    grid_df[bound_x] = gx.ravel()
    grid_df[bound_y] = gy.ravel()

    active_pipeline = cache_data['pipelines'][model_choice]
    raw_clf = active_pipeline.raw_model
    grid_probs = raw_clf.predict_proba(grid_df)[:, 1].reshape(gx.shape)

    fig_boundary = go.Figure()
    
    # 1. Background Soft Risk Contour Surface (Soft opacity so scatter points pop out)
    fig_boundary.add_trace(go.Contour(
        z=grid_probs,
        x=np.linspace(x_min, x_max, 45),
        y=np.linspace(y_min, y_max, 45),
        colorscale='YlOrRd',
        opacity=0.35,
        contours=dict(showlabels=True, labelfont=dict(size=10, color='#334155')),
        colorbar=dict(title='Fraud Risk p(x)', len=0.85, yanchor="middle", y=0.5)
    ))

    # 2. Stratified Points Selection (Guarantees Fraud Red Points Are ALWAYS Visible)
    # Pull ALL Fraud points in test set (or full dataset if test set has < 15 fraud points)
    fr_pts = test_sub_all[test_sub_all['Class'] == 1]
    if len(fr_pts) < 15:
        fr_pts = df[df['Class'] == 1]

    # Sample genuine points for readable scatter overlay
    gen_pts = test_sub_all[test_sub_all['Class'] == 0].sample(min(1200, len(test_sub_all[test_sub_all['Class'] == 0])), random_state=42)

    # Hover Text Format
    hover_text_gen = [
        f"🟢 <b>GENUINE TRANSACTION</b><br>{bound_x}: {x:.2f}<br>{bound_y}: {y:.2f}<br>Amount: ${amt:.2f}<br>Status: Class 0 (Genuine)"
        for x, y, amt in zip(gen_pts[bound_x], gen_pts[bound_y], gen_pts['Amount'])
    ]

    hover_text_fr = [
        f"🔴 <b>FRAUDULENT TRANSACTION</b><br>{bound_x}: {x:.2f}<br>{bound_y}: {y:.2f}<br>Amount: ${amt:.2f}<br>Status: Class 1 (Fraud)"
        for x, y, amt in zip(fr_pts[bound_x], fr_pts[bound_y], fr_pts['Amount'])
    ]

    # Add Genuine Scatter Points (Green)
    fig_boundary.add_trace(go.Scatter(
        x=gen_pts[bound_x], y=gen_pts[bound_y], mode='markers',
        name='Genuine Transactions (Class 0)',
        marker=dict(color='#059669', opacity=0.55, size=5),
        text=hover_text_gen, hoverinfo='text'
    ))

    # Add Fraud Scatter Points (Prominent Red Diamonds with Black Border)
    fig_boundary.add_trace(go.Scatter(
        x=fr_pts[bound_x], y=fr_pts[bound_y], mode='markers',
        name=f'Fraud Transactions (Class 1, n={len(fr_pts)})',
        marker=dict(color='#FF0033', opacity=1.0, size=11, symbol='diamond', line=dict(color='#000000', width=1.8)),
        text=hover_text_fr, hoverinfo='text'
    ))

    fig_boundary.update_layout(
        title=dict(
            text=f"2D Decision Surface: {bound_x} vs {bound_y} ({model_choice.upper()} Classifier)",
            font=dict(size=15, color='#002147'),
            x=0.0, xanchor='left'
        ),
        xaxis_title=f"{bound_x} ({PCA_FEATURE_DESCRIPTIONS.get(bound_x, '')})",
        yaxis_title=f"{bound_y} ({PCA_FEATURE_DESCRIPTIONS.get(bound_y, '')})",
        template="plotly_white",
        height=480,
        margin=dict(l=60, r=100, t=95, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color='#0F172A')
        ),
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_boundary, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # MODULE 2: FEATURE SEPARATION & MODEL FEATURE IMPORTANCE (STACKED VERTICALLY BELOW)
    # ---------------------------------------------------------
    st.markdown("### 🔍 Feature Separation & Model Feature Importance Inspector")
    st.markdown("Explore how raw features differ between Genuine and Fraudulent transactions, and see **how machine learning models weigh feature importance** to separate class boundaries:")

    # Section 2A: Feature Density Distribution Overlay
    st.markdown("#### 1. Feature Distribution Overlay (Genuine vs Fraud)")
    selected_feat = st.selectbox(
        "Select Feature Component for KDE Density Comparison",
        options=['V14', 'V17', 'V12', 'V11', 'V4', 'V10', 'V16', 'Amount'],
        index=0,
        format_func=lambda x: f"{x} — {PCA_FEATURE_DESCRIPTIONS.get(x, x)}"
    )
    
    fig_dist = go.Figure()
    gen_vals = df[df['Class'] == 0][selected_feat]
    fr_vals = df[df['Class'] == 1][selected_feat]
    
    fig_dist.add_trace(go.Histogram(
        x=gen_vals, name='Genuine (Class 0)', opacity=0.65, marker_color='#10B981',
        nbinsx=40, histnorm='probability density'
    ))
    fig_dist.add_trace(go.Histogram(
        x=fr_vals, name='Fraud (Class 1)', opacity=0.75, marker_color='#EF4444',
        nbinsx=40, histnorm='probability density'
    ))
    
    fig_dist.update_layout(
        title=f"KDE Overlay: {selected_feat} ({PCA_FEATURE_DESCRIPTIONS.get(selected_feat, '')})",
        xaxis_title=selected_feat,
        yaxis_title="Probability Density",
        barmode='overlay',
        template="plotly_white",
        height=360,
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Statistical Cohen's d separation metric
    mean_gen = np.mean(gen_vals)
    mean_fr = np.mean(fr_vals)
    std_pooled = np.sqrt((np.var(gen_vals) + np.var(fr_vals)) / 2.0)
    cohen_d = abs(mean_gen - mean_fr) / std_pooled if std_pooled > 0 else 0.0
    st.caption(f"📊 **Separation Distance (Cohen's d):** `{cohen_d:.2f}` — " + ("Strong predictive separation signal!" if cohen_d > 1.0 else "Moderate separation signal."))

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2B: Model Feature Importance (Placed Below with Sky Blue Styling)
    st.markdown(f"#### 2. Model Feature Importance ({model_choice.upper()} Classifier)")
    st.markdown("Relative predictive feature weights extracted from the trained model pipeline:")
    
    feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    if hasattr(raw_clf, 'feature_importances_'):
        importances = raw_clf.feature_importances_
    elif hasattr(raw_clf, 'coef_'):
        importances = np.abs(raw_clf.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    fi_df = pd.DataFrame({
        'Feature Code': feature_names,
        'Domain Feature': [f"{x} ({PCA_FEATURE_DESCRIPTIONS.get(x, x)})" for x in feature_names],
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)

    fig_fi = px.bar(
        fi_df, x='Importance', y='Domain Feature', orientation='h',
        color='Importance', color_continuous_scale=['#BAE6FD', '#0284C7', '#0369A1'],
        title=f"Top 10 Feature Importances ({model_choice.upper()} Model)"
    )
    fig_fi.update_layout(
        template="plotly_white",
        paper_bgcolor="#F0F9FF",
        plot_bgcolor="#E0F2FE",
        height=400,
        coloraxis_showscale=False,
        font=dict(color='#0369A1', size=13),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # MODULE 3: FILTERED DATASET EXPLORER WITH FEATURE MAPPING
    # ---------------------------------------------------------
    st.markdown("### 📋 Interactive Dataset Explorer with Feature Mapping")
    
    # Column renaming map
    column_rename_map = {
        'S.No': 'S.No', 'Time': 'Time (sec)', 'Amount': 'Amount ($)',
        'V1': 'Cardholder Age/Tenure (V1)', 'V2': 'Location Disparity (V2)', 'V3': 'Merchant Risk Score (V3)',
        'V4': 'Transaction Velocity (V4)', 'V5': 'CNP Verification Flag (V5)', 'V6': 'Device IP Anomaly (V6)',
        'V7': 'Spending Divergence (V7)', 'V8': 'Cross-Border Currency (V8)', 'V9': 'Billing Match Rating (V9)',
        'V10': 'High-Risk MCC Score (V10)', 'V11': 'Failed Auth Counter (V11)', 'V12': 'Behavioral Biometrics (V12)',
        'V13': 'POS Tech Signature (V13)', 'V14': 'Transaction Velocity Anomaly (V14)', 'V15': 'Mobile Wallet Risk (V15)',
        'V16': 'Issuer Country Risk (V16)', 'V17': 'Geolocation Distance Delta (V17)', 'V18': 'Subscription Deviation (V18)',
        'V19': 'Time Since High-Value Tx (V19)', 'V20': 'Balance Ratio (V20)', 'V21': 'IP Subnet Score (V21)',
        'V22': 'Support Inquiry Spike (V22)', 'V23': 'Virtual Card Flag (V23)', 'V24': 'Off-Hours Indicator (V24)',
        'V25': 'Session Fingerprint (V25)', 'V26': 'Peer Expense StdDev (V26)', 'V27': 'Multi-Card Binding (V27)',
        'V28': 'Behavioral Entropy (V28)', 'Class': 'Transaction Status'
    }

    # Search & Filter Controls
    col_flt1, col_flt2, col_flt3 = st.columns([1, 1, 1])
    with col_flt1:
        class_filter = st.selectbox("Filter by Class", options=["All", "0 (Genuine Only)", "1 (Fraud Only)"])
    with col_flt2:
        view_mode = st.radio(
            "Table Column View",
            options=["relevant", "all"],
            format_func=lambda x: "Key Relevant Fraud Signals (Recommended)" if x == "relevant" else "All 28 Feature Components",
            horizontal=True
        )
    with col_flt3:
        max_amt = float(df['Amount'].max())
        amt_range = st.slider("Filter Amount Range ($)", 0.0, min(2000.0, max_amt), (0.0, min(2000.0, max_amt)))

    # Filter Data
    filtered_df = df.copy()
    if class_filter == "0 (Genuine Only)":
        filtered_df = filtered_df[filtered_df['Class'] == 0]
    elif class_filter == "1 (Fraud Only)":
        filtered_df = filtered_df[filtered_df['Class'] == 1]
    
    filtered_df = filtered_df[(filtered_df['Amount'] >= amt_range[0]) & (filtered_df['Amount'] <= amt_range[1])]

    show_df = filtered_df.head(100).copy()
    show_df.insert(0, 'S.No', range(1, len(show_df) + 1))
    show_df['Class'] = show_df['Class'].map({0: '0 (Genuine)', 1: '1 (Fraud)'})
    show_df_renamed = show_df.rename(columns=column_rename_map)

    if view_mode == "relevant":
        relevant_cols = [
            'S.No', 'Time (sec)', 'Amount ($)',
            'Transaction Velocity Anomaly (V14)', 'Geolocation Distance Delta (V17)',
            'Behavioral Biometrics (V12)', 'Failed Auth Counter (V11)',
            'Transaction Velocity (V4)', 'Transaction Status'
        ]
        disp_df = show_df_renamed[relevant_cols]
    else:
        disp_df = show_df_renamed

    st.dataframe(
        disp_df.style.format({
            'Amount ($)': '${:.2f}',
            'Time (sec)': '{:.0f}'
        }),
        height=350,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("📖 View Complete PCA Feature Glossary (V1 – V28 Descriptions)"):
        feat_df = pd.DataFrame(list(PCA_FEATURE_DESCRIPTIONS.items()), columns=['PCA Component Code', 'Original Domain Feature Proxy'])
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    with st.expander("📚 Academic Reference: Class Imbalance & Fraud Benchmarking (Dal Pozzolo et al., 2015)"):
        st.markdown(r"""
        **Citation:**
        - Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Fraud Detection*. IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 159-166.
        
        **Key Takeaways:**
        - Real-world credit card datasets exhibit extreme class imbalance ($\sim 0.172\%$ positive rate).
        - Accuracy is a misleading metric; models must be evaluated on cost-sensitive loss and Precision-Recall Area Under Curve (PR-AUC).
        """)

# =========================================================
# TAB 3: THRESHOLD LAB (ELKAN'S COST-SENSITIVE ENGINE)
# =========================================================
with tab3:
    st.markdown("## ⚙️ Threshold Lab: Elkan's (2001) Cost-Sensitive Engine")
    st.markdown("Deriving the optimal decision boundary $\\tau^*$ from economic business costs rather than default $\\tau = 0.50$:")
    
    st.latex(r"\tau^* = \frac{C(FP)}{C(FP) + C(FN)}")

    # Core Concepts & Term Explanations Briefing Card
    st.markdown(r"""
    <div class="briefing-card">
        <h4>💡 Essential Economic Terms & Concepts</h4>
        <ul>
            <li><b>C(FP) — Cost of False Positive:</b> Financial cost incurred when a genuine transaction is falsely flagged as fraud (customer friction, SMS OTP verification fee, manual agent review, brand reputation loss).</li>
            <li><b>C(FN) — Cost of False Negative:</b> Direct monetary loss incurred when a fraudulent transaction is missed (chargeback loss, stolen transaction amount paid to fraudster).</li>
            <li><b>Optimal Threshold (τ*):</b> The mathematically optimal decision boundary calculated by Elkan (2001) that minimizes total monetary loss.</li>
            <li><b>Confusion Matrix Losses:</b> True Negatives (TN) cost £0; False Positives (FP) cost $FP \times C(FP)$; False Negatives (FN) cost $FN \times C(FN)$.</li>
            <li><b>Cost-Curve Minimization:</b> Plots total financial loss (£) across all thresholds $\tau \in [0, 1]$, demonstrating that default $\tau = 0.50$ causes severe financial loss compared to optimal $\tau^*$.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Cost Input Sliders
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        c_fp = st.slider("Cost of False Positive C(FP) (£)", min_value=1.0, max_value=500.0, value=25.0, step=1.0, help="Cost of customer friction, SMS OTP, and manual review.")
    with col_s2:
        c_fn = st.slider("Cost of False Negative C(FN) (£)", min_value=50.0, max_value=10000.0, value=1000.0, step=50.0, help="Direct monetary loss from undetected fraud.")

    tau_star = compute_elkan_threshold(c_fp, c_fn)
    cost_info = evaluate_threshold_cost(y_test.values, y_probas, tau_star, c_fp, c_fn)

    # Executive Output Metrics
    st.markdown("---")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Optimal Threshold (τ*)", f"{tau_star:.4f}")
    b2.metric("Total Expected Loss", f"£{cost_info['total_cost']:,.2f}")
    b3.metric("False Positives (FP Cost)", f"{cost_info['FP']:,} (£{cost_info['FP']*c_fp:,.0f})")
    b4.metric("False Negatives (FN Cost)", f"{cost_info['FN']:,} (£{cost_info['FN']*c_fn:,.0f})")
    b5.metric("F1-Score at τ*", f"{cost_info['f1_score']:.3f}")

    st.markdown("---")

    # Stacked Visualizations (Neat and Un-compact)
    st.markdown("### 🔢 1. Dynamic Confusion Matrix with Financial Loss Breakdown")
    st.markdown("Shows actual vs predicted transaction counts at Elkan's optimal threshold $\\tau^* = {:.4f}$".format(tau_star))
    
    cm_data = [[cost_info['TN'], cost_info['FP']], [cost_info['FN'], cost_info['TP']]]
    fig_cm = px.imshow(
        cm_data,
        x=['Pred Genuine (Pass)', 'Pred Fraud (Block)'],
        y=['Actual Genuine', 'Actual Fraud'],
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix at τ* = {tau_star:.4f}"
    )
    fig_cm.update_layout(template="plotly_white", height=380, coloraxis_showscale=False, font=dict(color='#0F172A'))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📉 2. Continuous Cost-Curve Minimization Plot")
    st.markdown(r"Demonstrates total monetary loss (£) across the entire threshold spectrum $\tau \in [0, 1]$. The golden star marks $\tau^*$:")
    
    curve_df = run_threshold_cost_curve(y_test.values, y_probas, c_fp, c_fn)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df['Threshold'], y=curve_df['Total_Cost'], mode='lines', name='Total Financial Loss (£)', line=dict(color='#EF4444', width=3)))
    fig_curve.add_trace(go.Scatter(x=[tau_star], y=[cost_info['total_cost']], mode='markers+text', name=f'Global Minimum τ*={tau_star:.4f}', marker=dict(color='#D97706', size=14, symbol='star'), text=[f"τ*={tau_star:.4f}"], textposition="top center"))
    
    fig_curve.update_layout(
        title=f"Expected Business Cost vs Decision Threshold τ",
        xaxis_title="Decision Threshold τ",
        yaxis_title="Total Expected Loss (£)",
        template="plotly_white",
        height=400,
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    with st.expander("📚 Academic Underpinning: Elkan's (2001) Cost-Sensitive Learning Theorem"):
        st.markdown(r"""
        **Academic Citation:**
        - Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning*. Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI-01), 973-978.
        
        **Mathematical Derivation:**
        Let $p(x) = P(y=1|x)$ be the calibrated probability of fraud. The expected cost of classifying transaction $x$ as positive (fraud) vs negative (genuine) is:
        $$\mathbb{E}[\text{Cost}|\text{Predict Fraud}] = (1 - p(x)) \cdot C(FP)$$
        $$\mathbb{E}[\text{Cost}|\text{Predict Genuine}] = p(x) \cdot C(FN)$$
        Predicting fraud is optimal when $\mathbb{E}[\text{Cost}|\text{Predict Fraud}] \le \mathbb{E}[\text{Cost}|\text{Predict Genuine}]$, yielding:
        $$\tau^* = \frac{C(FP)}{C(FP) + C(FN)}$$
        """)

# =========================================================
# TAB 4: PROBABILITY CALIBRATION & RELIABILITY ANALYSIS
# =========================================================
with tab4:
    st.markdown("## 📈 Probability Calibration & Reliability Analysis")
    st.markdown("Transforming arbitrary machine learning scores into true, empirical posterior fraud probabilities $P(Y=1|X=x)$.")

    # Briefing & Term Definitions Card
    st.markdown(r"""
    <div class="briefing-card">
        <h4>💡 Essential Probability Calibration Concepts</h4>
        <ul>
            <li><b>Probability Calibration:</b> Adjusting model output scores so that a predicted probability of 0.10 means exactly 10 out of 100 transactions are truly fraudulent ($P(Y=1|p(x)=p) \approx p$).</li>
            <li><b>Platt Scaling (Sigmoid):</b> Fits a parametric logistic curve $p(x) = \frac{1}{1 + \exp(A \cdot s(x) + B)}$ to map raw model scores to calibrated probabilities.</li>
            <li><b>Isotonic Regression:</b> Fits a non-parametric, piecewise monotonic step function. Highly effective for tree ensembles.</li>
            <li><b>Brier Score Loss:</b> Mean squared error between predicted probabilities $p_i$ and actual outcomes $y_i \in \{0, 1\}$. Lower is better ($\text{Brier} = \frac{1}{N}\sum (p_i - y_i)^2$).</li>
            <li><b>Log-Loss (Cross-Entropy):</b> Evaluates probability logarithmic accuracy. Lower is better.</li>
            <li><b>Precision-Recall Area Under Curve (PR-AUC):</b> Evaluates classification performance under severe class imbalance ($0.172\%$). Higher is better.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    metrics_all = cache_data['metrics'][model_choice]
    cal_colors = {'raw': '#EF4444', 'platt': '#2563EB', 'isotonic': '#10B981'}
    cal_names = {'raw': 'Raw Uncalibrated', 'platt': 'Platt Scaling (Sigmoid)', 'isotonic': 'Isotonic Regression'}

    # Linear Graph 1: Reliability Diagram
    st.markdown("### 🎯 1. Reliability Diagram (Calibration Curves)")
    st.markdown("Plots predicted fraud probability $p(x)$ against observed empirical fraud fraction. The diagonal dashed line ($y=x$) represents **perfect calibration**:")
    
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration (y=x)', line=dict(color='#64748B', dash='dash')))
    
    for mode in ['raw', 'platt', 'isotonic']:
        m = metrics_all[mode]
        fig_rel.add_trace(go.Scatter(
            x=m['reliability_pred'], y=m['reliability_true'], mode='lines+markers',
            name=f"{cal_names[mode]} (Brier={m['brier_score']:.5f})",
            line=dict(color=cal_colors[mode], width=2.5)
        ))

    fig_rel.update_layout(
        title=f"Calibration Reliability Curves ({model_choice.upper()} Model)",
        xaxis_title="Mean Predicted Fraud Probability p(x)",
        yaxis_title="Empirical Observed Fraud Fraction",
        template="plotly_white",
        height=420,
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Linear Graph 2: Precision-Recall Curve
    st.markdown("### 📊 2. Precision-Recall Curves (Imbalanced Evaluation)")
    st.markdown(r"Measures precision (Positive Predictive Value) vs recall (Sensitivity) across all decision thresholds under severe $0.172\%$ fraud imbalance:")
    
    fig_pr = go.Figure()
    for mode in ['raw', 'platt', 'isotonic']:
        m = metrics_all[mode]
        fig_pr.add_trace(go.Scatter(
            x=m['recall'], y=m['precision'], mode='lines',
            name=f"{cal_names[mode]} (PR-AUC={m['pr_auc']:.4f})",
            line=dict(color=cal_colors[mode], width=2.5)
        ))
    fig_pr.update_layout(
        title=f"Precision-Recall Curves ({model_choice.upper()} Model)",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        template="plotly_white",
        height=420,
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quantitative Metrics Table
    st.markdown("### 📋 3. Quantitative Probability Calibration Metrics Comparison")
    cal_table_data = []
    for mode in ['raw', 'platt', 'isotonic']:
        m = metrics_all[mode]
        cal_table_data.append({
            'Calibration Technique': cal_names[mode],
            'Brier Score Loss (Lower is Better)': f"{m['brier_score']:.6f}",
            'Log-Loss / Cross-Entropy (Lower is Better)': f"{m['log_loss']:.5f}",
            'PR-AUC / Average Precision (Higher is Better)': f"{m['pr_auc']:.4f}"
        })
    st.table(pd.DataFrame(cal_table_data))

    with st.expander("📚 Academic Underpinning: Probability Calibration (Niculescu-Mizil & Caruana 2005; Guo et al. 2017)"):
        st.markdown(r"""
        **Academic Citations:**
        - Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. ICML '05, 625-632.
        - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML '17.
        - Davis, J., & Goadrich, M. (2006). *The relationship between Precision-Recall and ROC curves*. ICML '06.
        """)

# =========================================================
# TAB 5: THE CONTESTED ZONE & SENSITIVITY SWEEP
# =========================================================
with tab5:
    st.markdown("## 🔍 The Contested Zone & Stakeholder Sensitivity Sweep")
    st.markdown("Quantifying operational decision indeterminacy across competing institutional stakeholders.")

    # Core Concepts & Term Explanations Briefing Card
    st.markdown(r"""
    <div class="briefing-card">
        <h4>💡 Essential Contested Zone Concepts & Terms</h4>
        <ul>
            <li><b>Lower Threshold (τ_min):</b> The decision cutoff preferred by customer-friction-averse stakeholders (e.g. Customer Support / Sales) who assign a high penalty to false alarms.</li>
            <li><b>Upper Threshold (τ_max):</b> The decision cutoff preferred by fraud-loss-averse stakeholders (e.g. Risk Management / Audit) who assign a high penalty to missed fraud.</li>
            <li><b>The Contested Zone [τ_min, τ_max]:</b> The Sorites indeterminacy interval $p(x) \in [\tau_{\text{min}}, \tau_{\text{max}}]$ where transaction labels flip depending on stakeholder cost preference.</li>
            <li><b>Financial Volume at Stake:</b> Total monetary sum of transactions trapped within the contested risk zone.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col_cz_in1, col_cz_in2 = st.columns(2)
    with col_cz_in1:
        c_fp_stake_min = st.number_input("Risk-Averse Stakeholder C(FP) (£)", value=10.0, step=5.0)
    with col_cz_in2:
        c_fp_stake_max = st.number_input("Customer-Friction-Averse Stakeholder C(FP) (£)", value=100.0, step=10.0)

    c_fn_base = 1000.0
    cz_data = analyze_contested_zone(df.iloc[len(df)-len(y_test):], y_probas, (c_fp_stake_min, c_fp_stake_max), c_fn_base)

    st.markdown("---")
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Lower Threshold τ_min", f"{cz_data['tau_min']:.4f}")
    cm2.metric("Upper Threshold τ_max", f"{cz_data['tau_max']:.4f}")
    cm3.metric("Contested Transactions", f"{cz_data['contested_count']:,} ({cz_data['proportion']*100:.2f}%)")
    cm4.metric("Financial Volume at Stake", f"£{cz_data['contested_amount_total']:,.2f}")

    st.markdown("---")

    # Contested Zone Interactive Scatter Plot
    st.markdown("### 🌌 Interactive Contested Zone Scatter Plot")
    
    test_sub = df.iloc[len(df)-len(y_test):].copy()
    test_sub['proba'] = y_probas
    test_sub['contested'] = (test_sub['proba'] >= cz_data['tau_min']) & (test_sub['proba'] < cz_data['tau_max'])
    
    # Sample for performance
    scatter_df = test_sub.sample(min(3000, len(test_sub)), random_state=42).copy()
    
    fig_cz = go.Figure()

    # Genuine non-contested
    gen_mask = (~scatter_df['contested']) & (scatter_df['proba'] < cz_data['tau_min'])
    fig_cz.add_trace(go.Scatter(
        x=scatter_df[gen_mask]['Amount'], y=scatter_df[gen_mask]['proba'],
        mode='markers', name='Approved Genuine',
        marker=dict(color='#10B981', opacity=0.6, size=5)
    ))

    # Fraud non-contested
    fr_mask = (~scatter_df['contested']) & (scatter_df['proba'] >= cz_data['tau_max'])
    fig_cz.add_trace(go.Scatter(
        x=scatter_df[fr_mask]['Amount'], y=scatter_df[fr_mask]['proba'],
        mode='markers', name='Blocked Fraud',
        marker=dict(color='#EF4444', opacity=0.7, size=6)
    ))

    # Contested Zone Amber
    cz_mask = scatter_df['contested']
    fig_cz.add_trace(go.Scatter(
        x=scatter_df[cz_mask]['Amount'], y=scatter_df[cz_mask]['proba'],
        mode='markers', name='CONTESTED ZONE (Sorites Region)',
        marker=dict(color='#F59E0B', opacity=0.9, size=9, line=dict(color='#0F172A', width=1)),
        text=[f"Amount: £{amt:.2f}<br>Fraud Risk p: {p:.4f}<br>Status: CONTESTED" for amt, p in zip(scatter_df[cz_mask]['Amount'], scatter_df[cz_mask]['proba'])],
        hoverinfo='text'
    ))

    # Add threshold horizontal shading
    fig_cz.add_hrect(
        y0=cz_data['tau_min'], y1=cz_data['tau_max'],
        fillcolor="#F59E0B", opacity=0.15, line_width=1, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Contested Zone [{cz_data['tau_min']:.4f}, {cz_data['tau_max']:.4f}]", annotation_position="top left"
    )

    fig_cz.update_layout(
        title="Transaction Amount vs Calibrated Fraud Risk Score p(x)",
        xaxis_title="Transaction Amount (£)",
        yaxis_title="Calibrated Fraud Probability p(x)",
        template="plotly_white",
        height=450,
        font=dict(color='#0F172A')
    )
    st.plotly_chart(fig_cz, use_container_width=True)

    st.markdown("---")

    # CSV Data Export
    if not cz_data['contested_df'].empty:
        st.markdown("### 📥 Download Contested Zone Transactions Export")
        st.markdown(f"Download the **{len(cz_data['contested_df']):,}** transactions currently trapped in the Sorites indeterminacy zone for audit review:")
        
        csv_bytes = cz_data['contested_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Export contested_transactions_export.csv",
            data=csv_bytes,
            file_name="contested_transactions_export.csv",
            mime="text/csv"
        )

    with st.expander("📚 Academic Summary: Decision Indeterminacy & Stakeholder Conflict"):
        st.markdown("""
        **Academic Citation:**
        - Wright, C. (1975). *On the Coherence of Vague Predicates*. Synthese, 30(3/4), 325-365.
        
        **Conclusion for M.Sc. Project COMP702:**
        The Contested Zone quantitatively isolates the financial volume and transaction cohort subject to stakeholder disagreement.
        By combining probability calibration with cost-sensitive threshold selection, financial institutions transform arbitrary binary cutoffs into transparent, economically audited decision boundaries.
        """)
