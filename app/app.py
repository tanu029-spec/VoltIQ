"""
VoltIQ: EV Battery Health Intelligence Dashboard
Streamlit Application Entry Point
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.predict import SOHPredictor
from src.feature_config import FEATURE_SET_TIER_1_SENSOR_ONLY

try:
    from app.components.ui_cards import load_custom_css, render_kpi_card, render_disclaimer
    from app.components.charts import (
        plot_fleet_soh_comparison,
        plot_battery_degradation_dual,
        plot_battery_sensor_dynamics,
        plot_ica_curves,
        plot_oof_true_vs_predicted,
        plot_feature_importance_bar,
        plot_soh_gauge,
    )
except (ModuleNotFoundError, ImportError):
    from components.ui_cards import load_custom_css, render_kpi_card, render_disclaimer
    from components.charts import (
        plot_fleet_soh_comparison,
        plot_battery_degradation_dual,
        plot_battery_sensor_dynamics,
        plot_ica_curves,
        plot_oof_true_vs_predicted,
        plot_feature_importance_bar,
        plot_soh_gauge,
    )

st.set_page_config(
    page_title="VoltIQ | EV Battery Health Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# Data & Model Caching
# ------------------------------------------------------------------------------
@st.cache_data
def load_dashboard_datasets():
    """Load core processed features and metrics datasets."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"

    features_df = pd.read_csv(data_dir / "battery_health_features.csv")
    ica_df = pd.read_csv(data_dir / "battery_ica_features.csv")

    model_comp_path = metrics_dir / "model_comparison.csv"
    model_comp_df = pd.read_csv(model_comp_path) if model_comp_path.is_file() else pd.DataFrame()

    fold_metrics_path = metrics_dir / "fold_metrics.csv"
    fold_metrics_df = pd.read_csv(fold_metrics_path) if fold_metrics_path.is_file() else pd.DataFrame()

    oof_path = metrics_dir / "oof_predictions.csv"
    oof_df = pd.read_csv(oof_path) if oof_path.is_file() else pd.DataFrame()

    feat_imp_path = metrics_dir / "feature_importance.csv"
    feat_imp_df = pd.read_csv(feat_imp_path) if feat_imp_path.is_file() else pd.DataFrame()

    return features_df, ica_df, model_comp_df, fold_metrics_df, oof_df, feat_imp_df


@st.cache_resource
def load_tier1_predictor():
    """Load and cache the trained Tier 1 Random Forest model for inference."""
    model_path = PROJECT_ROOT / "models" / "tier1_randomforest_model.joblib"
    if not model_path.is_file():
        return None
    predictor = SOHPredictor(model_path=model_path)
    predictor.load_model()
    return predictor


def main():
    load_custom_css()

    # Load data
    try:
        features_df, ica_df, model_comp_df, fold_metrics_df, oof_df, feat_imp_df = load_dashboard_datasets()
    except Exception as e:
        st.error(f"Error loading dashboard datasets: {e}")
        st.stop()

    predictor = load_tier1_predictor()

    # --------------------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("<h2 style='color:#00F0FF; margin-bottom:0;'>⚡ VoltIQ</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#8D99AE;'>EV Battery Health Intelligence</p>", unsafe_allow_html=True)
        st.markdown("---")

        nav_choice = st.radio(
            "Navigation",
            [
                "Overview",
                "Battery Health Explorer",
                "Model Performance",
                "SOH Prediction Demo",
                "About & Methodology",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Dataset Scope:**")
        st.caption("Authentic NASA Ames PCoE Commercial 18650 Li-ion Cells (`B0005`, `B0006`, `B0007`, `B0018`).")
        st.caption("636 Discharge Cycles | 185k Telemetry Points")

        st.markdown("---")
        st.caption("VoltIQ v1.0 • Academic Portfolio Project")

    # Header banner
    st.markdown("<h1 class='main-header'>VoltIQ | EV Battery Health Intelligence</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#ADB5BD; font-size:1.05rem; margin-top:-10px; margin-bottom:25px;'>"
        "Data-driven State of Health (SOH) estimation and degradation intelligence built on NASA lithium-ion battery cycling data."
        "</p>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # SECTION A: OVERVIEW
    # --------------------------------------------------------------------------
    if nav_choice == "Overview":
        st.subheader("Fleet Health & Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Battery Cells Analyzed", "4 Cells", "Commercial 18650 (LCO)", "#00F0FF")
        with col2:
            render_kpi_card("Total Discharge Cycles", f"{len(features_df):,} Cycles", "2.0A constant discharge", "#48CAE4")
        with col3:
            avg_soh = features_df["soh_percentage"].mean()
            render_kpi_card("Average Fleet SOH", f"{avg_soh:.2f}%", "Baseline-calibrated", "#06D6A0")
        with col4:
            min_soh = features_df["soh_percentage"].min()
            render_kpi_card("Lowest Observed SOH", f"{min_soh:.2f}%", "Cell B0006 (End of test)", "#EF476F")

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive SOH chart
        st.plotly_chart(plot_fleet_soh_comparison(features_df), use_container_width=True)

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("### The NASA Li-Ion Battery Aging Repository")
            st.write(
                """
                The **NASA Ames Prognostics Center of Excellence (PCoE)** battery dataset is a cornerstone 
                benchmark in electrochemical prognostics. Four commercial 18650 lithium-ion cells with nominal 
                rated capacities of **2.0 Ah** were continuously cycled at room temperature (24°C) under constant-current 
                discharge (2.0A) down to predetermined cutoff voltage thresholds.
                
                Testing was terminated when cells reached their **End-of-Life (EOL)** criterion (30% capacity fade, or 70–80% SOH).
                """
            )
        with col_b:
            st.markdown("### Experimental Protocol Summary")
            proto_df = pd.DataFrame({
                "Battery ID": ["B0005", "B0006", "B0007", "B0018"],
                "Cutoff Voltage": ["2.7 V", "2.5 V", "2.2 V", "2.5 V"],
                "Cycles": [168, 168, 168, 132],
                "Initial Capacity": ["1.8565 Ah", "2.0353 Ah", "1.8911 Ah", "1.8550 Ah"],
                "Final SOH": ["71.38%", "58.25%", "75.75%", "72.29%"],
            })
            st.dataframe(proto_df, use_container_width=True, hide_index=True)

        render_disclaimer(
            "VoltIQ uses individually calibrated cell baselines where each battery's State of Health begins at exactly 100.0% "
            "on its initial valid discharge cycle."
        )

    # --------------------------------------------------------------------------
    # SECTION B: BATTERY HEALTH EXPLORER
    # --------------------------------------------------------------------------
    elif nav_choice == "Battery Health Explorer":
        st.subheader("Battery Health Explorer")

        selected_battery = st.selectbox(
            "Select Physical Battery Cell:",
            ["B0005", "B0006", "B0007", "B0018"],
            index=0,
        )

        cell_data = features_df[features_df["battery_id"] == selected_battery].sort_values("cycle_index")
        init_cap = cell_data.iloc[0]["discharge_capacity_ah"]
        final_cap = cell_data.iloc[-1]["discharge_capacity_ah"]
        final_soh = cell_data.iloc[-1]["soh_percentage"]
        fade_pct = 100.0 - final_soh
        recovery_cnt = int(cell_data["capacity_recovery_flag"].sum())
        cutoff_dict = {"B0005": "2.7 V", "B0006": "2.5 V", "B0007": "2.2 V", "B0018": "2.5 V"}

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Initial Baseline", f"{init_cap:.4f} Ah", "Cycle 1 Baseline", "#00F0FF")
        with c2:
            render_kpi_card("Final Measured Capacity", f"{final_cap:.4f} Ah", f"Loss: {init_cap - final_cap:.4f} Ah", "#FFB703")
        with c3:
            render_kpi_card("Final Observed SOH", f"{final_soh:.2f}%", f"Total Fade: {fade_pct:.2f}%", "#06D6A0" if final_soh > 70 else "#EF476F")
        with c4:
            render_kpi_card("Recovery Jumps", f"{recovery_cnt} Events", "> +0.02 Ah after rest", "#9D4EDD")

        st.plotly_chart(plot_battery_degradation_dual(features_df, selected_battery), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Sensor Dynamics", "Incremental Capacity Analysis (ICA)", "Cycle Telemetry Table"])

        with tab1:
            st.plotly_chart(plot_battery_sensor_dynamics(features_df, selected_battery), use_container_width=True)
            st.caption("Telemetry reveals progressive mean voltage decay and thermal dispersion as internal resistance escalates.")

        with tab2:
            st.plotly_chart(plot_ica_curves(ica_df, selected_battery), use_container_width=True)
            st.caption(
                "ICA curves (dQ/dV vs Voltage) derived via 10 mV uniform interpolation and Gaussian smoothing (σ=2.0). "
                "Notice the pronounced attenuation of the phase transition peak around 3.75V as the cell ages."
            )

        with tab3:
            st.write(f"**Discharge Cycle Telemetry Records for Cell {selected_battery} ({len(cell_data)} cycles):**")
            display_cols = [
                "cycle_index", "soh_percentage", "discharge_capacity_ah",
                "voltage_mean", "temperature_max", "temperature_std", "current_load_mean"
            ]
            st.dataframe(cell_data[display_cols], use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # SECTION C: MODEL PERFORMANCE
    # --------------------------------------------------------------------------
    elif nav_choice == "Model Performance":
        st.subheader("Machine Learning Benchmarks & Cross-Battery Validation")

        st.markdown(
            """
            To evaluate true generalization to unseen battery packs, all models were evaluated using 
            **`GroupKFold(n_splits=4)` with `battery_id` grouping (Leave-One-Battery-Out)**. 
            No random row-level splitting was permitted, ensuring zero cycle leakage between training and testing.
            """
        )

        if not model_comp_df.empty:
            st.markdown("### Benchmark Results Across Feature Tiers")
            
            tier_filter = st.radio(
                "Select Feature Benchmark Tier:",
                ["All Tiers", "Tier 1 (Sensor-Only)", "Tier 2 (Sensor + Age)", "Tier 3 (Full Diagnostic)"],
                horizontal=True,
            )

            filtered_comp = model_comp_df if tier_filter == "All Tiers" else model_comp_df[model_comp_df["feature_tier"] == tier_filter]
            
            show_cols = ["feature_tier", "model_name", "features_count", "oof_rmse", "oof_mae", "oof_r2", "fold_rmse_mean"]
            rename_map = {
                "feature_tier": "Feature Tier",
                "model_name": "Model Architecture",
                "features_count": "Features",
                "oof_rmse": "OOF RMSE (%)",
                "oof_mae": "OOF MAE (%)",
                "oof_r2": "OOF R²",
                "fold_rmse_mean": "Mean Fold RMSE (%)",
            }
            styled_df = filtered_comp[show_cols].rename(columns=rename_map).sort_values("OOF RMSE (%)")
            st.dataframe(
                styled_df.style.format({
                    "OOF RMSE (%)": "{:.3f}%",
                    "OOF MAE (%)": "{:.3f}%",
                    "OOF R²": "{:.3f}",
                    "Mean Fold RMSE (%)": "{:.3f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("### Selected Experimental Model: Tier 1 Sensor-Only Random Forest")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi_card("Overall OOF RMSE", "3.711%", "Cross-Battery Pooled", "#00F0FF")
        with c2:
            render_kpi_card("Overall OOF MAE", "2.991%", "Cross-Battery Pooled", "#48CAE4")
        with c3:
            render_kpi_card("Coefficient of Determination", "0.885 R²", "Strong Cross-Cell Generalization", "#06D6A0")

        # Battery-wise fold breakdown table
        st.markdown("#### Battery-Wise Validation Breakdown (Leave-One-Battery-Out)")
        t1_fold_data = pd.DataFrame({
            "Held-Out Cell": ["B0005", "B0006", "B0007", "B0018", "Pooled Overall"],
            "Cutoff Voltage": ["2.7 V", "2.5 V", "2.2 V", "2.5 V", "All Protocols"],
            "Test Cycles": [168, 168, 168, 132, 636],
            "Holdout RMSE": ["2.363%", "5.174%", "3.400%", "3.237%", "3.711%"],
            "Holdout MAE": ["1.993%", "4.169%", "3.126%", "2.592%", "2.991%"],
            "Holdout R²": ["0.947", "0.824", "0.840", "0.849", "0.885"],
        })
        st.dataframe(t1_fold_data, use_container_width=True, hide_index=True)

        if not oof_df.empty:
            st.plotly_chart(plot_oof_true_vs_predicted(oof_df), use_container_width=True)

        if not feat_imp_df.empty:
            st.plotly_chart(plot_feature_importance_bar(feat_imp_df, top_n=10), use_container_width=True)

        render_disclaimer(
            "Understanding Negative R²: The DummyRegressor baseline produced R² = -0.127. A negative R² occurs when "
            "prediction error exceeds the target's fold variance, proving that predicting the training mean on an unseen "
            "battery cell is worse than predicting the unseen cell's own mean.",
            is_warning=False,
        )

    # --------------------------------------------------------------------------
    # SECTION D: SOH PREDICTION DEMO
    # --------------------------------------------------------------------------
    elif nav_choice == "SOH Prediction Demo":
        st.subheader("Historical Cycle SOH Estimation Demo")

        render_disclaimer(
            "Demonstration Context: This interactive module runs the trained Tier 1 Random Forest model on an "
            "existing labeled cycle from the NASA dataset using only its 21 approved sensor-only features. "
            "This is a historical demonstration to verify model response, NOT a real-time EV telemetry stream or forecasting tool."
        )

        if predictor is None:
            st.error("Model artifact not found at `models/tier1_randomforest_model.joblib`. Please run Phase 6 training first.")
            st.stop()

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel_bat = st.selectbox("Select Battery Cell:", ["B0005", "B0006", "B0007", "B0018"], index=0)
        
        bat_cycles = features_df[features_df["battery_id"] == sel_bat].sort_values("cycle_index")
        min_c = int(bat_cycles["cycle_index"].min())
        max_c = int(bat_cycles["cycle_index"].max())

        with col_sel2:
            sel_cycle = st.slider("Select Historical Cycle Index:", min_value=min_c, max_value=max_c, value=min_c)

        cycle_row = bat_cycles[bat_cycles["cycle_index"] == sel_cycle].iloc[0]
        actual_soh = float(cycle_row["soh_percentage"])

        # Run real model prediction
        cycle_df = pd.DataFrame([cycle_row])
        pred_soh = float(predictor.predict(cycle_df)[0])
        abs_err = abs(actual_soh - pred_soh)

        col_g, col_m = st.columns([1, 1])

        with col_g:
            st.plotly_chart(plot_soh_gauge(pred_soh, actual_soh), use_container_width=True)

        with col_m:
            st.markdown("#### Health Estimation Results")
            k1, k2 = st.columns(2)
            with k1:
                render_kpi_card("Model Predicted SOH", f"{pred_soh:.2f}%", "Tier 1 Random Forest", "#00F0FF")
            with k2:
                render_kpi_card("Actual Measured SOH", f"{actual_soh:.2f}%", "Historical Ground Truth", "#06D6A0")

            k3, k4 = st.columns(2)
            with k3:
                render_kpi_card("Absolute Error", f"{abs_err:.2f}%", f"{abs_err / actual_soh * 100:.2f}% Relative", "#FFB703")
            with k4:
                status_color = "#06D6A0" if pred_soh >= 90 else ("#FFB703" if pred_soh >= 80 else "#EF476F")
                status_text = "Healthy (>90%)" if pred_soh >= 90 else ("Degraded (80-90%)" if pred_soh >= 80 else "End-of-Life (<80%)")
                render_kpi_card("Battery Health State", status_text, "EOL Cutoff: 80% SOH", status_color)

        st.markdown("---")
        st.markdown(f"#### Input Sensor Telemetry Features (Cycle {sel_cycle} of Cell {sel_bat})")
        st.caption("The 21 sensor-only statistical features fed into the model for this estimation:")

        sensor_data = []
        for feat in FEATURE_SET_TIER_1_SENSOR_ONLY:
            sensor_data.append({"Feature Name": feat, "Recorded Value": f"{cycle_row[feat]:.4f}"})
        
        sensor_df = pd.DataFrame(sensor_data)
        st.dataframe(sensor_df.T, use_container_width=True)

    # --------------------------------------------------------------------------
    # SECTION E: ABOUT & METHODOLOGY
    # --------------------------------------------------------------------------
    elif nav_choice == "About & Methodology":
        st.subheader("About VoltIQ & Scientific Methodology")

        st.markdown(
            """
            ### 1. Project Motivation
            In electric vehicles (EVs), the battery pack is the single most valuable, safety-critical asset. 
            Accurately estimating **State of Health (SOH)** is essential for calculating remaining driving range, 
            preventing thermal runaway, scheduling second-life recycling, and determining asset residual valuation.
            
            VoltIQ is an end-to-end Data Science project built from raw experimental telemetry provided by the 
            **NASA Ames Prognostics Center of Excellence (PCoE)**.
            """
        )

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("### 2. Electrochemical Formulations")
            st.markdown(
                """
                - **Initial Capacity Baseline:** Established separately for each cell from its first chronological cycle ($C_{\\text{initial}}$).
                - **State of Health (SOH %):**
                  $$\\text{SOH} (\\%) = \\left( \\frac{C_{\\text{discharge}}}{C_{\\text{initial}}} \\right) \\times 100$$
                - **Capacity Fade (%):**
                  $$\\text{Capacity Fade} (\\%) = 100 - \\text{SOH}$$
                - **Capacity Recovery:** Legitimate electrochemical capacity rebound ($> +0.02\\text{ Ah}$) 
                  resulting from lithium concentration gradient relaxation during prolonged rest and EIS testing intervals.
                """
            )

        with col_m2:
            st.markdown("### 3. Leakage Prevention & GroupKFold")
            st.markdown(
                """
                - **The Cluster Effect ($N_{\\text{cells}} = 4$):** Although the dataset contains 636 cycles, they originate 
                  from only 4 commercial cells. Naive random K-Fold cross validation would cause severe data leakage by 
                  mixing correlated cycles from the same cell across train and test splits.
                - **Leave-One-Battery-Out:** `GroupKFold(n_splits=4)` holds out an entire physical cell in each fold.
                - **Target Proxy Quarantine:** Discharge duration ($Q = I \\times t$) correlates with capacity at $r = 0.9523$. 
                  Because duration is only known *after* complete discharge, it is quarantined from real-time BMS models.
                """
            )

        st.markdown("---")
        st.markdown("### 4. Scientific Limitations & Production Readiness")

        render_disclaimer(
            "VoltIQ is an academic Data Science portfolio demonstration. The models are trained on 4 experimental commercial "
            "18650 cells cycled at constant room temperature (24°C) under constant-current discharge. Real EV battery packs "
            "operate under highly dynamic drive cycles (WLTP/EPA), regenerative braking pulses, extreme ambient temperatures "
            "(-20°C to 45°C), and incomplete partial charging. This model cannot be deployed in production EVs without fleet-scale validation.",
            is_warning=True,
        )


if __name__ == "__main__":
    main()
