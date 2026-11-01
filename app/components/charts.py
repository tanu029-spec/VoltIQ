"""
VoltIQ Dashboard: Interactive Plotly Charting Engine
Generates publication-grade, automotive-themed interactive visualizations for battery telemetry.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Automotive dark palette
THEME_COLORS = {
    "background": "#151D36",
    "paper": "#0B132B",
    "text": "#E0E6ED",
    "grid": "rgba(255, 255, 255, 0.08)",
    "cyan": "#00F0FF",
    "yellow": "#FFB703",
    "green": "#06D6A0",
    "red": "#EF476F",
    "purple": "#9D4EDD",
}

BATTERY_PALETTE = {
    "B0005": "#00F0FF",  # Electric Cyan
    "B0006": "#FFB703",  # Amber Orange
    "B0007": "#06D6A0",  # Emerald Green
    "B0018": "#EF476F",  # Crimson Red
}


def _apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent high-contrast dark automotive styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#FFFFFF", family="sans-serif")),
        template="plotly_dark",
        paper_bgcolor=THEME_COLORS["paper"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], size=12),
        margin=dict(l=50, r=40, t=50, b=50),
        legend=dict(
            bgcolor="rgba(28, 37, 65, 0.8)",
            bordercolor="rgba(0, 240, 255, 0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=THEME_COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=THEME_COLORS["grid"], zeroline=False),
    )
    return fig


def plot_fleet_soh_comparison(features_df: pd.DataFrame) -> go.Figure:
    """
    Generate interactive SOH degradation trajectory comparing all four cells.
    """
    fig = go.Figure()

    for b_id, grp in features_df.groupby("battery_id"):
        init_cap = grp.iloc[0]["discharge_capacity_ah"]
        fig.add_trace(
            go.Scatter(
                x=grp["cycle_index"],
                y=grp["soh_percentage"],
                mode="lines+markers",
                name=f"Cell {b_id} (Init: {init_cap:.3f} Ah)",
                line=dict(color=BATTERY_PALETTE.get(b_id, THEME_COLORS["cyan"]), width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f"<b>Cell {b_id}</b><br>"
                    + "Cycle: %{x}<br>"
                    + "SOH: %{y:.2f}%<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # 80% EOL Threshold Line
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color=THEME_COLORS["red"],
        line_width=1.5,
        annotation_text="80% SOH End-of-Life (EOL)",
        annotation_position="bottom right",
        annotation_font_color=THEME_COLORS["red"],
    )

    fig.update_xaxes(title="Discharge Cycle Index")
    fig.update_yaxes(title="State of Health (SOH %)", range=[50, 105])
    return _apply_dark_theme(fig, "Fleet SOH Degradation Trajectories vs Cycle Aging")


def plot_battery_degradation_dual(features_df: pd.DataFrame, battery_id: str) -> go.Figure:
    """
    Plot SOH (%) and Discharge Capacity (Ah) fade for a selected battery cell.
    """
    df_cell = features_df[features_df["battery_id"] == battery_id].sort_values("cycle_index")
    cell_color = BATTERY_PALETTE.get(battery_id, THEME_COLORS["cyan"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # SOH line
    fig.add_trace(
        go.Scatter(
            x=df_cell["cycle_index"],
            y=df_cell["soh_percentage"],
            name="State of Health (%)",
            line=dict(color=cell_color, width=2.5),
            hovertemplate="Cycle %{x}<br>SOH: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    # Capacity line
    fig.add_trace(
        go.Scatter(
            x=df_cell["cycle_index"],
            y=df_cell["discharge_capacity_ah"],
            name="Discharge Capacity (Ah)",
            line=dict(color=THEME_COLORS["green"], width=1.8, dash="dot"),
            hovertemplate="Cycle %{x}<br>Capacity: %{y:.4f} Ah<extra></extra>",
        ),
        secondary_y=True,
    )

    # Highlight capacity recovery jumps
    recoveries = df_cell[df_cell["capacity_recovery_flag"] == True]
    if not recoveries.empty:
        fig.add_trace(
            go.Scatter(
                x=recoveries["cycle_index"],
                y=recoveries["soh_percentage"],
                mode="markers",
                name="Capacity Recovery Event (> +0.02 Ah)",
                marker=dict(symbol="diamond", size=8, color="#FFFFFF", line=dict(color=cell_color, width=1.5)),
                hovertemplate="Recovery Jump at Cycle %{x}<br>SOH: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    # 80% EOL line
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color=THEME_COLORS["red"],
        line_width=1.2,
        annotation_text="80% SOH EOL",
        annotation_position="bottom right",
        secondary_y=False,
    )

    fig.update_xaxes(title="Discharge Cycle Index")
    fig.update_yaxes(title_text="State of Health (SOH %)", secondary_y=False)
    fig.update_yaxes(title_text="Capacity (Ah)", secondary_y=True)
    return _apply_dark_theme(fig, f"Cell {battery_id}: Degradation Trajectory & Capacity Recovery")


def plot_battery_sensor_dynamics(features_df: pd.DataFrame, battery_id: str) -> go.Figure:
    """
    Plot evolution of voltage, temperature, and current load statistics across cycles.
    """
    df_cell = features_df[features_df["battery_id"] == battery_id].sort_values("cycle_index")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Mean Discharge Voltage (V)",
            "Peak Cycle Temperature (°C)",
            "Average Operating Load (A)",
        ),
    )

    cell_color = BATTERY_PALETTE.get(battery_id, THEME_COLORS["cyan"])

    # Mean Voltage
    fig.add_trace(
        go.Scatter(
            x=df_cell["cycle_index"],
            y=df_cell["voltage_mean"],
            line=dict(color=cell_color, width=2),
            name="Voltage Mean (V)",
            hovertemplate="Cycle %{x}<br>Voltage Mean: %{y:.3f} V<extra></extra>",
        ),
        row=1, col=1,
    )

    # Temperature Max
    fig.add_trace(
        go.Scatter(
            x=df_cell["cycle_index"],
            y=df_cell["temperature_max"],
            line=dict(color=THEME_COLORS["yellow"], width=2),
            name="Temp Peak (°C)",
            hovertemplate="Cycle %{x}<br>Temp Max: %{y:.2f} °C<extra></extra>",
        ),
        row=2, col=1,
    )

    # Current Load Mean
    fig.add_trace(
        go.Scatter(
            x=df_cell["cycle_index"],
            y=df_cell["current_load_mean"],
            line=dict(color=THEME_COLORS["green"], width=2),
            name="Load Mean (A)",
            hovertemplate="Cycle %{x}<br>Load Mean: %{y:.3f} A<extra></extra>",
        ),
        row=3, col=1,
    )

    fig.update_xaxes(title="Discharge Cycle Index", row=3, col=1)
    return _apply_dark_theme(fig, f"Cell {battery_id}: Key Sensor Dynamics Over Aging")


def plot_ica_curves(ica_df: pd.DataFrame, battery_id: str) -> go.Figure:
    """
    Plot Incremental Capacity Analysis (dQ/dV vs Voltage) curves across life stages.
    """
    cell_ica = ica_df[ica_df["battery_id"] == battery_id]
    cycle_col = "cycle_number" if "cycle_number" in cell_ica.columns else "cycle_index"
    dq_dv_col = "dq_dv" if "dq_dv" in cell_ica.columns else "ica_d_q_d_v"

    fig = go.Figure()
    if cycle_col not in cell_ica.columns or cell_ica.empty:
        return _apply_dark_theme(fig, f"No ICA data available for {battery_id}")

    cycles = sorted(cell_ica[cycle_col].unique())
    if not cycles:
        return _apply_dark_theme(fig, f"No ICA data available for {battery_id}")

    # Select early (cycle 1), mid (~50%), and late (~100%) cycles
    sel_cycles = [cycles[0], cycles[len(cycles) // 2], cycles[-1]]
    colors = [THEME_COLORS["cyan"], THEME_COLORS["yellow"], THEME_COLORS["red"]]
    labels = ["Initial (Fresh)", "Mid-Life", "End-of-Life (Aged)"]

    for c_val, col, lbl in zip(sel_cycles, colors, labels):
        sub = cell_ica[cell_ica[cycle_col] == c_val].sort_values("voltage_grid_v")
        fig.add_trace(
            go.Scatter(
                x=sub["voltage_grid_v"],
                y=sub[dq_dv_col],
                mode="lines",
                name=f"Cycle {c_val} ({lbl})",
                line=dict(color=col, width=2.5),
                hovertemplate="Voltage: %{x:.3f} V<br>dQ/dV: %{y:.2f} Ah/V<extra></extra>",
            )
        )

    fig.update_xaxes(title="Terminal Voltage (V)", range=[3.2, 4.0])
    fig.update_yaxes(title="Incremental Capacity dQ/dV (Ah/V)")
    return _apply_dark_theme(fig, f"Cell {battery_id}: Incremental Capacity Curve (dQ/dV) Evolution")


def plot_oof_true_vs_predicted(oof_df: pd.DataFrame) -> go.Figure:
    """
    Plot True vs Predicted SOH from out-of-fold validation.
    """
    if "pred_Tier 1_RandomForest" in oof_df.columns:
        pred_col = "pred_Tier 1_RandomForest"
    elif "pred_Tier1_RandomForest" in oof_df.columns:
        pred_col = "pred_Tier1_RandomForest"
    else:
        pred_cols = [c for c in oof_df.columns if c.startswith("pred_")]
        pred_col = pred_cols[0] if pred_cols else None

    fig = go.Figure()
    if pred_col is None:
        return _apply_dark_theme(fig, "No OOF prediction data available")

    for b_id, grp in oof_df.groupby("battery_id"):
        fig.add_trace(
            go.Scatter(
                x=grp["soh_percentage"],
                y=grp[pred_col],
                mode="markers",
                name=f"Cell {b_id}",
                marker=dict(size=6, color=BATTERY_PALETTE.get(b_id, THEME_COLORS["cyan"]), opacity=0.75),
                hovertemplate=(
                    f"<b>Cell {b_id}</b><br>"
                    + "True SOH: %{x:.2f}%<br>"
                    + "Pred SOH: %{y:.2f}%<br>"
                    + "Error: %{customdata:.2f}%<extra></extra>"
                ),
                customdata=np.abs(grp["soh_percentage"] - grp[pred_col]),
            )
        )

    # 1:1 reference line
    fig.add_trace(
        go.Scatter(
            x=[50, 105],
            y=[50, 105],
            mode="lines",
            name="Perfect 1:1 Agreement",
            line=dict(color="#FFFFFF", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    fig.update_xaxes(title="Ground Truth SOH (%)", range=[55, 105])
    fig.update_yaxes(title="Predicted SOH (%)", range=[55, 105])
    return _apply_dark_theme(fig, "Out-of-Fold Cross-Battery Validation: True vs Predicted SOH")


def plot_feature_importance_bar(imp_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of top predictive features.
    """
    top_df = imp_df.head(top_n).sort_values("importance_mdi", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df["importance_mdi"] * 100.0,
            y=top_df["feature"],
            orientation="h",
            marker=dict(
                color=top_df["importance_mdi"],
                colorscale="Viridis",
                line=dict(color="rgba(0, 240, 255, 0.5)", width=1),
            ),
            hovertemplate="Feature: %{y}<br>Importance: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_xaxes(title="Mean Decrease in Impurity (MDI Importance %)")
    fig.update_yaxes(title="")
    return _apply_dark_theme(fig, f"Top {top_n} Sensor Features by Relative Predictive Power")


def plot_soh_gauge(predicted_soh: float, actual_soh: Optional[float] = None) -> go.Figure:
    """
    Render a circular SOH health indicator gauge with color zones.
    """
    delta_dict = None
    if actual_soh is not None:
        delta_dict = {"reference": actual_soh, "increasing": {"color": THEME_COLORS["cyan"]}}

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number" + ("+delta" if actual_soh is not None else ""),
            value=predicted_soh,
            number={"suffix": "%", "font": {"size": 36, "color": "#FFFFFF"}},
            delta=delta_dict,
            title={"text": "Estimated Battery SOH", "font": {"size": 16, "color": "#8D99AE"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8D99AE"},
                "bar": {"color": THEME_COLORS["cyan"], "thickness": 0.28},
                "bgcolor": "rgba(28, 37, 65, 0.6)",
                "borderwidth": 1,
                "bordercolor": "rgba(0, 240, 255, 0.2)",
                "steps": [
                    {"range": [0, 80], "color": "rgba(239, 71, 111, 0.35)"},      # End of life
                    {"range": [80, 90], "color": "rgba(255, 183, 3, 0.35)"},      # Moderate degradation
                    {"range": [90, 100], "color": "rgba(6, 214, 160, 0.35)"},     # Healthy
                ],
                "threshold": {
                    "line": {"color": THEME_COLORS["red"], "width": 3},
                    "thickness": 0.75,
                    "value": 80.0,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME_COLORS["paper"],
        plot_bgcolor=THEME_COLORS["paper"],
        margin=dict(l=30, r=30, t=50, b=30),
        height=280,
    )
    return fig
