"""
VoltIQ: Exploratory Data Analysis (EDA) Module
Provides analytical routines, statistical aggregation, correlation inspection,
and publication-quality automotive visualisations for battery degradation intelligence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.feature_config import (
    SENSOR_ONLY_FEATURES,
    CYCLE_AGE_FEATURES,
    TARGET_PROXY_FEATURES,
    TARGET_COLUMNS,
)

logger = logging.getLogger(__name__)

# Automotive & Engineering visual style tokens
BATTERY_COLORS = {
    "B0005": "#1f77b4",  # Deep Blue
    "B0006": "#ff7f0e",  # Safety Orange
    "B0007": "#2ca02c",  # Forest Green
    "B0018": "#d62728",  # Crimson Red
}


class BatteryEDA:
    """
    Automates statistical computation, degradation trend analysis,
    and publication-grade visualization for the VoltIQ system.
    """

    def __init__(
        self,
        features_path: Union[str, Path] = "data/processed/battery_health_features.csv",
        ica_path: Union[str, Path] = "data/processed/battery_ica_features.csv",
        figures_dir: Union[str, Path] = "results/figures/eda",
        results_dir: Union[str, Path] = "results",
    ):
        self.features_path = Path(features_path)
        self.ica_path = Path(ica_path)
        self.figures_dir = Path(figures_dir)
        self.results_dir = Path(results_dir)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the engineered battery health features and ICA curves.
        """
        if not self.features_path.is_file():
            raise FileNotFoundError(f"Feature dataset not found: {self.features_path}")
        if not self.ica_path.is_file():
            raise FileNotFoundError(f"ICA dataset not found: {self.ica_path}")

        features_df = pd.read_csv(self.features_path)
        ica_curves_df = pd.read_csv(self.ica_path)

        return features_df, ica_curves_df

    def compute_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute descriptive statistics (mean, median, std, min, max, quartiles)
        globally and stratified by battery cell.
        """
        key_vars = [
            "soh_percentage",
            "discharge_capacity_ah",
            "voltage_mean",
            "voltage_min",
            "voltage_max",
            "temperature_mean",
            "temperature_max",
            "discharge_duration_seconds",
            "ica_peak_value",
            "ica_area",
        ]

        def _get_stats(series: pd.Series) -> Dict[str, float]:
            vals = series.dropna().values
            return {
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }

        global_stats = {col: _get_stats(df[col]) for col in key_vars if col in df.columns}

        per_battery_stats = {}
        for b_id, grp in df.groupby("battery_id"):
            per_battery_stats[b_id] = {col: _get_stats(grp[col]) for col in key_vars if col in grp.columns}

        return {
            "total_records": len(df),
            "global_statistics": global_stats,
            "battery_wise_statistics": per_battery_stats,
        }

    def compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute Pearson and Spearman correlations between features and SOH,
        and assess cross-feature collinearity.
        """
        numeric_df = df.select_dtypes(include=np.number)
        pearson_corr = numeric_df.corr(method="pearson")
        spearman_corr = numeric_df.corr(method="spearman")

        soh_pearson = pearson_corr["soh_percentage"].drop("soh_percentage").to_dict()
        soh_spearman = spearman_corr["soh_percentage"].drop("soh_percentage").to_dict()

        # Identify pairs with collinearity |r| >= 0.90
        collinear_pairs: List[Dict[str, Any]] = []
        cols = list(pearson_corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                r_val = float(pearson_corr.loc[c1, c2])
                if abs(r_val) >= 0.90:
                    collinear_pairs.append({"feature_1": c1, "feature_2": c2, "pearson_r": r_val})

        return {
            "soh_pearson": {k: float(v) for k, v in soh_pearson.items() if not np.isnan(v)},
            "soh_spearman": {k: float(v) for k, v in soh_spearman.items() if not np.isnan(v)},
            "highly_collinear_pairs_count": len(collinear_pairs),
            "highly_collinear_pairs": collinear_pairs,
        }

    def analyze_battery_differences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Examine observational differences across B0005, B0006, B0007, and B0018
        without drawing unsupported speculative conclusions.
        """
        b_summary = {}
        for b_id, grp in df.groupby("battery_id"):
            grp_sorted = grp.sort_values("cycle_index")
            init_cap = float(grp_sorted.iloc[0]["discharge_capacity_ah"])
            final_cap = float(grp_sorted.iloc[-1]["discharge_capacity_ah"])
            init_soh = float(grp_sorted.iloc[0]["soh_percentage"])
            final_soh = float(grp_sorted.iloc[-1]["soh_percentage"])
            cycles = int(len(grp))
            avg_fade_per_cycle = float((init_soh - final_soh) / (cycles - 1)) if cycles > 1 else 0.0

            CUTOFF_VOLTAGES = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}
            b_summary[b_id] = {
                "discharge_cycles_count": cycles,
                "cutoff_voltage_v": CUTOFF_VOLTAGES.get(b_id, 0.0),
                "initial_capacity_ah": init_cap,
                "final_capacity_ah": final_cap,
                "absolute_capacity_loss_ah": init_cap - final_cap,
                "soh_min": float(grp["soh_percentage"].min()),
                "soh_max": float(grp["soh_percentage"].max()),
                "avg_soh_loss_per_cycle": avg_fade_per_cycle,
                "mean_temperature_c": float(grp["temperature_mean"].mean()),
                "peak_temperature_c": float(grp["temperature_max"].max()),
                "mean_duration_s": float(grp["discharge_duration_seconds"].mean()),
                "recovery_cycles_count": int(grp["capacity_recovery_flag"].sum()),
            }

        return b_summary

    def generate_all_plots(
        self, features_df: pd.DataFrame, ica_curves_df: pd.DataFrame
    ) -> List[Path]:
        """
        Generate all publication-quality automotive visualisations requested in Part D.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        saved_plots: List[Path] = []

        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 15,
        })

        # 1. SOH Degradation Curves by Battery
        p1 = self.figures_dir / "soh_degradation_curves.png"
        plt.figure(figsize=(10, 5.5))
        for b_id, grp in features_df.groupby("battery_id"):
            plt.plot(
                grp["cycle_index"],
                grp["soh_percentage"],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=f"Cell {b_id} (Initial: {grp.iloc[0]['initial_capacity_ah']:.3f} Ah)",
                color=BATTERY_COLORS.get(b_id),
            )
        plt.axhline(80.0, color="crimson", linestyle="--", linewidth=1.5, label="80% SOH End-of-Life (EOL) Threshold")
        plt.title("State of Health (SOH %) Trajectories Across 18650 Battery Cells")
        plt.xlabel("Discharge Cycle Index (Sequential)")
        plt.ylabel("State of Health (SOH %)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p1, dpi=300)
        plt.close()
        saved_plots.append(p1)

        # 2. Capacity Fade Comparison
        p2 = self.figures_dir / "capacity_fade_comparison.png"
        plt.figure(figsize=(10, 5.5))
        for b_id, grp in features_df.groupby("battery_id"):
            plt.plot(
                grp["cycle_index"],
                grp["capacity_fade_percentage"],
                marker="s",
                markersize=2.5,
                linewidth=1.5,
                label=f"Cell {b_id}",
                color=BATTERY_COLORS.get(b_id),
            )
        plt.axhline(20.0, color="crimson", linestyle="--", linewidth=1.5, label="20% Capacity Fade EOL Limit")
        plt.title("Cumulative Capacity Fade (%) Comparison Across Aging Cycles")
        plt.xlabel("Discharge Cycle Index")
        plt.ylabel("Capacity Fade (%)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p2, dpi=300)
        plt.close()
        saved_plots.append(p2)

        # 3. Battery-Wise SOH Distributions
        p3 = self.figures_dir / "battery_wise_soh_distributions.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.boxplot(
            data=features_df,
            x="battery_id",
            y="soh_percentage",
            ax=axes[0],
            hue="battery_id",
            palette=BATTERY_COLORS,
            legend=False,
        )
        axes[0].set_title("SOH Distribution Across Cells (Boxplot)")
        axes[0].set_ylabel("State of Health (%)")
        axes[0].axhline(80.0, color="crimson", linestyle="--", label="80% EOL")
        axes[0].legend()

        sns.violinplot(
            data=features_df,
            x="battery_id",
            y="soh_percentage",
            ax=axes[1],
            hue="battery_id",
            palette=BATTERY_COLORS,
            inner="quartile",
            legend=False,
        )
        axes[1].set_title("SOH Density Distribution Across Cells (Violin Plot)")
        axes[1].set_ylabel("State of Health (%)")
        axes[1].axhline(80.0, color="crimson", linestyle="--")

        plt.suptitle("Battery-Wise SOH Dispersion and Interquartile Spread", y=1.02)
        plt.tight_layout()
        plt.savefig(p3, dpi=300)
        plt.close()
        saved_plots.append(p3)

        # 4. Voltage, Current, and Temperature Trends
        p4 = self.figures_dir / "sensor_trends_voltage_current_temperature.png"
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for b_id, grp in features_df.groupby("battery_id"):
            c = BATTERY_COLORS.get(b_id)
            axes[0].plot(grp["cycle_index"], grp["voltage_mean"], color=c, label=f"Cell {b_id}", linewidth=1.5)
            axes[1].plot(grp["cycle_index"], grp["current_mean"], color=c, label=f"Cell {b_id}", linewidth=1.5)
            axes[2].plot(grp["cycle_index"], grp["temperature_max"], color=c, label=f"Cell {b_id}", linewidth=1.5)

        axes[0].set_title("Mean Terminal Voltage vs Cycle Index")
        axes[0].set_ylabel("Voltage (V)")
        axes[0].legend(loc="lower left")

        axes[1].set_title("Discharge Current Stability vs Cycle Index")
        axes[1].set_ylabel("Current (A)")
        axes[1].set_ylim(-2.1, -1.9)

        axes[2].set_title("Maximum Cell Temperature vs Cycle Index (Thermal Escalation)")
        axes[2].set_ylabel("Peak Temp (°C)")
        axes[2].set_xlabel("Discharge Cycle Index")

        plt.suptitle("Sensor Telemetry Trajectories Over Experimental Life", y=1.00)
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close()
        saved_plots.append(p4)

        # 5. Capacity versus Discharge Duration (Target Proxy Analysis)
        p5 = self.figures_dir / "capacity_vs_discharge_duration.png"
        plt.figure(figsize=(9, 5.5))
        sns.scatterplot(
            data=features_df,
            x="discharge_duration_seconds",
            y="discharge_capacity_ah",
            hue="battery_id",
            palette=BATTERY_COLORS,
            alpha=0.75,
            s=45,
        )
        # Overlay overall linear regression trend
        m, b = np.polyfit(features_df["discharge_duration_seconds"], features_df["discharge_capacity_ah"], 1)
        dur_range = np.linspace(features_df["discharge_duration_seconds"].min(), features_df["discharge_duration_seconds"].max(), 100)
        r_val = features_df["discharge_duration_seconds"].corr(features_df["discharge_capacity_ah"])
        plt.plot(dur_range, m * dur_range + b, color="black", linestyle="--", linewidth=1.5, label=f"Linear Fit ($r = {r_val:.4f}$)")

        plt.title("Discharge Duration vs Capacity: High Collinearity & Target Proxy Audit")
        plt.xlabel("Discharge Duration to Cutoff (seconds)")
        plt.ylabel("Measured Discharge Capacity (Ah)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p5, dpi=300)
        plt.close()
        saved_plots.append(p5)

        # 6. Capacity Recovery Events
        p6 = self.figures_dir / "capacity_recovery_events.png"
        plt.figure(figsize=(11, 5.5))
        for b_id, grp in features_df.groupby("battery_id"):
            c = BATTERY_COLORS.get(b_id)
            plt.plot(grp["cycle_index"], grp["capacity_change_ah"], color=c, alpha=0.4, linewidth=1.0)
            recs = grp[grp["capacity_recovery_flag"]]
            if len(recs) > 0:
                plt.scatter(recs["cycle_index"], recs["capacity_change_ah"], color=c, s=50, edgecolors="black", label=f"{b_id} Recoveries (N={len(recs)})")

        plt.axhline(0.02, color="magenta", linestyle="--", linewidth=1.5, label="Recovery Threshold (+0.02 Ah)")
        plt.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
        plt.title("Electrochemical Capacity Recovery Jumps Following Rest/EIS Intervals")
        plt.xlabel("Discharge Cycle Index")
        plt.ylabel("Cycle-to-Cycle Capacity Change $\\Delta C$ (Ah)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(p6, dpi=300)
        plt.close()
        saved_plots.append(p6)

        # 7. Correlation Heatmap (Grouped by Category)
        p7 = self.figures_dir / "correlation_heatmap_grouped.png"
        selected_cols = [
            "soh_percentage",
            "cycle_index",
            "voltage_mean",
            "voltage_median",
            "voltage_std",
            "temperature_mean",
            "temperature_max",
            "temperature_range",
            "current_std",
            "discharge_duration_seconds",
            "ica_peak_value",
            "ica_area",
        ]
        corr_mat = features_df[selected_cols].corr()

        plt.figure(figsize=(11, 9))
        sns.heatmap(
            corr_mat,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"label": "Pearson Correlation Coefficient ($r$)"},
            square=True,
            linewidths=0.5,
        )
        plt.title("Correlation Matrix of SOH and Representative Engineered Features")
        plt.tight_layout()
        plt.savefig(p7, dpi=300)
        plt.close()
        saved_plots.append(p7)

        # 8. Feature Distribution Plots for Important Sensor Variables
        p8 = self.figures_dir / "sensor_feature_distributions.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        sns.histplot(data=features_df, x="voltage_mean", hue="battery_id", palette=BATTERY_COLORS, kde=True, ax=axes[0, 0], legend=False)
        axes[0, 0].set_title("Mean Terminal Voltage Distribution")
        axes[0, 0].set_xlabel("Mean Voltage (V)")

        sns.histplot(data=features_df, x="temperature_max", hue="battery_id", palette=BATTERY_COLORS, kde=True, ax=axes[0, 1], legend=False)
        axes[0, 1].set_title("Maximum Cell Temperature Distribution")
        axes[0, 1].set_xlabel("Peak Temperature (°C)")

        sns.histplot(data=features_df, x="temperature_range", hue="battery_id", palette=BATTERY_COLORS, kde=True, ax=axes[1, 0], legend=False)
        axes[1, 0].set_title("Thermal Rise ($\\Delta T$) Distribution")
        axes[1, 0].set_xlabel("Temperature Rise (°C)")

        sns.histplot(data=features_df, x="voltage_std", hue="battery_id", palette=BATTERY_COLORS, kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("Voltage Standard Deviation Distribution")
        axes[1, 1].set_xlabel("Voltage Std (V)")

        plt.suptitle("Statistical Sensor Feature Distributions Across Battery Cells", y=0.99)
        plt.tight_layout()
        plt.savefig(p8, dpi=300)
        plt.close()
        saved_plots.append(p8)

        # 9. Comparative Degradation for B0018
        p9 = self.figures_dir / "b0018_comparative_degradation.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for b_id, grp in features_df.groupby("battery_id"):
            c = BATTERY_COLORS.get(b_id)
            alpha = 1.0 if b_id == "B0018" else 0.5
            lw = 2.5 if b_id == "B0018" else 1.2
            axes[0].plot(grp["cycle_index"], grp["discharge_capacity_ah"], color=c, alpha=alpha, linewidth=lw, label=f"Cell {b_id}")
            axes[1].plot(grp["cycle_index"], grp["soh_percentage"], color=c, alpha=alpha, linewidth=lw, label=f"Cell {b_id}")

        axes[0].set_title("Capacity Trajectory: Cell B0018 vs Others")
        axes[0].set_xlabel("Sequential Cycle Index")
        axes[0].set_ylabel("Discharge Capacity (Ah)")
        axes[0].legend()

        axes[1].set_title("SOH (%) Trajectory: Cell B0018 vs Others")
        axes[1].set_xlabel("Sequential Cycle Index")
        axes[1].set_ylabel("State of Health (%)")
        axes[1].axhline(80.0, color="crimson", linestyle="--", label="80% EOL")
        axes[1].legend()

        plt.suptitle("Observational Comparison of Cell B0018 vs B0005, B0006, B0007", y=1.02)
        plt.tight_layout()
        plt.savefig(p9, dpi=300)
        plt.close()
        saved_plots.append(p9)

        return saved_plots

    def export_summary_json(
        self,
        stats: dict,
        correlations: dict,
        b_diffs: dict,
        target_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Export all statistical metrics, correlations, and battery comparisons to JSON.
        """
        out_path = Path(target_path) if target_path else self.results_dir / "metrics" / "eda_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "descriptive_statistics": stats,
            "correlations": correlations,
            "battery_differences": b_diffs,
            "sample_counts": {
                "total_cycles": stats["total_records"],
                "batteries_count": len(b_diffs),
                "battery_ids": list(b_diffs.keys()),
            },
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return out_path
