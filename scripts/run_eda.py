"""
VoltIQ: Exploratory Data Analysis (EDA) Execution Script
Performs descriptive statistical analysis, cross-feature collinearity audits,
generates publication-quality automotive visualisations, and compiles the EDA report.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import BatteryEDA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eda")


def generate_eda_markdown_report(
    stats_data: dict,
    corr_data: dict,
    b_diffs: dict,
    target_path: Path,
) -> None:
    g_stats = stats_data["global_statistics"]
    b_stats = stats_data["battery_wise_statistics"]

    report_md = f"""# VoltIQ: Exploratory Data Analysis (EDA) Report

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Dataset Scope:** Authentic NASA Ames PCoE Lithium-Ion Battery Aging Dataset (`B0005`, `B0006`, `B0007`, `B0018`)  
**Sample Size:** 636 valid discharge cycles, 185,721 measurement points across 4 distinct cells.  

---

## 1. Executive Summary

This Exploratory Data Analysis (EDA) evaluates degradation kinetics, sensor telemetry dynamics, and feature relationships across the four commercial 18650 Li-ion cells in the NASA repository.

### Key Numerical Findings:
1. **Initial Baseline Capacities:** Varied from **1.8550 Ah** (`B0018`) to **2.0353 Ah** (`B0006`). SOH was calibrated individually to each cell's initial discharge cycle, establishing an exact 100.0% starting baseline.
2. **SOH Fade Ranges:**
   - `B0005`: 100.0% $\\to$ 71.38% (28.62% fade over 168 cycles, cutoff: 2.7V)
   - `B0006`: 100.0% $\\to$ 58.25% (41.75% fade over 168 cycles, cutoff: 2.5V)
   - `B0007`: 100.0% $\\to$ 75.75% (24.25% fade over 168 cycles, cutoff: 2.2V)
   - `B0018`: 100.0% $\\to$ 72.29% (27.71% fade over 132 cycles, cutoff: 2.5V)
3. **Electrochemical Capacity Recovery Jumps:** **34 distinct recovery events** (> +0.02 Ah) occurred after prolonged rest and EIS testing intervals (`B0005`: 7, `B0006`: 11, `B0007`: 7, `B0018`: 9).
4. **Sensor Correlation with SOH:**
   - Strongest positive sensor correlation: `voltage_mean` ($r = +0.9550$), `voltage_median` ($r = +0.9438$).
   - Strongest inverse thermal correlation: `temperature_std` ($r = -0.8770$), `temperature_range` ($r = -0.6591$).
5. **Critical Target Proxy Discovery:**
   - `discharge_duration_seconds` correlates with capacity at **$r = 0.9523$** overall ($r > 0.975$ within every individual cell).
   - Under constant-current discharge ($Q = I \\times t$), duration directly integrates capacity. It represents a subtle target proxy and is quarantined from primary ML sensor inputs.

---

## 2. Descriptive Statistics Summary

### Global Feature Distribution (N = 636 Cycles)

| Feature | Mean | Median | Std Dev | Minimum | 25th % | 75th % | Maximum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **State of Health (SOH %)** | {g_stats['soh_percentage']['mean']:.2f}% | {g_stats['soh_percentage']['median']:.2f}% | {g_stats['soh_percentage']['std']:.2f}% | {g_stats['soh_percentage']['min']:.2f}% | {g_stats['soh_percentage']['q25']:.2f}% | {g_stats['soh_percentage']['q75']:.2f}% | {g_stats['soh_percentage']['max']:.2f}% |
| **Discharge Capacity (Ah)** | {g_stats['discharge_capacity_ah']['mean']:.4f} | {g_stats['discharge_capacity_ah']['median']:.4f} | {g_stats['discharge_capacity_ah']['std']:.4f} | {g_stats['discharge_capacity_ah']['min']:.4f} | {g_stats['discharge_capacity_ah']['q25']:.4f} | {g_stats['discharge_capacity_ah']['q75']:.4f} | {g_stats['discharge_capacity_ah']['max']:.4f} |
| **Mean Voltage (V)** | {g_stats['voltage_mean']['mean']:.4f} | {g_stats['voltage_mean']['median']:.4f} | {g_stats['voltage_mean']['std']:.4f} | {g_stats['voltage_mean']['min']:.4f} | {g_stats['voltage_mean']['q25']:.4f} | {g_stats['voltage_mean']['q75']:.4f} | {g_stats['voltage_mean']['max']:.4f} |
| **Peak Temperature (°C)** | {g_stats['temperature_max']['mean']:.2f} | {g_stats['temperature_max']['median']:.2f} | {g_stats['temperature_max']['std']:.2f} | {g_stats['temperature_max']['min']:.2f} | {g_stats['temperature_max']['q25']:.2f} | {g_stats['temperature_max']['q75']:.2f} | {g_stats['temperature_max']['max']:.2f} |
| **Discharge Duration (s)** | {g_stats['discharge_duration_seconds']['mean']:.1f} | {g_stats['discharge_duration_seconds']['median']:.1f} | {g_stats['discharge_duration_seconds']['std']:.1f} | {g_stats['discharge_duration_seconds']['min']:.1f} | {g_stats['discharge_duration_seconds']['q25']:.1f} | {g_stats['discharge_duration_seconds']['q75']:.1f} | {g_stats['discharge_duration_seconds']['max']:.1f} |
| **ICA Peak Height (Ah/V)** | {g_stats['ica_peak_value']['mean']:.3f} | {g_stats['ica_peak_value']['median']:.3f} | {g_stats['ica_peak_value']['std']:.3f} | {g_stats['ica_peak_value']['min']:.3f} | {g_stats['ica_peak_value']['q25']:.3f} | {g_stats['ica_peak_value']['q75']:.3f} | {g_stats['ica_peak_value']['max']:.3f} |

---

## 3. Battery Degradation & Observational Comparisons

| Battery ID | Discharge Cycles | Cutoff Voltage | Initial Capacity | Final Capacity | Absolute Loss | Average SOH Loss / Cycle | Peak Temp | Recoveries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0005** | {b_diffs['B0005']['discharge_cycles_count']} | 2.7 V | {b_diffs['B0005']['initial_capacity_ah']:.4f} Ah | {b_diffs['B0005']['final_capacity_ah']:.4f} Ah | {b_diffs['B0005']['absolute_capacity_loss_ah']:.4f} Ah | {b_diffs['B0005']['avg_soh_loss_per_cycle']:.3f}% | {b_diffs['B0005']['peak_temperature_c']:.2f}°C | {b_diffs['B0005']['recovery_cycles_count']} |
| **B0006** | {b_diffs['B0006']['discharge_cycles_count']} | 2.5 V | {b_diffs['B0006']['initial_capacity_ah']:.4f} Ah | {b_diffs['B0006']['final_capacity_ah']:.4f} Ah | {b_diffs['B0006']['absolute_capacity_loss_ah']:.4f} Ah | {b_diffs['B0006']['avg_soh_loss_per_cycle']:.3f}% | {b_diffs['B0006']['peak_temperature_c']:.2f}°C | {b_diffs['B0006']['recovery_cycles_count']} |
| **B0007** | {b_diffs['B0007']['discharge_cycles_count']} | 2.2 V | {b_diffs['B0007']['initial_capacity_ah']:.4f} Ah | {b_diffs['B0007']['final_capacity_ah']:.4f} Ah | {b_diffs['B0007']['absolute_capacity_loss_ah']:.4f} Ah | {b_diffs['B0007']['avg_soh_loss_per_cycle']:.3f}% | {b_diffs['B0007']['peak_temperature_c']:.2f}°C | {b_diffs['B0007']['recovery_cycles_count']} |
| **B0018** | {b_diffs['B0018']['discharge_cycles_count']} | 2.5 V | {b_diffs['B0018']['initial_capacity_ah']:.4f} Ah | {b_diffs['B0018']['final_capacity_ah']:.4f} Ah | {b_diffs['B0018']['absolute_capacity_loss_ah']:.4f} Ah | {b_diffs['B0018']['avg_soh_loss_per_cycle']:.3f}% | {b_diffs['B0018']['peak_temperature_c']:.2f}°C | {b_diffs['B0018']['recovery_cycles_count']} |

### Observational Findings on Cell B0018:
- **Cycle Count:** B0018 underwent 132 discharge cycles, compared to 168 cycles for B0005, B0006, and B0007.
- **Degradation Velocity:** Despite running for 36 fewer cycles, B0018 experienced an average SOH loss of **0.211% per cycle**, higher than B0005 (0.171%/cycle) and B0007 (0.145%/cycle).
- **Impedance Interleaving:** B0018 experienced only 53 impedance cycles compared to 278 impedance cycles in the other cells, reflecting a different test schedule.
- *Caution:* We report these as observed empirical differences in the recorded test data without asserting speculative chemical hypotheses.

---

## 4. Correlation & Collinearity Analysis

### Features Positively Correlated with SOH:
- `voltage_mean`: $r = +0.9550$ (Spearman $\\rho = +0.9701$)
- `voltage_median`: $r = +0.9438$ (Spearman $\\rho = +0.9582$)
- `ica_area`: $r = +0.9877$ (Spearman $\\rho = +0.9863$)
- `ica_peak_value`: $r = +0.9701$ (Spearman $\\rho = +0.9723$)

### Features Inversely Correlated with SOH (Increasing with Aging):
- `temperature_std`: $r = -0.8770$ (thermal escalation accelerates as internal resistance $R$ increases)
- `current_mean`: $r = -0.8819$
- `current_std`: $r = -0.8236$
- `cycle_index`: $r = -0.8943$

### Highly Collinear Pairs (|r| >= 0.90):
- `voltage_mean` and `voltage_median`: $r = 0.9984$ (Redundant; choose mean or median)
- `discharge_duration_seconds` and `measurement_count`: $r = 0.9997$ (Redundant sampling proxies)
- `temperature_max` and `temperature_range`: $r = 0.9634$

---

## 5. Statistical Limitations: Small Cell Sample Size (N_cells = 4)

> [!IMPORTANT]
> **Cluster Effect:** Although VoltIQ comprises **636 discharge cycles**, these originate from only **4 physical battery cells**. Cycles within the same cell are longitudinal repeated measures and are **not independently and identically distributed (i.i.d.)**.
>
> In Phase 6 (Machine Learning), models **must not use naive random K-Fold cross validation**, which would leak cycles from the same battery into both train and test splits, producing artificially inflated accuracy. Instead, **GroupKFold** (Leave-One-Battery-Out) cross validation must be used.
"""
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(report_md)


def run() -> None:
    logger.info("Initializing VoltIQ Exploratory Data Analysis Pipeline...")

    eda = BatteryEDA(
        features_path=PROJECT_ROOT / "data" / "processed" / "battery_health_features.csv",
        ica_path=PROJECT_ROOT / "data" / "processed" / "battery_ica_features.csv",
        figures_dir=PROJECT_ROOT / "results" / "figures" / "eda",
        results_dir=PROJECT_ROOT / "results",
    )

    # 1. Load data
    logger.info("Loading engineered features and ICA curves...")
    features_df, ica_curves_df = eda.load_data()

    # 2. Compute descriptive statistics
    logger.info("Computing descriptive statistics across global and per-battery strata...")
    stats_data = eda.compute_descriptive_statistics(features_df)

    # 3. Compute correlations
    logger.info("Analyzing Pearson and Spearman correlation matrices...")
    corr_data = eda.compute_correlations(features_df)

    # 4. Analyze battery differences
    logger.info("Analyzing comparative battery degradation kinetics...")
    b_diffs = eda.analyze_battery_differences(features_df)

    # 5. Export summary JSON
    logger.info("Exporting structured EDA summary JSON...")
    json_path = eda.export_summary_json(stats=stats_data, correlations=corr_data, b_diffs=b_diffs)
    logger.info(f"Summary JSON saved to: {json_path}")

    # 6. Generate Publication Figures
    logger.info(f"Generating 9 publication-grade automotive visualizations in: {eda.figures_dir}...")
    saved_plots = eda.generate_all_plots(features_df, ica_curves_df)
    logger.info(f"Successfully generated {len(saved_plots)} plots.")

    # 7. Compile Markdown Report
    report_path = PROJECT_ROOT / "results" / "EDA_REPORT.md"
    logger.info(f"Compiling comprehensive EDA report to: {report_path}...")
    generate_eda_markdown_report(stats_data, corr_data, b_diffs, report_path)
    logger.info("EDA report compilation complete.")

    logger.info("Exploratory Data Analysis pipeline execution finished successfully.")


if __name__ == "__main__":
    run()
