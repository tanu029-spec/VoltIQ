# VoltIQ: Exploratory Data Analysis (EDA) Report

**Generated:** 2026-11-01T22:34:46.607098+00:00  
**Dataset Scope:** Authentic NASA Ames PCoE Lithium-Ion Battery Aging Dataset (`B0005`, `B0006`, `B0007`, `B0018`)  
**Sample Size:** 636 valid discharge cycles, 185,721 measurement points across 4 distinct cells.  

---

## 1. Executive Summary

This Exploratory Data Analysis (EDA) evaluates degradation kinetics, sensor telemetry dynamics, and feature relationships across the four commercial 18650 Li-ion cells in the NASA repository.

### Key Numerical Findings:
1. **Initial Baseline Capacities:** Varied from **1.8550 Ah** (`B0018`) to **2.0353 Ah** (`B0006`). SOH was calibrated individually to each cell's initial discharge cycle, establishing an exact 100.0% starting baseline.
2. **SOH Fade Ranges:**
   - `B0005`: 100.0% $\to$ 71.38% (28.62% fade over 168 cycles, cutoff: 2.7V)
   - `B0006`: 100.0% $\to$ 58.25% (41.75% fade over 168 cycles, cutoff: 2.5V)
   - `B0007`: 100.0% $\to$ 75.75% (24.25% fade over 168 cycles, cutoff: 2.2V)
   - `B0018`: 100.0% $\to$ 72.29% (27.71% fade over 132 cycles, cutoff: 2.5V)
3. **Electrochemical Capacity Recovery Jumps:** **34 distinct recovery events** (> +0.02 Ah) occurred after prolonged rest and EIS testing intervals (`B0005`: 7, `B0006`: 11, `B0007`: 7, `B0018`: 9).
4. **Sensor Correlation with SOH:**
   - Strongest positive sensor correlation: `voltage_mean` ($r = +0.9550$), `voltage_median` ($r = +0.9438$).
   - Strongest inverse thermal correlation: `temperature_std` ($r = -0.8770$), `temperature_range` ($r = -0.6591$).
5. **Critical Target Proxy Discovery:**
   - `discharge_duration_seconds` correlates with capacity at **$r = 0.9523$** overall ($r > 0.975$ within every individual cell).
   - Under constant-current discharge ($Q = I \times t$), duration directly integrates capacity. It represents a subtle target proxy and is quarantined from primary ML sensor inputs.

---

## 2. Descriptive Statistics Summary

### Global Feature Distribution (N = 636 Cycles)

| Feature | Mean | Median | Std Dev | Minimum | 25th % | 75th % | Maximum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **State of Health (SOH %)** | 82.85% | 82.32% | 10.93% | 56.69% | 74.81% | 92.53% | 100.00% |
| **Discharge Capacity (Ah)** | 1.5817 | 1.5597 | 0.1988 | 1.1538 | 1.4211 | 1.7635 | 2.0353 |
| **Mean Voltage (V)** | 3.4986 | 3.5025 | 0.0475 | 3.4039 | 3.4667 | 3.5411 | 3.5736 |
| **Peak Temperature (°C)** | 39.57 | 39.67 | 1.44 | 36.37 | 38.37 | 40.87 | 42.33 |
| **Discharge Duration (s)** | 3117.0 | 3084.3 | 242.2 | 2742.8 | 2892.0 | 3311.8 | 3690.2 |
| **ICA Peak Height (Ah/V)** | 3.923 | 3.769 | 0.981 | 2.103 | 3.079 | 4.894 | 5.922 |

---

## 3. Battery Degradation & Observational Comparisons

| Battery ID | Discharge Cycles | Cutoff Voltage | Initial Capacity | Final Capacity | Absolute Loss | Average SOH Loss / Cycle | Peak Temp | Recoveries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0005** | 168 | 2.7 V | 1.8565 Ah | 1.3251 Ah | 0.5314 Ah | 0.171% | 41.45°C | 7 |
| **B0006** | 168 | 2.5 V | 2.0353 Ah | 1.1857 Ah | 0.8497 Ah | 0.250% | 42.01°C | 11 |
| **B0007** | 168 | 2.2 V | 1.8911 Ah | 1.4325 Ah | 0.4586 Ah | 0.145% | 42.33°C | 7 |
| **B0018** | 132 | 2.5 V | 1.8550 Ah | 1.3411 Ah | 0.5140 Ah | 0.211% | 38.88°C | 9 |

### Observational Findings on Cell B0018:
- **Cycle Count:** B0018 underwent 132 discharge cycles, compared to 168 cycles for B0005, B0006, and B0007.
- **Degradation Velocity:** Despite running for 36 fewer cycles, B0018 experienced an average SOH loss of **0.211% per cycle**, higher than B0005 (0.171%/cycle) and B0007 (0.145%/cycle).
- **Impedance Interleaving:** B0018 experienced only 53 impedance cycles compared to 278 impedance cycles in the other cells, reflecting a different test schedule.
- *Caution:* We report these as observed empirical differences in the recorded test data without asserting speculative chemical hypotheses.

---

## 4. Correlation & Collinearity Analysis

### Features Positively Correlated with SOH:
- `voltage_mean`: $r = +0.9550$ (Spearman $\rho = +0.9701$)
- `voltage_median`: $r = +0.9438$ (Spearman $\rho = +0.9582$)
- `ica_area`: $r = +0.9877$ (Spearman $\rho = +0.9863$)
- `ica_peak_value`: $r = +0.9701$ (Spearman $\rho = +0.9723$)

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
