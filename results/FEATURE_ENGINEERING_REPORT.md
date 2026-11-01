# VoltIQ: Feature Engineering & Battery Health Report

**Generated:** 2026-11-01T22:20:25.771645+00:00  
**Status:** ✅ Feature Derivation & Leakage Validation Complete

---

## 1. Executive Summary

Phase 4 of the VoltIQ pipeline derived domain-informed battery health indicators, comprehensive sensor statistical aggregations, cycle-to-cycle degradation metrics, and advanced Incremental Capacity Analysis (ICA) features from the cleaned NASA dataset.

| Metric | Result |
| :--- | :--- |
| **Total Processed Records** | **636** discharge cycles |
| **Batteries Accounted For** | **4** (`B0005`, `B0006`, `B0007`, `B0018`) |
| **Total Features Engineered** | **47** columns |
| **Permitted Machine Learning Inputs** | **30** features (`ML_FEATURE_COLUMNS`) |
| **Target Variables & Leakage Exclusions** | **10** columns strictly quarantined |
| **High-Resolution ICA Curve Records** | **51,516** points across 81 voltage intervals |

---

## 2. Battery-Wise Health Indicators & SOH Degradation

Baseline initial capacity was established individually using the first valid chronological discharge cycle for each cell, rather than assuming a generic 2.0 Ah rated capacity:

| Battery ID | Discharge Cycles | Measured Initial Capacity | Final Observed Capacity | SOH Range | Total Recoveries (> 0.02 Ah) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0005** | 168 | 1.8565 Ah | 1.3251 Ah | 69.35% – 100.00% | 7 |
| **B0006** | 168 | 2.0353 Ah | 1.1857 Ah | 56.69% – 100.00% | 11 |
| **B0007** | 168 | 1.8911 Ah | 1.4325 Ah | 74.06% – 100.00% | 7 |
| **B0018** | 132 | 1.8550 Ah | 1.3411 Ah | 72.29% – 100.00% | 9 |

---

## 3. Incremental Capacity Analysis (ICA: dQ/dV) Feasibility & Findings

1. **Feasibility:**
   ICA was successfully and reliably extracted using uniform voltage grid interpolation (10 mV steps between 4.00V and 3.20V) coupled with mild Gaussian smoothing ($\\sigma=2.0$).
2. **Physical Findings:**
   - **Peak Height Attenuation:** For fresh cells (Cycle 2), the primary $dQ/dV$ peak reaches $\\sim 5.1\\text{ Ah/V}$. As cycling progresses to EOL, this peak steadily decreases to $\\sim 3.7\\text{ Ah/V}$, quantifying Loss of Active Material (LAM) and Loss of Lithium Inventory (LLI).
   - **Peak Potential Shift:** The voltage peak position shifts toward lower voltages (from $3.50\\text{V}$ down to $3.47\\text{V}$), directly reflecting ohmic and charge-transfer resistance growth.
3. **Data Limitations Handled:**
   - Raw consecutive measurements contain repeated voltages ($\\Delta V = 0$) and small measurement fluctuations that produce division-by-zero errors when naively differentiated. The uniform grid interpolation completely resolves this numerical instability without fabricating data.

---

## 4. Machine Learning Target Leakage Prevention

To ensure strict validity in subsequent Machine Learning modeling, features are strictly partitioned:

- **Permitted Inputs (`ML_FEATURE_COLUMNS`):**
  Only features observable in an operational vehicle (terminal voltage statistics, current profile dispersion, temperature dynamics, cycle duration, sampling intervals, and derived ICA indicators).
- **Quarantined Leakage Columns (`LEAKAGE_COLUMNS`):**
  `discharge_capacity_ah`, `initial_capacity_ah`, `soh_percentage`, `capacity_fade_percentage`, `capacity_loss_ah`, `relative_capacity_retention`, `capacity_change_ah`, `capacity_change_percentage`, `soh_change_percentage`, `capacity_recovery_flag`.
- **Metadata Grouping:**
  `battery_id` is retained strictly for GroupKFold cross-validation partitioning and is quarantined from model input matrices to prevent cell ID memorization.
