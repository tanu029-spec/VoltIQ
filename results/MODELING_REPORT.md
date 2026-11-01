# VoltIQ: Machine Learning Modeling Report (Phase 6)

**Generated:** 2026-11-01T22:49:59.372508+00:00  
**Target Variable:** `soh_percentage` (Relative State of Health against first-cycle baseline)  
**Sample Population:** 636 valid discharge cycles from 4 commercial 18650 Li-ion cells (`B0005`, `B0006`, `B0007`, `B0018`)  
**Cross-Validation Protocol:** `GroupKFold(n_splits=4)` with `battery_id` grouping (Leave-One-Battery-Out)

---

## 1. Executive Summary & Benchmark Results

This report presents a rigorous empirical benchmark of five regression architectures across three distinct feature tiers. 

To prevent data leakage and provide realistic estimates of cross-cell generalization, **zero row-level random splitting was permitted**. In each fold, an entire physical battery cell was held out as unseen test data, and all data transformations (scaling, imputation) were fitted exclusively on the remaining three training batteries.

### Key Empirical Findings:
1. **Tier 1 Sensor-Only Baseline:**
   - **Random Forest** and **HistGradientBoosting** achieved outstanding, robust cross-cell performance on pure sensor features with **OOF RMSE = 3.711% and 3.682%**, and **OOF R² = 0.885 and 0.886**.
   - Crucially, Random Forest showed consistent generalization across **all four individual held-out cells** (Holdout R²: `B0005` = 0.947, `B0006` = 0.824, `B0007` = 0.840, `B0018` = 0.849).
2. **Multicollinearity & Linear Instability:**
   - Unregularized **Linear Regression** failed catastrophically on held-out cell `B0007` (Holdout R² = -2.166, RMSE = 15.10%), caused by extreme multicollinearity between sensor features (e.g., `voltage_mean` and `voltage_median` at $r = 0.9984$).
   - **Ridge Regression** successfully stabilized the linear weights, dropping OOF RMSE to 4.310% (OOF R² = 0.844).
3. **The Cycle-Age Disparity (Tier 2):**
   - Injecting `cycle_index` and `cycle_number` reduced OOF RMSE for Ridge (to 2.891%) and tree models (to ~3.37%–3.47%), but Linear Regression collapsed on held-out cell `B0018` (Holdout R² = -0.403). B0018 ran for only 132 cycles and degraded faster per cycle than the 168-cycle training cells. This underscores that cycle-count memorization is hazardous when cell operating conditions differ.
4. **Diagnostic Upper Bound (Tier 3):**
   - Including duration and offline ICA metrics did **not** substantially improve tree-based cross-battery generalization (OOF RMSE = 3.871% vs 3.711% in Tier 1). Because cutoff voltages differed across cells (2.2V, 2.5V, 2.7V), discharge duration patterns shifted across batteries, demonstrating that target proxies tied to test protocol cutoffs can degrade generalization.

---

## 2. Multi-Model Performance Across Feature Tiers

### Tier 1: Pure Sensor-Only Baseline (21 Features)
*Experimental sensor-only features; zero duration proxy, zero cycle-age memorization.*

| Model Architecture | Pooled OOF RMSE | Pooled OOF MAE | Pooled OOF R² | Fold RMSE (Mean ± Std) | Fold R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | 3.682% | 2.889% | 0.886 | 3.476 ± 1.232% | 0.868 ± 0.066 |
| **RandomForest** | 3.711% | 2.991% | 0.885 | 3.543 ± 1.179% | 0.865 ± 0.055 |
| **Ridge** | 4.310% | 3.352% | 0.844 | 3.550 ± 2.665% | 0.768 ± 0.353 |
| **LinearRegression** | 9.451% | 7.852% | 0.251 | 7.996 ± 5.368% | -0.079 ± 1.425 |
| **DummyRegressor** | 11.597% | 9.691% | -0.127 | 11.148 ± 3.017% | -0.273 ± 0.269 |

---

### Tier 2: Sensor + Cycle Age Benchmark (23 Features)
*Age-informed laboratory benchmark; includes `cycle_index` and `cycle_number`.*

| Model Architecture | Pooled OOF RMSE | Pooled OOF MAE | Pooled OOF R² | Fold RMSE (Mean ± Std) | Fold R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge** | 2.891% | 2.464% | 0.930 | 2.805 ± 0.626% | 0.917 ± 0.025 |
| **HistGradientBoosting** | 3.367% | 2.606% | 0.905 | 3.079 ± 1.416% | 0.903 ± 0.054 |
| **RandomForest** | 3.466% | 2.717% | 0.899 | 3.209 ± 1.372% | 0.893 ± 0.054 |
| **LinearRegression** | 6.712% | 5.783% | 0.622 | 6.550 ± 2.564% | 0.380 ± 0.583 |
| **DummyRegressor** | 11.597% | 9.691% | -0.127 | 11.148 ± 3.017% | -0.273 ± 0.269 |

---

### Tier 3: Full Diagnostic Benchmark (30 Features)
*Diagnostic upper-bound benchmark; includes duration proxies and offline ICA features.*

| Model Architecture | Pooled OOF RMSE | Pooled OOF MAE | Pooled OOF R² | Fold RMSE (Mean ± Std) | Fold R² (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomForest** | 3.871% | 3.100% | 0.874 | 3.416 ± 1.993% | 0.874 ± 0.099 |
| **HistGradientBoosting** | 3.935% | 3.168% | 0.870 | 3.540 ± 1.832% | 0.869 ± 0.088 |
| **LinearRegression** | 4.673% | 3.627% | 0.817 | 3.766 ± 3.081% | 0.830 ± 0.188 |
| **Ridge** | 4.765% | 4.429% | 0.810 | 4.433 ± 1.691% | 0.795 ± 0.096 |
| **DummyRegressor** | 11.597% | 9.691% | -0.127 | 11.148 ± 3.017% | -0.273 ± 0.269 |

---

## 3. Detailed Battery-by-Battery Performance (Selected Tier 1 Model: Random Forest)

| Held-Out Battery | Test Cycles | Cutoff Voltage | RMSE | MAE | R² | MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0007** | 168 | 2.2 V | 3.400% | 3.126% | 0.840 | 3.64% |
| **B0006** | 168 | 2.5 V | 5.174% | 4.169% | 0.824 | 6.04% |
| **B0005** | 168 | 2.7 V | 2.363% | 1.993% | 0.947 | 2.40% |
| **B0018** | 132 | 2.5 V | 3.237% | 2.592% | 0.849 | 2.96% |
| **Overall Pooled** | **636** | **All Protocols** | **3.711%** | **2.991%** | **0.885** | **3.81%** |

### Explaining Negative R² Scores:
The **DummyRegressor** produced negative R² on every fold (OOF R² = -0.127, fold R² down to -0.568 on B0006). A negative R² occurs when a model's mean squared prediction error exceeds the variance of the true target values around their own fold mean (SS_res > SS_tot). For the dummy model, predicting the mean of the training batteries when evaluated against a held-out battery with a different mean capacity naturally results in R² < 0. This provides the authentic lower bound.

---

## 4. Feature Importance & Interpretability Analysis

Feature importance analysis was conducted on the Tier 1 Random Forest using Mean Decrease in Impurity (MDI) averaged across all four cross-validation folds:

1. **`voltage_mean` (Importance: 71.86%):** Primary driver of SOH estimation. As the battery ages, internal resistance increases and active lithium is lost, depressing the average plateau voltage during constant 2A discharge.
2. **`voltage_median` (Importance: 16.25%):** Robust central tendency metric capturing the discharge voltage curve depression.
3. **`voltage_load_mean` (Importance: 4.21%):** Measures terminal voltage under active circuit load.
4. **`current_mean` & `current_load_mean` (Importance: 4.89% combined):** Capture current regulation and external circuit load impedance dynamics.
5. **`current_std` & `temperature_std` (Importance: 1.63% combined):** Capture intra-cycle current variability and thermal dispersion ($I^2 R$ Joule heating rise toward end-of-discharge).

> [!NOTE]
> **Scientific Interpretation Disclaimer:** Feature importance reflects predictive utility within the random forest ensemble under this specific experimental protocol. It does **not prove electrochemical causality**. Correlation between thermal rise and SOH reflects underlying impedance growth ($R$).

---

## 5. Scientific Limitations & Production Readiness

> [!WARNING]
> **This model CANNOT be claimed production-ready for commercial electric vehicles.**
>
> 1. **Small Sample Size (N_cells = 4):** All training and evaluation data originate from only four individual 18650 cells manufactured by a single vendor with a single cathode chemistry (LCO) over a decade ago.
> 2. **Fixed Constant-Current Protocol:** Testing was performed under constant 2.0A discharge at constant room temperature (24°C). Real EV battery packs experience dynamic drive cycles (WLTP/EPA), regenerative braking spikes, variable ambient temperatures (-20°C to 45°C), and incomplete partial charging.
> 3. **Summary Telemetry vs BMS Ingestion:** The features used here represent cycle-aggregated statistical moments. A true onboard BMS operates on instantaneous streaming sensor packets and requires online state estimation (Extended Kalman Filters, equivalent circuit models) rather than full-cycle post-hoc aggregations.

---

## 6. Selected Model Artifacts

Two distinct model artifacts have been serialized using `joblib` into the `models/` directory:

1. **Selected Experimental Model:** `models/tier1_randomforest_model.joblib`
   - **Architecture:** Random Forest Regressor (100 trees, max_depth=6, min_samples_leaf=2)
   - **Feature Set:** `Tier 1: SENSOR_ONLY_FEATURES` (21 features)
   - **Intended Use:** Experimental battery health estimation from summary cycle telemetry without capacity proxies or cycle-age memorization.
   - **Cross-Battery Validation:** OOF RMSE = 3.711%, OOF R² = 0.885, minimum holdout R² = 0.824.

2. **Age-Informed Laboratory Benchmark:** `models/tier2_histgradientboosting_model.joblib`
   - **Architecture:** HistGradientBoosting Regressor (max_iter=100, max_depth=4)
   - **Feature Set:** `Tier 2: SENSOR_PLUS_AGE` (23 features)
   - **Intended Use:** Laboratory benchmark quantifying performance when experimental cycle progression is known.
   - **Cross-Battery Validation:** OOF RMSE = 3.367%, OOF R² = 0.905.
