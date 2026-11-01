# VoltIQ | Interactive Streamlit Dashboard User Guide

**Application Title:** VoltIQ | EV Battery Health Intelligence  
**Entry Point:** `app/app.py`  
**Execution Command:**  
```powershell
.\venv\Scripts\python.exe -m streamlit run app/app.py
```

---

## 1. Executive Summary & Design Aesthetics

The **VoltIQ Dashboard** is an interactive automotive data science web application designed to explore degradation phenomena, evaluate machine learning models, and demonstrate State of Health (SOH) inference using authentic cycling data from the **NASA Ames Prognostics Center of Excellence (PCoE)** repository.

### Visual Design & Aesthetics:
- **Color Theme:** Dark navy chassis (`#0B132B`, `#151D36`, `#1C2541`) with high-contrast electric cyan accents (`#00F0FF`, `#48CAE4`), emerald green (`#06D6A0`), amber orange (`#FFB703`), and crimson red (`#EF476F`).
- **Typography:** Modern clean sans-serif with high contrast and legible data labels.
- **Charts:** Interactive Plotly graphics with custom hover cards, zoom/pan capability, and synchronized color schemes across all four battery cells (`B0005`: Cyan, `B0006`: Amber, `B0007`: Emerald, `B0018`: Crimson).
- **Component Architecture:** Modular, lightweight design separated into `app/components/ui_cards.py` (custom styling and KPI cards) and `app/components/charts.py` (Plotly charting engine).

---

## 2. Walkthrough of Dashboard Sections

The application features five primary analytical views accessible via the sidebar navigation:

### A. Fleet Health & Dataset Overview
- **Fleet KPI Metrics:**
  - Battery Cells Analyzed: **4 Cells** (`B0005`, `B0006`, `B0007`, `B0018`).
  - Total Discharge Cycles: **636 Cycles** under 2.0A constant discharge load.
  - Average Fleet SOH: **82.85%** across all 636 cycles.
  - Lowest Observed SOH: **56.69%** (Cell B0006 at terminal cycle 168).
- **Fleet Degradation Trajectory Chart:**
  Interactive Plotly plot comparing SOH (%) vs discharge cycle index across all four cells, with an annotated **80% SOH End-of-Life (EOL)** boundary.
- **Dataset Context & Protocol Table:**
  Outlines commercial 18650 LCO specifications, room temperature (24°C) test conditions, and cutoff voltage differences (2.7V, 2.5V, 2.2V, 2.5V).

---

### B. Battery Health Explorer
- **Interactive Selector:** Dropdown to isolate individual cells (`B0005`, `B0006`, `B0007`, `B0018`).
- **Dynamic Cell KPIs:**
  - Initial Baseline Discharge Capacity ($C_{\text{initial}}$ Ah).
  - Final Measured Capacity ($C_{\text{final}}$ Ah) and Absolute Loss.
  - Final Observed SOH (%) and Capacity Fade (%).
  - Total Capacity Recovery Jumps ($> +0.02\text{ Ah}$) after rest.
- **Dual-Axis Degradation Plot:**
  Tracks SOH (%) and Discharge Capacity (Ah) simultaneously, with diamond markers highlighting **electrochemical capacity recovery events** occurring after prolonged rest and impedance spectroscopy.
- **Analytical Sub-Tabs:**
  1. *Sensor Dynamics:* Multi-panel tracking of Mean Discharge Voltage (V), Peak Cycle Temperature (°C), and Average Load Current (A) over aging.
  2. *Incremental Capacity Analysis (ICA):* $dQ/dV$ curves across early, mid-life, and end-of-life cycles showing phase transition peak attenuation around 3.75V.
  3. *Cycle Telemetry Table:* Scrollable tabular view of raw cycle features for granular inspection.

---

### C. Model Performance & Validation
- **Benchmark Matrix:**
  Comprehensive multi-tier table summarizing all 5 regression architectures across the 3 audited feature tiers (15 evaluated models):
  - Tier 1 (Sensor-Only Baseline — 21 features).
  - Tier 2 (Sensor + Cycle Age — 23 features).
  - Tier 3 (Full Diagnostic — 30 features).
- **Selected Experimental Model:**
  - **Random Forest Regressor (Tier 1 Experimental Sensor-Only):** Highlighted with KPI cards showing pooled Out-of-Fold RMSE = **3.711%**, MAE = **2.991%**, and R² = **0.885**.
- **Battery-Wise Validation Table (Leave-One-Battery-Out):**
  Breaks down holdout RMSE, MAE, R², and MAPE for each individual held-out battery cell, showing consistent generalization across all four cells:
  - Held-out `B0005`: RMSE = **2.363%**, MAE = **1.993%**, R² = **0.947**
  - Held-out `B0006`: RMSE = **5.174%**, MAE = **4.170%**, R² = **0.824**
  - Held-out `B0007`: RMSE = **3.400%**, MAE = **3.126%**, R² = **0.840**
  - Held-out `B0018`: RMSE = **3.237%**, MAE = **2.592%**, R² = **0.849**
- **Interactive True vs. Predicted SOH Plot:**
  Scatter plot of out-of-fold predictions against ground truth with a 1:1 reference line.
- **Feature Importance Ranking:**
  Horizontal bar chart displaying the top sensor features by Mean Decrease in Impurity (MDI), demonstrating that `voltage_mean` (71.86%) and `voltage_median` (16.25%) dominate health estimation.
- **Educational Explainer Boxes:**
  - Explains the **Cluster Effect** ($N_{\text{cells}} = 4$) and why `GroupKFold` is essential.
  - Explains why the DummyRegressor baseline produced negative R² ($R^2 = -0.127$).

---

### D. SOH Prediction Demo (Historical Cycle Estimation)
- **Interactive Cycle Selector:**
  Users select a battery cell and move a slider to pick any of its historical discharge cycles ($1 \dots N$).
- **Live Model Inference:**
  The saved Tier 1 Random Forest model (`models/tier1_randomforest_model.joblib`) is invoked in real time via `SOHPredictor` to predict SOH using only the cycle's 21 sensor-only features.
- **Visual Health Display:**
  - Circular Plotly SOH Gauge with color thresholds (Green: $>90\%$, Yellow: $80\% - 90\%$, Red: $<80\%$).
  - KPI cards showing Model Predicted SOH, Actual Ground Truth SOH, Absolute Error, and Health State Classification.
- **Input Feature Inspector:**
  Displays the exact 21 sensor features fed into the model for the selected cycle.
- **Prominent Demonstration Disclaimer:**
  Explicitly clarifies that this is a historical evaluation demo using existing labeled laboratory data, NOT a live EV telemetry feed or forecasting tool.

---

### E. About & Methodology
- **Electrochemical Principles:**
  Documents formulas for SOH, Capacity Fade, and Capacity Recovery.
- **Data Leakage Quarantine:**
  Explains why discharge duration ($Q = I \times t$) was quarantined as a target proxy.
- **Scientific Limitations:**
  Explicitly warns that the model is trained on only four 18650 cells under constant room temperature cycling and cannot be claimed production-ready for commercial EVs without dynamic drive-cycle and fleet-scale validation.

---

## 3. Technical Architecture & Performance Optimization

1. **In-Memory Caching:**
   - Data loading is wrapped in `@st.cache_data`, ensuring CSV files are read only once upon application launch.
   - Model artifact loading is wrapped in `@st.cache_resource`, keeping the trained `SOHPredictor` pipeline in memory for instant millisecond inference.
2. **Zero Model Retraining:**
   The dashboard strictly loads pre-computed metrics and pre-trained `.joblib` model artifacts. No training or hyperparameter searching occurs within the dashboard.
3. **Data Protection:**
   The underlying raw `.mat` files and processed CSVs are accessed in read-only mode and are never modified by the dashboard.

---

## 4. Local Verification & Launch

To launch the dashboard on a Windows development system:
```powershell
# Open terminal in VoltIQ project root
cd <path-to-VoltIQ>

# Launch the Streamlit application
.\venv\Scripts\python.exe -m streamlit run app/app.py
```
The application will start on local port `http://localhost:8501`.
