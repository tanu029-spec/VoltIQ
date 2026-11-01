# ⚡ VoltIQ: EV Battery Health & Intelligence System

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Data Science and Machine Learning engineering application that analyzes lithium-ion battery aging dynamics, estimates battery **State of Health (SOH)**, studies degradation mechanisms across charge-discharge cycles, and delivers interactive analytics through an interactive Streamlit dashboard.

---

## 📌 Project Overview & Objectives

In electric vehicles (EVs) and energy storage systems (ESS), battery degradation directly impacts driving range, reliability, safety, and residual asset valuation. Lithium-ion cells undergo irreversible chemical and mechanical degradation during cycling—such as loss of lithium inventory (LLI) and loss of active material (LAM)—which causes capacity fade and internal resistance escalation.

**VoltIQ delivers:**
1. **Multi-Cycle Telemetry Extraction:** Ingestion of raw MATLAB `.mat` cycle data (charging CC/CV, discharging under constant load, and electrochemical impedance spectroscopy).
2. **Feature Engineering & Health Indicators (HI):** Extraction of SOH, degradation rate curves, terminal voltage decay, thermal rise, and incremental capacity features ($dQ/dV$).
3. **SOH Regression Modeling:** Machine Learning models (Linear, Tree-based ensembles, Support Vector Regression) trained to accurately predict SOH across cycle life.
4. **Interactive Intelligence Dashboard:** Streamlit-powered analytical platform providing battery comparison, degradation trend projections, and real-time cycle health inference.

---

## 🔬 Dataset: NASA Ames Battery Aging Repository

This project utilizes experimental degradation data provided by the **NASA Ames Prognostics Center of Excellence (PCoE)**:
- **Source:** [NASA Li-Ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- **Cell Specifications:** Commercial 18650-size Lithium-ion cells (nominal rated capacity: **2.0 Ah**, nominal voltage: **3.7 V**).
- **Core Experimental Battery Cells:**
  - `B0005`: Cycled at room temperature (24°C) with 2A discharge current down to 2.7V cutoff.
  - `B0006`: Cycled at room temperature (24°C) with 2A discharge current down to 2.5V cutoff.
  - `B0007`: Cycled at room temperature (24°C) with 2A discharge current down to 2.2V cutoff.
  - `B0018`: Cycled at room temperature (24°C) with accelerated degradation cycling down to 2.5V cutoff.
- **End-of-Life (EOL) Criterion:** 30% fade in rated capacity (reaching 1.4 Ah or 70–80% SOH).

---

## 📁 Repository Structure

```text
VoltIQ/
├── data/
│   ├── raw/                # NASA raw .mat files (B0005.mat, B0006.mat, etc.)
│   └── processed/          # Clean tabular datasets (Parquet / CSV)
├── notebooks/              # Jupyter notebooks for interactive research & EDA
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── data_loader.py      # .mat file parser and cycle extractor
│   ├── preprocessing.py    # Cycle cleaning and tabular structuring
│   ├── feature_engineering.py # Health indicators and SOH calculation
│   ├── eda.py              # Degradation curves and statistical analysis
│   ├── train.py            # Model training & hyperparameter tuning
│   ├── evaluation.py       # Regression metrics (RMSE, MAE, R², MAPE)
│   └── predict.py          # Inference pipeline for new cycle inputs
├── scripts/                # Execution and automation scripts
├── models/                 # Serialized model artifacts (.joblib)
├── results/
│   ├── figures/            # Exported degradation and evaluation figures
│   └── metrics/            # Model performance reports (JSON/CSV)
├── app/
│   ├── app.py              # Streamlit dashboard entry point
│   └── components/         # Reusable UI cards, tables, and chart components
├── tests/                  # Unit and integration test suite
├── docs/                   # Engineering documentation and data dictionaries
├── requirements.txt        # Production dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

---

## 🛠️ Technology Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.13+ |
| **Data Processing & SciPy** | Pandas, NumPy, SciPy (`scipy.io.loadmat`) |
| **Machine Learning** | Scikit-learn, Joblib |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Testing & Quality** | Pytest |
| **Version Control** | Git & GitHub |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python (3.11+ / 3.13 recommended) and Git are installed.

### 2. Clone and Setup Environment
```powershell
# Open terminal in project root
git clone https://github.com/<your-username>/VoltIQ.git
cd VoltIQ

# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install -r requirements.txt
```

### 3. Placing NASA Dataset Files
Place the downloaded MATLAB dataset files directly into the raw data directory:
```text
VoltIQ/data/raw/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

---

## 📈 Planned Development Roadmap

1. [x] **Phase 1: Environment & Architecture Setup** (Directory scaffolding, virtual environment, dependency management, git setup).
2. [ ] **Phase 2: Data Ingestion & Extraction** (Load `.mat` structures, extract discharge/charge time-series, map cycle metadata).
3. [ ] **Phase 3: Data Preprocessing & Cleaning** (Handle anomalies, timestamp alignment, structured tabular representation).
4. [ ] **Phase 4: Feature Engineering & Health Indicator Derivation** (SOH labeling, ICA curves, thermal/voltage decay statistics).
5. [ ] **Phase 5: Exploratory Data Analysis (EDA)** (Capacity fade visualization, comparative cycle decay, correlation analysis).
6. [ ] **Phase 6: Machine Learning Modeling** (Model training, cross-validation across battery cells, hyperparameter optimization).
7. [ ] **Phase 7: Evaluation & Model Serialization** (RMSE, MAE, R², model export).
8. [ ] **Phase 8: Interactive Streamlit Dashboard** (Real-time SOH estimation, interactive telemetry charts, battery diagnostic reports).

---

## 📄 License
This project is open-source under the MIT License.
