# VoltIQ: Machine Learning Feature Suitability & Leakage Audit

This document details the critical audit of engineered features conducted in **Phase 5 (Exploratory Data Analysis)**. It classifies all candidate inputs into three distinct categories, identifies subtle target proxy risks, and defines standardized benchmark feature tiers for future machine learning model training.

---

## 1. Executive Rationale & Motivation

In electric vehicle (EV) battery prognostics, an AI model trained on laboratory data can easily achieve unrealistically high test metrics ($R^2 > 0.99$) by exploiting **target proxies** or **laboratory-specific artifacts** rather than learning true physical degradation phenomena.

A feature is considered a **target proxy** if:
1. It is mathematically derived from the target variable ($Q$ or $\text{SOH}$).
2. Under experimental conditions, it is effectively equivalent to integrating the target (e.g., discharge duration under constant-current load).
3. It cannot be observed in an operational EV during partial charges/discharges.

---

## 2. Three-Tiered Feature Classification

### Group 1: Pure Sensor-Only Features (`SENSOR_ONLY_FEATURES`)
*Definition:* Features derived strictly from instantaneous or cycle-level voltage, current, temperature, and load telemetry.

- **Voltage Metrics:** `voltage_mean`, `voltage_median`, `voltage_min`, `voltage_max`, `voltage_std`, `voltage_range`
- **Current Metrics:** `current_mean`, `current_median`, `current_min`, `current_max`, `current_std`, `current_range`
- **Temperature Metrics:** `temperature_mean`, `temperature_median`, `temperature_min`, `temperature_max`, `temperature_std`, `temperature_range`
- **Load Metrics:** `current_load_mean`, `voltage_load_mean`
- **Thermal Dynamics:** `temperature_change_from_previous_cycle`

**Suitability:** ✅ **High / Primary Recommended Set.**
These features reflect real-time physical properties (ohmic drop, polarization, Joule heating) observable by an onboard Battery Management System (BMS). They carry zero mathematical dependence on total cycle capacity.

---

### Group 2: Cycle-Age Features (`CYCLE_AGE_FEATURES`)
*Definition:* Variables tracking chronological progression through the laboratory test.

- `cycle_number` (Original 1-based chronological index)
- `cycle_index` (Normalized sequential discharge count $1 \dots N$)

**Audit Findings & Warnings:**
- In laboratory aging under fixed cycling protocols, `cycle_index` strongly correlates with SOH ($r = -0.8943$).
- **Risk:** In an operational EV, battery aging depends on depth of discharge (DOD), temperature history, fast-charging frequency, and calendar resting. A model trained on `cycle_index` risks memorizing a laboratory decay curve rather than inferring electrochemical state.
- **Suitability:** ⚠️ **Secondary Benchmark Only.** Must be separated into a distinct evaluation tier to determine if the model is genuinely learning sensor dynamics or merely tracking cycle count.

---

### Group 3: Potential Target Proxies & Capacity-Derived Features (`TARGET_PROXY_FEATURES`)
*Definition:* Features derived from or strongly collinear with discharge capacity integration.

#### 1. `discharge_duration_seconds` & `measurement_count`:
- Under a constant-current discharge protocol ($I \approx -2.0\text{ A}$), total capacity is physically defined as:
  $$Q = \int_0^{t_{\text{end}}} -I(\tau) d\tau \approx |I| \times t_{\text{duration}}$$
- **Empirical Correlation:** `discharge_duration_seconds` correlates with capacity at **$r = 0.9755$** (B0005), **$r = 0.9931$** (B0006), **$r = 0.9820$** (B0007), and **$r = 0.9926$** (B0018).
- **Risk:** Total discharge duration is known **only after the battery is completely empty**. Feeding duration to an SOH estimator constitutes subtle target leakage.

#### 2. Incremental Capacity Analysis (`ica_peak_value`, `ica_peak_voltage`, `ica_area`):
- ICA ($dQ/dV$) is computed by differentiating cumulative capacity $Q(t)$ with respect to voltage $V$.
- `ica_area` ($\int_{3.4}^{3.8} \frac{dQ}{dV} dV$) directly represents the capacity discharged in that voltage window, correlating with SOH at **$r = 0.9877$**.
- While valuable for offline battery health diagnostics in maintenance depots, it requires accurate coulomb-counting integration across a broad voltage sweep.

**Suitability:** ⚠️ **Quarantined from Primary Sensor Set.** Reserved strictly for diagnostic benchmarking.

---

## 3. Recommended Benchmark Evaluation Tiers

To provide scientifically sound model comparisons in Phase 6, three evaluation tiers are defined in `src/feature_config.py`:

| Benchmark Tier | Feature Set Identifier | Included Features | Purpose & Real-World Realism |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Primary Baseline)** | `FEATURE_SET_TIER_1_SENSOR_ONLY` | 21 sensor statistics | **Operational BMS Model:** Pure physical signals, zero duration proxy, zero lab-cycle memorization. |
| **Tier 2 (Sensor + Age)** | `FEATURE_SET_TIER_2_SENSOR_PLUS_AGE` | 23 features (Tier 1 + `cycle_index`, `cycle_number`) | **Lab Benchmark:** Tests marginal predictive value of knowing laboratory cycle progression. |
| **Tier 3 (Full Informative)** | `FEATURE_SET_TIER_3_FULL_INFORMATIVE` | 30 features (Tier 2 + Duration + ICA metrics) | **Upper Bound:** Quantifies artificial performance inflation caused by proxy features. |
