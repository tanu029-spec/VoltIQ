# VoltIQ: Feature Dictionary

This document details the engineered battery health features, mathematical formulas, physical units, descriptions, source measurements, and strict Machine Learning eligibility rules to prevent target leakage.

---

## 1. Primary Regression Target Variables (Never Inputs for ML)

| Feature Name | Formula / Definition | Unit | Description | Suitable for ML Input? |
| :--- | :--- | :--- | :--- | :--- |
| `soh_percentage` | $(C_{\text{discharge}} / C_{\text{initial}}) \times 100$ | % | Primary target. State of Health of the cell. | ❌ NO (Target) |
| `discharge_capacity_ah`| Integrated discharge load | Ah | Direct target proxy. Measured capacity. | ❌ NO (Target Leakage) |
| `capacity_fade_percentage`| $100.0 - \text{SOH}$ | % | Direct mathematical inverse of SOH. | ❌ NO (Target Leakage) |
| `capacity_loss_ah` | $C_{\text{initial}} - C_{\text{discharge}}$ | Ah | Absolute capacity degradation from initial baseline. | ❌ NO (Target Leakage) |
| `relative_capacity_retention`| $C_{\text{discharge}} / C_{\text{initial}}$ | Ratio [0, 1] | Direct ratio representation of SOH. | ❌ NO (Target Leakage) |
| `capacity_change_ah` | $C_k - C_{k-1}$ | Ah | Cycle-to-cycle capacity delta. | ❌ NO (Target Leakage) |
| `capacity_change_percentage`| $(C_k - C_{k-1}) / C_{k-1} \times 100$ | % | Percentage capacity change from previous cycle. | ❌ NO (Target Leakage) |
| `soh_change_percentage`| $\text{SOH}_k - \text{SOH}_{k-1}$ | % | Delta SOH from previous cycle. | ❌ NO (Target Leakage) |
| `capacity_recovery_flag`| $C_k - C_{k-1} > 0.02\text{ Ah}$ | Boolean | Indicator for capacity jumps during rest periods. | ❌ NO (Target Leakage) |

---

## 2. Permitted Machine Learning Features (`ML_FEATURE_COLUMNS`)

These features can be extracted from BMS voltage, current, and temperature telemetry in an operational vehicle without prior knowledge of cell capacity.

### A. Cycle Progression
| Feature Name | Definition | Unit | Description |
| :--- | :--- | :--- | :--- |
| `cycle_number` | Raw experiment cycle index | Count | Chronological 1-based cycle index in NASA dataset. |
| `cycle_index` | Sequential discharge count | Count | Normalized sequential cycle count (1 to $N$). |

### B. Terminal Voltage Features
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `voltage_mean` | $\frac{1}{N}\sum V_t$ | V | Mean terminal voltage during discharge. |
| `voltage_median` | $\text{Median}(V_t)$ | V | Median terminal voltage. |
| `voltage_min` | $\min(V_t)$ | V | Cutoff / terminal minimum voltage reached. |
| `voltage_max` | $\max(V_t)$ | V | Open-circuit / initial discharge voltage. |
| `voltage_std` | $\text{Std}(V_t)$ | V | Standard deviation of voltage trajectory. |
| `voltage_range`| $V_{\max} - V_{\min}$ | V | Total voltage span traversed during discharge. |

### C. Current Telemetry Features
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `current_mean` | $\frac{1}{N}\sum I_t$ | A | Mean measured discharge current (~ -2.0 A). |
| `current_median`| $\text{Median}(I_t)$ | A | Median measured current. |
| `current_min` | $\min(I_t)$ | A | Peak negative load surge. |
| `current_max` | $\max(I_t)$ | A | Standby / rest current prior to load switch. |
| `current_std` | $\text{Std}(I_t)$ | A | Current stability dispersion. |
| `current_range`| $I_{\max} - I_{\min}$ | A | Total current span during cycle. |

### D. Operating Temperature Features
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `temperature_mean` | $\frac{1}{N}\sum T_t$ | °C | Average cell surface temperature. |
| `temperature_median`| $\text{Median}(T_t)$ | °C | Median cell temperature. |
| `temperature_min` | $\min(T_t)$ | °C | Initial temperature before Joule heating (~24°C). |
| `temperature_max` | $\max(T_t)$ | °C | Maximum thermal peak achieved during discharge. |
| `temperature_std` | $\text{Std}(T_t)$ | °C | Thermal dispersion over the cycle. |
| `temperature_range`| $T_{\max} - T_{\min}$ | °C | Total thermal elevation during discharge ($\Delta T$). |

### E. Load Telemetry Features
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `current_load_mean` | $\frac{1}{N}\sum I_{\text{load}}$ | A | Mean programmable electronic load current. |
| `voltage_load_mean` | $\frac{1}{N}\sum V_{\text{load}}$ | V | Mean voltage sensed across load terminals. |

### F. Temporal & Sampling Features
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `discharge_duration_seconds`| $\max(t)$ | s | Total elapsed discharge time to cutoff. |
| `measurement_count` | $N$ | Count | Total high-frequency telemetry samples recorded. |
| `average_sampling_interval` | $\Delta t_{\text{avg}} = \frac{t_{\max}}{N-1}$ | s | Average sampling interval between telemetry frames. |
| `cycle_duration_change` | $\text{Dur}_k - \text{Dur}_{k-1}$ | s | Cycle-to-cycle change in discharge duration. |
| `temperature_change_from_previous_cycle` | $T_{\text{mean}, k} - T_{\text{mean}, k-1}$ | °C | Cycle-to-cycle change in average thermal response. |

### G. Incremental Capacity Analysis (ICA: $dQ/dV$)
| Feature Name | Formula | Unit | Description |
| :--- | :--- | :--- | :--- |
| `ica_peak_value` | $\max(dQ/dV)$ in $[3.35\text{V}, 3.75\text{V}]$ | Ah/V | Amplitude of the primary phase-transition peak. |
| `ica_peak_voltage` | $\text{argmax}_V(dQ/dV)$ | V | Voltage position of the primary peak. |
| `ica_area` | $\int_{3.4}^{3.8} (dQ/dV) dV$ | Ah | Integrated plateau area under the $dQ/dV$ curve. |
