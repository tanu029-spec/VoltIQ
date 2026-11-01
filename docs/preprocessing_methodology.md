# VoltIQ: Data Preprocessing & Anomaly Detection Methodology

This document outlines the engineering rationale, statistical rules, and domain-informed validation criteria applied during Phase 3 of the VoltIQ pipeline.

---

## 1. Guiding Principles

1. **Preservation of Raw & Extracted Datasets:**
   The original extracted datasets (`data/processed/battery_cycles.csv` and `data/processed/battery_measurements.csv`) remain completely untouched to maintain reproducibility and lineage back to the raw MATLAB archives.
2. **Non-Destructive Anomaly Handling:**
   In battery prognostics, degradation phenomena (capacity fade, accelerated voltage drop, internal heating) can easily be misclassified as "anomalies" by naive statistical filters. Rather than deleting these critical data points, the pipeline attaches boolean diagnostic flags to each observation.
3. **Deterministic Type Enforcement:**
   All features are cast to standardized, memory-efficient numeric and categorical types (`float64`, `int64`, `string`) and sorted chronologically.

---

## 2. Validation Checks

The preprocessing pipeline audits the dataset across 6 primary dimensions:
- **Schema & Dimension Verification:** Confirms all expected fields are present with appropriate non-null row counts.
- **Null and Infinite Checks:** Verifies zero NaN, Null, or infinite values in numeric time series.
- **Uniqueness Audits:** Ensures primary composite keys `(battery_id, cycle_number)` for cycles and `(battery_id, cycle_number, time_seconds)` for measurements have zero collisions.
- **Temporal Monotonicity:** Verifies elapsed time is strictly non-decreasing within each discharge cycle ($\Delta t \ge 0$).
- **Cross-Table Array Alignment:** Ensures cycle-level sample counts precisely equal the number of telemetry rows present in the measurement table.
- **Physical Boundary Verification:** Validates that capacity, voltage, current, and temperature fall within realistic physical limits for 18650 Li-ion cells.

---

## 3. Anomaly Tagging Heuristics

### Cycle-Level Flags:
- **`is_capacity_iqr_outlier`:** Statistical outlier based on Tukey's Interquartile Range ($Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}$) calculated per battery.
- **`is_capacity_recovery`:** Identifies cycles where discharge capacity jumped upward by $> 0.02\text{ Ah}$ relative to the preceding cycle, representing electrochemical relaxation during rest periods.
- **`is_duration_iqr_outlier`:** Identifies cycles where the total elapsed discharge duration was statistically abnormal.

### Measurement-Level Flags:
- **`is_voltage_physical_anomaly`:** Terminal voltage $< 1.5\text{ V}$ or $> 4.35\text{ V}$.
- **`is_below_nominal_cutoff`:** Voltage dips below nominal cell cutoff thresholds ($2.7\text{ V}$ for B0005, $2.5\text{ V}$ for B0006/B0018, $2.2\text{ V}$ for B0007), identifying the final load trigger point.
- **`is_current_spike`:** Deviation $> 0.20\text{ A}$ from nominal $-2.0\text{ A}$ current during active discharge ($t > 30\text{ s}$).
- **`is_temperature_physical_anomaly`:** Temperature $< 18^\circ\text{C}$ or $> 45^\circ\text{C}$.
- **`is_load_sign_inverted`:** Flags electronic load records with negative sign convention resulting from historical DAQ configuration.
