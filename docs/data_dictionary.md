# VoltIQ: Data Dictionary

This document defines the schema, data types, physical measurement units, source MATLAB mappings, and domain semantics for all datasets produced by the VoltIQ data ingestion pipeline.

---

## 1. Cycle-Level Table (`data/processed/battery_cycles.csv`)

One record per valid discharge cycle across the battery cells.

| Column Name | Data Type | Units | Source MATLAB Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| `battery_id` | `string` | N/A | Root MAT key (`B0005`, `B0006`, etc.) | Unique battery cell identifier. |
| `cycle_number` | `integer` | Count | Cycle struct array index (`i + 1`) | Chronological 1-based cycle index in the experiment. |
| `cycle_type` | `string` | N/A | `cycle.type` (`'discharge'`) | Operational regime type. |
| `ambient_temperature` | `float` | °C | `cycle.ambient_temperature` | Environmental chamber setpoint during the test. |
| `discharge_capacity_ah` | `float` | Ampere-hours (Ah) | `cycle.data.Capacity` | Total integrated discharge capacity measured for the cycle. |
| `measurement_count` | `integer` | Count | `length(cycle.data.Time)` | Number of time-series telemetry samples recorded in this cycle. |

---

## 2. Measurement-Level Table (`data/processed/battery_measurements.csv`)

High-frequency time-series telemetry sampled during each discharge cycle.

| Column Name | Data Type | Units | Source MATLAB Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| `battery_id` | `string` | N/A | Root MAT key | Unique battery cell identifier. |
| `cycle_number` | `integer` | Count | Cycle struct array index | Reference to the parent discharge cycle. |
| `time_seconds` | `float` | Seconds (s) | `cycle.data.Time` | Relative elapsed time since the beginning of the discharge cycle. |
| `voltage_measured_v` | `float` | Volts (V) | `cycle.data.Voltage_measured` | Battery cell terminal voltage measured at the given timestamp. |
| `current_measured_a` | `float` | Amperes (A) | `cycle.data.Current_measured` | Current flow across cell terminals (negative denotes discharge). |
| `temperature_measured_c`| `float` | °C | `cycle.data.Temperature_measured` | Cell surface temperature recorded via thermocouple. |
| `current_load_a` | `float` | Amperes (A) | `cycle.data.Current_load` | Current drawn by the programmable electronic load. |
| `voltage_load_v` | `float` | Volts (V) | `cycle.data.Voltage_load` | Voltage measured across the electronic load terminals. |
