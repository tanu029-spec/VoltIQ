# VoltIQ: NASA Battery Dataset Ingestion Report

**Generated:** 2026-11-01T21:53:51.010092+00:00  
**Status:** ✅ Successfully Ingested & Validated

---

## 1. Executive Summary

The VoltIQ data ingestion pipeline extracted cycle-level summaries and high-frequency telemetry from authentic NASA Ames Prognostics Center of Excellence (PCoE) Lithium-Ion battery aging datasets (`.mat` files). All four targeted experimental cells (**B0005**, **B0006**, **B0007**, **B0018**) were parsed and converted into normalized tabular formats.

| Metric | Result |
| :--- | :--- |
| **Batteries Processed** | 4 (`B0005`, `B0006`, `B0007`, `B0018`) |
| **Total Experimental Cycles in MATLAB** | 2,167 cycles |
| **Extracted Discharge Cycles** | **636** cycles |
| **Total Measurement Records** | **185,721** rows |
| **Skipped Malformed Cycles** | **0** |
| **Duplicate Cycle Records** | **0** |
| **Measurement Alignment Mismatches** | **0** |

---

## 2. Cycle Breakdown per Battery

| Battery ID | Charge Cycles | Discharge Cycles | Impedance Cycles | Total Cycles | Discharge Measurement Rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0005** | 170 | 168 | 278 | 616 | 50,285 |
| **B0006** | 170 | 168 | 278 | 616 | 50,285 |
| **B0007** | 170 | 168 | 278 | 616 | 50,285 |
| **B0018** | 134 | 132 | 53 | 319 | 34,866 |
| **Total** | **644** | **636** | **887** | **2,167** | **185,721** |

---

## 3. Physical Value Boundaries & Data Quality

Physical bounds confirmed valid operating ranges for 18650-size lithium-ion cells under CC-CV charge and constant-current discharge:

| Feature | Minimum | Maximum | Expected Range | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Discharge Capacity (Ah)** | `1.1538` | `2.0353` | 1.0 – 2.1 Ah | ✅ Valid |
| **Voltage Measured (V)** | `1.7370` | `4.2333` | 1.7 – 4.3 V | ✅ Valid |
| **Current Measured (A)** | `-2.0291` | `0.0143` | -2.1 – 0.1 A | ✅ Valid |
| **Temperature Measured (°C)** | `22.35` | `42.33` | 20 – 45 °C | ✅ Valid |

### Missing Values Check
- **`battery_cycles.csv`:** Zero missing values across all columns.
- **`battery_measurements.csv`:** Zero missing values across all columns.

---

## 4. Generated Output Files

1. **`data/processed/battery_cycles.csv`**
   - Size: `30,946` bytes
   - Records: 636 rows
2. **`data/processed/battery_measurements.csv`**
   - Size: `16,368,578` bytes (15.61 MB)
   - Records: 185,721 rows
