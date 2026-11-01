"""
VoltIQ: Data Preprocessing, Quality Assessment, and Anomaly Detection Module
Performs comprehensive data validation, non-destructive data cleaning,
statistical and physical anomaly flagging, and diagnostic report generation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BatteryPreprocessor:
    """
    Validates, audits, flags anomalies, and prepares clean datasets
    from raw extracted NASA battery aging cycle and measurement records.
    """

    NOMINAL_CAPACITY_AH = 2.0
    NOMINAL_DISCHARGE_CURRENT_A = -2.0
    VOLTAGE_MIN_PHYSICAL_V = 1.5
    VOLTAGE_MAX_PHYSICAL_V = 4.35
    TEMP_MIN_PHYSICAL_C = 18.0
    TEMP_MAX_PHYSICAL_C = 45.0
    RECOVERY_THRESHOLD_AH = 0.02

    def __init__(
        self,
        processed_data_dir: Union[str, Path] = "data/processed",
        results_dir: Union[str, Path] = "results",
    ):
        self.processed_data_dir = Path(processed_data_dir)
        self.results_dir = Path(results_dir)

    def load_extracted_data(
        self,
        cycles_file: Optional[Union[str, Path]] = None,
        measurements_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the original extracted cycle and measurement CSV files.
        """
        c_path = Path(cycles_file) if cycles_file else self.processed_data_dir / "battery_cycles.csv"
        m_path = Path(measurements_file) if measurements_file else self.processed_data_dir / "battery_measurements.csv"

        if not c_path.is_file():
            raise FileNotFoundError(f"Battery cycles file not found: {c_path}")
        if not m_path.is_file():
            raise FileNotFoundError(f"Battery measurements file not found: {m_path}")

        cycles_df = pd.read_csv(c_path)
        meas_df = pd.read_csv(m_path)

        return cycles_df, meas_df

    def validate_data(
        self, cycles_df: pd.DataFrame, meas_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Conduct thorough schema, completeness, monotonicity, and data integrity checks.
        """
        # Dimensions & Schema
        cycles_shape = list(cycles_df.shape)
        meas_shape = list(meas_df.shape)
        cycles_cols = list(cycles_df.columns)
        meas_cols = list(meas_df.columns)

        # Missing & Infinite Values
        cycles_nulls = {k: int(v) for k, v in cycles_df.isnull().sum().to_dict().items()}
        meas_nulls = {k: int(v) for k, v in meas_df.isnull().sum().to_dict().items()}

        cycles_num = cycles_df.select_dtypes(include=np.number)
        meas_num = meas_df.select_dtypes(include=np.number)
        cycles_infs = {k: int(v) for k, v in np.isinf(cycles_num).sum().to_dict().items()}
        meas_infs = {k: int(v) for k, v in np.isinf(meas_num).sum().to_dict().items()}

        # Duplicates
        exact_dup_cycles = int(cycles_df.duplicated().sum())
        key_dup_cycles = int(cycles_df.duplicated(subset=["battery_id", "cycle_number"]).sum())
        exact_dup_meas = int(meas_df.duplicated().sum())
        timestamp_dup_meas = int(
            meas_df.duplicated(subset=["battery_id", "cycle_number", "time_seconds"]).sum()
        )

        # Timestamp ordering (monotonicity) per cycle
        non_monotonic_cycles: List[Dict[str, Any]] = []
        for (b_id, c_num), group in meas_df.groupby(["battery_id", "cycle_number"]):
            times = group["time_seconds"].values
            if not np.all(np.diff(times) >= 0):
                non_monotonic_cycles.append({"battery_id": b_id, "cycle_number": int(c_num)})

        # Measurement count cross-validation
        meas_count_grouped = meas_df.groupby(["battery_id", "cycle_number"]).size().to_dict()
        count_mismatches: List[Dict[str, Any]] = []
        for _, row in cycles_df.iterrows():
            b_id = row["battery_id"]
            c_num = row["cycle_number"]
            expected = int(row["measurement_count"])
            actual = int(meas_count_grouped.get((b_id, c_num), 0))
            if expected != actual:
                count_mismatches.append({
                    "battery_id": b_id,
                    "cycle_number": int(c_num),
                    "expected": expected,
                    "actual": actual,
                })

        # Battery representation
        batteries = sorted(cycles_df["battery_id"].unique().tolist())
        cycles_per_battery = {k: int(v) for k, v in cycles_df.groupby("battery_id").size().to_dict().items()}
        meas_per_battery = {k: int(v) for k, v in meas_df.groupby("battery_id").size().to_dict().items()}

        # Capacity validity
        negative_caps = int((cycles_df["discharge_capacity_ah"] <= 0).sum())
        abnormal_caps = int((cycles_df["discharge_capacity_ah"] > 3.0).sum())

        return {
            "cycles_shape": cycles_shape,
            "meas_shape": meas_shape,
            "cycles_columns": cycles_cols,
            "meas_columns": meas_cols,
            "batteries_present": batteries,
            "cycles_per_battery": cycles_per_battery,
            "measurements_per_battery": meas_per_battery,
            "null_counts": {"cycles": cycles_nulls, "measurements": meas_nulls},
            "infinite_counts": {"cycles": cycles_infs, "measurements": meas_infs},
            "duplicates": {
                "exact_cycles": exact_dup_cycles,
                "cycle_keys": key_dup_cycles,
                "exact_measurements": exact_dup_meas,
                "timestamp_measurements": timestamp_dup_meas,
            },
            "temporal_integrity": {
                "non_monotonic_cycles_count": len(non_monotonic_cycles),
                "non_monotonic_cycles": non_monotonic_cycles,
            },
            "cross_table_alignment": {
                "measurement_count_mismatches": len(count_mismatches),
                "mismatches": count_mismatches,
            },
            "capacity_validity": {
                "non_positive_count": negative_caps,
                "unphysically_high_count": abnormal_caps,
                "min_capacity": float(cycles_df["discharge_capacity_ah"].min()),
                "max_capacity": float(cycles_df["discharge_capacity_ah"].max()),
            },
        }

    def clean_data(
        self, cycles_df: pd.DataFrame, meas_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Perform reproducible, non-destructive cleaning and type enforcement.
        Preserves original rows while standardizing types and sorting order.
        """
        audit_log: List[str] = []

        # 1. Work on explicit copies to preserve original datasets
        clean_cycles = cycles_df.copy()
        clean_meas = meas_df.copy()
        audit_log.append("Created working copies of extracted cycles and measurements DataFrames.")

        # 2. Enforce strict dtypes
        cycle_dtype_map = {
            "battery_id": "string",
            "cycle_number": "int64",
            "cycle_type": "string",
            "ambient_temperature": "float64",
            "discharge_capacity_ah": "float64",
            "measurement_count": "int64",
        }
        for col, dt in cycle_dtype_map.items():
            if col in clean_cycles.columns:
                clean_cycles[col] = clean_cycles[col].astype(dt)
        audit_log.append("Enforced explicit data types on cycle-level DataFrame.")

        meas_dtype_map = {
            "battery_id": "string",
            "cycle_number": "int64",
            "time_seconds": "float64",
            "voltage_measured_v": "float64",
            "current_measured_a": "float64",
            "temperature_measured_c": "float64",
            "current_load_a": "float64",
            "voltage_load_v": "float64",
        }
        for col, dt in meas_dtype_map.items():
            if col in clean_meas.columns:
                clean_meas[col] = clean_meas[col].astype(dt)
        audit_log.append("Enforced explicit data types on measurement-level DataFrame.")

        # 3. Deterministic sorting: chronological order by battery, cycle, and timestamp
        clean_cycles = clean_cycles.sort_values(by=["battery_id", "cycle_number"]).reset_index(drop=True)
        clean_meas = clean_meas.sort_values(by=["battery_id", "cycle_number", "time_seconds"]).reset_index(drop=True)
        audit_log.append("Sorted cycles and measurements deterministically by battery, cycle number, and elapsed time.")

        return clean_cycles, clean_meas, audit_log

    def flag_anomalies(
        self, cycles_df: pd.DataFrame, meas_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform multi-faceted anomaly detection using IQR, statistical bounds,
        and physical battery domain heuristics without deleting data.

        Returns:
            - cycles_with_flags: Cycle DataFrame with boolean anomaly flags.
            - meas_with_flags: Measurement DataFrame with boolean anomaly flags.
            - anomaly_summary: DataFrame summarizing anomaly counts and rates per battery.
        """
        c_flagged = cycles_df.copy()
        m_flagged = meas_df.copy()

        # ==========================================
        # 1. Cycle-Level Anomaly Flags
        # ==========================================

        # A. Capacity IQR Outlier per battery
        c_flagged["is_capacity_iqr_outlier"] = False
        for b_id, grp in c_flagged.groupby("battery_id"):
            caps = grp["discharge_capacity_ah"]
            q25, q75 = caps.quantile(0.25), caps.quantile(0.75)
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            outlier_mask = (grp["discharge_capacity_ah"] < lower) | (grp["discharge_capacity_ah"] > upper)
            c_flagged.loc[outlier_mask.index, "is_capacity_iqr_outlier"] = outlier_mask

        # B. Capacity Recovery (Electrochemical Relaxation Jump > 0.02 Ah)
        c_flagged["is_capacity_recovery"] = False
        for b_id, grp in c_flagged.groupby("battery_id"):
            caps = grp["discharge_capacity_ah"].values
            cap_diffs = np.zeros(len(caps))
            cap_diffs[1:] = caps[1:] - caps[:-1]
            recovery_mask = cap_diffs > self.RECOVERY_THRESHOLD_AH
            c_flagged.loc[grp.index, "is_capacity_recovery"] = recovery_mask

        # C. Duration Outlier per battery (from measurement time)
        cycle_durations = m_flagged.groupby(["battery_id", "cycle_number"])["time_seconds"].max().reset_index()
        cycle_durations = cycle_durations.rename(columns={"time_seconds": "cycle_duration_s"})
        c_flagged = pd.merge(c_flagged, cycle_durations, on=["battery_id", "cycle_number"], how="left")

        c_flagged["is_duration_iqr_outlier"] = False
        for b_id, grp in c_flagged.groupby("battery_id"):
            durs = grp["cycle_duration_s"]
            q25, q75 = durs.quantile(0.25), durs.quantile(0.75)
            iqr = q75 - q25
            outlier_mask = (durs < (q25 - 1.5 * iqr)) | (durs > (q75 + 1.5 * iqr))
            c_flagged.loc[outlier_mask.index, "is_duration_iqr_outlier"] = outlier_mask

        # ==========================================
        # 2. Measurement-Level Anomaly Flags
        # ==========================================

        # A. Physical Voltage Outlier (< 1.5V or > 4.35V)
        m_flagged["is_voltage_physical_anomaly"] = (
            (m_flagged["voltage_measured_v"] < self.VOLTAGE_MIN_PHYSICAL_V)
            | (m_flagged["voltage_measured_v"] > self.VOLTAGE_MAX_PHYSICAL_V)
        )

        # B. Cutoff Voltage Point (Terminal trigger sample where voltage reaches cutoff)
        nominal_cutoffs = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}
        m_flagged["is_below_nominal_cutoff"] = False
        for b_id, cutoff in nominal_cutoffs.items():
            mask = (m_flagged["battery_id"] == b_id) & (m_flagged["voltage_measured_v"] < cutoff)
            m_flagged.loc[mask, "is_below_nominal_cutoff"] = True

        # C. Current Steady-State Deviation (> 0.2A deviation during active discharge)
        # We evaluate after the initial ramp-up transient (t > 30s) and before cutoff relaxation
        active_discharge = (m_flagged["time_seconds"] > 30.0) & (m_flagged["current_measured_a"] < -0.5)
        m_flagged["is_current_spike"] = False
        curr_dev = np.abs(m_flagged["current_measured_a"] - self.NOMINAL_DISCHARGE_CURRENT_A)
        m_flagged.loc[active_discharge & (curr_dev > 0.20), "is_current_spike"] = True

        # D. Physical Temperature Anomaly (< 18°C or > 45°C)
        m_flagged["is_temperature_physical_anomaly"] = (
            (m_flagged["temperature_measured_c"] < self.TEMP_MIN_PHYSICAL_C)
            | (m_flagged["temperature_measured_c"] > self.TEMP_MAX_PHYSICAL_C)
        )

        # E. Inverted DAQ Load Sign (Historical DAQ sign reversal)
        m_flagged["is_load_sign_inverted"] = m_flagged["current_load_a"] < -0.5

        # ==========================================
        # 3. Anomaly Summary Compilation
        # ==========================================
        summary_rows = []
        for b_id in sorted(c_flagged["battery_id"].unique()):
            b_c = c_flagged[c_flagged["battery_id"] == b_id]
            b_m = m_flagged[m_flagged["battery_id"] == b_id]

            summary_rows.append({
                "battery_id": b_id,
                "total_cycles": len(b_c),
                "capacity_iqr_outliers": int(b_c["is_capacity_iqr_outlier"].sum()),
                "capacity_recoveries": int(b_c["is_capacity_recovery"].sum()),
                "duration_iqr_outliers": int(b_c["is_duration_iqr_outlier"].sum()),
                "total_measurements": len(b_m),
                "voltage_physical_anomalies": int(b_m["is_voltage_physical_anomaly"].sum()),
                "below_nominal_cutoff_samples": int(b_m["is_below_nominal_cutoff"].sum()),
                "current_spikes": int(b_m["is_current_spike"].sum()),
                "temperature_physical_anomalies": int(b_m["is_temperature_physical_anomaly"].sum()),
                "load_sign_inverted_samples": int(b_m["is_load_sign_inverted"].sum()),
            })

        # Overall row
        summary_rows.append({
            "battery_id": "ALL_BATTERIES",
            "total_cycles": len(c_flagged),
            "capacity_iqr_outliers": int(c_flagged["is_capacity_iqr_outlier"].sum()),
            "capacity_recoveries": int(c_flagged["is_capacity_recovery"].sum()),
            "duration_iqr_outliers": int(c_flagged["is_duration_iqr_outlier"].sum()),
            "total_measurements": len(m_flagged),
            "voltage_physical_anomalies": int(m_flagged["is_voltage_physical_anomaly"].sum()),
            "below_nominal_cutoff_samples": int(m_flagged["is_below_nominal_cutoff"].sum()),
            "current_spikes": int(m_flagged["is_current_spike"].sum()),
            "temperature_physical_anomalies": int(m_flagged["is_temperature_physical_anomaly"].sum()),
            "load_sign_inverted_samples": int(m_flagged["is_load_sign_inverted"].sum()),
        })

        anomaly_summary_df = pd.DataFrame(summary_rows)

        return c_flagged, m_flagged, anomaly_summary_df

    def export_cleaned_data(
        self,
        clean_cycles_df: pd.DataFrame,
        clean_meas_df: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Export the clean datasets with anomaly flags to the processed data directory.
        """
        target_dir = Path(output_dir) if output_dir else self.processed_data_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        cycles_out = target_dir / "clean_battery_cycles.csv"
        meas_out = target_dir / "clean_battery_measurements.csv"

        clean_cycles_df.to_csv(cycles_out, index=False)
        clean_meas_df.to_csv(meas_out, index=False)

        return cycles_out, meas_out
