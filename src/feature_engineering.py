"""
VoltIQ: Feature Engineering and Battery Health Indicator Derivation Module
Calculates battery State of Health (SOH), capacity fade, statistical sensor telemetry,
chronological cycle degradation metrics, and Incremental Capacity Analysis (ICA).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.feature_config import ML_FEATURE_COLUMNS, LEAKAGE_COLUMNS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms clean cycle records and telemetry time-series into a comprehensive
    battery health dataset with strict leakage prevention.
    """

    NOMINAL_RATED_CAPACITY_AH = 2.0
    RECOVERY_THRESHOLD_AH = 0.02
    ICA_VOLTAGE_GRID = np.linspace(4.00, 3.20, 81)  # 10mV grid from 4.00V down to 3.20V

    def __init__(
        self,
        clean_cycles_path: Union[str, Path] = "data/processed/clean_battery_cycles.csv",
        clean_measurements_path: Union[str, Path] = "data/processed/clean_battery_measurements.csv",
    ):
        self.clean_cycles_path = Path(clean_cycles_path)
        self.clean_measurements_path = Path(clean_measurements_path)

    @staticmethod
    def calculate_soh(current_capacity: float, initial_capacity: float) -> float:
        """
        Calculate State of Health (SOH) relative to measured initial discharge capacity.
        """
        if initial_capacity <= 0:
            raise ValueError(f"Initial capacity must be positive, got: {initial_capacity}")
        return float((current_capacity / initial_capacity) * 100.0)

    @staticmethod
    def calculate_capacity_fade(soh_percentage: float) -> float:
        """
        Calculate cumulative capacity fade percentage.
        """
        return float(100.0 - soh_percentage)

    def compute_health_indicators(self, cycles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive initial capacity baseline, SOH, capacity fade, and relative retention
        individually for each battery cell in chronological order.
        """
        df = cycles_df.copy()
        df = df.sort_values(by=["battery_id", "cycle_number"]).reset_index(drop=True)

        # 1. Determine baseline initial capacity per battery (first valid discharge cycle)
        initial_caps: Dict[str, float] = {}
        for b_id, grp in df.groupby("battery_id"):
            first_valid_cap = float(grp.iloc[0]["discharge_capacity_ah"])
            initial_caps[b_id] = first_valid_cap

        df["initial_capacity_ah"] = df["battery_id"].map(initial_caps)

        # 2. Cycle index: 1-based sequential discharge cycle number per battery
        df["cycle_index"] = df.groupby("battery_id").cumcount() + 1

        # 3. Health Indicators
        df["soh_percentage"] = (df["discharge_capacity_ah"] / df["initial_capacity_ah"]) * 100.0
        df["capacity_fade_percentage"] = 100.0 - df["soh_percentage"]
        df["capacity_loss_ah"] = df["initial_capacity_ah"] - df["discharge_capacity_ah"]
        df["relative_capacity_retention"] = df["discharge_capacity_ah"] / df["initial_capacity_ah"]

        return df

    def aggregate_sensor_features(self, meas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptive statistical aggregations across high-frequency sensor measurements
        for each (battery_id, cycle_number).
        """
        logger.info("Aggregating voltage, current, temperature, and load sensor features...")

        def _stats(series: pd.Series, prefix: str) -> Dict[str, float]:
            vals = series.values
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            return {
                f"{prefix}_mean": float(np.mean(vals)),
                f"{prefix}_median": float(np.median(vals)),
                f"{prefix}_min": vmin,
                f"{prefix}_max": vmax,
                f"{prefix}_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                f"{prefix}_range": float(vmax - vmin),
            }

        rows = []
        for (b_id, c_num), grp in meas_df.groupby(["battery_id", "cycle_number"], sort=False):
            t_vals = grp["time_seconds"].values
            duration = float(np.max(t_vals))
            m_count = len(grp)
            sampling_interval = float(duration / (m_count - 1)) if m_count > 1 else 0.0

            row: Dict[str, Any] = {
                "battery_id": b_id,
                "cycle_number": int(c_num),
                "discharge_duration_seconds": duration,
                "measurement_count": m_count,
                "average_sampling_interval": sampling_interval,
            }

            row.update(_stats(grp["voltage_measured_v"], "voltage"))
            row.update(_stats(grp["current_measured_a"], "current"))
            row.update(_stats(grp["temperature_measured_c"], "temperature"))

            # Load features
            row["current_load_mean"] = float(grp["current_load_a"].mean())
            row["voltage_load_mean"] = float(grp["voltage_load_v"].mean())

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_degradation_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute chronological cycle-to-cycle change metrics strictly within each battery.
        Prevents cross-battery contamination.
        """
        df = features_df.copy()
        df = df.sort_values(by=["battery_id", "cycle_number"]).reset_index(drop=True)

        numeric_delta_cols = [
            "capacity_change_ah",
            "capacity_change_percentage",
            "soh_change_percentage",
            "cycle_duration_change",
            "temperature_change_from_previous_cycle",
        ]
        for col in numeric_delta_cols:
            if col not in df.columns:
                df[col] = 0.0

        if "capacity_recovery_flag" not in df.columns:
            df["capacity_recovery_flag"] = False

        for b_id, grp in df.groupby("battery_id"):
            idxs = grp.index
            caps = grp["discharge_capacity_ah"].values
            sohs = grp["soh_percentage"].values
            durs = grp["discharge_duration_seconds"].values
            t_means = grp["temperature_mean"].values

            # Cycle-to-cycle capacity delta (C_k - C_{k-1})
            cap_diff = np.zeros(len(caps), dtype=np.float64)
            cap_diff[1:] = caps[1:] - caps[:-1]
            df.loc[idxs, "capacity_change_ah"] = cap_diff

            # Percentage capacity delta
            cap_pct_diff = np.zeros(len(caps), dtype=np.float64)
            cap_pct_diff[1:] = (cap_diff[1:] / caps[:-1]) * 100.0
            df.loc[idxs, "capacity_change_percentage"] = cap_pct_diff

            # SOH change percentage
            soh_diff = np.zeros(len(sohs), dtype=np.float64)
            soh_diff[1:] = sohs[1:] - sohs[:-1]
            df.loc[idxs, "soh_change_percentage"] = soh_diff

            # Capacity recovery flag (> 0.02 Ah jump)
            recovery_flag = cap_diff > self.RECOVERY_THRESHOLD_AH
            df.loc[idxs, "capacity_recovery_flag"] = recovery_flag

            # Duration delta
            dur_diff = np.zeros(len(durs), dtype=np.float64)
            dur_diff[1:] = durs[1:] - durs[:-1]
            df.loc[idxs, "cycle_duration_change"] = dur_diff

            # Mean temperature change from previous cycle
            temp_diff = np.zeros(len(t_means), dtype=np.float64)
            temp_diff[1:] = t_means[1:] - t_means[:-1]
            df.loc[idxs, "temperature_change_from_previous_cycle"] = temp_diff

        return df

    def compute_ica_features(
        self, meas_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Derives Incremental Capacity Analysis (ICA: dQ/dV) curves and extracts key peak metrics.

        Numerical Methodology:
        1. Selects the constant-current discharge window (V between 3.15V and 4.05V, t > 20s).
        2. Integrates cumulative discharge capacity Q(t) = integral(-I dt) in Ah.
        3. Sorts measurements by descending terminal voltage.
        4. Interpolates Q onto a uniform 10 mV voltage grid (4.00V down to 3.20V).
        5. Applies Gaussian smoothing (sigma=2.0) to eliminate quantization noise without peak distortion.
        6. Computes the numerical derivative dQ/dV using central differences.
        7. Extracts the primary peak height, peak voltage position, and integrated plateau area.

        Returns:
            - ica_summary_df: Cycle-level summary containing ica_peak_value, ica_peak_voltage, ica_area.
            - ica_curves_df: Long-form DataFrame containing complete dQ/dV curves across grid points.
        """
        logger.info("Computing Incremental Capacity Analysis (ICA: dQ/dV) features...")

        summary_rows: List[Dict[str, Any]] = []
        curve_records: List[Dict[str, Any]] = []

        v_grid = self.ICA_VOLTAGE_GRID

        for (b_id, c_num), grp in meas_df.groupby(["battery_id", "cycle_number"], sort=False):
            v = grp["voltage_measured_v"].values
            i = grp["current_measured_a"].values
            t = grp["time_seconds"].values

            # Cumulative capacity Q(t) in Ah
            dt = np.diff(t, prepend=0)
            q = np.cumsum(-i * dt) / 3600.0

            # Constant current discharge regime mask
            mask = (t > 20.0) & (v >= 3.15) & (v <= 4.05) & (i < -1.0)

            if np.sum(mask) < 20:
                # Insufficient points in window
                summary_rows.append({
                    "battery_id": b_id,
                    "cycle_number": int(c_num),
                    "ica_peak_value": np.nan,
                    "ica_peak_voltage": np.nan,
                    "ica_area": np.nan,
                })
                continue

            v_f = v[mask]
            q_f = q[mask]

            # Sort by descending voltage
            sort_idx = np.argsort(-v_f)
            v_s = v_f[sort_idx]
            q_s = q_f[sort_idx]

            # Interpolate Q onto uniform voltage grid
            q_interp = np.interp(-v_grid, -v_s, q_s)

            # Gaussian smoothing to prevent finite-difference noise amplification
            q_smooth = gaussian_filter1d(q_interp, sigma=2.0)

            # Numerical gradient: dv is negative, so -np.gradient gives positive dQ/dV
            dq_dv = -np.gradient(q_smooth, v_grid)

            # Restrict peak search to typical Li-ion phase transition window (3.35V to 3.75V)
            window_mask = (v_grid >= 3.35) & (v_grid <= 3.75)
            window_v = v_grid[window_mask]
            window_ica = dq_dv[window_mask]

            if len(window_ica) > 0:
                peak_idx = int(np.argmax(window_ica))
                peak_val = float(window_ica[peak_idx])
                peak_v = float(window_v[peak_idx])
            else:
                peak_val = float(np.max(dq_dv))
                peak_v = float(v_grid[np.argmax(dq_dv)])

            # Area under curve in the active plateau (3.40V to 3.80V)
            area_mask = (v_grid >= 3.40) & (v_grid <= 3.80)
            ica_area = float(np.trapezoid(dq_dv[area_mask], -v_grid[area_mask]))

            summary_rows.append({
                "battery_id": b_id,
                "cycle_number": int(c_num),
                "ica_peak_value": peak_val,
                "ica_peak_voltage": peak_v,
                "ica_area": ica_area,
            })

            # Save full curve points for representative reporting
            for v_val, ica_val in zip(v_grid, dq_dv):
                curve_records.append({
                    "battery_id": b_id,
                    "cycle_number": int(c_num),
                    "voltage_grid_v": float(v_val),
                    "dq_dv": float(ica_val),
                })

        ica_summary_df = pd.DataFrame(summary_rows)
        ica_curves_df = pd.DataFrame(curve_records)

        return ica_summary_df, ica_curves_df

    def build_feature_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrate complete feature derivation:
        1. Load clean datasets.
        2. Derive health indicators (SOH, capacity fade).
        3. Aggregate sensor statistics.
        4. Compute cycle-to-cycle degradation metrics.
        5. Extract ICA features.
        6. Merge into a consolidated, clean feature dataset.
        """
        logger.info(f"Loading clean cycles from: {self.clean_cycles_path}")
        clean_cycles = pd.read_csv(self.clean_cycles_path)
        logger.info(f"Loading clean measurements from: {self.clean_measurements_path}")
        clean_meas = pd.read_csv(self.clean_measurements_path)

        # 1. Health indicators
        hi_df = self.compute_health_indicators(clean_cycles)

        # 2. Sensor aggregations
        sensor_df = self.aggregate_sensor_features(clean_meas)

        # 3. Merge base features
        merged = pd.merge(
            hi_df,
            sensor_df,
            on=["battery_id", "cycle_number"],
            how="inner",
            suffixes=("", "_sensor"),
        )
        # Drop redundant measurement_count column if duplicated during merge
        if "measurement_count_sensor" in merged.columns:
            merged = merged.drop(columns=["measurement_count_sensor"])

        # 4. Degradation features
        degradation_df = self.compute_degradation_features(merged)

        # 5. Advanced ICA features
        ica_summary, ica_curves = self.compute_ica_features(clean_meas)
        features_df = pd.merge(
            degradation_df,
            ica_summary,
            on=["battery_id", "cycle_number"],
            how="left",
        )

        return features_df, ica_curves

    def export_features(
        self,
        features_df: pd.DataFrame,
        ica_curves_df: pd.DataFrame,
        output_dir: Union[str, Path] = "data/processed",
    ) -> Tuple[Path, Path]:
        """
        Save the engineered feature dataset and ICA curves dataset to CSV files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        features_csv = out_path / "battery_health_features.csv"
        ica_csv = out_path / "battery_ica_features.csv"

        features_df.to_csv(features_csv, index=False)
        ica_curves_df.to_csv(ica_csv, index=False)

        logger.info(f"Saved battery health features to: {features_csv}")
        logger.info(f"Saved battery ICA curves to: {ica_csv}")

        return features_csv, ica_csv
