"""
VoltIQ: Data Ingestion and MATLAB Structure Parsing Module
Parses and structures NASA Ames Lithium-Ion Battery Aging datasets (.mat).
Extracts discharge cycle metadata and high-frequency time-series measurements.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


class BatteryDataLoader:
    """
    Robust data loader and extractor for NASA Ames battery aging datasets.
    Navigates nested MATLAB structs and converts cycle measurements into clean DataFrames.
    """

    EXPECTED_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

    def __init__(self, raw_data_dir: Union[str, Path] = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    @staticmethod
    def _extract_scalar_str(val: Any) -> str:
        """
        Unpack nested MATLAB string arrays down to a clean lowercase string.
        """
        while isinstance(val, (np.ndarray, list)) and len(val) > 0:
            val = val[0]
        return str(val).strip().lower()

    @staticmethod
    def _extract_scalar_float(val: Any) -> float:
        """
        Unpack nested MATLAB numeric arrays down to a float.
        """
        while isinstance(val, (np.ndarray, list)) and len(val) > 0:
            val = val[0]
        return float(val)

    @staticmethod
    def _extract_1d_float_array(val: Any) -> np.ndarray:
        """
        Extract a 1D float64 numpy array from standard or object-nested MATLAB arrays.
        """
        if isinstance(val, np.ndarray) and val.dtype != object:
            return val.flatten().astype(np.float64)
        flat: List[float] = []

        def _recurse(x: Any) -> None:
            if isinstance(x, (np.ndarray, list, tuple)):
                for item in x:
                    _recurse(item)
            else:
                flat.append(float(x))

        _recurse(val)
        return np.array(flat, dtype=np.float64)

    def discover_raw_files(self) -> List[Path]:
        """
        Discover all .mat files in raw_data_dir that match expected NASA battery names.
        """
        if not self.raw_data_dir.exists():
            return []
        found_files = []
        for b_name in self.EXPECTED_BATTERIES:
            target_path = self.raw_data_dir / f"{b_name}.mat"
            if target_path.is_file():
                found_files.append(target_path)
        # Also include any other .mat files in the directory
        for p in sorted(self.raw_data_dir.glob("*.mat")):
            if p not in found_files:
                found_files.append(p)
        return found_files

    def check_raw_files(self) -> Dict[str, bool]:
        """
        Check existence of the standard four NASA .mat files.
        """
        return {
            f"{b_id}.mat": (self.raw_data_dir / f"{b_id}.mat").is_file()
            for b_id in self.EXPECTED_BATTERIES
        }

    def load_mat_file(self, file_path_or_id: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
        """
        Load a MATLAB file using scipy.io.loadmat and identify the battery ID.

        Returns:
            Tuple of (battery_id, mat_dict)
        """
        if isinstance(file_path_or_id, str) and not file_path_or_id.endswith(".mat"):
            file_path = self.raw_data_dir / f"{file_path_or_id}.mat"
        else:
            file_path = Path(file_path_or_id)

        if not file_path.is_file():
            raise FileNotFoundError(f"MATLAB file not found at: {file_path}")

        mat = scipy.io.loadmat(str(file_path))
        battery_id = self.get_battery_id(mat, default_id=file_path.stem)
        return battery_id, mat

    @staticmethod
    def get_battery_id(mat_dict: Dict[str, Any], default_id: str = "Unknown") -> str:
        """
        Identify the root battery identifier key in the loaded MATLAB dictionary.
        """
        candidate_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
        if not candidate_keys:
            raise KeyError("No valid battery structure key found in MATLAB dictionary.")
        # Typically the top-level variable matches the battery name (e.g. 'B0005')
        return candidate_keys[0]

    def extract_cycle_metadata(self, file_path_or_id: Union[str, Path]) -> Dict[str, Any]:
        """
        Inspect all cycles in a battery file and return cycle type breakdown and counts.
        """
        battery_id, mat = self.load_mat_file(file_path_or_id)
        cycle_arr = mat[battery_id]["cycle"][0, 0]
        n_cycles = cycle_arr.shape[1] if cycle_arr.ndim > 1 else cycle_arr.shape[0]

        type_counts: Dict[str, int] = {}
        for i in range(n_cycles):
            c = cycle_arr[0, i] if cycle_arr.ndim > 1 else cycle_arr[i]
            c_type = self._extract_scalar_str(c["type"])
            type_counts[c_type] = type_counts.get(c_type, 0) + 1

        return {
            "battery_id": battery_id,
            "total_cycles": n_cycles,
            "cycle_types": type_counts,
        }

    def extract_battery_data(
        self, file_path_or_id: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
        """
        Extract cycle-level metadata and time-series measurements for all valid discharge cycles.

        Returns:
            Tuple of:
                - cycles_df (DataFrame of cycle-level records)
                - measurements_df (DataFrame of time-series measurements)
                - skipped_cycles (List of dictionaries documenting any skipped cycles)
                - cycle_counts (Dictionary of cycle counts by type)
        """
        battery_id, mat = self.load_mat_file(file_path_or_id)
        cycle_arr = mat[battery_id]["cycle"][0, 0]
        n_cycles = cycle_arr.shape[1] if cycle_arr.ndim > 1 else cycle_arr.shape[0]

        cycle_records: List[Dict[str, Any]] = []
        measurement_records: Dict[str, List[Any]] = {
            "battery_id": [],
            "cycle_number": [],
            "time_seconds": [],
            "voltage_measured_v": [],
            "current_measured_a": [],
            "temperature_measured_c": [],
            "current_load_a": [],
            "voltage_load_v": [],
        }
        skipped_cycles: List[Dict[str, Any]] = []
        cycle_counts: Dict[str, int] = {}

        for i in range(n_cycles):
            c = cycle_arr[0, i] if cycle_arr.ndim > 1 else cycle_arr[i]
            orig_cycle_num = i + 1
            c_type = self._extract_scalar_str(c["type"])
            cycle_counts[c_type] = cycle_counts.get(c_type, 0) + 1

            if c_type != "discharge":
                continue

            # Ambient temperature
            amb_temp = self._extract_scalar_float(c["ambient_temperature"])

            # Navigate nested data struct safely
            if "data" not in c.dtype.names or c["data"].size == 0:
                skipped_cycles.append({
                    "battery_id": battery_id,
                    "cycle_number": orig_cycle_num,
                    "reason": "Missing 'data' struct in discharge cycle",
                })
                continue

            data_struct = c["data"][0, 0]
            data_fields = data_struct.dtype.names if data_struct.dtype.names else ()

            # Extract Capacity (in Ah)
            if "Capacity" not in data_fields or data_struct["Capacity"].size == 0:
                skipped_cycles.append({
                    "battery_id": battery_id,
                    "cycle_number": orig_cycle_num,
                    "reason": "Missing 'Capacity' field in discharge cycle data",
                })
                continue

            capacity_ah = self._extract_scalar_float(data_struct["Capacity"])

            # Extract required time-series measurements
            required_meas_fields = {
                "Time": "time_seconds",
                "Voltage_measured": "voltage_measured_v",
                "Current_measured": "current_measured_a",
                "Temperature_measured": "temperature_measured_c",
                "Current_load": "current_load_a",
                "Voltage_load": "voltage_load_v",
            }

            missing_meas_fields = [f for f in required_meas_fields.keys() if f not in data_fields]
            if missing_meas_fields:
                skipped_cycles.append({
                    "battery_id": battery_id,
                    "cycle_number": orig_cycle_num,
                    "reason": f"Missing measurement fields: {missing_meas_fields}",
                })
                continue

            # Extract arrays safely
            meas_arrays = {
                target_col: self._extract_1d_float_array(data_struct[mat_col])
                for mat_col, target_col in required_meas_fields.items()
            }

            # Check array length consistency
            lengths = {col: len(arr) for col, arr in meas_arrays.items()}
            unique_lengths = set(lengths.values())

            if len(unique_lengths) != 1 or 0 in unique_lengths:
                skipped_cycles.append({
                    "battery_id": battery_id,
                    "cycle_number": orig_cycle_num,
                    "reason": f"Inconsistent or zero measurement array lengths: {lengths}",
                })
                continue

            meas_count = len(meas_arrays["time_seconds"])

            # Record cycle-level data
            cycle_records.append({
                "battery_id": battery_id,
                "cycle_number": orig_cycle_num,
                "cycle_type": "discharge",
                "ambient_temperature": amb_temp,
                "discharge_capacity_ah": capacity_ah,
                "measurement_count": meas_count,
            })

            # Append measurement time-series
            measurement_records["battery_id"].extend([battery_id] * meas_count)
            measurement_records["cycle_number"].extend([orig_cycle_num] * meas_count)
            for target_col, arr in meas_arrays.items():
                measurement_records[target_col].extend(arr)

        cycles_df = pd.DataFrame(cycle_records)
        measurements_df = pd.DataFrame(measurement_records)

        return cycles_df, measurements_df, skipped_cycles, cycle_counts

    def extract_all(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """
        Process all discovered MATLAB files in the raw data directory.

        Returns:
            Tuple of:
                - all_cycles_df (Combined DataFrame of cycle records)
                - all_measurements_df (Combined DataFrame of measurement records)
                - all_skipped (Combined list of skipped cycles)
                - all_cycle_counts (Nested dict of cycle counts per battery)
        """
        raw_files = self.discover_raw_files()
        if not raw_files:
            raise FileNotFoundError(f"No .mat files found in {self.raw_data_dir}")

        cycles_dfs: List[pd.DataFrame] = []
        meas_dfs: List[pd.DataFrame] = []
        all_skipped: List[Dict[str, Any]] = []
        all_cycle_counts: Dict[str, Dict[str, int]] = {}

        for fpath in raw_files:
            cycles_df, meas_df, skipped, counts = self.extract_battery_data(fpath)
            b_id = fpath.stem
            cycles_dfs.append(cycles_df)
            meas_dfs.append(meas_df)
            all_skipped.extend(skipped)
            all_cycle_counts[b_id] = counts

        combined_cycles = pd.concat(cycles_dfs, ignore_index=True) if cycles_dfs else pd.DataFrame()
        combined_meas = pd.concat(meas_dfs, ignore_index=True) if meas_dfs else pd.DataFrame()

        return combined_cycles, combined_meas, all_skipped, all_cycle_counts

    def save_processed_data(
        self,
        cycles_df: pd.DataFrame,
        measurements_df: pd.DataFrame,
        output_dir: Union[str, Path] = "data/processed",
    ) -> Tuple[Path, Path]:
        """
        Save the processed cycles and measurements DataFrames to CSV format.

        Returns:
            Tuple of (cycles_csv_path, measurements_csv_path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cycles_file = out_path / "battery_cycles.csv"
        meas_file = out_path / "battery_measurements.csv"

        cycles_df.to_csv(cycles_file, index=False)
        measurements_df.to_csv(meas_file, index=False)

        return cycles_file, meas_file
