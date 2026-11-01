"""
VoltIQ: Feature Configuration and Data Leakage Prevention Module
Explicitly partitions engineered features into Machine Learning inputs,
supervisory target variables, analytical metrics, and metadata.
"""

from typing import List, Dict, Any

# Primary regression target variables
TARGET_COLUMNS: List[str] = [
    "soh_percentage",
    "discharge_capacity_ah",
]

# Secondary / alternative target variables
ALTERNATIVE_TARGET_COLUMNS: List[str] = [
    "capacity_fade_percentage",
    "capacity_loss_ah",
]

# LEAKAGE COLUMNS:
# Columns that directly compute or reveal capacity / SOH.
# MUST BE EXCLUDED from feature matrices (X) in all Machine Learning pipelines.
LEAKAGE_COLUMNS: List[str] = [
    "discharge_capacity_ah",
    "initial_capacity_ah",
    "soh_percentage",
    "capacity_fade_percentage",
    "capacity_loss_ah",
    "relative_capacity_retention",
    "capacity_change_ah",
    "capacity_change_percentage",
    "soh_change_percentage",
    "capacity_recovery_flag",
]

# METADATA & GROUPING IDENTIFIERS:
# Identifiers for splitting datasets (e.g., GroupKFold by battery_id) or tracking.
# Should NOT be passed as predictive numeric inputs to models.
METADATA_COLUMNS: List[str] = [
    "battery_id",
    "cycle_type",
]

# ==============================================================================
# AUDITED FEATURE CLASSIFICATION FOR MACHINE LEARNING (Phase 5 Audit)
# ==============================================================================

# 1. SENSOR-ONLY FEATURES:
# Pure physical measurements derived from voltage, current, temperature, and load telemetry.
# Free of cycle-age bias and zero mathematical dependency on capacity integration.
SENSOR_ONLY_FEATURES: List[str] = [
    # Voltage Statistics
    "voltage_mean",
    "voltage_median",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "voltage_range",
    # Current Statistics
    "current_mean",
    "current_median",
    "current_min",
    "current_max",
    "current_std",
    "current_range",
    # Temperature Statistics
    "temperature_mean",
    "temperature_median",
    "temperature_min",
    "temperature_max",
    "temperature_std",
    "temperature_range",
    # Load Telemetry
    "current_load_mean",
    "voltage_load_mean",
    # Thermal Dynamics
    "temperature_change_from_previous_cycle",
]

# 2. CYCLE-AGE FEATURES:
# Features that track chronological progression in the laboratory aging protocol.
# Note: In real-world EVs, cycle number does not directly translate to health without knowing DOD/C-rate.
CYCLE_AGE_FEATURES: List[str] = [
    "cycle_number",
    "cycle_index",
]

# 3. POTENTIAL TARGET PROXIES & CAPACITY-DERIVED FEATURES:
# Features that directly or indirectly encode discharge capacity:
# - discharge_duration_seconds: under constant 2A discharge, duration strongly proxies Q = I * t (r > 0.95).
# - measurement_count: proportional to duration at ~18.8s per sample (r > 0.90).
# - cycle_duration_change: delta of duration.
# - average_sampling_interval: duration / (count - 1).
# - ICA features (ica_peak_value, ica_peak_voltage, ica_area): require cumulative capacity Q(t) to derive dQ/dV.
TARGET_PROXY_FEATURES: List[str] = [
    "discharge_duration_seconds",
    "measurement_count",
    "average_sampling_interval",
    "cycle_duration_change",
    "ica_peak_value",
    "ica_peak_voltage",
    "ica_area",
]

# ==============================================================================
# TIERED FEATURE BENCHMARK SETS FOR CONTROLLED ML EVALUATION
# ==============================================================================

# Tier 1 (Recommended Primary Baseline): Pure Sensor-Only Features
# Strictly observable in real-time BMS, zero target proxy risk, zero lab-cycle memorization.
FEATURE_SET_TIER_1_SENSOR_ONLY: List[str] = list(SENSOR_ONLY_FEATURES)

# Tier 2: Sensor-Only + Cycle Age
# Allows evaluating whether knowing cycle progression provides marginal predictive value.
FEATURE_SET_TIER_2_SENSOR_PLUS_AGE: List[str] = SENSOR_ONLY_FEATURES + CYCLE_AGE_FEATURES

# Tier 3: Full Informative Feature Set (Includes Sensor + Age + Proxies/ICA)
# For comparative bench-testing to quantify how much proxy features artificially inflate accuracy.
FEATURE_SET_TIER_3_FULL_INFORMATIVE: List[str] = (
    SENSOR_ONLY_FEATURES + CYCLE_AGE_FEATURES + TARGET_PROXY_FEATURES
)

# Backward-compatible alias for existing pipelines
ML_FEATURE_COLUMNS: List[str] = list(FEATURE_SET_TIER_3_FULL_INFORMATIVE)


def validate_ml_feature_set(feature_columns: List[str]) -> Dict[str, Any]:
    """
    Validate that an input feature list contains zero target leakage columns.
    Raises ValueError if any leakage feature is detected.
    """
    detected_leakage = [col for col in feature_columns if col in LEAKAGE_COLUMNS]
    if detected_leakage:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! The following target-leaking columns were found: {detected_leakage}"
        )

    return {
        "status": "PASS",
        "total_features": len(feature_columns),
        "leakage_columns_count": 0,
        "is_safe_for_training": True,
    }
