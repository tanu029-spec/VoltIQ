"""
VoltIQ: Feature Engineering Module
Extracts domain-specific battery degradation indicators:
- State of Health (SOH = Current Capacity / Rated Capacity * 100)
- Incremental Capacity Analysis (ICA: dQ/dV curves)
- Voltage and temperature transient features
- Resistance / Impedance features
"""

import pandas as pd


class FeatureEngineer:
    """
    Computes domain features and health indicators from battery cycle data.
    """

    RATED_CAPACITY_AH = 2.0  # Nominal capacity for 18650 Li-ion cells in NASA dataset

    @staticmethod
    def calculate_soh(current_capacity: float, rated_capacity: float = RATED_CAPACITY_AH) -> float:
        """
        Calculate battery State of Health (SOH) as a percentage.
        """
        if rated_capacity <= 0:
            raise ValueError("Rated capacity must be greater than zero.")
        return (current_capacity / rated_capacity) * 100.0
