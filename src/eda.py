"""
VoltIQ: Exploratory Data Analysis (EDA) Module
Functions and scripts to explore degradation patterns, capacity fade curves,
cycle comparisons, and thermal profiles.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class BatteryEDA:
    """
    Visualization and statistical analysis routines for battery aging datasets.
    """

    def __init__(self, figures_dir: str = "results/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
