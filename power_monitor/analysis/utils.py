import pandas as pd
from pathlib import Path
import numpy as np


def load_power_logs(csv_dir):
    """
    Load all power_log_*.csv files and concatenate them into one DataFrame.
    Adds a run_index column (0, 1, 2, ...).
    """
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("power_log_*.csv"))

    if not files:
        raise FileNotFoundError(f"No power_log_*.csv files found in {csv_dir}")

    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["run_index"] = i
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def normalize_time(df, time_col="ros_time_s"):
    """
    Normalize time column to [0, 1] for a single run.
    """
    df = df.copy()
    t0 = df[time_col].iloc[0]
    t1 = df[time_col].iloc[-1]

    df["norm_t"] = (df[time_col] - t0) / (t1 - t0)
    return df


def resample_norm_time(df, value_col, bins=100):
    """
    Resample a value along normalized time [0, 1].
    Returns (values, norm_time_axis).
    """
    norm_axis = np.linspace(0, 1, bins)
    values = np.interp(norm_axis, df["norm_t"], df[value_col])
    return values, norm_axis