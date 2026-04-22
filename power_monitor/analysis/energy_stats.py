from pathlib import Path
from utils import load_power_logs

CSV_DIR = Path(__file__).parent / "csv_logs"
POWER_COL = "power_W"
TIME_COL = "ros_time_s"


def compute_energy(df):
    """
    Compute energy (J) via trapezoidal integration.
    """
    import numpy as np

    t = df[TIME_COL].to_numpy()
    p = df[POWER_COL].to_numpy()
    return np.trapz(p, t)


def main():
    df = load_power_logs(CSV_DIR)

    energies = []

    for run_idx in df["run_index"].unique():
        run_df = df[df["run_index"] == run_idx]
        E = compute_energy(run_df)
        energies.append(E)

        print(f"Run {run_idx}: {E:.2f} J")

    import numpy as np

    energies = np.array(energies)
    print("\nSummary:")
    print(f"Mean energy: {energies.mean():.2f} J")
    print(f"Std energy : {energies.std():.2f} J")


if __name__ == "__main__":
    main()