from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from utils import load_power_logs, normalize_time, resample_norm_time


CSV_DIR = Path(__file__).parent / "csv_logs"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

POWER_COL = "power_W"     # juster ved behov
TIME_COL = "ros_time_s"


def main():
    df = load_power_logs(CSV_DIR)

    all_resampled = []

    for run_idx in df["run_index"].unique():
        run_df = df[df["run_index"] == run_idx]
        run_df = normalize_time(run_df, TIME_COL)

        values, norm_t = resample_norm_time(
            run_df, POWER_COL, bins=200
        )
        all_resampled.append(values)

    all_resampled = np.vstack(all_resampled)

    mean_power = all_resampled.mean(axis=0)
    std_power = all_resampled.std(axis=0)

    # ---- Matplotlib plotting ----
    plt.figure(figsize=(8, 4))
    plt.plot(norm_t, mean_power, color="blue", label="Mean power")
    plt.fill_between(
        norm_t,
        mean_power - std_power,
        mean_power + std_power,
        color="blue",
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Normalized task progress")
    plt.ylabel("Power [W]")
    plt.title("Mean power usage across runs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "mean_power_vs_progress.png"
    plt.savefig(out, dpi=200)
    plt.show()

    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()