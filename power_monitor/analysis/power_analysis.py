from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CSV_DIR = Path(__file__).parent / "csv_logs"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TIME_COL = "ros_time_s"
ENERGY_COL = "energy_total_Wh"


def load_energy_metrics(csv_dir):
    """
    For each CSV file (one series), compute:
      - total energy [Wh]
      - duration [min]
      - average energy consumption [Wh/min]
    """
    csv_files = sorted(csv_dir.glob("power_log_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    series_idx = []
    total_energy = []
    energy_per_min = []
    durations_min = []

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)

        # Total energy (cumulative)
        E_total = df[ENERGY_COL].iloc[-1]

        # Duration
        t_start = df[TIME_COL].iloc[0]
        t_end = df[TIME_COL].iloc[-1]
        duration_s = t_end - t_start
        duration_min = duration_s / 60.0

        # Average energy consumption per minute
        E_per_min = E_total / duration_min

        series_idx.append(i)
        total_energy.append(E_total)
        energy_per_min.append(E_per_min)
        durations_min.append(duration_min)

    return series_idx, total_energy, energy_per_min, durations_min


def main():
    series_idx, total_energy, energy_per_min, durations_min = load_energy_metrics(CSV_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # =========================
    # Plot 1: Total energy + duration
    # =========================
    ax_energy = axes[0]
    ax_time = ax_energy.twinx()

    ax_energy.plot(series_idx, total_energy, marker="o",
                   label="Total energy [Wh]")
    ax_time.plot(series_idx, durations_min, marker="s",
                 linestyle="--", color="tab:gray",
                 label="Duration [min]")

    ax_energy.set_xlabel("Series index")
    ax_energy.set_ylabel("Total energy [Wh]")
    ax_time.set_ylabel("Duration [min]")

    ax_energy.set_title("Total energy and duration per series")
    ax_energy.grid(True)

    # Combined legend
    lines_1, labels_1 = ax_energy.get_legend_handles_labels()
    lines_2, labels_2 = ax_time.get_legend_handles_labels()
    ax_energy.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    # =========================
    # Plot 2: Average energy per minute
    # =========================
    axes[1].plot(series_idx, energy_per_min,
                 marker="o", color="tab:orange")
    axes[1].set_xlabel("Series index")
    axes[1].set_ylabel("Energy per minute [Wh/min]")
    axes[1].set_title("Average energy consumption per series")
    axes[1].grid(True)

    fig.tight_layout()

    out = FIG_DIR / "energy_per_series_comparison.png"
    fig.savefig(out, dpi=200)
    plt.show()

    print(f"Saved figure to {out}\n")

    # Numeric summary (useful for report tables)
    for i, E, EpM, T in zip(series_idx, total_energy, energy_per_min, durations_min):
        print(f"Series {i:02d}: "
              f"Total = {E:.4f} Wh, "
              f"Avg = {EpM:.4f} Wh/min, "
              f"Duration = {T:.2f} min")


if __name__ == "__main__":
    main()