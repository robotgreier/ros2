from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Configuration of directories for CSV logs inputs and figure outputs
CSV_DIR = Path(__file__).parent / "csv_logs"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Episode-level columns (as logged by power_logger_node)
EPISODE_COLS = [
    "episode_energy_total_Wh",
    "episode_energy_system_Wh",
    "episode_energy_fpga_Wh",
    "search_energy_total_Wh",
    "approach_energy_total_Wh",
    "search_time_total_s",
    "approach_time_total_s",
]


def load_episode_data(csv_dir: Path) -> pd.DataFrame:
    csv_files = sorted(csv_dir.glob("power_log_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        df["series"] = csv.stem
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ---------------- Plot 1: Episode energy ----------------

def plot_episode_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(df.index, df["episode_energy_total_Wh"],
            marker="o", label="Total")
    ax.plot(df.index, df["episode_energy_system_Wh"],
            marker="s", linestyle="--", label="System")
    ax.plot(df.index, df["episode_energy_fpga_Wh"],
            marker="^", linestyle="--", label="FPGA")

    ax.set_xlabel("Episode index")
    ax.set_ylabel("Energy [Wh]")
    ax.set_title("Episode energy consumption")
    ax.legend()
    ax.grid(True)

    out = FIG_DIR / "episode_energy_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.show()

    print(f"Saved {out}")


# ---------------- Plot 2: Search vs approach energy ----------------

def plot_search_vs_approach_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(df.index, df["search_energy_total_Wh"],
            marker="o", label="Search")
    ax.plot(df.index, df["approach_energy_total_Wh"],
            marker="s", label="Approach")

    ax.set_xlabel("Episode index")
    ax.set_ylabel("Energy [Wh]")
    ax.set_title("Search vs approach energy per episode")
    ax.legend()
    ax.grid(True)

    out = FIG_DIR / "search_vs_approach_energy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.show()

    print(f"Saved {out}")


# ---------------- Plot 3: Search vs approach time ----------------

def plot_search_vs_approach_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(df.index, df["search_time_total_s"],
            marker="o", label="Search")
    ax.plot(df.index, df["approach_time_total_s"],
            marker="s", label="Approach")

    ax.set_xlabel("Episode index")
    ax.set_ylabel("Time [s]")
    ax.set_title("Search vs approach time per episode")
    ax.legend()
    ax.grid(True)

    out = FIG_DIR / "search_vs_approach_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.show()

    print(f"Saved {out}")


# ---------------- Numeric summary ----------------

def print_summary(df: pd.DataFrame):
    print("\n=== Episode summary ===")
    for i, row in df.iterrows():
        print(
            f"Episode {i:02d} | "
            f"E_total={row.episode_energy_total_Wh:.3f} Wh "
            f"(sys={row.episode_energy_system_Wh:.3f}, "
            f"fpga={row.episode_energy_fpga_Wh:.3f}) | "
            f"Search: {row.search_energy_total_Wh:.3f} Wh / "
            f"{row.search_time_total_s:.1f} s | "
            f"Approach: {row.approach_energy_total_Wh:.3f} Wh / "
            f"{row.approach_time_total_s:.1f} s"
        )


# ---------------- Main ----------------

if __name__ == "__main__":
    df = load_episode_data(CSV_DIR)

    plot_episode_energy(df)
    plot_search_vs_approach_energy(df)
    plot_search_vs_approach_time(df)

    print_summary(df)
