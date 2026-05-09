from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

# Configuration of directories for CSV logs inputs and figure outputs

BASE_DIR = Path(__file__).parent
DATASETS = {
    "FPGA": BASE_DIR / "csv_logs/FPGA",
    "Python": BASE_DIR / "csv_logs/Python"
}

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load data
# Label data
# combine data
# Save plots to folder /figures

def load_dataset(folder, label):
    files = list(folder.glob("*.csv"))
    df_list = []

    for file in files:
        try:
            df = pd.read_csv(file)
            df["system_type"] = label  # tag system
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


# Load data
df_all = pd.concat(
    [load_dataset(path, name) for name, path in DATASETS.items()],
    ignore_index=True
)

# Statistical tests

# Split data

df_all["efficiency"] = df_all["episode_energy_total_Wh"] / df_all["episode_total_time_s"]

fpga = df_all[df_all["system_type"] == "FPGA"]
python_sys = df_all[df_all["system_type"] == "Python"]

def run_tests(metric):
    x = fpga[metric].dropna()
    y = python_sys[metric].dropna()

    print(f"\n=== {metric} ===")

    # Welch’s t-test
    t_stat, p_t = ttest_ind(x, y, equal_var=False)

    # Mann-Whitney U (non-parametric)
    u_stat, p_u = mannwhitneyu(x, y, alternative='two-sided')

    print(f"T-test p-value: {p_t:.5f}")
    print(f"Mann Whitney p-value: {p_u:.5f}")

    # Interpretation
    if p_t < 0.05:
        print("→ Significant difference (t-test)")
    else:
        print("→ No significant difference (t-test)")

    if p_u < 0.05:
        print("→ Significant difference (Mann Whitney)")
    else:
        print("→ No significant difference (Mann Whitney)")


# Efficiency test


run_tests("episode_energy_total_Wh")
run_tests("avg_power_total_W")
run_tests("efficiency")

# Save results to file
results = []

def collect_tests(metric):
    x = fpga[metric].dropna()
    y = python_sys[metric].dropna()

    t_stat, p_t = ttest_ind(x, y, equal_var=False)
    u_stat, p_u = mannwhitneyu(x, y, alternative='two-sided')

    results.append({
        "metric": metric,
        "t_test_p": p_t,
        "mannwhitney_p": p_u
    })


collect_tests("episode_energy_total_Wh")
collect_tests("avg_power_total_W")
collect_tests("efficiency")

df_results = pd.DataFrame(results)
df_results.to_csv(FIG_DIR / "statistical_tests.csv", index=False)

print(df_results)


# Clean and sort
df_all = df_all.dropna(subset=["episode_energy_total_Wh"])
df_all = df_all.sort_values(by="episode_start_ros_time_s")
df_all["episode_index"] = df_all.groupby("system_type").cumcount()

# Plot 1: Energy comparison FPGA and CPU
plt.figure()

for system in df_all["system_type"].unique():
    subset = df_all[df_all["system_type"] == system]

    plt.plot(
        subset["episode_index"],
        subset["episode_energy_total_Wh"],
        marker='o',
        label=system
    )

plt.xlabel("Episode")
plt.ylabel("Energy (Wh)")
plt.title("Energy per Episode Comparison")
plt.legend()

plt.savefig(FIG_DIR / "energy_per_episode.png")
plt.close()

# Plot 2: Boxplot
plt.figure()

df_all.boxplot(
    column="episode_energy_total_Wh",
    by="system_type"
)

plt.ylabel("Energy (Wh)")
plt.title("Energy Distribution per System")
plt.suptitle("")

plt.savefig(FIG_DIR / "energy_boxplot.png")
plt.close()

# Plot 3: Average Power

plt.figure()

for system in df_all["system_type"].unique():
    subset = df_all[df_all["system_type"] == system]

    plt.scatter(
        subset["episode_total_time_s"],
        subset["avg_power_total_W"],
        label=system
    )

plt.xlabel("Episode Time (s)")
plt.ylabel("Average Power (W)")
plt.title("Power vs Time")
plt.legend()

plt.savefig(FIG_DIR / "power_vs_time.png")
plt.close()

# Plot 4: Mean comparison

grouped = df_all.groupby("system_type")[[
    "episode_energy_total_Wh",
    "avg_power_total_W"
]].mean()

grouped.plot(kind="bar")

plt.ylabel("Average Value")
plt.title("Mean Energy and Power per System")

plt.savefig(FIG_DIR / "mean_comparison.png")
plt.close()

# Plot 5: Efficiency

df_all["efficiency"] = df_all["episode_energy_total_Wh"] / df_all["episode_total_time_s"]

plt.figure()

df_all.boxplot(column="efficiency", by="system_type")

plt.ylabel("Wh per second")
plt.title("Energy Efficiency Comparison")
plt.suptitle("")

plt.savefig(FIG_DIR / "efficiency_boxplot.png")
plt.close()
