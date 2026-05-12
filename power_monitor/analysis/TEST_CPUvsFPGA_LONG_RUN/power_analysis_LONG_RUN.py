from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

# Configuration of directories for CSV logs inputs and figure outputs

BASE_DIR = Path(__file__).parent
DATASETS = {
    "FPGA": BASE_DIR / "csv_logs/FPGA",
    "Python": BASE_DIR / "csv_logs/Python"
    #"FPGA": BASE_DIR / "TEST_CPUvsFPGA_energy/csv_logs/FPGA",
    #"Python": BASE_DIR / "TEST_CPUvsFPGA_energy/csv_logs/Python"
}

FIG_DIR = BASE_DIR / "figures"
#FIG_DIR = BASE_DIR / "TEST_CPUvsFPGA_energy/figures"
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

# ---------- identify baselines to isolate snn energy ----------
# FPGA idle (when Python runs SNN)
fpga_idle_mean = python_sys["episode_energy_fpga_Wh"].mean()

# CPU/system baseline (when FPGA runs SNN)
cpu_idle_mean = fpga["episode_energy_system_Wh"].mean()

print(f"\nFPGA idle mean energy: {fpga_idle_mean:.4f} Wh")
print(f"CPU baseline mean energy: {cpu_idle_mean:.4f} Wh")


# ---------- Isolated SNN energy ----------
# FPGA SNN energy
fpga["snn_energy_fpga_Wh"] = (
    fpga["episode_energy_fpga_Wh"] - fpga_idle_mean
)

# CPU SNN energy
python_sys["snn_energy_cpu_Wh"] = (
    python_sys["episode_energy_system_Wh"] - cpu_idle_mean
)
# ---------- combine snn energy for comparison ----------
snn_df = pd.concat([
    fpga[["snn_energy_fpga_Wh"]].rename(columns={"snn_energy_fpga_Wh": "snn_energy_Wh"}).assign(system="FPGA"),
    python_sys[["snn_energy_cpu_Wh"]].rename(columns={"snn_energy_cpu_Wh": "snn_energy_Wh"}).assign(system="CPU")
])

# ---------- statistical test on isolated SNN ----------
x = snn_df[snn_df["system"] == "FPGA"]["snn_energy_Wh"].dropna()
y = snn_df[snn_df["system"] == "CPU"]["snn_energy_Wh"].dropna()

t_stat, p_t = ttest_ind(x, y, equal_var=False)
u_stat, p_u = mannwhitneyu(x, y, alternative='two-sided')

print("\n=== SNN ENERGY COMPARISON ===")
print(f"T-test p-value: {p_t:.5f}")
print(f"Mann Whitney p-value: {p_u:.5f}")

# Boxplot: isolated SNN
plt.figure()
snn_df.boxplot(column="snn_energy_Wh", by="system")
plt.ylabel("SNN Energy (Wh)")
plt.title("Isolated SNN Energy Consumption")
plt.suptitle("")
plt.savefig(FIG_DIR / "snn_energy_boxplot.png")
plt.close()


# Add episode index for both systems
fpga = fpga.sort_values(by="episode_start_ros_time_s").copy()
python_sys = python_sys.sort_values(by="episode_start_ros_time_s").copy()

fpga["episode_index"] = range(len(fpga))
python_sys["episode_index"] = range(len(python_sys))

# Lineplot: SNN energy per episode
plt.figure()

plt.plot(
    fpga["episode_index"],
    fpga["snn_energy_fpga_Wh"],
    marker='o',
    label="FPGA SNN"
)

plt.plot(
    python_sys["episode_index"],
    python_sys["snn_energy_cpu_Wh"],
    marker='o',
    label="CPU SNN"
)

plt.xlabel("Episode")
plt.ylabel("SNN Energy (Wh)")
plt.title("SNN Energy per Episode")
plt.legend()

plt.savefig(FIG_DIR / "snn_energy_per_episode.png")
plt.close()


python_sys["snn_power_cpu_W"] = (
    python_sys["snn_energy_cpu_Wh"] / python_sys["episode_total_time_s"]
)

# Compute power
fpga["snn_power_fpga_W"] = (
    fpga["snn_energy_fpga_Wh"] / fpga["episode_total_time_s"]
)

python_sys["snn_power_cpu_W"] = (
    python_sys["snn_energy_cpu_Wh"] / python_sys["episode_total_time_s"]
)

# Plot: SNN power per episode
plt.figure()

plt.plot(
    fpga["episode_index"],
    fpga["snn_power_fpga_W"],
    marker='o',
    label="FPGA SNN"
)

plt.plot(
    python_sys["episode_index"],
    python_sys["snn_power_cpu_W"],
    marker='o',
    label="CPU SNN"
)

plt.xlabel("Episode")
plt.ylabel("SNN Power (W)")
plt.title("SNN Power per Episode")
plt.legend()

plt.savefig(FIG_DIR / "snn_power_per_episode.png")
plt.close()



# Plot: Energy comparison FPGA and CPU
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

# Plot: Boxplot - Energy Distribution per System
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

# Scatterlot: Average Power
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

# Barplot: Mean comparison
grouped = df_all.groupby("system_type")[[
    "episode_energy_total_Wh",
    "avg_power_total_W"
]].mean()

grouped.plot(kind="bar")

plt.ylabel("Average Value")
plt.title("Mean Energy and Power per System")

plt.savefig(FIG_DIR / "mean_comparison.png")
plt.close()

# Boxplot: Efficiency

df_all["efficiency"] = df_all["episode_energy_total_Wh"] / df_all["episode_total_time_s"]

plt.figure()

df_all.boxplot(column="efficiency", by="system_type")

plt.ylabel("Wh per second")
plt.title("Energy Efficiency Comparison")
plt.suptitle("")

plt.savefig(FIG_DIR / "efficiency_boxplot.png")
plt.close()
