#!/usr/bin/env python3
"""
Standalone latency and memory benchmark for the Python SNN.

Runs the same Layer class used by snn_node.py with the production
neuron/synapse parameters, but without any ROS dependency. Measures
the cost of one full learning timestep: forward() + apply_reward().

Usage:
    python3 benchmark.py                       # default: 10000 iters
    python3 benchmark.py --iters 50000 --csv out.csv
    python3 benchmark.py --no-feedback         # match HDL without feedback neuron
    python3 benchmark.py --profile             # also produce cProfile breakdown
    python3 benchmark.py --scaling-sweep       # sweep network sizes
"""

import argparse
import cProfile
import csv
import io
import os
import pstats
import sys
import time
import tracemalloc

import numpy as np

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

# Allow running this script directly from the scripts/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from python_snn_node.LIF_SNN_network import Layer  # noqa: E402


# Production parameters, copied from snn_node.py defaults.
NEURON_PARAMS = {
    "decay": 256,
    "threshold": 2048,
    "reset": 0,
    "refractory": 0,
}

SYNAPSE_PARAMS = {
    "lr_shift": 7,
    "w_init": -1,           # random init in [64, 192)
    "t_pre": 3,
    "t_post": 3,
    "tau_e_shift": 3,
    "dw_pos": 32,
    "dw_neg": 16,
    "w_min": 16,
    "w_max": 254,
    "mode": "rstdp",
}

# Matches the deployed topology: 27 sensor inputs + 1 feedback neuron -> 4 actions,
# i.e. 28 x 4 = 112 synapses, matching the FPGA implementation.
INPUT_SIZE = 27
OUTPUT_SIZE = 4


def percentile(values_ns, pct):
    return float(np.percentile(values_ns, pct)) / 1e3  # ns -> us


def summarize(name, values_ns):
    a = np.asarray(values_ns, dtype=np.float64)
    print(f"\n=== {name} (n={len(a)}) ===")
    print(f"  mean   : {a.mean()/1e3:>9.3f} us")
    print(f"  std    : {a.std()/1e3:>9.3f} us")
    print(f"  median : {np.median(a)/1e3:>9.3f} us")
    print(f"  p99    : {np.percentile(a, 99)/1e3:>9.3f} us")
    print(f"  min    : {a.min()/1e3:>9.3f} us")
    print(f"  max    : {a.max()/1e3:>9.3f} us")


def report_memory(net):
    """Static + dynamic memory footprint of the Layer."""
    print("\n=== Memory footprint ===")
    total_bytes = 0
    for attr in ("weights", "eligibility", "eligibility_snapshot",
                 "pre_timer", "post_timer", "mem", "pre_reset_mem",
                 "spk", "spike_count", "refractory_timer", "last_delta_w"):
        arr = getattr(net, attr, None)
        if arr is None:
            continue
        nbytes = arr.nbytes
        total_bytes += nbytes
        print(f"  {attr:<22s} shape={str(arr.shape):<10s} dtype={arr.dtype} "
              f"-> {nbytes:>6d} B")
    print(f"  {'TOTAL state arrays':<22s} {'':<10s} {'':>5}"
          f"      {total_bytes:>6d} B  ({total_bytes/1024:.2f} KB)")

    # FPGA-equivalent weight storage: 1 byte per synapse (uint8 in HDL)
    n_synapses = net.weights.size
    print(f"\n  weights:   {n_synapses} synapses x int32 = {net.weights.nbytes} B")
    print(f"  HDL equiv: {n_synapses} synapses x  uint8 = {n_synapses} B")


def build_layer(n_inputs, n_outputs, feedback):
    return Layer(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        neuron_params=NEURON_PARAMS,
        synapse_params=SYNAPSE_PARAMS,
        feedback=feedback,
    )


def run_timed_loop(net, inputs, warmup, iters, dopamine):
    """Run a warmup + timed loop. Returns (fwd_ns, rew_ns, wall_ns)."""
    perf_ns = time.perf_counter_ns
    fwd_ns = np.empty(iters, dtype=np.int64)
    rew_ns = np.empty(iters, dtype=np.int64)

    for i in range(warmup):
        out = net.forward(input_spikes=inputs[i])
        winner = net.winner_takes_all(out)
        if winner >= 0:
            net.apply_reward(dopamine=dopamine, winner_idx=winner)

    wall_t0 = perf_ns()
    for i in range(iters):
        x = inputs[warmup + i]
        t0 = perf_ns()
        out = net.forward(input_spikes=x)
        t1 = perf_ns()
        winner = net.winner_takes_all(out)
        t2 = perf_ns()
        if winner >= 0:
            net.apply_reward(dopamine=dopamine, winner_idx=winner)
        t3 = perf_ns()
        fwd_ns[i] = t1 - t0
        rew_ns[i] = t3 - t2
    wall_t1 = perf_ns()

    return fwd_ns, rew_ns, wall_t1 - wall_t0


def run_unprofiled_loop(net, inputs, warmup, iters, dopamine):
    """Same arithmetic path, no per-call timing, for cProfile to attribute time
    against the underlying Layer methods rather than perf_counter_ns calls."""
    for i in range(warmup):
        out = net.forward(input_spikes=inputs[i])
        winner = net.winner_takes_all(out)
        if winner >= 0:
            net.apply_reward(dopamine=dopamine, winner_idx=winner)
    for i in range(iters):
        x = inputs[warmup + i]
        out = net.forward(input_spikes=x)
        winner = net.winner_takes_all(out)
        if winner >= 0:
            net.apply_reward(dopamine=dopamine, winner_idx=winner)


def report_process(label="Process resources"):
    print(f"\n=== {label} ===")
    if not _HAVE_PSUTIL:
        # /proc fallback (Linux only)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmSize:", "VmPeak:")):
                        print(f"  {line.strip()}")
        except FileNotFoundError:
            print("  psutil not available and /proc/self/status not readable")
        return
    p = psutil.Process()
    mem = p.memory_info()
    print(f"  RSS     : {mem.rss/1024/1024:>8.2f} MB  (resident set size)")
    print(f"  VMS     : {mem.vms/1024/1024:>8.2f} MB  (total address space)")
    try:
        full = p.memory_full_info()
        print(f"  USS     : {full.uss/1024/1024:>8.2f} MB  (unique to this process)")
    except (psutil.AccessDenied, AttributeError):
        pass
    cpu = p.cpu_percent(interval=None)
    if cpu > 0:
        print(f"  CPU     : {cpu:>8.1f} %  of one core (averaged over timed loop)")


def report_profile(net, inputs, warmup, iters, dopamine):
    print("\n=== cProfile breakdown (HDL-submodule analog) ===")
    pr = cProfile.Profile()
    pr.enable()
    run_unprofiled_loop(net, inputs, warmup, iters, dopamine)
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative")
    stats.print_stats(15)
    lines = buf.getvalue().splitlines()
    # Print everything from the first 'ncalls' header line onward
    started = False
    for line in lines:
        if "ncalls" in line:
            started = True
        if started:
            print("  " + line)


def run_main_benchmark(args, feedback):
    np.random.seed(args.seed)
    net = build_layer(INPUT_SIZE, OUTPUT_SIZE, feedback)

    print(f"Platform: {os.uname().sysname} {os.uname().machine}")
    print(f"NumPy   : {np.__version__}")
    print(f"Layer   : {INPUT_SIZE} inputs ({'+1 feedback' if feedback else 'no feedback'})"
          f" -> {OUTPUT_SIZE} outputs")
    print(f"Synapses: {net.weights.size}  (mode={SYNAPSE_PARAMS['mode']})")
    print(f"Iters   : warmup={args.warmup}, timed={args.iters}, "
          f"spike_prob={args.spike_prob}, dopamine={args.dopamine}")

    total = args.warmup + args.iters
    inputs = (np.random.rand(total, INPUT_SIZE) < args.spike_prob).astype(np.int32)

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    if _HAVE_PSUTIL:
        psutil.Process().cpu_percent(interval=None)  # prime the meter

    fwd_ns, rew_ns, wall_ns = run_timed_loop(
        net, inputs, args.warmup, args.iters, args.dopamine)

    snap_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summarize("forward()", fwd_ns)
    summarize("apply_reward()", rew_ns)
    summarize("forward() + apply_reward()", fwd_ns + rew_ns)

    wall_us = wall_ns / 1e3
    print(f"\nWall-clock total: {wall_us/1e6:.3f} s "
          f"({wall_us/args.iters:.3f} us/iter avg incl. WTA + loop overhead)")
    throughput = args.iters / (wall_ns / 1e9)
    print(f"Throughput      : {throughput:,.0f} timesteps/sec")
    print(f"At 15 Hz target : {wall_us/args.iters/1e6*15*100:.4f}% of one timer period")

    print("\n=== Python-level allocations during timed loop ===")
    print(f"  current : {current/1024:.2f} KB")
    print(f"  peak    : {peak/1024:.2f} KB")
    diff = snap_after.compare_to(snap_before, "lineno")[:5]
    if any(d.size_diff > 0 for d in diff):
        print("  top growth sites:")
        for d in diff:
            if d.size_diff <= 0:
                continue
            print(f"    +{d.size_diff:>7d} B  {d.traceback.format()[-1].strip()}")

    report_process("Process-level cost (interpreter + NumPy + state)")
    report_memory(net)

    if args.profile:
        # Run a separate, smaller profile pass on the same inputs.
        prof_iters = min(args.iters, 5_000)
        report_profile(net, inputs, args.warmup, prof_iters, args.dopamine)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "forward_ns", "apply_reward_ns", "total_ns"])
            for i in range(args.iters):
                w.writerow([i, int(fwd_ns[i]), int(rew_ns[i]),
                            int(fwd_ns[i] + rew_ns[i])])
        print(f"\nPer-iteration timings written to {args.csv}")


def run_scaling_sweep(args, feedback):
    """Sweep network sizes to expose CPU's linear scaling vs FPGA's O(1)."""
    print("\n=== Scaling sweep (FPGA is O(1) in cycles; CPython is not) ===")
    sweeps = [
        # (n_inputs, n_outputs, label)
        (27,  4, "prod (28x4 = 112 syn)"),
        (55,  4, "2x inputs (56x4 = 224 syn)"),
        (111, 4, "4x inputs (112x4 = 448 syn)"),
        (223, 4, "8x inputs"),
        (27,  8, "2x outputs"),
        (27, 16, "4x outputs"),
        (27, 32, "8x outputs"),
        (111, 16, "16x scale-up"),
        (223, 32, "64x scale-up"),
    ]
    iters = args.sweep_iters
    warmup = max(100, args.warmup // 5)

    print(f"\n{'n_in':>5} {'n_out':>5} {'syn':>6} {'fwd mean (us)':>15} "
          f"{'fwd p99 (us)':>14} {'apply mean':>12} {'total mean':>12} {'note'}")
    print("  " + "-" * 90)
    results = []
    for n_in, n_out, note in sweeps:
        np.random.seed(args.seed)
        net = build_layer(n_in, n_out, feedback)
        total = warmup + iters
        inputs = (np.random.rand(total, n_in) < args.spike_prob).astype(np.int32)
        fwd_ns, rew_ns, _ = run_timed_loop(net, inputs, warmup, iters, args.dopamine)
        fwd_us = fwd_ns / 1e3
        rew_us = rew_ns / 1e3
        n_syn = net.weights.size
        print(f"{n_in:>5} {n_out:>5} {n_syn:>6} "
              f"{fwd_us.mean():>15.2f} {np.percentile(fwd_us, 99):>14.2f} "
              f"{rew_us.mean():>12.2f} {(fwd_us + rew_us).mean():>12.2f}  {note}")
        results.append((n_syn, float(fwd_us.mean()), float(rew_us.mean())))

    # Simple slope estimate over the input-sweep (fixed 4 outputs)
    input_sweep = [r for n_syn, fwd, _ in results
                   for r in [(n_syn, fwd)] if n_syn % 4 == 0]
    if len(input_sweep) >= 2:
        xs = np.array([s[0] for s in input_sweep], dtype=float)
        ys = np.array([s[1] for s in input_sweep], dtype=float)
        # Fit y = a + b*n  to expose the per-synapse marginal cost
        b, a = np.polyfit(xs, ys, 1)
        print(f"\n  Linear fit (fwd mean us vs n_synapses): "
              f"{a:.2f} us + {b*1000:.3f} ns/synapse")
        print(f"  Interpretation: a~constant NumPy/Python overhead floor, "
              f"b~per-synapse marginal cost.")
        print(f"  FPGA equivalent: ~1 clock cycle (~10 ns @ 100 MHz) regardless of n.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--spike-prob", type=float, default=0.2)
    parser.add_argument("--dopamine", type=int, default=1)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--profile", action="store_true",
                        help="Add cProfile breakdown of forward()/apply_reward() callees")
    parser.add_argument("--scaling-sweep", action="store_true",
                        help="Sweep network sizes to expose Python's linear scaling")
    parser.add_argument("--sweep-iters", type=int, default=2_000,
                        help="Iterations per sweep point (default 2000)")
    args = parser.parse_args()

    feedback = not args.no_feedback
    run_main_benchmark(args, feedback)
    if args.scaling_sweep:
        run_scaling_sweep(args, feedback)


if __name__ == "__main__":
    main()
