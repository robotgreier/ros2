"""Structural-tuning sweep.

Simulates the eligibility trace under candidate (t_post, t_pre, dw_pos, dw_neg,
tau_e_shift) parameter sets against the same spike pattern, then computes the
Δw distribution under realistic dopamine values. Goal: find a set that
produces a balanced LTP/LTD ratio (currently ~10:1 LTD-skewed).

Spike pattern is calibrated from the user's captured membrane traces
(post-fire ~46% on the dominant neuron, dense pre-fire ~60%).
"""

import random
import numpy as np

from python_snn_node.LIF_SNN_network import Synapse


N_TICKS = 4000
PRE_FIRE_PROB = 0.60
POST_FIRE_PROB = 0.46
N_RUNS = 16  # average over multiple synapses with independent spike streams
SEED = 0


def run_synapse(params, lr_shift, n_ticks=N_TICKS, seed=0):
    """Run one synapse for n_ticks with random pre/post spikes; return list of
    eligibility values across the run."""
    rng = random.Random(seed)
    syn = Synapse(lr_shift=lr_shift, w_init=128, **params)
    elig_trace = np.empty(n_ticks, dtype=np.int32)
    for t in range(n_ticks):
        syn.update_eligibility(
            int(rng.random() < PRE_FIRE_PROB),
            int(rng.random() < POST_FIRE_PROB),
        )
        elig_trace[t] = syn.eligibility
    return elig_trace


def run_population(params, lr_shift):
    """Concatenate eligibility from N_RUNS independent synapses."""
    out = []
    for r in range(N_RUNS):
        out.append(run_synapse(params, lr_shift, seed=SEED + r))
    return np.concatenate(out)


def dw_stats(elig, lr_shift, dop):
    dw = (elig.astype(np.int64) * dop) >> lr_shift
    pos = dw[dw > 0]
    neg = dw[dw < 0]
    return {
        'frac_nz': float((dw != 0).mean()),
        'n_pos': int(pos.size),
        'n_neg': int(neg.size),
        'mean_pos': float(pos.mean()) if pos.size else 0.0,
        'mean_neg': float(neg.mean()) if neg.size else 0.0,
        'max_abs': int(np.abs(dw).max()) if dw.size else 0,
    }


def summarise(name, params, lr_shift):
    elig = run_population(params, lr_shift)
    print(f"\n{name}")
    print(f"  params: {params}, lr_shift={lr_shift}")
    print(f"  eligibility: range [{elig.min()},{elig.max()}], "
          f"mean={elig.mean():+.2f}, "
          f"+:{(elig>0).sum()} / -:{(elig<0).sum()} "
          f"(ratio +/-:{(elig>0).sum() / max(1, (elig<0).sum()):.2f})")
    print(f"  {'dop':<6}{'frac_nz':<10}{'#LTP':<8}{'#LTD':<8}"
          f"{'mean_LTP':<11}{'mean_LTD':<11}{'max|Δw|':<8}")
    for dop in [+3, -3, -6]:
        s = dw_stats(elig, lr_shift, dop)
        print(f"  {dop:<+6}{s['frac_nz']:<10.3f}{s['n_pos']:<8}"
              f"{s['n_neg']:<8}{s['mean_pos']:<+11.2f}"
              f"{s['mean_neg']:<+11.2f}{s['max_abs']:<8}")


def main():
    base = dict(t_pre=3, t_post=3, tau_e_shift=4, dw_pos=5, dw_neg=5,
                w_min=16, w_max=254, mode='rstdp')

    summarise("CURRENT (baseline)", base, lr_shift=6)

    # Each candidate changes one or two things vs baseline
    summarise("t_post=1 (strict acausal window)",
              {**base, 't_post': 1}, lr_shift=6)

    summarise("dw_pos=15, dw_neg=5 (asymmetric weights)",
              {**base, 'dw_pos': 15, 'dw_neg': 5}, lr_shift=6)

    summarise("tau_e_shift=2 (faster decay)",
              {**base, 'tau_e_shift': 2}, lr_shift=6)

    summarise("COMBINED: t_post=1, dw_pos=10, lr_shift=4",
              {**base, 't_post': 1, 'dw_pos': 10}, lr_shift=4)

    summarise("COMBINED: t_post=1, tau_e_shift=2, lr_shift=4",
              {**base, 't_post': 1, 'tau_e_shift': 2}, lr_shift=4)

    summarise("COMBINED: t_post=1, dw_pos=10, tau_e_shift=3, lr_shift=4",
              {**base, 't_post': 1, 'dw_pos': 10, 'tau_e_shift': 3},
              lr_shift=4)

    summarise("CURRENT-FILE: t_post=1, tau_e_shift=2, dw_pos=7, lr_shift=6",
              {**base, 't_post': 1, 'tau_e_shift': 2, 'dw_pos': 7},
              lr_shift=6)

    summarise("dw_pos=12, lr_shift=6",
              {**base, 't_post': 1, 'tau_e_shift': 2, 'dw_pos': 12},
              lr_shift=6)
    summarise("dw_pos=16, lr_shift=6",
              {**base, 't_post': 1, 'tau_e_shift': 2, 'dw_pos': 16},
              lr_shift=6)
    summarise("dw_pos=7, lr_shift=5 (lower learning rate)",
              {**base, 't_post': 1, 'tau_e_shift': 2, 'dw_pos': 7},
              lr_shift=5)

    # Per-second Δw at 15 Hz, for the recommended structural change with
    # different lr_shifts. Assume dopamine is nonzero ~50% of ticks (rough
    # estimate based on dopamine_logic.py — many states return 0).
    print("\n" + "=" * 78)
    print("Δw per SECOND at 15 Hz (one apply_reward per tick on winner's row)")
    print("Assumes dopamine ~= +3 reward 30% of ticks, -3 to -6 punish 20%, "
          "0 otherwise")
    print("=" * 78)
    structural = {**base, 't_post': 1, 'tau_e_shift': 2, 'dw_pos': 7}
    for lr_shift in [3, 4, 5, 6, 7]:
        elig = run_population(structural, lr_shift)
        # Simulate 15 ticks/sec with mixed dopamine
        rng = np.random.default_rng(0)
        n_seconds_simulated = elig.size // 15
        per_sec = []
        for s in range(n_seconds_simulated):
            window = elig[s*15:(s+1)*15]
            dop_choices = rng.choice([+3, -3, -6, 0],
                                     size=window.size,
                                     p=[0.30, 0.10, 0.10, 0.50])
            dw_window = (window.astype(np.int64) * dop_choices) >> lr_shift
            per_sec.append(dw_window.sum())
        per_sec = np.array(per_sec)
        max_abs_per_tick = max(
            abs((elig * +3) >> lr_shift).max(),
            abs((elig * -6) >> lr_shift).max(),
        )
        print(f"\n  lr_shift={lr_shift}: per-tick max |Δw|={max_abs_per_tick}")
        print(f"  cumulative Δw per second:  "
              f"mean={per_sec.mean():+.1f}  "
              f"std={per_sec.std():.1f}  "
              f"p5={np.percentile(per_sec, 5):+.0f}  "
              f"p95={np.percentile(per_sec, 95):+.0f}  "
              f"max|Δw/s|={np.abs(per_sec).max()}")
        # Time to saturate from middle weight (128 -> 16 or 254, span 112)
        worst_drift_per_sec = max(abs(per_sec.mean()) + per_sec.std(),
                                  np.abs(per_sec).max() / 4)
        sat_time = 112 / max(1, worst_drift_per_sec)
        print(f"  approx time to drift 112 units (mid -> clamp): "
              f"{sat_time:.0f} s")


if __name__ == '__main__':
    main()
