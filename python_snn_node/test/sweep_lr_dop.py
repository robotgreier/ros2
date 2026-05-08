"""Sweep (lr_shift, dopamine_scale) against REAL eligibility-trace samples
captured from a live run.

For each (lr_shift, dop) pair, computes the realised Δw = (elig * dop) >> lr_shift
across all values in the captured frames and reports:
  - distribution of nonzero Δw per frame (how many of the 81 synapses move)
  - mean LTP magnitude (positive Δw)
  - mean LTD magnitude (negative Δw)
  - LTP/LTD asymmetry (since the captured traces are LTD-skewed)

Goal: find a tuning where typical action-tick dopamine (+3 reward, -3..-6
punish) produces a Δw of at most a few units on a meaningful fraction of
synapses, in both directions, without saturating.
"""

import numpy as np


# Eligibility frames captured from /python_snn_node/eligibility during a run.
# Each frame is 3 outputs * 27 inputs = 81 int32 values, flattened row-major.
ELIG_FRAMES_RAW = """
4 1 0 4 -4 -1 0 4 0 0 0 0 -4 -2 0 0 0 0 0 0 0 0 0 0 0 0 4 1 0 4 -4 -1 0 4 0 0 0 0 -4 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 -2 0 4 -1 0 0 4 0 0 0 0 -4 -2 0 0 0 0 0 0 0 0 0 0 0 0 4
0 -3 0 0 -8 -5 0 0 0 0 0 0 -8 -6 0 0 0 0 0 0 0 0 0 0 0 0 3 -10 -13 0 -4 -20 -10 0 -10 0 0 0 0 -15 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -6 0 0 -5 -4 0 0 0 0 0 0 -8 -6 0 0 0 0 0 0 0 0 0 0 0 0 3
-4 -7 0 0 -12 -9 0 0 0 0 0 0 -12 -10 0 0 0 0 0 0 0 0 0 0 0 0 -1 -9 -12 0 0 -18 -9 0 -4 0 0 0 0 -14 -3 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 -10 0 0 -9 -8 0 0 0 0 0 0 -12 -10 0 0 0 0 0 0 0 0 0 0 0 0 -1
-3 -11 0 0 -15 -8 0 -4 0 0 0 0 -15 -14 0 0 0 0 0 0 0 0 0 0 0 0 0 -8 -15 0 0 -21 -8 0 -8 0 0 0 0 -17 -7 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 -14 0 0 -13 -7 0 -4 0 0 0 0 -15 -14 0 0 0 0 0 0 0 0 0 0 0 0 0
-2 -10 0 0 -14 -7 0 -3 0 0 0 0 -14 -13 0 0 0 0 0 0 0 0 0 0 0 0 0 -2 -14 0 4 -19 -2 0 -7 0 0 0 0 -15 -6 0 0 0 0 0 0 0 0 0 0 0 0 0 1 -8 0 4 -7 -1 0 0 0 0 0 0 -9 -8 0 0 0 0 0 0 0 0 0 0 0 0 4
-1 -9 0 0 -13 -6 0 -2 0 0 0 0 -13 -12 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -17 0 3 -22 -1 0 -11 0 0 0 0 -18 -10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -12 0 3 -11 0 0 -4 0 0 0 0 -13 -12 0 0 0 0 0 0 0 0 0 0 0 0 3
0 -8 0 0 -12 -5 0 -1 0 0 0 0 -12 -11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -15 0 2 -15 0 0 -10 0 0 0 0 -16 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 -15 0 2 -10 0 0 -8 0 0 0 0 -16 -15 -4 0 0 0 0 0 0 0 0 0 0 0 -1
0 -7 0 0 -11 -4 0 0 0 0 0 0 -11 -10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -14 0 1 -14 -4 0 -14 0 0 0 0 -19 -13 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 -14 0 1 -9 -4 0 -12 0 0 0 0 -19 -18 -3 0 0 0 0 0 0 0 0 0 0 0 0
0 -6 0 0 -10 -3 0 0 0 0 0 0 -10 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 4 -8 0 0 -13 -3 0 -13 0 0 0 0 -17 -12 4 0 0 0 0 0 0 0 0 0 0 0 0 1 -8 0 0 -3 0 0 -6 0 0 0 0 -13 -12 1 0 0 0 0 0 0 0 0 0 0 0 4
0 -5 0 0 -9 -2 0 0 0 0 0 0 -9 -8 0 0 0 0 0 0 0 0 0 0 0 0 0 3 -7 0 0 -16 -7 0 -16 0 0 0 0 -20 -15 3 0 0 0 0 0 0 0 0 0 0 0 0 0 -7 0 0 -7 -4 0 -10 0 0 0 0 -16 -15 0 0 0 0 0 0 0 0 0 0 0 0 3
0 -4 0 0 -8 -1 0 0 0 0 0 0 -8 -7 0 0 0 0 0 0 0 0 0 0 0 0 0 2 -6 0 0 -15 -1 0 -10 0 0 0 0 -18 -14 2 0 0 0 0 0 0 0 0 0 0 0 0 0 -6 0 0 -11 -3 0 -9 0 0 0 0 -19 -18 -4 0 0 0 0 0 0 0 0 0 0 0 -1
0 -3 0 0 -7 0 0 0 0 0 0 0 -7 -6 0 0 0 0 0 0 0 0 0 0 0 0 0 1 -5 0 0 -18 -5 0 -14 0 0 0 0 -21 -17 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 -15 -7 0 -13 0 0 0 0 -22 -21 -8 0 0 0 0 0 0 0 0 0 0 0 0
0 -2 0 0 -6 0 0 0 0 0 0 0 -6 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 0 0 -12 -4 0 -8 0 0 0 0 -19 -15 -1 0 0 0 0 0 0 0 0 0 0 0 0 4 -4 0 0 -9 -1 0 -7 0 0 0 0 -15 -15 -2 0 0 0 0 0 0 0 0 0 0 0 4
0 -1 0 0 -5 0 0 0 0 0 0 0 -5 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 -3 0 0 -11 -8 0 -7 0 0 0 0 -22 -18 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 0 0 -8 -5 0 -6 0 0 0 0 -18 -18 -6 0 0 0 0 0 0 0 0 0 0 0 3
4 0 0 0 0 4 0 4 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 4 -3 -2 0 0 -5 -7 0 -1 0 0 0 0 -20 -16 -4 0 0 0 0 0 0 0 0 0 0 0 0 -4 -2 0 0 -7 -9 0 -5 0 0 0 0 -21 -21 -10 0 0 0 0 0 0 0 0 0 0 0 -1
3 -4 0 0 0 3 0 3 0 0 0 0 -4 -4 0 0 0 0 0 0 0 0 0 0 0 0 3 -2 -6 0 0 -4 -6 0 0 0 0 0 0 -23 -19 -8 0 0 0 0 0 0 0 0 0 0 0 0 -3 -6 0 0 -6 -8 0 -4 0 0 0 0 -24 -24 -14 0 0 0 0 0 0 0 0 0 0 0 0
2 -8 0 0 0 2 0 2 0 0 0 0 -8 -8 -4 0 0 0 0 0 0 0 0 0 0 0 -1 2 -5 0 0 -3 0 0 0 0 0 0 0 -21 -17 -7 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 -5 -2 0 -3 0 0 0 0 -17 -17 -8 0 0 0 0 0 0 0 0 0 0 0 4
-2 -12 0 0 -4 -2 0 -2 0 0 0 0 -12 -12 -3 0 0 0 0 0 0 0 0 0 0 0 0 -2 -9 0 0 -7 -4 0 -4 0 0 0 0 -24 -20 -6 0 0 0 0 0 0 0 0 0 0 0 0 -3 -4 0 0 -9 -6 0 -7 0 0 0 0 -20 -20 -7 0 0 0 0 0 0 0 0 0 0 0 3
-1 -11 0 0 -3 -1 0 -1 0 0 0 0 -11 -11 -2 0 0 0 0 0 0 0 0 0 0 0 0 2 -8 0 0 -6 0 0 -3 0 0 0 0 -22 -18 0 0 0 0 0 0 0 0 0 0 0 0 0 -2 -8 0 0 -13 -5 0 -11 0 0 0 0 -23 -23 -6 0 0 0 0 0 0 0 0 0 0 0 -1
0 -10 0 0 -2 0 0 0 0 0 0 0 -10 -10 -1 0 0 0 0 0 0 0 0 0 0 0 0 1 -12 0 0 -10 0 0 -7 0 0 0 0 -25 -21 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -12 0 0 -16 -4 0 -15 0 0 0 0 -26 -26 -5 0 0 0 0 0 0 0 0 0 0 0 0
0 -9 0 0 -1 0 0 0 0 0 0 0 -9 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -11 0 0 -4 0 0 -1 0 0 0 0 -23 -19 0 0 0 0 0 0 0 0 0 0 0 0 0 3 -6 0 0 -10 0 0 -9 0 0 0 0 -19 -19 -4 0 0 0 0 0 0 0 0 0 0 0 4
0 -8 0 0 0 0 0 0 0 0 0 0 -8 -8 0 0 0 0 0 0 0 0 0 0 0 0 0 -4 -15 0 0 -3 -4 0 0 0 0 0 0 -26 -22 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -10 0 0 -9 -4 0 -8 0 0 0 0 -22 -22 -3 0 0 0 0 0 0 0 0 0 0 0 3
0 -7 0 0 0 0 0 0 0 0 0 0 -7 -7 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 -14 0 0 1 -3 0 4 0 0 0 0 -24 -15 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -14 0 0 -8 -8 0 -7 0 0 0 0 -25 -20 -2 0 0 0 0 0 0 0 0 0 0 0 -1
"""


def parse_frames():
    frames = []
    for line in ELIG_FRAMES_RAW.strip().splitlines():
        vals = [int(x) for x in line.split()]
        if len(vals) == 81:
            frames.append(np.array(vals, dtype=np.int32))
    return np.stack(frames)  # (n_frames, 81)


def arith_rshift(x, k):
    """Floor-toward-negative-infinity right shift, matching Python's >>."""
    return x >> k


def stats_for(frames, lr_shift, dop):
    dw = arith_rshift(frames * dop, lr_shift)
    nonzero = dw[dw != 0]
    pos = dw[dw > 0]
    neg = dw[dw < 0]
    return {
        'frac_nonzero': nonzero.size / dw.size,
        'mean_pos_dw': pos.mean() if pos.size else 0.0,
        'mean_neg_dw': neg.mean() if neg.size else 0.0,
        'max_abs_dw': np.abs(dw).max() if dw.size else 0,
        'n_pos': pos.size,
        'n_neg': neg.size,
    }


def main():
    frames = parse_frames()
    print(f"Loaded {frames.shape[0]} eligibility frames, "
          f"{frames.shape[1]} synapses each.")
    print(f"Eligibility range: [{frames.min()}, {frames.max()}], "
          f"mean = {frames.mean():.2f}, "
          f"frac nonzero = {(frames != 0).mean():.2f}")
    print(f"Sign skew: {(frames > 0).sum()} positive, "
          f"{(frames < 0).sum()} negative -> "
          f"trace is HEAVILY LTD-biased.\n")

    print("Δw stats per (lr_shift, dop). frac_nz = fraction of synapses "
          "that move; LTP/LTD = mean Δw on those that move that way.\n")
    print(f"{'lr':<4}{'dop':<6}{'frac_nz':<10}{'LTP_mean':<11}"
          f"{'LTD_mean':<11}{'#LTP':<7}{'#LTD':<7}{'max|Δw|':<8}")
    print("-" * 70)
    for lr_shift in [3, 4, 5, 6, 7]:
        for dop in [+3, -3, -6]:
            s = stats_for(frames, lr_shift, dop)
            print(f"{lr_shift:<4}{dop:<+6}"
                  f"{s['frac_nonzero']:<10.3f}"
                  f"{s['mean_pos_dw']:<+11.2f}"
                  f"{s['mean_neg_dw']:<+11.2f}"
                  f"{s['n_pos']:<7}{s['n_neg']:<7}"
                  f"{s['max_abs_dw']:<8}")
        print()


if __name__ == '__main__':
    main()
