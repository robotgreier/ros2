# encoding_node/proximity_bracket_encoder.py
import math
from typing import List


class ProximityBracketEncoder:
    """
    Thermometric distance encoding.

    Produces n_dist_bits spikes with evenly-spaced thresholds from
    dist_max_m down to dist_max_m / n_dist_bits.  Each bit is 1 when
    the distance is closer than its threshold, so a near object saturates
    all bits and a far object fires none.

    n_dist_bits=1 is identical to the old binary encoding with
    threshold = dist_max_m.

    Returns all-zero on invalid/infinite distance (sensor not available).
    """

    def __init__(self, n_dist_bits: int, dist_max_m: float):
        if n_dist_bits < 1:
            raise ValueError("n_dist_bits must be >= 1")
        if dist_max_m <= 0.0:
            raise ValueError("dist_max_m must be positive")

        self.n_dist_bits = n_dist_bits
        self.dist_max_m = dist_max_m

        step = dist_max_m / n_dist_bits
        self.thresholds: List[float] = [
            dist_max_m - i * step for i in range(n_dist_bits)
        ]

    def reset(self) -> None:
        pass  # stateless encoding; kept for API compatibility

    def update(self, d: float) -> List[int]:
        """
        Feed one distance reading.

        Returns a list of n_dist_bits bits.  Bit i is 1 when d < thresholds[i].
        Returns all-zero for invalid (non-finite or non-positive) readings.
        """
        if not math.isfinite(d) or d <= 0.0:
            return [0] * self.n_dist_bits

        return [1 if d < thr else 0 for thr in self.thresholds]
