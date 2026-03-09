# encoding_node/proximity_bracket_encoder.py
from bisect import bisect_right
import math
from typing import List, Optional


class ProximityBracketEncoder:
    """
    Converts a distance value into a discrete bracket/bin and emits a spike (1)
    only when the bracket changes.
    """

    def __init__(
        self,
        bin_edges: List[float],
        *,
        init_no_spike: bool = True,
        inf_as_far: bool = True,
    ):
        if not bin_edges:
            raise ValueError("bin_edges must be non-empty")
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]:
                raise ValueError("bin_edges must be strictly ascending")

        self.bin_edges = list(bin_edges)
        self.init_no_spike = init_no_spike
        self.inf_as_far = inf_as_far
        self._prev_bin: Optional[int] = None

    def reset(self) -> None:
        self._prev_bin = None

    def distance_to_bin(self, d: float) -> Optional[int]:
        # Return None for invalid values unless configured otherwise.
        if not math.isfinite(d):
            if math.isinf(d) and d > 0 and self.inf_as_far:
                # Treat +inf as very far -> last bin
                return len(self.bin_edges)
            return None
        if d <= 0.0:
            return None
        return bisect_right(self.bin_edges, d)

    def update(self, d: float) -> int:
        """
        Feed one distance reading. Returns 1 only if bracket changes, else 0.
        """
        b = self.distance_to_bin(d)
        if b is None:
            return 0

        if self._prev_bin is None:
            self._prev_bin = b
            return 0 if self.init_no_spike else 1

        if b != self._prev_bin:
            self._prev_bin = b
            return 1

        return 0
