from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from std_msgs.msg import Int32MultiArray


@dataclass
class GridShape:
    rows: int
    cols: int


class KeypointsGridEncoder:
    """
    Converts an Int32MultiArray grid of keypoint counts (row-major) into an
    event-driven spike vector.

    A spike fires (1) when the digitized threshold-level of a bin changes
    between consecutive frames — matching the offline encode_events() logic:

        threshold_edges = np.linspace(1, threshold_max, n_threshold_levels)
        spike[i] = 1 if digitize(counts[i]) != digitize(prev_counts[i])

    Edges start at 1 so empty bins (count=0) land in digitize index 0,
    making 0 <-> nonzero transitions detectable as spikes.

    On the very first message (no previous frame) all spikes are 0.
    """

    def __init__(self, n_threshold_levels: int, threshold_max: int):
        if n_threshold_levels < 1:
            raise ValueError("n_threshold_levels must be >= 1")
        if threshold_max < 1:
            raise ValueError("threshold_max must be >= 1")

        self.n_threshold_levels = n_threshold_levels
        self.threshold_max = threshold_max

        # Edges start at 1 (see docstring)
        self.threshold_edges = np.linspace(1, threshold_max, n_threshold_levels)

        self._prev_idx: Optional[np.ndarray] = None
        self._shape: Optional[GridShape] = None

    def reset(self) -> None:
        """Clear inter-frame state (e.g. on episode reset)."""
        self._prev_idx = None

    def _infer_shape(self, msg: Int32MultiArray) -> Optional[GridShape]:
        if msg.layout is None or msg.layout.dim is None:
            return None
        if len(msg.layout.dim) < 2:
            return None
        rows = int(msg.layout.dim[0].size)
        cols = int(msg.layout.dim[1].size)
        if rows <= 0 or cols <= 0:
            return None
        return GridShape(rows=rows, cols=cols)

    def update_from_msg(self, msg: Int32MultiArray) -> Tuple[List[int], Optional[GridShape]]:
        shape = self._infer_shape(msg)
        if shape is not None:
            self._shape = shape

        counts = np.asarray(msg.data, dtype=np.float64)
        new_idx = np.digitize(counts, self.threshold_edges)

        if self._prev_idx is None or len(self._prev_idx) != len(new_idx):
            spikes = [0] * len(counts)
        else:
            spikes = (new_idx != self._prev_idx).astype(np.int8).tolist()

        self._prev_idx = new_idx
        return spikes, self._shape
