from __future__ import annotations

import numpy as np
from typing import List


# Task states (match task_manager)
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


class ArucoDirectionEncoder:
    """
    One-hot encodes ArUco x_norm into n_aruco_bins lateral zones.

    x_norm is expected as a signed value centred at 0 (range [-1, 1]),
    where -1 is the left edge and +1 is the right edge.  It is remapped
    to [0, 1] internally before binning.

    Fires in all task states whenever detect_flag >= 0.5 (tag visible).
    Returns all-zero when no tag is detected.
    """

    def __init__(self, n_aruco_bins: int):
        if n_aruco_bins < 1:
            raise ValueError("n_aruco_bins must be >= 1")
        self.n_aruco_bins = n_aruco_bins
        self._bin_edges = np.linspace(0.0, 1.0, n_aruco_bins + 1)

    def encode(self, *, state: int, detect_flag: float, x_norm: float) -> List[int]:
        if detect_flag < 0.5:
            return [0] * self.n_aruco_bins

        # Remap signed [-1, 1] -> [0, 1]
        x_01 = x_norm * 0.5 + 0.5
        x_01 = max(0.0, min(1.0, x_01))

        idx = int(np.digitize(x_01, self._bin_edges)) - 1
        idx = max(0, min(idx, self.n_aruco_bins - 1))

        spikes = [0] * self.n_aruco_bins
        spikes[idx] = 1
        return spikes
