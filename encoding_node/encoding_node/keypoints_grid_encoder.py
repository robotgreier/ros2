from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from std_msgs.msg import Int32MultiArray


@dataclass
class GridShape:
    rows: int
    cols: int


class KeypointsGridEncoder:
    """
    Converts an Int32MultiArray grid of keypoint counts (row-major) into a spike vector:
      spike[i] = 1 if counts[i] >= threshold else 0

    Output preserves input order exactly (msg.data order).
    """

    def __init__(self, threshold: int):
        self.set_threshold(threshold)
        self._shape: Optional[GridShape] = None

    def set_threshold(self, threshold: int) -> None:
        threshold = int(threshold)
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.threshold = threshold

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

        # Preserve order exactly
        spikes = [1 if int(v) >= self.threshold else 0 for v in msg.data]
        return spikes, self._shape
