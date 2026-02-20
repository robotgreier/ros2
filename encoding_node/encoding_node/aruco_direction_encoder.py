from __future__ import annotations

from dataclasses import dataclass
from typing import List


# Task states (match task_manager)
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


@dataclass
class ArucoDirConfig:
    center_tol_item: float
    center_tol_dropoff: float


class ArucoDirectionEncoder:
    """
    Produces a 3-bit code (as a list of ints) based on ArUco x_norm and state:

      000 = no aruco in sight (or state is SEARCH)
      100 = aruco left of center
      010 = aruco centered (within tolerance)
      001 = aruco right of center

    State gating:
      SEARCH_ITEM / SEARCH_DROPOFF -> always 000
      APPROACH_ITEM -> use center_tol_item
      APPROACH_DROPOFF -> use center_tol_dropoff
    """

    def __init__(self, center_tol_item: float, center_tol_dropoff: float):
        if center_tol_item < 0.0 or center_tol_dropoff < 0.0:
            raise ValueError("center tolerances must be >= 0")
        self.cfg = ArucoDirConfig(
            center_tol_item=float(center_tol_item),
            center_tol_dropoff=float(center_tol_dropoff),
        )

    def set_tolerances(self, center_tol_item: float, center_tol_dropoff: float) -> None:
        if center_tol_item < 0.0 or center_tol_dropoff < 0.0:
            raise ValueError("center tolerances must be >= 0")
        self.cfg.center_tol_item = float(center_tol_item)
        self.cfg.center_tol_dropoff = float(center_tol_dropoff)

    @staticmethod
    def _code_none() -> List[int]:
        return [0, 0, 0]

    @staticmethod
    def _code_left() -> List[int]:
        return [1, 0, 0]

    @staticmethod
    def _code_center() -> List[int]:
        return [0, 1, 0]

    @staticmethod
    def _code_right() -> List[int]:
        return [0, 0, 1]

    def encode(self, *, state: int, detect_flag: float, x_norm: float) -> List[int]:
        # Gate by state
        if state in (SEARCH_ITEM, SEARCH_DROPOFF):
            return self._code_none()

        if state not in (APPROACH_ITEM, APPROACH_DROPOFF):
            return self._code_none()

        # During approach: if no detection, output none
        if detect_flag < 0.5:
            return self._code_none()

        tol = self.cfg.center_tol_item if state == APPROACH_ITEM else self.cfg.center_tol_dropoff

        if abs(x_norm) <= tol:
            return self._code_center()
        if x_norm < -tol:
            return self._code_left()
        return self._code_right()