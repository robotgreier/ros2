from typing import Dict, List, Optional, Tuple


class DopamineComputer:
    """
    Computes a signed dopamine reward from:
    - ArUco direction bits
    - selected action
    - proximity stop flag

    Version 1 includes only the currently active reward logic.
    Energy-aware logic will be added later as a separate layer.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def decode_object_bits(obj_bits: List[int]) -> Tuple[bool, Optional[int]]:
        """
        Decode one-hot lateral position from a bit vector.

        For 3 bits:
          [1,0,0] -> left   (-1)
          [0,1,0] -> center ( 0)
          [0,0,1] -> right  (+1)
          [0,0,0] -> not seen

        Works for any odd number of bins.
        """
        n = len(obj_bits)

        for i, bit in enumerate(obj_bits):
            if bit == 1:
                return True, i - (n // 2)

        return False, None

    def step(
        self,
        obj_bits: List[int],
        action_idx: int,
        proximity_stop: bool,
        task_state: Optional[int] = None,
    ) -> Tuple[int, Dict[str, int]]:
        """
        Compute dopamine reward for the current action.

        action_idx convention:
          0 = LEFT
          1 = FORWARD
          2 = RIGHT
          3 = BACKWARD

        Returns:
          dopamine: signed integer reward
          comps: dict with named reward component(s)
        """
        seen, pos = self.decode_object_bits(obj_bits)
        comps: Dict[str, int] = {}

        # Base reward
        if seen and pos is not None and (
            (pos == 0 and action_idx == 1) or
            (pos < 0 and action_idx == 0) or
            (pos > 0 and action_idx == 2)
        ):
            dopamine = 1
        elif not seen and action_idx in (0, 2, 3):
            dopamine = 1
        elif seen and action_idx == 3:
            dopamine = -4
        else:
            dopamine = -3

        if not seen and dopamine > 0:
            comps["searching"] = dopamine
        else:
            comps["align_action"] = dopamine

        # Proximity stop overrides base reward
        if proximity_stop:
            if action_idx == 3:
                dopamine = 3
                comps["proximity_stop"] = dopamine
            else:
                dopamine = -6
                comps["proximity_stop"] = dopamine

        return dopamine, comps