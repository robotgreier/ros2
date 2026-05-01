from typing import Dict, List, Optional, Tuple


class DopamineComputer:
    """
    Computes a signed dopamine reward from:
    - ArUco direction bits
    - selected action
    - proximity stop flag

    Adds a small search reward when no ArUco is visible:
    turn left for a while, then move forward briefly, then repeat.
    """

    def __init__(self) -> None:
        # Counts how long we have been searching with no ArUco visible
        self.search_counter = 0

        # Used mostly for debugging/logging reward components
        self.search_phase = "turn"

    @staticmethod
    def decode_object_bits(obj_bits: List[int]) -> Tuple[bool, Optional[int]]:
        """
        Decode one-hot lateral position.

        [1,0,0] -> left   (-1)
        [0,1,0] -> center ( 0)
        [0,0,1] -> right  (+1)
        [0,0,0] -> not seen
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
        action_idx convention:
          0 = LEFT
          1 = FORWARD
          2 = RIGHT
          3 = BACKWARD
        """

        seen, pos = self.decode_object_bits(obj_bits)
        comps: Dict[str, int] = {}

        # If we see an ArUco, stop/reset search pattern
        if seen:
            self.search_counter = 0
            self.search_phase = "turn"

        # Existing reward: correct action when ArUco is visible
        if seen and pos is not None and (
            (pos == 0 and action_idx == 1) or
            (pos < 0 and action_idx == 0) or
            (pos > 0 and action_idx == 2)
        ):
            dopamine = 6

        # New reward: structured search when no ArUco is visible
        elif not seen:
            self.search_counter += 1

            # Tune these numbers depending on your control loop speed
            turn_steps = 30       # reward LEFT for this many steps
            forward_steps = 10    # then reward FORWARD for this many steps

            cycle_length = turn_steps + forward_steps
            phase_step = self.search_counter % cycle_length

            if phase_step < turn_steps:
                self.search_phase = "turn"
                dopamine = 0 if action_idx == 0 else 0
            else:
                self.search_phase = "forward"
                dopamine = 0 if action_idx == 1 else 0

        # Existing penalty: backing away while ArUco is visible
        elif seen and action_idx == 3:
            dopamine = -2

        # Existing default: no reward / no penalty
        else:
            dopamine = 0

        # Label reward component for debugging
        if not seen and dopamine > 0:
            comps[f"search_{self.search_phase}"] = dopamine
        else:
            comps["align_action"] = dopamine

        # Existing proximity override
        if proximity_stop:
            if action_idx == 3:
                dopamine = 4
                comps["proximity_stop"] = dopamine
            else:
                dopamine = -3
                comps["proximity_stop"] = dopamine

        return dopamine, comps