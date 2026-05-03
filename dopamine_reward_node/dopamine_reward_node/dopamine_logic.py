from typing import Dict, List, Optional, Tuple


class DopamineComputer:
    """
    Computes a signed dopamine reward using a strict priority system:
      1. proximity_stop  — wall avoidance dominates all other signals
      2. ArUco visible   — alignment rewards, uncontaminated
      3. searching       — weak shaping to avoid silent zeros during exploration
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
        dopamine = 0

        # Priority 1: wall avoidance
        if proximity_stop:
            if action_idx == 3:         # BACKWARD — escape
                dopamine = 2
            elif action_idx in (0, 2):  # LEFT / RIGHT — escape
                dopamine = 3
            else:                       # FORWARD — into the wall
                dopamine = -3
            comps["proximity"] = dopamine

        # Priority 2: ArUco visible — alignment rewards.
        elif seen:
            if pos == 0:
                if action_idx == 1:     # centered → drive forward
                    dopamine = 6
                elif action_idx == 3:   # centered → backing away
                    dopamine = -2
                else:                   # turning in place when centered
                    dopamine = 0
            elif pos is not None and pos < 0:   # target is left
                if action_idx == 0:     # correct: turn left
                    dopamine = 3
                elif action_idx == 2:   # wrong: turn right
                    dopamine = -2
            elif pos is not None and pos > 0:   # target is right
                if action_idx == 2:     # correct: turn right
                    dopamine = 3
                elif action_idx == 0:   # wrong: turn left
                    dopamine = -2
            comps["align"] = dopamine

        # Priority 3: searching — no ArUco
        # Weak shaping so the SNN has a gradient during search instead of
        # silent zeros that produce no learning signal at all.
        else:
            if action_idx == 1:         # FORWARD — explore open space
                dopamine = 0
            elif action_idx in (0, 2):  # LEFT / RIGHT — scan for target
                dopamine = 1
            else:                       # BACKWARD — retreating during search
                dopamine = -1
            comps[f"search_{self.search_phase}"] = dopamine

        return dopamine, comps