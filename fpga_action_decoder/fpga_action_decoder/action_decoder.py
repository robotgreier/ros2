from typing import List, Optional


class ActionDecoder:
    """
    Decodes a one-hot FPGA action spike vector into an action index.

    Action convention:
      0 = LEFT
      1 = BACKWARD
      2 = RIGHT
      3 = FORWARD
    """

    VALID_ACTIONS = {0, 1, 2, 3}

    def decode_one_hot(self, spikes: List[int]) -> Optional[int]:
        """
        Return the winning action index if the vector is valid one-hot.
        Return None if invalid.
        """
        if len(spikes) != 4:
            return None

        active = [i for i, bit in enumerate(spikes) if bit == 1]

        if len(active) != 1:
            return None

        action = active[0]
        if action not in self.VALID_ACTIONS:
            return None

        return action