# ---- Grab event codes (match grab_node) ----
EVENT_IDLE = 0
EVENT_GRABBED = 1
EVENT_DROPPED = 2
EVENT_BUSY = 3

# ---- Task states (match task_manager) ----
SEARCH_ITEM = 0
APPROACH_ITEM = 1
SEARCH_DROPOFF = 2
APPROACH_DROPOFF = 3


class DopamineComputer:
    """
    Reward shaping for continuous pick-deliver-repeat.

    Returns (dopamine, comps) per timestep where dopamine is a signed integer
    and comps is a dict mapping component names to their signed integer contributions.

    Priority (highest overrides lowest):
      1. Grab/drop events
      2. State transitions (progress / regression)
      3. Proximity stop penalty
      4. Lost-target penalty
      5. Alignment/action match (base signal via reward_signal())
    """

    def __init__(self, lost_grace_ticks: int = 5):
        self.prev_task_state: int | None = None
        self.prev_seen: bool = False
        self.lost_ticks: int = 0
        self.lost_grace_ticks = lost_grace_ticks
        self._lost_penalized = False

    @staticmethod
    def decode_object_bits(obj_bits: list[int]):
        """
        One-hot encoding over N horizontal zones (works for any odd N).
        For N=5:
          10000 = far left  (-2)
          01000 = left      (-1)
          00100 = center    ( 0)
          00010 = right     (+1)
          00001 = far right (+2)
          00000 = none
        Returns: (seen: bool, pos: int/None)
        """
        n = len(obj_bits)
        for i, bit in enumerate(obj_bits):
            if bit == 1:
                return True, i - (n // 2)
        return False, None

    def step(
        self,
        obj_bits: list[int],         # [L,C,R]
        proximity_spike: int,        # 0/1 (distance bracket change spike)
        action_idx: int,             # 0=LEFT, 1=FORWARD, 2=RIGHT
        task_state: int | None,      # UInt8 or None if not yet received
        grab_event: int,             # UInt8
        proximity_stop: bool,        # Bool
    ):
        seen, pos = self.decode_object_bits(obj_bits)
        comps: dict[str, int] = {}

        # Priority 5 (base): alignment / action match
        if seen and pos is not None and (
            (pos == 0 and action_idx == 1) or
            (pos  < 0 and action_idx == 0) or
            (pos  > 0 and action_idx == 2)
        ):
            dopamine = 2   # correct action: reward
        elif not seen and action_idx in (0, 2):
            dopamine = 1   # searching: weak reward
        else:
            dopamine = -2  # everything else: punish

        if not seen and dopamine > 0:
            comps["searching"] = dopamine
        else:
            comps["align_action"] = dopamine

        """# Priority 4: lost-target penalty
        if seen:
            self.lost_ticks = 0
            self._lost_penalized = False
        else:
            if self.prev_seen:
                self.lost_ticks += 1
                if self.lost_ticks > self.lost_grace_ticks:
                    dopamine = -1
                    comps["lost_target"] = dopamine"""

        """# Priority 3: proximity stop penalty
        if proximity_stop:
            dopamine = -1
            comps["proximity_stop"] = dopamine"""

        """# Priority 2: state transitions
        if task_state is not None:
            if self.prev_task_state is not None:
                prev, curr = self.prev_task_state, task_state
                forward = {(0, 1), (1, 2), (2, 3)}
                regress  = {(1, 0), (3, 2)}

                if (prev, curr) in forward:
                    dopamine = +1
                    comps["state_progress"] = dopamine
                elif (prev, curr) in regress:
                    dopamine = -1
                    comps["state_regress"] = dopamine

            self.prev_task_state = task_state"""

        """# Priority 1: grab/drop events
        if grab_event == EVENT_GRABBED:
            dopamine = +1
            comps["grabbed"] = dopamine
        elif grab_event == EVENT_DROPPED:
            dopamine = +1
            comps["dropped"] = dopamine"""

        # Commented out: gated proximity spike reward
        # gated = (
        #     (task_state in (APPROACH_ITEM, APPROACH_DROPOFF))
        #     and seen and (pos == 0)
        #     and (not proximity_stop)
        # )
        # if gated and proximity_spike == 1:
        #     dopamine = +1
        #     comps["prox_spike_gated"] = dopamine

        #self.prev_seen = seen
        return dopamine, comps
