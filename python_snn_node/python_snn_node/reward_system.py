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


def reward_signal(seen, pos, action_idx, reward_shift, punish_shift):
    """Returns (dopamine_shift, dopamine_sign, dopamine_enable)."""
    if seen and (
        (pos == 0  and action_idx == 1) or
        (pos == -1 and action_idx == 0) or
        (pos == +1 and action_idx == 2)
    ):
        return reward_shift, 1, 1   # correct alignment action: reward

    if not seen and action_idx in (0, 2):
        return 0, 1, 1              # searching: weak reward

    return punish_shift, 0, 1      # everything else: punish


class DopamineComputer:
    """
    Reward shaping for continuous pick-deliver-repeat.

    Returns (dopamine_shift, dopamine_sign, dopamine_enable, comps) per timestep.

    Priority (highest overrides lowest):
      1. Grab/drop events
      2. State transitions (progress / regression)
      3. Proximity stop penalty
      4. Lost-target penalty
      5. Alignment/action match (base signal via reward_signal())

    Components without a clean shift/sign/enable analog are commented out.
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
        Encoding:
          000 = none
          001 = right
          010 = center
          100 = left
        Returns: (seen: bool, pos: -1/0/+1/None)
        """
        l, c, r = obj_bits
        if l == 1 and c == 0 and r == 0:
            return True, -1
        if l == 0 and c == 1 and r == 0:
            return True, 0
        if l == 0 and c == 0 and r == 1:
            return True, +1
        return False, None

    def step(
        self,
        obj_bits: list[int],         # [L,C,R]
        proximity_spike: int,        # 0/1 (distance bracket change spike) — unused while prox_spike_gated is commented out
        action_idx: int,             # 0=LEFT, 1=FORWARD, 2=RIGHT
        task_state: int | None,      # UInt8 or None if not yet received
        grab_event: int,             # UInt8
        proximity_stop: bool,        # Bool
        reward_shift: int = 2,       # left-shift for reward magnitude
        punish_shift: int = 0,       # left-shift for punishment magnitude
    ):
        seen, pos = self.decode_object_bits(obj_bits)
        comps: dict[str, tuple] = {}

        # Priority 5 (base): alignment / action match
        d_shift, d_sign, d_enable = reward_signal(seen, pos, action_idx, reward_shift, punish_shift)
        if not seen and d_sign == 1:
            comps["searching"] = (d_shift, d_sign, d_enable)
        else:
            comps["align_action"] = (d_shift, d_sign, d_enable)

        # Priority 4: lost-target penalty
        if seen:
            self.lost_ticks = 0
            self._lost_penalized = False
        else:
            if self.prev_seen:
                self.lost_ticks += 1
                if self.lost_ticks > self.lost_grace_ticks:
                    d_shift, d_sign, d_enable = punish_shift, 0, 1
                    comps["lost_target"] = (d_shift, d_sign, d_enable)

        # Priority 3: proximity stop penalty
        if proximity_stop:
            d_shift, d_sign, d_enable = punish_shift, 0, 1
            comps["proximity_stop"] = (d_shift, d_sign, d_enable)

        # Priority 2: state transitions
        if task_state is not None:
            if self.prev_task_state is not None:
                prev, curr = self.prev_task_state, task_state
                forward = {(0, 1), (1, 2), (2, 3)}
                regress  = {(1, 0), (3, 2)}
                # reset_ok = {(3, 0)}  # allowed reset, no reward or penalty

                if (prev, curr) in forward:
                    d_shift, d_sign, d_enable = reward_shift, 1, 1
                    comps["state_progress"] = (d_shift, d_sign, d_enable)
                elif (prev, curr) in regress:
                    d_shift, d_sign, d_enable = punish_shift, 0, 1
                    comps["state_regress"] = (d_shift, d_sign, d_enable)

            self.prev_task_state = task_state

        # Priority 1: grab/drop events
        if grab_event == EVENT_GRABBED:
            d_shift, d_sign, d_enable = reward_shift, 1, 1
            comps["grabbed"] = (d_shift, d_sign, d_enable)
        elif grab_event == EVENT_DROPPED:
            d_shift, d_sign, d_enable = reward_shift, 1, 1
            comps["dropped"] = (d_shift, d_sign, d_enable)

        # Commented out: gated proximity spike reward
        # Does not map cleanly to a single (shift, sign, enable) signal —
        # multi-condition gating is better handled at the call site if needed.
        # gated = (
        #     (task_state in (APPROACH_ITEM, APPROACH_DROPOFF))
        #     and seen and (pos == 0)
        #     and (not proximity_stop)
        # )
        # if gated and proximity_spike == 1:
        #     d_shift, d_sign, d_enable = 0, 1, 1
        #     comps["prox_spike_gated"] = (d_shift, d_sign, d_enable)

        self.prev_seen = seen
        return d_shift, d_sign, d_enable, comps
