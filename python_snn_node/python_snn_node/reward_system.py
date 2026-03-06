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
    Reward shaping for continuous pick-deliver-repeat:

    - Dense shaping:
      * alignment reward from object_rec (L/C/R/none)
      * action-match reward (turn toward target; go forward when centered)
      * gated proximity-bracket spike reward during APPROACH + centered

    - Sparse rewards:
      * state progress reward (0->1->2->3)
      * grab/drop success reward (from grab_node event)

    - Penalties:
      * losing target after having it (after grace ticks)
      * proximity_stop (near collision / unsafe)
      * regressions that indicate failure (1->0, 3->2)
      * IMPORTANT: 3->0 reset is allowed (no penalty).
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
        proximity_spike: int,        # 0/1 (distance bracket change spike)
        action_idx: int,             # 0=LEFT, 1=FORWARD, 2=RIGHT
        task_state: int | None,      # UInt8 or None if not yet received
        grab_event: int,             # UInt8
        proximity_stop: bool         # Bool
    ):
        seen, pos = self.decode_object_bits(obj_bits)

        dopamine = 0.0
        comps: dict[str, float] = {}

        # (1) Alignment shaping: prefer target centered, mild penalty if unseen
        if not seen:
            comps["align"] = -0.02
        elif pos == 0:
            comps["align"] = +0.05
        else:
            comps["align"] = +0.01
        dopamine += comps["align"]

        # (2) Action-match shaping: reward actions that correct alignment / approach
        act = 0.0
        if seen:
            if pos == 0 and action_idx == 1:
                act += 0.10  # centered + forward
            elif pos == -1 and action_idx == 0:
                act += 0.06  # left-of-center + turn left
            elif pos == +1 and action_idx == 2:
                act += 0.06  # right-of-center + turn right
        else:
            if action_idx in (0, 2):
                act += 0.02  # turning while searching
        comps["action_match"] = act
        dopamine += act

        # (3) Lost-target penalty: if target disappears after being seen
        if seen:
            self.lost_ticks = 0
            self._lost_penalized = False
        else:
            if self.prev_seen:
                self.lost_ticks += 1
                if self.lost_ticks > self.lost_grace_ticks:
                    if not self._lost_penalized:
                        dopamine -= 0.25
                        comps["lost_once"] = -0.25
                        self._lost_penalized = True
                    dopamine -= 0.01
                    comps["lost_tick"] = comps.get("lost_tick", 0.0) - 0.01

        # (4) State transition rewards (loop-aware): 3->0 reset is allowed
        if task_state is not None:
            if self.prev_task_state is not None:
                prev, curr = self.prev_task_state, task_state

                forward = {(0, 1), (1, 2), (2, 3)}
                reset_ok = {(3, 0)}
                regress = {(1, 0), (3, 2)}

                if (prev, curr) in forward:
                    dopamine += 0.5
                    comps["state_progress"] = +0.5
                elif (prev, curr) in regress:
                    dopamine -= 0.5
                    comps["state_regress"] = -0.5
                elif (prev, curr) in reset_ok:
                    comps["state_reset_ok"] = 0.0
                else:
                    comps["state_other"] = 0.0

            self.prev_task_state = task_state

        # (5) Grab/drop success rewards (big, sparse)
        if grab_event == EVENT_GRABBED:
            dopamine += 2.0
            comps["grabbed"] = +2.0
        elif grab_event == EVENT_DROPPED:
            dopamine += 2.0
            comps["dropped"] = +2.0

        # (6) Proximity stop penalty: near collision / unsafe driving
        if proximity_stop:
            dopamine -= 0.4
            comps["proximity_stop"] = -0.4

        # (7) Gated proximity spike reward: only during APPROACH + target centered
        # This uses your "higher spike frequency when close" idea, but avoids wall-farming.
        gated = (
            (task_state in (APPROACH_ITEM, APPROACH_DROPOFF))
            and seen and (pos == 0)
            and (not proximity_stop)
        )
        if gated and proximity_spike == 1:
            dopamine += 0.08
            comps["prox_spike_gated"] = +0.08

        self.prev_seen = seen
        return dopamine, comps