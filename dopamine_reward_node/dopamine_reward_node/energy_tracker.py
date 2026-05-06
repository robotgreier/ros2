from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EnergyPhaseResult:
    pickup_idx: int
    phase_idx: int
    energy_joules: float
    duration_s: float
    average_joules: Optional[float]
    delta_joules: Optional[float]


class EnergyTracker:
    """
    Tracks energy used in 12 phase categories:
      3 pickups x 4 phases

    Phase definitions:
      phase 0: SEARCH_ITEM      -> APPROACH_ITEM
      phase 1: APPROACH_ITEM    -> SEARCH_DROPOFF
      phase 2: SEARCH_DROPOFF   -> APPROACH_DROPOFF
      phase 3: APPROACH_DROPOFF -> SEARCH_ITEM

    The tracker:
    - starts timing/energy when the phase start state is first entered
    - ignores oscillation/regression inside the phase
    - completes on first entry to the phase end state
    - maintains running averages per pickup/phase
    """

    SEARCH_ITEM = 0
    APPROACH_ITEM = 1
    SEARCH_DROPOFF = 2
    APPROACH_DROPOFF = 3

    PHASE_START_STATES = [SEARCH_ITEM, APPROACH_ITEM, SEARCH_DROPOFF, APPROACH_DROPOFF]
    PHASE_END_STATES   = [APPROACH_ITEM, SEARCH_DROPOFF, APPROACH_DROPOFF, SEARCH_ITEM]

    def __init__(self) -> None:
        # Running stats: 3 pickups x 4 phases
        self.running_avg: List[List[Optional[float]]] = [
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
        ]
        self.sample_count: List[List[int]] = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        self.phase_start_time_s: Optional[float] = None

        # Episode/phase state
        self.current_pickup_idx: int = 0
        self.current_phase_idx: Optional[int] = None
        self.phase_active: bool = False

        # Energy integration state
        self.phase_energy_joules: float = 0.0
        self.last_power_w: Optional[float] = None
        self.last_power_time_s: Optional[float] = None

        # Task state tracking
        self.prev_task_state: Optional[int] = None

        # Phase tracking
        self.next_phase_idx: int = 0

    def reset_episode(self) -> None:
        """
        Reset episode-local progress but keep learned averages.
        """
        self.current_pickup_idx = 0
        self.current_phase_idx = None
        self.phase_active = False
        self.phase_energy_joules = 0.0
        self.last_power_w = None
        self.last_power_time_s = None
        self.prev_task_state = None
        self.next_phase_idx = 0
        self.phase_start_time_s = None

    def update_power(self, power_w: float, now_s: float) -> None:
        """
        Integrate energy over time while a phase is active.
        """
        if self.last_power_time_s is None:
            self.last_power_time_s = now_s
            self.last_power_w = power_w
            return

        dt = now_s - self.last_power_time_s
        if dt < 0.0:
            dt = 0.0

        if self.phase_active and self.last_power_w is not None:
            self.phase_energy_joules += self.last_power_w * dt

        self.last_power_time_s = now_s
        self.last_power_w = power_w

    def on_task_state(self, task_state: int) -> Optional[EnergyPhaseResult]:
        """
        Process a task-state update.

        Returns an EnergyPhaseResult when a phase completes, else None.
        """
        # If no phase is active, see if we should start one
        if not self.phase_active:
            phase_to_start = self._expected_phase_start_for_current_pickup()
            if phase_to_start is not None:
                start_state = self.PHASE_START_STATES[phase_to_start]
                if task_state == start_state:
                    self.current_phase_idx = phase_to_start
                    self.phase_active = True
                    self.phase_energy_joules = 0.0
                    self.phase_start_time_s = self.last_power_time_s

        result = None

        # If a phase is active, see if it has reached its first valid end state
        if self.phase_active and self.current_phase_idx is not None:
            end_state = self.PHASE_END_STATES[self.current_phase_idx]
            if task_state == end_state:
                result = self._complete_current_phase()

        self.prev_task_state = task_state
        return result

    def _expected_phase_start_for_current_pickup(self) -> Optional[int]:
        """
        Return the next phase expected in the current pickup.
        """
        if 0 <= self.current_pickup_idx <= 2 and 0 <= self.next_phase_idx <= 3:
            return self.next_phase_idx
        return None

    def _complete_current_phase(self) -> EnergyPhaseResult:
        """
        Finalize the current phase, update running average, and advance progress.
        """
        pickup = self.current_pickup_idx
        phase = self.current_phase_idx
        energy = self.phase_energy_joules

        prev_avg = self.running_avg[pickup][phase]
        prev_count = self.sample_count[pickup][phase]

        if prev_avg is None:
            new_avg = energy
            delta = None
        else:
            delta = energy - prev_avg
            new_avg = ((prev_avg * prev_count) + energy) / (prev_count + 1)

        self.running_avg[pickup][phase] = new_avg
        self.sample_count[pickup][phase] = prev_count + 1

        end_time_s = self.last_power_time_s

        if self.phase_start_time_s is not None and end_time_s is not None:
            start_time_s = self.phase_start_time_s
            duration_s = max(0.0, end_time_s - start_time_s)
        else:
            start_time_s = 0.0
            end_time_s = 0.0
            duration_s = 0.0

        result = EnergyPhaseResult(
            pickup_idx=pickup,
            phase_idx=phase,
            energy_joules=energy,
            duration_s=duration_s,
            average_joules=prev_avg,
            delta_joules=delta,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )

        # Advance to next phase / pickup
        if phase < 3:
            self.next_phase_idx += 1
        else:
            self.next_phase_idx = 0
            if self.current_pickup_idx < 2:
                self.current_pickup_idx += 1
        
        # Mark this phase as completed
        self.phase_active = False
        self.current_phase_idx = None
        self.phase_energy_joules = 0.0
        self.phase_start_time_s = None

        return result