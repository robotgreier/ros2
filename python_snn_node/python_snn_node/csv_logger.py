import csv
import queue
import threading
from datetime import datetime

import os

"""
SNN logging and reward structure
-------------
Logging Modes
-------------
The SNN node supports three logging modes that determine when a row is written to the CSV log. Each row contains:
•	input spike vector
•	network output spikes
•	winner neuron
•	action decision
•	dopamine (reward) signals
Mode A — Input-Triggered Logging
Purpose: Analyze network behavior in response to specific inputs.
•	One CSV row is written for each new input vector received.
•	The logged input is paired with the next network output produced by the timer loop.
•	This mode answers the question:
“Given this sensory input, what did the network decide?”
Best for:
•	supervised learning analysis
•	training dataset creation
•	debugging input→action mapping
________________________________________
Mode B — Timer-Based Logging
Purpose: Observe continuous system behavior over time.
•	One CSV row is written every timer cycle (e.g., 30 Hz).
•	Logs the latest input state and current network output.
•	Produces a steady stream of data.
Best for:
•	time-series plots
•	behavioral analysis
•	monitoring system stability
________________________________________
Mode C — Event-Based Logging
Purpose: Capture meaningful changes while keeping logs compact.
•	A row is written only when something changes, such as:
o	input spike pattern changes
o	winner neuron changes
o	action decision changes
o	dopamine signal changes
This results in sparse logs that highlight important events.
Best for:
•	event-driven SNN analysis
•	debugging behavioral transitions
•	minimizing log size during long runs
________________________________________
Dopamine Reward Signals
The SNN uses a biologically inspired dopamine signal to evaluate behavior.
Dopamine is calculated each timestep and is composed of several components that represent different behavioral objectives.
The total dopamine is the sum of the individual signals.
________________________________________
1. Alignment Reward (dop_align)
Encourages the robot to keep detected objects centered in the camera frame.
Behavior rewarded:
•	object centered → higher reward
•	object visible but off-center → small reward
•	no object visible → small penalty
Purpose:
Helps the robot learn to visually track targets.
________________________________________
2. Action-Match Reward (dop_action)
Rewards actions that correct alignment or move toward a target.
Examples:
•	object left of center + turning left → reward
•	object centered + moving forward → reward
•	turning while searching for objects → small reward
Purpose:
Encourages consistent sensor→action behavior.
________________________________________
3. Lost Target Penalty (dop_lost)
Penalizes the robot when it loses sight of a previously detected object.
Two penalties exist:
•	lost_once – larger penalty when target is first lost
•	lost_tick – small penalty for each timestep the target remains lost
Purpose:
Discourages behaviors that lose track of objects.
________________________________________
4. State Transition Reward (dop_state)
Rewards progress through the task pipeline.
Task states:
State	Meaning
0	SEARCH_ITEM
1	APPROACH_ITEM
2	SEARCH_DROPOFF
3	APPROACH_DROPOFF
Rewards:
•	forward progress (0→1→2→3) → positive reward
•	regression (e.g. losing item) → penalty
•	reset (3→0 after delivery) → neutral
Purpose:
Encourages successful completion of the pickup-delivery cycle.
________________________________________
5. Grab/Drop Success Reward (dop_grabdrop)
Large reward for successful manipulation events.
Events:
•	object successfully grabbed
•	object successfully dropped at the target
Purpose:
Reinforces completion of key task milestones.
________________________________________
6. Proximity Stop Penalty (dop_prox_stop)
Penalty when the emergency stop system activates due to a nearby obstacle.
Purpose:
Discourages unsafe navigation and collisions.
________________________________________
7. Proximity Approach Reward (dop_prox_approach)
Small reward when the robot moves closer to the target.
Based on spikes from the ultrasonic distance brackets.
Reward is applied only when:
•	robot is in an APPROACH state
•	target object is visible and centered
•	proximity stop is not active
Purpose:
Encourages controlled approach toward objects.
________________________________________
Logged Dopamine Signals
Each CSV row contains:
Column	Meaning
dopamine_total	Total dopamine value for the timestep
dop_align	Alignment reward
dop_action	Action-matching reward
dop_lost	Lost-target penalties
dop_state	Task progression reward
dop_grabdrop	Successful manipulation reward
dop_prox_stop	Obstacle avoidance penalty
dop_prox_approach	Approach progress reward



CSV files are written asynchronously (non-blocking) to:
    ~/.ros/snn_logs/
One file is created per run with timestamped filename.
"""

class CsvAsyncLogger:
    def __init__(self, filepath: str, header: list[str], queue_size: int = 5000, flush_hz: float = 10.0):
        self.filepath = filepath
        self.header = header
        self.q = queue.Queue(maxsize=queue_size)
        self.flush_hz = max(flush_hz, 0.1)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped = 0

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f = open(filepath, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.header)
        self._f.flush()

        self._thread.start()

    def push(self, row: list):
        try:
            self.q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        # Flush every 1/flush_hz seconds
        period = 1.0 / self.flush_hz
        next_flush = datetime.now().timestamp() + period

        while not self._stop.is_set():
            try:
                row = self.q.get(timeout=0.1)
                self._w.writerow(row)
            except queue.Empty:
                pass

            now = datetime.now().timestamp()
            if now >= next_flush:
                self._f.flush()
                next_flush = now + period

        # Final drain + close
        while True:
            try:
                row = self.q.get_nowait()
                self._w.writerow(row)
            except queue.Empty:
                break
        self._f.flush()
        self._f.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)