import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import (UInt8,
                           Float32,
                             Bool,
                               UInt8MultiArray,
                                 Int32,
                                   Int32MultiArray,
                                     String)

from geometry_msgs.msg import Twist

from .LIF_SNN_network import SNNLayer

from ament_index_python.packages import get_package_share_directory
import os

import csv
import queue
import threading
from datetime import datetime

ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT"] # index 0=LEFT, 1=FORWARD, 2=RIGHT

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


class RewardComputer:
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

        reward = 0.0
        comps: dict[str, float] = {}

        # (1) Alignment shaping: prefer target centered, mild penalty if unseen
        if not seen:
            comps["align"] = -0.02
        elif pos == 0:
            comps["align"] = +0.05
        else:
            comps["align"] = +0.01
        reward += comps["align"]

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
        reward += act

        # (3) Lost-target penalty: if target disappears after being seen
        if seen:
            self.lost_ticks = 0
            self._lost_penalized = False
        else:
            if self.prev_seen:
                self.lost_ticks += 1
                if self.lost_ticks > self.lost_grace_ticks:
                    if not self._lost_penalized:
                        reward -= 0.25
                        comps["lost_once"] = -0.25
                        self._lost_penalized = True
                    reward -= 0.01
                    comps["lost_tick"] = comps.get("lost_tick", 0.0) - 0.01

        # (4) State transition rewards (loop-aware): 3->0 reset is allowed
        if task_state is not None:
            if self.prev_task_state is not None:
                prev, curr = self.prev_task_state, task_state

                forward = {(0, 1), (1, 2), (2, 3)}
                reset_ok = {(3, 0)}
                regress = {(1, 0), (3, 2)}

                if (prev, curr) in forward:
                    reward += 0.5
                    comps["state_progress"] = +0.5
                elif (prev, curr) in regress:
                    reward -= 0.5
                    comps["state_regress"] = -0.5
                elif (prev, curr) in reset_ok:
                    comps["state_reset_ok"] = 0.0
                else:
                    comps["state_other"] = 0.0

            self.prev_task_state = task_state

        # (5) Grab/drop success rewards (big, sparse)
        if grab_event == EVENT_GRABBED:
            reward += 2.0
            comps["grabbed"] = +2.0
        elif grab_event == EVENT_DROPPED:
            reward += 2.0
            comps["dropped"] = +2.0

        # (6) Proximity stop penalty: near collision / unsafe driving
        if proximity_stop:
            reward -= 0.4
            comps["proximity_stop"] = -0.4

        # (7) Gated proximity spike reward: only during APPROACH + target centered
        # This uses your "higher spike frequency when close" idea, but avoids wall-farming.
        gated = (
            (task_state in (APPROACH_ITEM, APPROACH_DROPOFF))
            and seen and (pos == 0)
            and (not proximity_stop)
        )
        if gated and proximity_spike == 1:
            reward += 0.08
            comps["prox_spike_gated"] = +0.08

        self.prev_seen = seen
        return reward, comps


"""
CSV LOGGING MODES

The SNN node supports three logging modes controlled by the ROS2 parameter:
    log_mode := "A" | "B" | "C"

Mode A — Input-triggered logging (Input → Next Output pairing)
---------------------------------------------------------------
• One CSV row per NEW input message.
• The input vector is paired with the NEXT network output
  produced by the timer loop.
• Best for supervised-style analysis:
    "Given this input, what did the network do?"

Mode B — Timer-based logging (Fixed rate, e.g., 30 Hz)
-------------------------------------------------------
• One CSV row per timer tick.
• Logs the latest input state and the current output.
• Produces steady-rate data (e.g., 30 rows/sec).
• Best for time-series analysis and plotting behavior over time.

Mode C — Event-based logging (Change detection)
-----------------------------------------------
• Logs only when something changes.
• Default change criteria:
    - Input vector changes OR
    - Winner neuron changes OR
    - Decision string changes
• Produces compact datasets.
• Best for event-driven / sparse SNN analysis.

All modes log:
    t_input_ns
    input_0 ... input_15
    t_output_ns
    winner
    decision
    spikes_0 ... spikes_N

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

class SNNNode(Node):
    """
    Python SNN node:
    - Reads packed spikes (0/1) from /snn/input (UInt8MultiArray)
    - Runs LIF SNN with dopamine learning and publishes:
        * /cmd_vel/snn (geometry_msgs/Twist) for robot control
        * /snn/decision (string) with the action name (LEFT, FORWARD, RIGHT)
        * /snn/winner (Int32)
        * /snn/spikes (INT32MuliArray) for debugging (spikes of output neurons)
    - Training mode: listens to /snn/correct_output (Int32), uses dopamine learning to adjust synaptic weights.

    """
    def __init__(self):
        super().__init__('python_snn_node')

        # --- Parameters ---
        # Input/Output Setup
        self.declare_parameter('input_mode', 'packed')
        self.declare_parameter('input_topic', '/snn/input')
        self.declare_parameter('pack_order', ['keypoints_grid','proximity', 'object_rec'])

        # Channel sizes (matches YAML)
        self.declare_parameter('keypoints_grid_size', 12)
        self.declare_parameter('proximity_size', 1)
        self.declare_parameter('object_rec_size', 3)

        # Action parameters
        self.declare_parameter('num_actions', 3)

        # Driving parameters
        self.declare_parameter('timer_hz', 30.0)
        self.declare_parameter('idle_timeout_sec', 1.0)
        self.declare_parameter('use_proximity_for_stop', False)
        self.declare_parameter('proximity_stop_active_high', True)

        # Robot speed parameters
        self.declare_parameter('forward_speed', 0.25)
        self.declare_parameter('turn_speed', 0.6)

        # Neuron parameters (Changed to lowercase 'threshold')
        self.declare_parameter('decay', 0.75)
        self.declare_parameter('threshold', 4.0)
        self.declare_parameter('reset', 0.0)

        # Synapse & Learning parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('learning_rate', 0.250)
        self.declare_parameter('initial_weight', 0.3)
        self.declare_parameter('t_pre', 3.0)
        self.declare_parameter('t_post', 3.0)
        self.declare_parameter('tau_e_shift', 4.0)
        self.declare_parameter('dw_pos', 0.25)
        self.declare_parameter('dw_neg', 0.03125)
        self.declare_parameter('min_weight', 0.03125)
        self.declare_parameter('max_weight', 1.0)
        self.declare_parameter('dopamine_correct', 1.0)
        self.declare_parameter('dopamine_wrong', -0.5)
        self.declare_parameter('learning_mode', 'rstdp')
        self.declare_parameter('seed', 42)

        # ---- CSV logging ----
        self.declare_parameter('log_enable', False)
        self.declare_parameter('log_mode', 'A')  # 'A', 'B', or 'C'
        self.declare_parameter('log_dir', '')    # default resolved to ~/.ros/snn_logs
        self.declare_parameter('log_queue_size', 5000)
        self.declare_parameter('log_flush_hz', 10.0)

        # ---- Reward-related topics ----
        self.declare_parameter('task_state_topic', '/task/state')
        self.declare_parameter('grab_event_topic', '/grab_node/event')
        self.declare_parameter('proximity_stop_topic', '/proximity_stop')
        self.declare_parameter('lost_grace_ticks', 5)

        self.task_state: int | None = None
        self.grab_event: int = EVENT_IDLE
        self.proximity_stop: bool = False

        self.create_subscription(
            UInt8,
            self.get_parameter('task_state_topic').value,
            self._on_task_state,
            10
        )
        self.create_subscription(
            UInt8,
            self.get_parameter('grab_event_topic').value,
            self._on_grab_event,
            10
        )
        self.create_subscription(
            Bool,
            self.get_parameter('proximity_stop_topic').value,
            self._on_proximity_stop,
            10
        )

        # Reward publisher for plotting/debugging
        self.pub_reward = self.create_publisher(Float32, '/snn/reward', 10)

        self.rewarder = RewardComputer(
            lost_grace_ticks=int(self.get_parameter('lost_grace_ticks').value)
        )


        def _on_task_state(self, msg: UInt8):
            self.task_state = int(msg.data)


        def _on_grab_event(self, msg: UInt8):
            self.grab_event = int(msg.data)


        def _on_proximity_stop(self, msg: Bool):
            self.proximity_stop = bool(msg.data)

        # --- Read Parameters ---
        self.input_mode = str(self.get_parameter('input_mode').value).lower().strip()
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.pack_order = list(self.get_parameter('pack_order').value)

        self.channel_sizes = {
            'proximity': int(self.get_parameter('proximity_size').value),
            'keypoints_grid': int(self.get_parameter('keypoints_grid_size').value),
            'object_rec': int(self.get_parameter('object_rec_size').value)
        }

        self.timer_hz = float(self.get_parameter('timer_hz').value)
        self.idle_timeout_sec = float(self.get_parameter('idle_timeout_sec').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)

        self.num_actions = int(self.get_parameter('num_actions').value)
        self.output_size = self.num_actions

        # Neuron params
        self.decay = float(self.get_parameter('decay').value)
        self.threshold = float(self.get_parameter('threshold').value)
        self.reset = float(self.get_parameter('reset').value)

        # Synapse params
        self.training_mode = bool(self.get_parameter('training_mode').value)
        self.learning_rate = float(self.get_parameter('learning_rate').value)
        self.initial_weight = float(self.get_parameter('initial_weight').value)
        self.t_pre = float(self.get_parameter('t_pre').value)
        self.t_post = float(self.get_parameter('t_post').value)
        self.tau_e_shift = float(self.get_parameter('tau_e_shift').value)
        self.dw_pos = float(self.get_parameter('dw_pos').value)
        self.dw_neg = float(self.get_parameter('dw_neg').value)
        self.max_weight = float(self.get_parameter('max_weight').value)
        self.min_weight = float(self.get_parameter('min_weight').value)
        self.dopamine_correct = float(self.get_parameter('dopamine_correct').value)
        self.dopamine_wrong = float(self.get_parameter('dopamine_wrong').value)
        self.learning_mode = str(self.get_parameter('learning_mode').value)
        
        seed = self.get_parameter('seed').value
        if seed is not None:
            np.random.seed(int(seed))

        # --- Derived Logic ---
        self.segment_offsets = self._compute_offsets(self.pack_order, self.channel_sizes)
        self.input_size = sum(self.channel_sizes.get(name, 0) for name in self.pack_order)

        neuron_params = {"decay": self.decay, "threshold": self.threshold, "reset": self.reset}
        synapse_params = {
            "learning_rate": self.learning_rate, "w_init": self.initial_weight, 
            "t_pre": self.t_pre, "t_post": self.t_post, "tau_e_shift": self.tau_e_shift, 
            "dw_pos": self.dw_pos, "dw_neg": self.dw_neg, 
            "w_min": self.min_weight, "w_max": self.max_weight, 
            "learning_mode": self.learning_mode
        }

        self.network = SNNLayer(
            n_inputs=self.input_size, 
            n_outputs=self.output_size, 
            neuron_params=neuron_params, 
            synapse_params=synapse_params
        )

        share_dir = get_package_share_directory('python_snn_node')
        weight_path = os.path.join(share_dir, 'config', 'weights.mem')

        self.network.load_weights(weight_file=weight_path, scale=127)

        # --- State and Communication ---
        self.last_input_stamp = self.get_clock().now()
        self.last_vector = np.zeros(self.input_size, dtype=np.float32)
        self.correct_output = -1

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(UInt8MultiArray, self.input_topic, self.cb_packed, qos_sensor)
        self.create_subscription(Int32, '/snn/correct_output', self.cb_correct, 10)

        self.pub_winner = self.create_publisher(Int32, '/snn/winner', 10)
        self.pub_spikes = self.create_publisher(Int32MultiArray, '/snn/spikes', 10)
        self.pub_decision = self.create_publisher(String, '/snn/decision', 10)
        
        self.declare_parameter('cmd_vel_topic', '/cmd_vel/snn')
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)

        period = 1.0 / max(self.timer_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(f"SNN Node initialized: {self.input_size} in -> {self.output_size} out")

        # ---- Logging setup ----
        self.log_enable = bool(self.get_parameter('log_enable').value)
        self.log_mode = str(self.get_parameter('log_mode').value).upper().strip()
        log_dir = str(self.get_parameter('log_dir').value).strip()
        if log_dir == '':
            log_dir = os.path.expanduser('~/.ros/snn_logs')

        self.logger_csv = None
        self.pending_input = None  # (t_input_ns, input_vector_list)
        self.last_logged = None    # used by mode C

        if self.log_enable:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(log_dir, f"snn_log_{ts}.csv")

            # Build header
            header = []
            header.append("t_input_ns")
            header += [f"input_{i}" for i in range(self.input_size)]
            header.append("t_output_ns")
            header.append("winner")
            header.append("decision")
            header += [f"spikes_{i}" for i in range(self.output_size)]

            self.logger_csv = CsvAsyncLogger(
                filepath=filepath,
                header=header,
                queue_size=int(self.get_parameter('log_queue_size').value),
                flush_hz=float(self.get_parameter('log_flush_hz').value),
            )

            self.get_logger().info(f"CSV logging enabled -> {filepath} (mode {self.log_mode})")

    ### For training ###
    def _extract_object_bits_from_last(self) -> list[int]:
        """Extract [L,C,R] from the last 3 packed input bits."""
        # self.last_vector is float32, values are 0.0/1.0 from cb_packed
        v = self.last_vector
        if v is None or v.size < 3:
            return [0, 0, 0]
        return [int(v[-3]), int(v[-2]), int(v[-1])]


    def _extract_proximity_spike_from_last(self) -> int:
        """
        Extract proximity bit from packed input.
        With pack_order ['keypoints_grid','proximity','object_rec'] and sizes 12,1,3:
        proximity is index 12.
        If your pack_order/sizes change, update this to compute offsets dynamically.
        """
        v = self.last_vector
        if v is None or v.size < 13:
            return 0
        return int(v[12])

    ### /For training ###

    # Helpers 
    def _compute_offsets(self, order, sizes):
        offsets = {}
        pos = 0
        for name in order:
            size = sizes.get(name, 0)
            offsets[name] = (pos, pos + size)
            pos += size
        return offsets

    # Callbacks
    def cb_packed(self, msg: UInt8MultiArray):
        data = np.array(msg.data, dtype=np.uint8)
        if data.size != self.input_size:
            self.get_logger().warn(
                f"/snn/input len={data.size} != expected {self.input_size} "
                f"(pack_order={self.pack_order}, sizes={self.channel_sizes})"
            )
            return

        # Convert 0/1 uint8 spikes to float32 and store in last_vector
        self.last_vector[:] = data.astype(np.float32)
        self.last_input_stamp = self.get_clock().now()

        if self.logger_csv is not None and self.log_mode == "A":
            t_input_ns = int(self.last_input_stamp.nanoseconds)
            # store as plain Python list so it won't change later
            self.pending_input = (t_input_ns, self.last_vector.astype(int).tolist())

    def cb_correct(self, msg: Int32):
        self.correct_output = int(msg.data)

    # Timer
    def on_timer(self):
        age_sec = (self.get_clock().now() - self.last_input_stamp).nanoseconds * 1e-9
        if age_sec > self.idle_timeout_sec:
            self.publish_stop(f"stale input {age_sec:.2f}s > {self.idle_timeout_sec}s")
            return



        ##### Run network #####

        # Forward pass
        output_spikes = self.network.forward(input_spikes=self.last_vector)

        # Find idx of winning neuron
        winner_idx = self.network.winner_takes_all(output_spikes=output_spikes)
        
        """# Apply reward, uncomment for reward based training
        self.network.apply_reward(dopamine=0, winner_idx=winner_idx)"""


        # Normal actuation
        decision = self.publish_cmd_from_winner(int(winner_idx), force_stop=False)

        # Debug publish winners and spikes
        self.pub_winner.publish(Int32(data=int(winner_idx)))

        spk_msg = Int32MultiArray()
        spk_msg.data = [int(x) for x in output_spikes]
        self.pub_spikes.publish(spk_msg)

        # ---- Reward computation ----
        obj_bits = self._extract_object_bits_from_last()
        prox_spike = self._extract_proximity_spike_from_last()

        reward, reward_comps = self.rewarder.step(
            obj_bits=obj_bits,
            proximity_spike=prox_spike,
            action_idx=int(winner_idx),
            task_state=self.task_state,
            grab_event=self.grab_event,
            proximity_stop=self.proximity_stop
        )

        self.pub_reward.publish(Float32(data=float(reward)))

        # ---- Optional: apply reward to learning rule (when you enable training) ----
        # self.network.apply_reward(dopamine=reward, winner_idx=int(winner_idx))

        # ---- Add to your existing logger ----
        # Recommended: store reward + some context so you can tune later.
        # If you already log pending_input, just append these fields to the same row.
        # Example (pseudo):
        # self.logger_csv.write_row(..., reward=reward, state=self.task_state,
        #                           grab_event=self.grab_event,
        #                           proximity_stop=int(self.proximity_stop),
        #                           **reward_comps)

        if self.logger_csv is not None:
            t_output_ns = int(self.get_clock().now().nanoseconds)
            winner = int(winner_idx)
            spikes_list = [int(x) for x in output_spikes]  # length = output_size

            # Choose input to log depending on mode
            if self.log_mode == "A":
                if self.pending_input is None:
                    return  # no new input since last log
                t_input_ns, input_list = self.pending_input
                self.pending_input = None  # consume it

                row = [t_input_ns] + input_list + [t_output_ns, winner, decision] + spikes_list
                self.logger_csv.push(row)

            elif self.log_mode == "B":
                # always log latest input (even if unchanged)
                t_input_ns = int(self.last_input_stamp.nanoseconds)
                input_list = self.last_vector.astype(int).tolist()
                row = [t_input_ns] + input_list + [t_output_ns, winner, decision] + spikes_list
                self.logger_csv.push(row)

            elif self.log_mode == "C":
                # log only when something changes (simple default: winner OR any input bit change)
                t_input_ns = int(self.last_input_stamp.nanoseconds)
                input_list = self.last_vector.astype(int).tolist()

                signature = (tuple(input_list), winner, decision, tuple(spikes_list))
                if self.last_logged != signature:
                    row = [t_input_ns] + input_list + [t_output_ns, winner, decision] + spikes_list
                    self.logger_csv.push(row)
                    self.last_logged = signature

    def on_proximity_penalty(self, winner_idx: int):
        """
        Called when /proximity_stop triggers. winner_idx is what the policy wanted to do.
        Implement your punishment here once we confirm how net.step applies dopamine.
        """
        self.network.apply_reward(dopamine=self.dopamine_wrong, winner_idx=winner_idx)


    def publish_cmd_from_winner(self, winner_idx: int, force_stop: bool = False):
        cmd = Twist()
        decision = "IDLE"

        if force_stop:
            decision = "STOP_PROXIMITY"
        else:
            if winner_idx == 0:      # LEFT
                cmd.linear.x = 0.0
                cmd.angular.z = +self.turn_speed
                decision = ACTION_NAMES[0]
            elif winner_idx == 1:    # FORWARD
                cmd.linear.x = self.forward_speed
                cmd.angular.z = 0.0
                decision = ACTION_NAMES[1]
            elif winner_idx == 2:    # RIGHT
                cmd.linear.x = 0.0
                cmd.angular.z = -self.turn_speed
                decision = ACTION_NAMES[2]
            else:
                decision = "UNKNOWN"

        self.cmd_vel_pub.publish(cmd)
        self.pub_decision.publish(String(data=decision))

        return decision

    def publish_stop(self, reason: str = ""):
        if reason:
            self.get_logger().warn(f"STOP: {reason}")
        self.cmd_vel_pub.publish(Twist())
        self.pub_decision.publish(String(data="IDLE"))

    def destroy_node(self):
        if self.logger_csv is not None:
            self.logger_csv.close()
            self.get_logger().info(f"CSV logger closed (dropped_rows={self.logger_csv.dropped})")
        super().destroy_node()


def main():
    rclpy.init()
    node = SNNNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()