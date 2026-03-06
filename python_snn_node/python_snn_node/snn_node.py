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
from .reward_system import DopamineComputer, EVENT_IDLE
from .csv_logger import CsvAsyncLogger

from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT"] # index 0=LEFT, 1=FORWARD, 2=RIGHT

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

        # Dopamine publisher for plotting/debugging
        self.pub_dopamine = self.create_publisher(Float32, '/snn/dopamine', 10)

        self.dopamine_computer = DopamineComputer(
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
            header.append("dopamine_total")
            header.append("dop_align")
            header.append("dop_action")
            header.append("dop_lost")
            header.append("dop_state")
            header.append("dop_grabdrop")
            header.append("dop_prox_stop")
            header.append("dop_prox_approach")
            header += [f"spikes_{i}" for i in range(self.output_size)]

            self.logger_csv = CsvAsyncLogger(
                filepath=filepath,
                header=header,
                queue_size=int(self.get_parameter('log_queue_size').value),
                flush_hz=float(self.get_parameter('log_flush_hz').value),
            )

            self.get_logger().info(f"CSV logging enabled -> {filepath} (mode {self.log_mode})")

    def _on_task_state(self, msg: UInt8):
        self.task_state = int(msg.data)


    def _on_grab_event(self, msg: UInt8):
        self.grab_event = int(msg.data)


    def _on_proximity_stop(self, msg: Bool):
        self.proximity_stop = bool(msg.data)

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

        dopamine, dopamine_comps = self.dopamine_computer.step(
            obj_bits=obj_bits,
            proximity_spike=prox_spike,
            action_idx=int(winner_idx),
            task_state=self.task_state,
            grab_event=self.grab_event,
            proximity_stop=self.proximity_stop
        )

        # ---- Dopamine component breakdown for logging ----
        dop_align = float(dopamine_comps.get("align", 0.0))
        dop_action = float(dopamine_comps.get("action_match", 0.0))

        dop_lost = (
            float(dopamine_comps.get("lost_once", 0.0)) +
            float(dopamine_comps.get("lost_tick", 0.0))
        )

        dop_state = (
            float(dopamine_comps.get("state_progress", 0.0)) +
            float(dopamine_comps.get("state_regress", 0.0)) +
            float(dopamine_comps.get("state_reset_ok", 0.0)) +
            float(dopamine_comps.get("state_other", 0.0))
        )

        dop_grabdrop = (
            float(dopamine_comps.get("grabbed", 0.0)) +
            float(dopamine_comps.get("dropped", 0.0))
        )

        dop_prox_stop = float(dopamine_comps.get("proximity_stop", 0.0))
        dop_prox_approach = float(dopamine_comps.get("prox_spike_gated", 0.0))

        self.pub_dopamine.publish(Float32(data=float(dopamine)))

        # ---- Optional: apply dopamine to learning rule (when you enable training) ----
        # self.network.apply_reward(dopamine=dopamine, winner_idx=int(winner_idx))

        # ---- Add to your existing logger ----
        # Recommended: store reward + some context so you can tune later.
        # If you already log pending_input, just append these fields to the same row.
        # Example (pseudo):
        # self.logger_csv.write_row(..., reward=dopamine, state=self.task_state,
        #                           grab_event=self.grab_event,
        #                           proximity_stop=int(self.proximity_stop),
        #                           **reward_comps)

        if self.logger_csv is not None:
            t_output_ns = int(self.get_clock().now().nanoseconds)
            winner = int(winner_idx)
            spikes_list = [int(x) for x in output_spikes]  # length = output_size

            self.get_logger().info(f"Mode A check: pending_input is None? {self.pending_input is None}")

            # Choose input to log depending on mode
            if self.log_mode == "A":
                self.get_logger().info(f"Mode A check: pending_input is None? {self.pending_input is None}")
                if self.pending_input is None:
                    return  # no new input since last log
                t_input_ns, input_list = self.pending_input
                self.pending_input = None  # consume it

                row = (
                        [t_input_ns]
                        + input_list
                        + [
                            t_output_ns,
                            winner,
                            decision,
                            float(dopamine),
                            dop_align,
                            dop_action,
                            dop_lost,
                            dop_state,
                            dop_grabdrop,
                            dop_prox_stop,
                            dop_prox_approach,
                        ]
                        + spikes_list
                    )
                self.get_logger().info(f"Pushing CSV row with len={len(row)}")
                self.logger_csv.push(row)

            elif self.log_mode == "B":
                # always log latest input (even if unchanged)
                t_input_ns = int(self.last_input_stamp.nanoseconds)
                input_list = self.last_vector.astype(int).tolist()
                row = (
                    [t_input_ns]
                    + input_list
                    + [
                        t_output_ns,
                        winner,
                        decision,
                        float(dopamine),
                        dop_align,
                        dop_action,
                        dop_lost,
                        dop_state,
                        dop_grabdrop,
                        dop_prox_stop,
                        dop_prox_approach,
                    ]
                    + spikes_list
                )

                self.get_logger().info(f"Pushing CSV row with len={len(row)}")
                self.logger_csv.push(row)

            elif self.log_mode == "C":
                # log only when something changes
                t_input_ns = int(self.last_input_stamp.nanoseconds)
                input_list = self.last_vector.astype(int).tolist()

                signature = (
                    tuple(input_list),
                    winner,
                    decision,
                    round(float(dopamine), 6),
                    dop_align,
                    dop_action,
                    dop_lost,
                    dop_state,
                    dop_grabdrop,
                    dop_prox_stop,
                    dop_prox_approach,
                    tuple(spikes_list),
                )

                if self.last_logged != signature:
                    row = (
                        [t_input_ns]
                        + input_list
                        + [
                            t_output_ns,
                            winner,
                            decision,
                            float(dopamine),
                            dop_align,
                            dop_action,
                            dop_lost,
                            dop_state,
                            dop_grabdrop,
                            dop_prox_stop,
                            dop_prox_approach,
                        ]
                        + spikes_list
                    )

                    self.get_logger().info(f"Pushing CSV row with len={len(row)}")
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