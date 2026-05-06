import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import (UInt8,
                           Bool,
                           UInt8MultiArray,
                           Int16,
                           Int32,
                           Int32MultiArray,
                           String)
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range

from geometry_msgs.msg import Twist

from .LIF_SNN_network import SNNLayer
from .csv_logger import CsvAsyncLogger

from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

from taskbot_interfaces.srv import SaveWeights
#from taskbot_interfaces.msg import Int32Array
from pathlib import Path

EVENT_IDLE = 0

ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT", "BACKWARD"]  # index 0=LEFT, 1=FORWARD, 2=RIGHT, 3=BACKWARD

class SNNNode(Node):
    """
    Python SNN node:
    - Reads packed spikes (0/1) from /snn/input (UInt8MultiArray)
    - Runs LIF SNN with dopamine learning and publishes:
        * /cmd_vel/snn (geometry_msgs/Twist) for robot control
        * /snn/decision (string) with the action name (LEFT, FORWARD, RIGHT, BACKWARD)
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
        self.declare_parameter('kp_rows', 4)
        self.declare_parameter('kp_cols', 6)
        self.declare_parameter('proximity_size', 4)
        self.declare_parameter('object_rec_size', 3)

        # Action parameters
        self.declare_parameter('num_actions', 4)

        # Driving parameters
        self.declare_parameter('timer_hz', 15.0)
        self.declare_parameter('idle_timeout_sec', 1.0)
        self.declare_parameter('use_proximity_for_stop', True)
        self.declare_parameter('proximity_stop_active_high', True)

        # Robot speed parameters
        self.declare_parameter('forward_speed', 0.125)
        self.declare_parameter('turn_speed', 0.3)

        # Neuron parameters (integer-scaled)
        self.declare_parameter('decay', 256)
        self.declare_parameter('threshold', 2048)
        self.declare_parameter('reset', 0)
        self.declare_parameter('refractory', 0)

        # Synapse & Learning parameters
        self.declare_parameter('lr_shift', 7)
        self.declare_parameter('initial_weight', -1)
        self.declare_parameter('t_pre', 3)
        self.declare_parameter('t_post', 3)
        self.declare_parameter('tau_e_shift', 3)
        self.declare_parameter('dw_pos', 32)
        self.declare_parameter('dw_neg', 16)
        self.declare_parameter('min_weight', 16)
        self.declare_parameter('max_weight', 254)
        self.declare_parameter('learning_mode', 'rstdp')
        self.declare_parameter('feedback', True)
        self.declare_parameter('seed', 42)

        # ---- CSV logging ----
        self.declare_parameter('log_enable', True)
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
        self.last_distance_m: float = float('nan')

        # ---- For episode weight logging ----
        self.declare_parameter(
            'weights_log_dir',
            "/opt/robot_ws/src/ros2/weights_logs/"
        )
        self.weights_log_dir = Path(self.get_parameter('weights_log_dir').value).expanduser()
        self.weights_log_dir.mkdir(parents=True, exist_ok=True)

        self.save_weights_srv = self.create_service(
            SaveWeights,
            '/save_weights',
            self.handle_save_weights
        )

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

        self.create_subscription(
            Int16,
            '/reward/dopamine',
            self._on_reward_dopamine,
            10
        )
                

        # --- Read Parameters ---
        self.input_mode = str(self.get_parameter('input_mode').value).lower().strip()
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.pack_order = list(self.get_parameter('pack_order').value)

        kp_rows = int(self.get_parameter('kp_rows').value)
        kp_cols = int(self.get_parameter('kp_cols').value)
        self.channel_sizes = {
            'proximity': int(self.get_parameter('proximity_size').value),
            'keypoints_grid': kp_rows * kp_cols,
            'object_rec': int(self.get_parameter('object_rec_size').value)
        }

        self.timer_hz = float(self.get_parameter('timer_hz').value)
        self.idle_timeout_sec = float(self.get_parameter('idle_timeout_sec').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)

        self.num_actions = int(self.get_parameter('num_actions').value)
        self.output_size = self.num_actions

        # Neuron params
        self.decay = int(self.get_parameter('decay').value)
        self.threshold = int(self.get_parameter('threshold').value)
        self.reset = int(self.get_parameter('reset').value)
        self.refractory = int(self.get_parameter('refractory').value)

        # Synapse params
        self.lr_shift = int(self.get_parameter('lr_shift').value)
        self.initial_weight = int(self.get_parameter('initial_weight').value)
        self.t_pre = int(self.get_parameter('t_pre').value)
        self.t_post = int(self.get_parameter('t_post').value)
        self.tau_e_shift = int(self.get_parameter('tau_e_shift').value)
        self.dw_pos = int(self.get_parameter('dw_pos').value)
        self.dw_neg = int(self.get_parameter('dw_neg').value)
        self.max_weight = int(self.get_parameter('max_weight').value)
        self.min_weight = int(self.get_parameter('min_weight').value)
        self.learning_mode = str(self.get_parameter('learning_mode').value)
        self.feedback = bool(self.get_parameter('feedback').value)
        
        seed = self.get_parameter('seed').value
        if seed is not None:
            np.random.seed(int(seed))

        # --- Derived Logic ---
        self.segment_offsets = self._compute_offsets(self.pack_order, self.channel_sizes)
        self.input_size = sum(self.channel_sizes.get(name, 0) for name in self.pack_order)



        #### Initialize SNN Layer ####
        neuron_params = {"decay": self.decay, "threshold": self.threshold, "reset": self.reset, "refractory": self.refractory}
        synapse_params = {
            "lr_shift": self.lr_shift, "w_init": self.initial_weight,
            "t_pre": self.t_pre, "t_post": self.t_post, "tau_e_shift": self.tau_e_shift,
            "dw_pos": self.dw_pos, "dw_neg": self.dw_neg,
            "w_min": self.min_weight, "w_max": self.max_weight,
            "mode": self.learning_mode
        }

        self.network = SNNLayer(
            n_inputs=self.input_size,
            n_outputs=self.output_size,
            neuron_params=neuron_params,
            synapse_params=synapse_params,
            feedback=self.feedback
        )

        # ---- Shared weights location (used by both Python SNN and FPGA system) ----
        self.declare_parameter(
            'weights_base_dir',
            '/opt/robot_ws/src/ros2/weights_logs'
        )

        weights_base_dir = Path(
            self.get_parameter('weights_base_dir').value
        ).expanduser()

        weights_base_dir.mkdir(parents=True, exist_ok=True)

        self.weights_current_file = weights_base_dir / "weights_current.mem"
        self.weights_log_dir = weights_base_dir / "episode_logs"
        self.weights_log_dir.mkdir(parents=True, exist_ok=True)

        weight_path = str(self.weights_current_file)

        if os.path.exists(weight_path):
            try:
                self.network.load_weights(weight_file=weight_path)
                self.get_logger().info(f"Loaded weights from {weight_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load weights: {e}")
        else:
            self.get_logger().warn(
                f"No weights file found at {weight_path}, using initial weights"
            )

        self.weights_log_dir = self.weights_current_file.parent / "episode_logs"
        self.weights_log_dir.mkdir(parents=True, exist_ok=True)

        ###########################################

        # --- State and Communication ---
        self.last_input_stamp = self.get_clock().now()
        self.last_vector = np.zeros(self.input_size, dtype=np.int32)
        self.correct_output = -1
        self.latest_dopamine = 0
        self.previous_winner_idx = -1

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(UInt8MultiArray, self.input_topic, self.cb_packed, qos_sensor)
        self.create_subscription(Range, '/ultrasonic/front/scan', self._on_scan, qos_sensor) # Endre Range til LaserScan ved Gazebo
        self.create_subscription(Int32, '/snn/correct_output', self.cb_correct, 10)

        self.pub_winner = self.create_publisher(Int32, '/snn/winner', 10)
        self.pub_spikes = self.create_publisher(Int32MultiArray, '/snn/spikes', 10)
        self.pub_decision = self.create_publisher(String, '/snn/decision', 10)

        qos_monitor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_mem = self.create_publisher(Int32MultiArray, '/snn/mem', qos_monitor)
        self.pub_weights = self.create_publisher(Int32MultiArray, '/snn/weights', qos_monitor)
        self.pub_eligibility = self.create_publisher(Int32MultiArray, '/snn/eligibility', qos_monitor)
        self.pub_delta_w = self.create_publisher(Int32MultiArray, '/snn/delta_w', qos_monitor)
        
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
            header.append("distance_m")
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

    def _on_scan(self, msg: Range):
        d = msg.range
        self.last_distance_m = d if 0.0 < d < float('inf') else float('nan')


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

        # Convert 0/1 uint8 spikes to int32 and store in last_vector
        self.last_vector[:] = data.astype(np.int32)
        self.last_input_stamp = self.get_clock().now()

        if self.logger_csv is not None and self.log_mode == "A":
            t_input_ns = int(self.last_input_stamp.nanoseconds)
            # store as plain Python list so it won't change later
            self.pending_input = (t_input_ns, self.last_vector.astype(int).tolist())

    def cb_correct(self, msg: Int32):
        self.correct_output = int(msg.data)

    def _on_reward_dopamine(self, msg: Int16):
        self.latest_dopamine = int(msg.data)

    # Timer
    def on_timer(self):
        age_sec = (self.get_clock().now() - self.last_input_stamp).nanoseconds * 1e-9
        if age_sec > self.idle_timeout_sec:
            self.publish_stop(f"stale input {age_sec:.2f}s > {self.idle_timeout_sec}s")
            return


        ##### Run network #####

        # Forward pass
        output_spikes = self.network.forward(input_spikes=self.last_vector)

        self.pub_mem.publish(Int32MultiArray(data=self.network.pre_reset_mem.tolist())) ###

        # Find idx of winning neuron
        winner_idx = self.network.winner_takes_all(output_spikes=output_spikes)
        
        # Normal actuation
        decision = self.publish_cmd_from_winner(int(winner_idx), force_stop=False)

        # Debug publish winners and spikes
        self.pub_winner.publish(Int32(data=int(winner_idx)))

        spk_msg = Int32MultiArray()
        spk_msg.data = [int(x) for x in output_spikes]
        self.pub_spikes.publish(spk_msg)

        # Reward now comes from dopamine_reward_node via /reward/dopamine
        dopamine = int(self.latest_dopamine)
        dopamine_float = float(dopamine)

        # Component-wise internal reward breakdown is no longer computed here
        dop_align = 0.0
        dop_action = 0.0
        dop_lost = 0.0
        dop_state = 0.0
        dop_grabdrop = 0.0
        dop_prox_stop = 0.0
        dop_prox_approach = 0.0

        # Apply reward to the PREVIOUS winner using the eligibility snapshot captured
        # at the end of that tick — before this tick's post-spike LTD accumulates.
        if self.learning_mode == 'rstdp' and self.previous_winner_idx >= 0:
            self.network.apply_reward(dopamine=dopamine, winner_idx=self.previous_winner_idx)
        self.previous_winner_idx = int(winner_idx)

        self.pub_weights.publish(Int32MultiArray(data=self.network.weights.flatten().tolist()))  ###
        self.pub_eligibility.publish(Int32MultiArray(data=self.network.eligibility.flatten().tolist()))  ###
        self.pub_delta_w.publish(Int32MultiArray(data=self.network.last_delta_w.flatten().tolist())) ###

        ## Logging ##

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
                            dopamine_float,
                            dop_align,
                            dop_action,
                            dop_lost,
                            dop_state,
                            dop_grabdrop,
                            dop_prox_stop,
                            dop_prox_approach,
                            self.last_distance_m,
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
                        dopamine_float,
                        dop_align,
                        dop_action,
                        dop_lost,
                        dop_state,
                        dop_grabdrop,
                        dop_prox_stop,
                        dop_prox_approach,
                        self.last_distance_m,
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
                    round(dopamine_float, 6),
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
                            dopamine_float,
                            dop_align,
                            dop_action,
                            dop_lost,
                            dop_state,
                            dop_grabdrop,
                            dop_prox_stop,
                            dop_prox_approach,
                            self.last_distance_m,
                        ]
                        + spikes_list
                    )

                    self.get_logger().info(f"Pushing CSV row with len={len(row)}")
                    self.logger_csv.push(row)
                    self.last_logged = signature

    #### Publishing helpers ####

    def publish_cmd_from_winner(self, winner_idx: int, force_stop: bool = False):
        cmd = Twist()
        decision = "IDLE"

        if force_stop:
            decision = "STOP_PROXIMITY"
        elif winner_idx < 0:
            decision = "IDLE"
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
            elif winner_idx == 3:    # BACKWARD
                cmd.linear.x = -self.forward_speed
                cmd.angular.z = 0.0
                decision = ACTION_NAMES[3]
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

    def save_weights(self, weight_file):
        """
        Save weights to a hex .mem file, one 8-bit value per line.
        Atomic save: write temp file first, then replace final file.
        """
        weight_file = Path(weight_file).expanduser()
        weight_file.parent.mkdir(parents=True, exist_ok=True)

        weights = self.network.get_weights()
        flat_weights = weights.flatten()

        tmp_file = weight_file.with_suffix(weight_file.suffix + ".tmp")

        with open(tmp_file, "w") as f:
            for value in flat_weights:
                f.write(f"{int(value) & 0xFF:02X}\n")

        tmp_file.replace(weight_file)

    def handle_save_weights(self, request, response):
        try:
            filename = request.filename.strip()

            if not filename:
                response.success = False
                response.message = "Filename was empty"
                return response

            if filename == "weights_current.mem":
                full_path = self.weights_current_file
            else:
                full_path = self.weights_log_dir / filename

            self.save_weights(str(full_path))

            response.success = True
            response.message = f"Saved weights to {full_path}"
            self.get_logger().info(response.message)

        except Exception as e:
            response.success = False
            response.message = f"Failed to save weights: {e}"
            self.get_logger().error(response.message)

        return response

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