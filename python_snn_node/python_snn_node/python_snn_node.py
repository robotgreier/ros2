import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import (
    Int32, 
    Int32MultiArray, 
    UInt8MultiArray, 
    String,
    float32,
)

from geometry_msgs.msg import Twist

from python_snn_node.network import LIFNetwork

ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT"] # index 0=LEFT, 1=FORWARD, 2=RIGHT

class SNNNode(Node):
    """
    Python SNN node:
    - Reads packed spikes (0/1) from /snn/input (UInt8MultiArray)
    - Runs LIF SNN with dopamine learning and publishes:
        * /cmd_vel (geometry_msgs/Twist) for robot control
        * /snn/decision (string) with the action name (LEFT, FORWARD, RIGHT)
        * /snn/winner (Int32)
        * /snn/spikes (INT32MuliArray) for debugging (spikes of output neurons)
    - Training mode: listens to /snn/correct_output (Int32), uses dopamine learning to adjust synaptic weights.

    """
    def __init__(self):
        super().__init__('python_snn_node')
        # Params
        self.declare_parameter('input_mode', 'packed') # 'packed' default, 'separate' if we change the input topics to separate.
        self.declare_parameter('input_topic', '/snn/input') # Topic published by encoding_node
        self.declare_parameter('pack_order', ['proximety', 'keypoints_grid']) # Order of input features, must match encoder_node's packing

        # Channel sizes, must match encoder_node's output sizes
        self.declare_parameter('proximity_size', 1) # If distance changes from one bin to another.
        self.declare_parameter('keypoints_grid_size', 12) # 4x3 grid

        # Action parameters / output neurons
        self.declare_parameter('num_actions', 3) # LEFT, FORWARD, RIGHT (0, 1, 2)

        # Driving parameters / robustness
        self.declare_parameter('timer_hz', 30.0) # Rate of main loop, and publishing cmd_vel
        self.declare_parameter('idle_timeout_sec', 1.0) # stop if no input received for this amount of seconds
        
        # We should consider this emergency stop logic. If proximity sensor detects something close, we want to stop the robot immediately, regardless of the SNN output.
        self.declare_parameter('use_proximity_for_stop', False) # if True: emergency stop is activated based on proximity sensor. Need new topic for this.
        self.declare_parameter('proximity_stop_active_high', True) # True: stop (close) is 1, False: stop (close) is 0. Emergency stop logic.
        

        # Robot speed parameters
        self.declare_parameter('forward_speed', 0.25) # m/s 
        self.declare_parameter('turn_speed', 0.6) # rad/s 

        # Learning parameters
        self.declare_parameter('training_mode', True) # If True: listen to /snn/correct_output and update weight. If False: run SNN without learning (evaluation)
        self.declare_parameter('dopamine_correct', 1.0) # Reward for correct action
        self.declare_parameter('dopamine_wrong', 0.5) # punishment for wrong action
        self.declare_parameter('dopamine_nofire', 0.1) # punishment for not firing any output neurons (prevent inactivity)
        self.declare_parameter('seed', 42) # Random seed for reproducibility

        # Read parameters
        self.input_mode = str(self.get_parameter('input_mode').value).lower().strip()
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.pack_order = list(self.get_parameter('pack_order').value)

        # Channel sizes
        self.channel_sizes = {
            'proximity': int(self.get_parameter('proximity_size').value),
            'keypoints_grid': int(self.get_parameter('keypoints_grid_size').value),
            # add more channels here when we add more input features to /snn/input
        }    

        self.num_actions = int(self.get_parameter('num_actions').value)
        self.timer_hz = float(self.get_parameter('timer_hz').value)
        self.idle_timeout_sec = float(self.get_parameter('idle_timeout_sec').value)

        # We should consider this emergency stop logic. Need a new topic for this.
        self.use_proximity_for_stop = bool(self.get_parameter('use_proximity_for_stop').value)
        self.proximity_stop_active_high = bool(self.get_parameter('proximity_stop_active_high').value)
        
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        
        self.training_mode = bool(self.get_parameter('training_mode').value)
        self.dop_correct = float(self.get_parameter('dopamine_correct').value)
        self.dop_wrong   = float(self.get_parameter('dopamine_wrong').value)
        self.dop_nofire  = float(self.get_parameter('dopamine_nofire').value)
        seed = int(self.get_parameter('seed').value)

        # Derived input and output sizes
        self.segment_offsets = self._compute_offsets(self.pack_order, self.channel_sizes)
        self.input_size = sum(self.channel_sizes.get(name, 0) for name in self.pack_order)
        self.output_size = self.num_actions

        # SNN initialization
        self.net = LIFNetwork(self.input_size, self.output_size, seed=seed)

        # State variables
        self.last_input_stamp = self.get_clock().now()
        self.last_vector = np.zeros(self.input_size, dtype=np.float32)
        self.correct_output = -1 # -1 means no correct output available

        # Quality of Service QoS for subscribers and publishers
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriptions - check input mode and subscribe to the appropriate topic 
        if self.input_mode != 'packed':
            self.get_logger().warn("input_mode != 'packed' er ikke aktivert i denne implementasjonen. "
                                   "Bytter til 'packed'.")
            self.input_mode = 'packed'
        
        # encoder_node publishes packed spikes as UInt8MultiArray
        self.create_subscription(UInt8MultiArray, self.input_topic, self.cb_packed, qos_sensor)

        # Training signal for dopamine learning
        self.create_subscription(Int32, '/snn/correct_output', self.cb_correct, 10)

        # Publishers
        self.pub_winner = self.create_publisher(Int32, '/snn/winner', 10) # publish the winner output neuron index for debugging and visualization
        self.pub_spikes = self.create_publisher(Int32MultiArray, '/snn/spikes', 10) # publish spikes of output neurons for debugging and visualization
        self.pub_decision = self.create_publisher(String, '/snn/decision', 10) # publish the action name LEFT, FORWARD, RIGHT for debugging and visualization
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # publish motor commands to the robot

        # Timer for main loop and cmd_vel publishing
        period = 1.0 / max(self.timer_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        # Log parameters at startup
        self.get_logger().info(
            f"SNN init: mode=packed, input_topic={self.input_topic}, "
            f"input_size={self.input_size}, pack_order={self.pack_order}, "
            f"channel_sizes={self.channel_sizes}, num_actions={self.num_actions}, "
            f"use_proximity_for_stop={self.use_proximity_for_stop}"
        )

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

        def cb_correct(self, msg: Int32):
            self.correct_output = int(msg.data)

        # Timer
        def on_timer(self):
            age_sec = (self.get_clock().now() - self.last_input_stamp).nanoseconds * 1e-9
            if age_sec > self.idle_timeout_sec:
                self.publish_stop(f"stale input {age_sec:.2f}s > {self.idle_timeout_sec}s")
                return

            input_vec = self.last_vector.tolist()
            correct = self.correct_output if self.training_mode else -1

        # Run SNN one step
        winner, dop = self.net.step(
            input_vec,
            correct_output=correct,
            dopamine_correct=self.dop_correct,
            dopamine_wrong=self.dop_wrong,
            dopamine_nofire=self.dop_nofire
        )

        # Debug publish winners and spikes
        self.pub_winner.publish(Int32(data=int(winner)))

        spk_msg = Int32MultiArray()
        spk_msg.data = self._get_output_spikes_safe()
        self.pub_spikes.publish(spk_msg)

        # We should consider this emergency stop logic
        force_stop = self._proximity_stop() if self.use_proximity_for_stop else False
        
        # publish cmd_vel based on winner, or stop if proximity stop is activated
        self.publish_cmd_from_winner(int(winner), force_stop=force_stop)

    def _get_output_spikes_safe(self):
        """
        Get spikes from output neurons, but catch exceptions, or return "winner-only" spikes if something goes wrong.
        """
        try:
            return [int(n.spk) for n in self.net.output_neurons]
        except Exception:
            # fallback: winner=1, others=0
            out = [0] * self.output_size
            # winner published in /snn/winner, to avoid mismatch with internal SNN state.
            return out

    def _proximity_stop(self) -> bool:
        """
        Reads the proximity channel (if defined) and activates emergency stop
        Use topic name "proximity_stop" to get this part of the code to work 
        """
        # First check if "proximity_stop" is defined, otherwise fall back to 'proximity'
        name = 'proximity_stop' if 'proximity_stop' in self.segment.offsets else 'proximity'

        if name not in self.segment_offsets:
            return False
        s, e = self.segment_offsets[name]
        chunk = self.last_vector[s:e]
        if chunk.size == 0:
            return False

        val = chunk[0] > 0.5  # 0/1
        return bool(val) if self.proximity_stop_active_high else (not bool(val))

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

    def publish_stop(self, reason: str = ""):
        if reason:
            self.get_logger().warn(f"STOP: {reason}")
        self.cmd_vel_pub.publish(Twist())
        self.pub_decision.publish(String(data="IDLE"))


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