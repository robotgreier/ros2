import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
import numpy as np
from std_msgs.msg import UInt8MultiArray, String


class SpikeTrainPublisher(Node):
    """
    Publishes deterministic spike trains to /snn/input for controlled SNN
    energy measurements. Emits phase markers (idle_pre, active, idle_post,
    done) on /experiment/phase as a latched topic so downstream loggers
    always see the current phase regardless of subscribe time.

    Phases:
    1. idle_pre   (zeros)
    2. active     (random spikes, seeded for reproducibility)
    3. idle_post  (zeros)
    4. done       (terminal marker, published once before timer cancels)
    """

    def __init__(self):
        super().__init__('spike_train_publisher')

        # ---- Parameters ----
        self.declare_parameter('output_topic', '/snn/input')
        self.declare_parameter('phase_topic', '/experiment/phase')
        self.declare_parameter('freq', 15.0)
        self.declare_parameter('num_neurons', 27)
        self.declare_parameter('spike_prob', 0.1)
        self.declare_parameter('idle_pre_duration', 30.0)
        self.declare_parameter('active_duration', 300.0)
        self.declare_parameter('idle_post_duration', 30.0)
        self.declare_parameter('seed', 42)

        self.output_topic = self.get_parameter('output_topic').value
        self.phase_topic = self.get_parameter('phase_topic').value
        self.freq = float(self.get_parameter('freq').value)
        self.num_neurons = int(self.get_parameter('num_neurons').value)
        self.spike_prob = float(self.get_parameter('spike_prob').value)
        self.idle_pre_duration = float(self.get_parameter('idle_pre_duration').value)
        self.active_duration = float(self.get_parameter('active_duration').value)
        self.idle_post_duration = float(self.get_parameter('idle_post_duration').value)
        self.seed = int(self.get_parameter('seed').value)

        # ---- State ----
        self.period = 1.0 / self.freq
        self.rng = np.random.default_rng(self.seed)
        self.last_phase = None
        self.last_logged_sec = -1
        self.total_spikes = 0

        # ---- Publishers ----
        self.spike_pub = self.create_publisher(
            UInt8MultiArray,
            self.output_topic,
            10,
        )

        # Latched phase: TRANSIENT_LOCAL so a logger started after this node
        # still receives the current phase on subscription.
        phase_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.phase_pub = self.create_publisher(
            String,
            self.phase_topic,
            phase_qos,
        )

        # ---- Timing ----
        self.start_time = self.now()
        self.timer = self.create_timer(self.period, self.publish_spikes)

        self.get_logger().info(
            f"SpikeTrainPublisher started: {self.num_neurons} neurons @ "
            f"{self.freq:.1f} Hz, phases "
            f"{self.idle_pre_duration:.0f}/{self.active_duration:.0f}/"
            f"{self.idle_post_duration:.0f} s, seed={self.seed}"
        )

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def get_phase(self, t):
        if t < self.idle_pre_duration:
            return "idle_pre"
        elif t < self.idle_pre_duration + self.active_duration:
            return "active"
        elif t < self.idle_pre_duration + self.active_duration + self.idle_post_duration:
            return "idle_post"
        else:
            return "done"

    def publish_phase(self, phase):
        msg = String()
        msg.data = phase
        self.phase_pub.publish(msg)

    def publish_spikes(self):
        t = self.now() - self.start_time
        phase = self.get_phase(t)

        # Publish phase only on transition; latched QoS carries the last value
        # to late subscribers.
        if phase != self.last_phase:
            self.publish_phase(phase)
            self.get_logger().info(
                f"Phase -> {phase} at t={t:.2f}s (total_spikes={self.total_spikes})"
            )
            self.last_phase = phase

        if phase == "done":
            self.get_logger().info(
                f"Experiment finished. Total spikes: {self.total_spikes}"
            )
            self.timer.cancel()
            return

        if phase.startswith("idle"):
            spikes = np.zeros(self.num_neurons, dtype=np.uint8)
        else:
            spikes = (self.rng.random(self.num_neurons) < self.spike_prob).astype(np.uint8)

        self.total_spikes += int(spikes.sum())

        msg = UInt8MultiArray()
        msg.data = spikes.tolist()
        self.spike_pub.publish(msg)

        # Heartbeat every 30 s so an aborted run still leaves a count.
        current_sec = int(t)
        if current_sec % 30 == 0 and current_sec != self.last_logged_sec:
            self.get_logger().info(
                f"t={t:.1f}s phase={phase} total_spikes={self.total_spikes}"
            )
            self.last_logged_sec = current_sec


def main():
    rclpy.init()
    node = SpikeTrainPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
