
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import UInt8MultiArray, String


class SpikeTrainPublisher(Node):
    """
    Publishes spike trains to /snn/input for controlled snn energy measurements on the different snn implementations.

    Phases:
    1. idle_pre  (zeros)
    2. active    (random spikes)
    3. idle_post (zeros)
    """

    def __init__(self):
        super().__init__('spike_train_publisher')

        # ---- Logging helpers ----
        self.last_logged_sec = -1
        self.total_spikes = 0

        # ---- Publishers ----
        self.spike_pub = self.create_publisher(
            UInt8MultiArray,
            '/snn/input',
            10
        )

        self.phase_pub = self.create_publisher(
            String,
            '/experiment/phase',
            10
        )

        # ---- Timing ----
        self.freq = 15.0                # Same frequence as fps on camera
        self.period = 1.0 / self.freq

        self.idle_pre_duration = 30.0   # 0.5 min
        self.active_duration = 300.0    # 5.0 min
        self.idle_post_duration = 30.0  # 0.5 min

        self.start_time = self.now()

        # ---- System size ----
        self.num_neurons = 26   # remember to adjust according to the system (see my_ros2_bringup/config-> params.yaml)
        self.spike_prob = 0.1

        # Reproducibility
        np.random.seed(42)

        self.timer = self.create_timer(self.period, self.publish_spikes)

        self.get_logger().info("SpikeTrainPublisher started")

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

    def publish_spikes(self):
        t = self.now() - self.start_time
        phase = self.get_phase(t)

        if phase == "done":
            self.get_logger().info(
                f"Experiment finished. Total spikes: {self.total_spikes}"
            )
            self.timer.cancel()
            return

        # ---- Publish phase ----
        phase_msg = String()
        phase_msg.data = phase
        self.phase_pub.publish(phase_msg)

        # ---- Generate spikes ----
        if phase.startswith("idle"):
            spikes = np.zeros(self.num_neurons, dtype=np.uint8)
        else:
            spikes = (np.random.rand(self.num_neurons) < self.spike_prob).astype(np.uint8)

        self.total_spikes += int(spikes.sum())

        # ---- Publish spikes ----
        msg = UInt8MultiArray()
        msg.data = spikes.tolist()
        self.spike_pub.publish(msg)

        # ---- Clean logging every 10 seconds ----
        current_sec = int(t)
        if current_sec % 10 == 0 and current_sec != self.last_logged_sec:
            self.get_logger().info(f"Phase: {phase}, t={t:.1f}s")
            self.last_logged_sec = current_sec


def main():
    rclpy.init()
    node = SpikeTrainPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()