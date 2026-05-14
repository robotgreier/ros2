import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import UInt8MultiArray


class SpikeTrainPublisher(Node):
    """
    Publishes spike trains to /snn/input for controlled energy testing of SNN network on FPGA and CPU.

    Phases:
    1. Idle (all zeros)
    2. Active (random spikes)
    3. Idle (all zeros)

    Keeps constant publish rate to avoid triggering idle timeout in snn_node.
    """

    def __init__(self):
        super().__init__('spike_train_publisher')

        self.publisher_ = self.create_publisher(
            UInt8MultiArray,
            '/snn/input',
            10
        )

        # ---- Timing ----
        self.freq = 15.0
        self.period = 1.0 / self.freq

        self.idle_pre_duration = 30.0   # seconds
        self.active_duration = 60.0     # seconds
        self.idle_post_duration = 30.0  # seconds

        self.start_time = self.now()

        # ---- Match parameters to system ----
        self.num_neurons = 26 

        # Spike probability (can tune)
        self.spike_prob = 0.1

        # Reproducibility
        np.random.seed(42)

        self.timer = self.create_timer(self.period, self.publish_spikes)

        self.get_logger().info("SpikeTrainPublisher started")

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

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
            self.get_logger().info("Experiment finished")
            self.timer.cancel()
            return

        # ---- Generate spikes based on phase ----
        if phase.startswith("idle"):
            spikes = np.zeros(self.num_neurons, dtype=np.uint8)

        elif phase == "active":
            spikes = (np.random.rand(self.num_neurons) < self.spike_prob).astype(np.uint8)

        # ---- Publish ----
        msg = UInt8MultiArray()
        msg.data = spikes.tolist()
        self.publisher_.publish(msg)

        # Optional: log transitions
        if int(t) % 10 == 0:  # log every 10 sec
            self.get_logger().info(f"Phase: {phase}, t={t:.1f}s")


def main():
    rclpy.init()
    node = SpikeTrainPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
