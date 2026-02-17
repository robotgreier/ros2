
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray
import numpy as np

from python_snn_node.network import LIFNetwork

class SNNNode(Node):
    def __init__(self):
        super().__init__('python_snn_node')
        # Params
        self.declare_parameter('keypoint_size', 10)
        self.declare_parameter('object_size', 4)
        self.declare_parameter('timer_hz', 50.0)
        self.declare_parameter('dopamine_correct', 1.0)
        self.declare_parameter('dopamine_wrong', 0.5)
        self.declare_parameter('dopamine_nofire', 0.1)
        self.declare_parameter('seed', 42)

        kp = self.get_parameter('keypoint_size').value
        obj = self.get_parameter('object_size').value
        self.timer_hz = self.get_parameter('timer_hz').value
        self.dop_correct = self.get_parameter('dopamine_correct').value
        self.dop_wrong   = self.get_parameter('dopamine_wrong').value
        self.dop_nofire  = self.get_parameter('dopamine_nofire').value
        seed = self.get_parameter('seed').value

        self.input_size = kp + 1 + obj
        self.output_size = obj

        self.net = LIFNetwork(self.input_size, self.output_size, seed=seed)

        # IO buffers
        self.keypoints = np.zeros(kp, dtype=float)
        self.distance = np.array([0.0], dtype=float)
        self.object_onehot = np.zeros(obj, dtype=int)
        self.correct_output = 0

        # Subs
        self.create_subscription(Float32MultiArray, '/keypoints', self.cb_keypoints, 10)
        self.create_subscription(Float32MultiArray, '/distance', self.cb_distance, 10)
        self.create_subscription(Int32MultiArray,   '/object_onehot', self.cb_obj, 10)
        self.create_subscription(Int32, '/snn/correct_output', self.cb_correct, 10)

        # Pubs
        self.pub_winner = self.create_publisher(Int32, '/snn/winner', 10)
        self.pub_spikes = self.create_publisher(Int32MultiArray, '/snn/spikes', 10)

        # Timer
        period = 1.0 / max(self.timer_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

    def cb_keypoints(self, msg):  # Float32MultiArray
        arr = np.array(msg.data, dtype=float)
        if arr.size == self.keypoints.size:
            self.keypoints[:] = arr

    def cb_distance(self, msg):   # Float32MultiArray with one value
        if len(msg.data) > 0:
            self.distance[0] = float(msg.data[0])

    def cb_obj(self, msg):        # Int32MultiArray (one-hot)
        arr = np.array(msg.data, dtype=int)
        if arr.size == self.object_onehot.size:
            self.object_onehot[:] = arr

    def cb_correct(self, msg):    # Int32
        self.correct_output = int(msg.data)

    def on_timer(self):
        input_vec = np.concatenate([self.keypoints, self.distance, self.object_onehot]).tolist()
        winner, dop = self.net.step(
            input_vec,
            correct_output=self.correct_output,
            dopamine_correct=self.dop_correct,
            dopamine_wrong=self.dop_wrong,
            dopamine_nofire=self.dop_nofire
        )

        # Publish outputs
        win_msg = Int32()
        win_msg.data = int(winner)
        self.pub_winner.publish(win_msg)

        spk_msg = Int32MultiArray()
        spk_msg.data = [int(n.spk) for n in self.net.output_neurons]
        self.pub_spikes.publish(spk_msg)

def main():
    rclpy.init()
    node = SNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
