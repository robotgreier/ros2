import json
from typing import List, Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8, UInt8MultiArray, Bool, Int16, Int32, String, Empty
from geometry_msgs.msg import Vector3

from .energy_tracker import EnergyTracker

from .dopamine_logic import DopamineComputer

from dopamine_interfaces.msg import PhaseEnergyResult

EVENT_DROPPED = 2

class DopamineRewardNode(Node):
    def __init__(self) -> None:
        super().__init__("dopamine_reward_node")

        # Publishers
        self.reward_pub = self.create_publisher(Int16, "/reward/dopamine", 10)
        self.debug_pub = self.create_publisher(String, "/reward/debug", 10)
        self.phase_pub = self.create_publisher(UInt8MultiArray, "/task/phase", 10)
        self.phase_result_pub = self.create_publisher(PhaseEnergyResult, "/task/phase_result", 10)

        # Subscribers
        self.create_subscription(UInt8, "/task/state", self.task_state_cb, 10)
        self.create_subscription(Bool, "/proximity_stop", self.proximity_stop_cb, 10)
        self.create_subscription(UInt8MultiArray, "/snn/aruco_dir", self.aruco_dir_cb, 10)
        self.create_subscription(Int32, "/snn/winner", self.winner_cb, 10)
        self.create_subscription(Vector3, "/battery/status", self.battery_cb, 10)
        self.create_subscription(Empty, "/episode_complete", self.episode_cb, 10)
        self.create_subscription(UInt8, "/grab_node/event", self.grab_event_cb, 10)

        # Latest inputs
        self.task_state: Optional[int] = None
        self.proximity_stop: bool = False
        self.aruco_dir: List[int] = [0, 0, 0]

        # Reward logic
        self.dopamine = DopamineComputer()

        self.get_logger().info(
            "dopamine_reward_node started. Waiting for /snn/winner, /snn/aruco_dir, /proximity_stop, and /task/state"
        )

        #Energy tracker
        self.energy_tracker = EnergyTracker()

        self.declare_parameter("use_energy_reward", True)
        self.declare_parameter("energy_reward_positive", 2)  
        self.declare_parameter("energy_reward_negative", -2)

        self.use_energy_reward = self.get_parameter("use_energy_reward").value
        self.energy_reward_positive = int(self.get_parameter("energy_reward_positive").value)
        self.energy_reward_negative = int(self.get_parameter("energy_reward_negative").value)

        self.pending_energy_reward = 0
        self.pending_energy_debug = ""

    def task_state_cb(self, msg: UInt8) -> None:
        state = int(msg.data)
        self.task_state = state

        self.get_logger().info(f"[TaskState] received state={state}")

        result = self.energy_tracker.on_task_state(state)

        if result is not None:
            self.get_logger().info(
                f"[Energy] pickup={result.pickup_idx}, "
                f"phase={result.phase_idx}, "
                f"E={result.energy_joules:.2f}J, "
                f"avg={result.average_joules}"
            )

            self.publish_phase_result(result)

            self.pending_energy_reward = 0
            self.pending_energy_debug = ""

            if self.use_energy_reward and result.average_joules is not None:
                if result.energy_joules < result.average_joules:
                    self.pending_energy_reward = self.energy_reward_positive
                elif result.energy_joules > result.average_joules:
                    self.pending_energy_reward = self.energy_reward_negative

                self.pending_energy_debug = (
                    f"pickup={result.pickup_idx}, phase={result.phase_idx}, "
                    f"E={result.energy_joules:.2f}, avg={result.average_joules:.2f}, "
                    f"energy_reward={self.pending_energy_reward}"
                )

            self.publish_phase()

    def proximity_stop_cb(self, msg: Bool) -> None:
        self.proximity_stop = bool(msg.data)

    def aruco_dir_cb(self, msg: UInt8MultiArray) -> None:
        data = list(msg.data)

        if len(data) == 0:
            self.get_logger().warn("Received empty /snn/aruco_dir message")
            return

        self.aruco_dir = data

    def winner_cb(self, msg: Int32) -> None:
        action_idx = int(msg.data)

        if action_idx not in (0, 1, 2, 3):
            self.get_logger().warn(f"Ignoring invalid winner/action: {action_idx}")
            return

        if len(self.aruco_dir) == 0:
            self.get_logger().warn("No aruco direction data available yet")
            return

        reward, comps = self.dopamine.step(
            obj_bits=self.aruco_dir,
            action_idx=action_idx,
            proximity_stop=self.proximity_stop,
            task_state=self.task_state,
        )

        energy_reward = self.pending_energy_reward
        final_reward = reward + energy_reward

        if energy_reward != 0:
            comps["energy_phase"] = energy_reward

        reward_msg = Int16()
        reward_msg.data = final_reward
        self.reward_pub.publish(reward_msg)

        debug_msg = String()
        debug_msg.data = json.dumps(
            {
                "task_state": self.task_state,
                "aruco_dir": self.aruco_dir,
                "proximity_stop": self.proximity_stop,
                "action_idx": action_idx,
                "base_reward": reward,
                "energy_reward": energy_reward,
                "final_reward": final_reward,
                "components": comps,
            }
        )
        self.debug_pub.publish(debug_msg)

        self.pending_energy_reward = 0
        self.pending_energy_debug = ""

        self.get_logger().info(
            f"winner={action_idx}, reward={reward}, comps={comps}"
        )

    def battery_cb(self, msg: Vector3) -> None:
        power_w = float(msg.x)

        now_s = self.get_clock().now().nanoseconds / 1e9

        self.energy_tracker.update_power(power_w, now_s)    

    def episode_cb(self, msg: Empty) -> None:
        self.get_logger().info("Episode complete → resetting energy tracker")
        self.energy_tracker.reset_episode()

    def grab_event_cb(self, msg: UInt8) -> None:
        event = int(msg.data)

        if event != EVENT_DROPPED:
            return

        result = self.energy_tracker.force_complete_phase(
            expected_phase_idx=3
        )

        if result is None:
            return

        self.get_logger().info(
            f"[Energy] drop event completed phase: "
            f"pickup={result.pickup_idx}, "
            f"phase={result.phase_idx}, "
            f"E={result.energy_joules:.2f}J, "
            f"duration={result.duration_s:.2f}s"
        )

        self.publish_phase_result(result)
        self.publish_phase()

    def publish_phase(self) -> None:
        msg = UInt8MultiArray()

        pickup_idx = int(self.energy_tracker.current_pickup_idx)

        if self.energy_tracker.current_phase_idx is None:
            phase_idx = 255
        else:
            phase_idx = int(self.energy_tracker.current_phase_idx)

        phase_active = 1 if self.energy_tracker.phase_active else 0

        msg.data = [pickup_idx, phase_idx, phase_active]
        self.phase_pub.publish(msg)
    
    def publish_phase_result(self, result) -> None:
        msg = PhaseEnergyResult()

        msg.pickup_idx = int(result.pickup_idx)
        msg.phase_idx = int(result.phase_idx)

        msg.reward_energy_joules = float(result.energy_joules)
        msg.reward_energy_wh = float(result.energy_joules) / 3600.0
        msg.start_time_s = float(result.start_time_s)
        msg.end_time_s = float(result.end_time_s)
        msg.duration_s = float(result.duration_s)

        msg.average_joules = (
            float(result.average_joules)
            if result.average_joules is not None
            else -1.0
        )

        msg.delta_joules = (
            float(result.delta_joules)
            if result.delta_joules is not None
            else -1.0
        )

        self.phase_result_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DopamineRewardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down dopamine_reward_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()