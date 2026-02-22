#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


@dataclass
class SourceState:
    name: str
    topic: str
    timeout_sec: float
    priority: int
    last_msg: Twist
    last_stamp: Time
    has_msg: bool = False


class CmdArbiter(Node):
    """
    Minimal command arbiter:
      - Multiple Twist inputs -> one /cmd_vel output
      - Priority + timeout arbitration
      - Hard safety stop using /proximity_stop Bool
    """

    def __init__(self):
        super().__init__("cmd_arbiter")

        # Parameters (topics)
        self.declare_parameter("out_topic", "/cmd_vel")
        self.declare_parameter("proximity_stop_topic", "/proximity_stop")

        # Sources: names must match the keys in the arrays below (same order).
        self.declare_parameter("source_names", ["grab", "teleop", "snn"])
        self.declare_parameter("source_topics", ["/cmd_vel/grab", "/cmd_vel/teleop", "/cmd_vel/snn"])
        self.declare_parameter("source_priorities", [100, 50, 10])  # higher wins
        self.declare_parameter("source_timeouts_sec", [0.5, 0.5, 0.5])

        # Behavior
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("proximity_stop_active_high", True)
        self.declare_parameter("publish_zero_when_idle", True)

        self.out_topic = self.get_parameter("out_topic").value
        self.prox_topic = self.get_parameter("proximity_stop_topic").value

        names: List[str] = list(self.get_parameter("source_names").value)
        topics: List[str] = list(self.get_parameter("source_topics").value)
        priorities: List[int] = list(self.get_parameter("source_priorities").value)
        timeouts: List[float] = list(self.get_parameter("source_timeouts_sec").value)

        if not (len(names) == len(topics) == len(priorities) == len(timeouts)):
            raise RuntimeError("source_* parameters must have the same length")

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.prox_active_high = bool(self.get_parameter("proximity_stop_active_high").value)
        self.publish_zero_when_idle = bool(self.get_parameter("publish_zero_when_idle").value)

        # Internal: proximity stop
        self.proximity_stop = False

        # Publisher to the *real* cmd_vel
        self.pub = self.create_publisher(Twist, self.out_topic, 10)

        # Track sources
        now = self.get_clock().now()
        self.sources: Dict[str, SourceState] = {}
        for n, t, p, to in zip(names, topics, priorities, timeouts):
            self.sources[n] = SourceState(
                name=n,
                topic=t,
                timeout_sec=float(to),
                priority=int(p),
                last_msg=Twist(),
                last_stamp=now,
                has_msg=False,
            )

        # Subscriptions for each source
        # (Use closures to bind source name)
        for name in names:
            topic = self.sources[name].topic
            self.create_subscription(
                Twist,
                topic,
                lambda msg, n=name: self.cb_source(n, msg),
                10,
            )

        # Proximity stop subscription
        self.create_subscription(Bool, self.prox_topic, self.cb_proximity_stop, 10)

        # Timer to publish at a steady rate
        period = 1.0 / max(1e-6, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.last_published_source: Optional[str] = None

        self.get_logger().info(
            "cmd_arbiter started\n"
            f"  out_topic: {self.out_topic}\n"
            f"  proximity_stop_topic: {self.prox_topic}\n"
            f"  sources:\n" +
            "\n".join(
                [f"    - {s.name}: topic={s.topic} priority={s.priority} timeout={s.timeout_sec}s"
                 for s in sorted(self.sources.values(), key=lambda x: -x.priority)]
            )
        )

    def cb_source(self, name: str, msg: Twist) -> None:
        s = self.sources[name]
        s.last_msg = msg
        s.last_stamp = self.get_clock().now()
        s.has_msg = True

    def cb_proximity_stop(self, msg: Bool) -> None:
        val = bool(msg.data)
        self.proximity_stop = val if self.prox_active_high else (not val)

    def is_fresh(self, s: SourceState, now: Time) -> bool:
        if not s.has_msg:
            return False
        age = now - s.last_stamp
        return age <= Duration(seconds=s.timeout_sec)

    def pick_source(self, now: Time) -> Optional[SourceState]:
        fresh = [s for s in self.sources.values() if self.is_fresh(s, now)]
        if not fresh:
            return None
        # Highest priority wins
        fresh.sort(key=lambda x: x.priority, reverse=True)
        return fresh[0]

    def publish_zero(self) -> None:
        self.pub.publish(Twist())

    def publish_turn_only(self, cmd: Twist) -> None:
        """
        Safety behavior: allow turning away, but prevent forward motion.
        """
        safe = Twist()
        safe.linear.x = 0.0
        safe.linear.y = 0.0
        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        safe.angular.z = cmd.angular.z  # keep yaw turn
        self.pub.publish(safe)

    def on_timer(self) -> None:
        # Safety gate: block linear motion but allow turning away
        if self.proximity_stop:
            now = self.get_clock().now()
            chosen = self.pick_source(now)

            if self.last_published_source != "PROX_TURN_ONLY":
                self.get_logger().warn("Proximity stop active -> blocking linear.x, allowing angular.z")
                self.last_published_source = "PROX_TURN_ONLY"

            if chosen is None:
                # No fresh source -> no safe turn command available
                self.publish_zero()
                return

            self.publish_turn_only(chosen.last_msg)
            return

        now = self.get_clock().now()
        chosen = self.pick_source(now)

        if chosen is None:
            if self.publish_zero_when_idle:
                if self.last_published_source != "IDLE":
                    self.get_logger().info("No fresh cmd sources -> IDLE (zero Twist)")
                    self.last_published_source = "IDLE"
                self.publish_zero()
            return

        # Publish chosen command
        if self.last_published_source != chosen.name:
            self.get_logger().info(f"Arbiter chose source: {chosen.name}")
            self.last_published_source = chosen.name

        self.pub.publish(chosen.last_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
