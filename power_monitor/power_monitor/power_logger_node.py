import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, UInt8MultiArray, Empty
from geometry_msgs.msg import Vector3
import csv
import os
from datetime import datetime
import math
import uuid
from dopamine_interfaces.msg import PhaseEnergyResult

BATTERY_WH = 49.02  # LiHV 3S: 11.4 V * 4.3 Ah = 49.02 Wh

STATE_NAMES = {
    0: "SEARCH_ITEM",
    1: "APPROACH_ITEM",
    2: "SEARCH_DROPOFF",
    3: "APPROACH_DROPOFF",
}

class PowerLogger(Node):
    def __init__(self):
        super().__init__('power_logger')

        # Phase tracking
        self.current_pickup = None
        self.current_phase = None
        self.phase_active = False

        # Run metadata for logging
        self.run_id = str(uuid.uuid4())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "/opt/robot_ws/src/ros2/power_monitor/analysis/csv_logs/Python"
        #log_dir = "/opt/robot_ws/src/ros2/power_monitor/analysis/csv_logs/FPGA"
        os.makedirs(log_dir, exist_ok=True)
        self.filename = f"{log_dir}/power_log_{timestamp}.csv"

        self.get_logger().info(f"Run ID: {self.run_id}")
        self.get_logger().info(f"Logging to CSV: {self.filename}")

        # CSV setup
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.episode_has_data = False

        header = [
            "run_id",
            "episode_id",
            "episode_start_ros_time_s",
            "episode_end_ros_time_s",
            "episode_total_time_s",

            "episode_energy_total_Wh",
            "episode_energy_system_Wh",
            "episode_energy_fpga_Wh",

            "search_energy_total_Wh",
            "approach_energy_total_Wh",

            "search_time_total_s",
            "approach_time_total_s",

            "avg_power_total_W",
            "avg_power_system_W",
            "avg_power_fpga_W",
        ]

        for prefix in ["E_total", "E_system", "E_fpga", "time"]:
            unit = "Wh" if prefix.startswith("E") else "s"

            for pickup in range(3):
                for phase in range(4):
                    header.append(f"{prefix}_p{pickup}_ph{phase}_{unit}")

        self.writer.writerow(header)
        self.file.flush()

        self.reset_episode_data()

        self.episode_id = 0

        #self.get_logger().info(f"Logging to CSV: {self.filename}")

        # latest filtered inputs
        self.system = None
        self.fpga = None
        self.system_time = None
        self.fpga_time = None
        self.current_state = None
        #self.episode_id = None
        self.last_time = None
        self.power_samples = []
        self.power_sample_keep_s = 1800.0  # keep last 30 minutes

        # battery status publisher
        self.battery_pub = self.create_publisher(
            Vector3, "/battery/status", 10
        )

        # subscriptions (already filtered upstream)
        self.create_subscription(
            Vector3, "/system/power", self.cb_system, 10
        )
        self.create_subscription(
            Vector3, "/fpga/power", self.cb_fpga, 10
        )

        self.create_subscription(
            UInt8, "/task/state", self.cb_state, 10
        )

        self.create_subscription(
            UInt8MultiArray, "/task/phase", self.cb_phase, 10
        )

        self.create_subscription(
            Empty, "/episode_complete", self.episode_cb, 10
        )
        
        self.create_subscription(
            PhaseEnergyResult, "/task/phase_result", self.cb_phase_result, 10
        )

    # ---------------- Utility function to get ROS time in seconds ----------------
    def now_ros_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- CSV logging ----------------

    
        
    def write_episode_row(self):
        now = self.now_ros_seconds()
        total_time = now - self.episode_start_time

        # --- Search vs Approach ---
        search_energy = 0.0
        approach_energy = 0.0
        search_time = 0.0
        approach_time = 0.0

        for p in range(3):
            for ph in range(4):
                if ph in (0, 2):  # SEARCH phases
                    search_energy += self.energy_total_phase[p][ph]
                    search_time += self.time_phase[p][ph]
                elif ph in (1, 3):  # APPROACH phases
                    approach_energy += self.energy_total_phase[p][ph]
                    approach_time += self.time_phase[p][ph]

        # --- Average power ---
        avg_power_total = (
            self.episode_energy_total / total_time * 3600.0
            if total_time > 0 else 0.0
        )
        avg_power_system = (
            self.episode_energy_system / total_time * 3600.0
            if total_time > 0 else 0.0
        )
        avg_power_fpga = (
            self.episode_energy_fpga / total_time * 3600.0
            if total_time > 0 else 0.0
        )

        # --- Base row ---
        row = [
            self.run_id,
            self.episode_id,
            self.episode_start_time,
            now,
            total_time,

            self.episode_energy_total,
            self.episode_energy_system,
            self.episode_energy_fpga,

            search_energy,
            approach_energy,

            search_time,
            approach_time,

            avg_power_total,
            avg_power_system,
            avg_power_fpga,
        ]

        # --- Flatten phase data ---
        for prefix in ["total", "system", "fpga", "time"]:
            for p in range(3):
                for ph in range(4):
                    if prefix == "total":
                        row.append(self.energy_total_phase[p][ph])
                    elif prefix == "system":
                        row.append(self.energy_system_phase[p][ph])
                    elif prefix == "fpga":
                        row.append(self.energy_fpga_phase[p][ph])
                    elif prefix == "time":
                        row.append(self.time_phase[p][ph])

        # --- Write ---
        self.writer.writerow(row)
        self.file.flush()

        #self.get_logger().info(
        #    f"[EPISODE {self.episode_id}] "
        #    f"E_total={self.episode_energy_total:.3f}Wh, "
        #    f"time={total_time:.2f}s"
        #)
    # ---------------- Battery calculations ----------------

    def voltage_to_percentage(self, voltage):
        """
        LiHV 3S battery voltage to percentage based on typical discharge curve.
        """

      
        cell_v = voltage / 3.0

        table = [
            (4.35, 100),
            (4.20, 80),
            (4.10, 60),
            (3.90, 40),
            (3.70, 20),
            (3.50, 0),
        ]

        if cell_v >= table[0][0]:
            return 100.0
        if cell_v <= table[-1][0]:
            return 0.0

        for (v1, p1), (v2, p2) in zip(table, table[1:]):
            if v2 <= cell_v <= v1:
                return p2 + (p1 - p2) * (cell_v - v2) / (v1 - v2)

        return 0.0

    def estimate_runtime_minutes(self, voltage, current):
        if current < 0.05:
            return math.inf

        percent = self.voltage_to_percentage(voltage) / 100.0
        remaining_wh = percent * BATTERY_WH
        power = voltage * current

        if power <= 0.5:
            return math.inf

        return (remaining_wh / power) * 60.0

    # ---------------- Callbacks ----------------

    def cb_state(self, msg: UInt8):
        self.current_state = msg.data

    def cb_system(self, msg):
        """
        msg.x = V (filtered)
        msg.y = I (filtered)
        msg.z = P (filtered)
        """
        self.system = msg
        self.system_time = self.now_ros_seconds()
        self.try_publish_battery("system", msg)

    def cb_fpga(self, msg: Vector3):
        self.fpga = msg
        self.fpga_time = self.now_ros_seconds()
        self.try_publish_battery("fpga", msg)

    def cb_phase(self, msg: UInt8MultiArray):
        data = list(msg.data)

        if len(data) < 3:
            return

        pickup, phase, active = data

        self.current_pickup = pickup
        self.current_phase = None if phase == 255 else phase
        self.phase_active = bool(active)

    def episode_cb(self, msg: Empty):
        if self.episode_has_data:
            self.write_episode_row()
        self.episode_id += 1
        self.reset_episode_data()

    def cb_phase_result(self, msg: PhaseEnergyResult):
        p = int(msg.pickup_idx)
        ph = int(msg.phase_idx)

        if not (0 <= p < 3 and 0 <= ph < 4):
            self.get_logger().warn(f"Invalid phase_result indices: p={p}, ph={ph}")
            return

        t_start = float(msg.start_time_s)
        t_end = float(msg.end_time_s)

        if t_end <= t_start:
            self.get_logger().warn(f"Invalid phase times: start={t_start}, end={t_end}")
            return

        E_total = 0.0
        E_system = 0.0
        E_fpga = 0.0

        prev_sample = None

        for sample in self.power_samples:
            t, P_system, P_fpga, P_total = sample

            if prev_sample is None:
                prev_sample = sample
                continue

            t_prev, P_sys_prev, P_fpga_prev, P_tot_prev = prev_sample

            # Check if this interval overlaps with phase window
            if t <= t_start:
                prev_sample = sample
                continue

            if t_prev >= t_end:
                break

            # Clip interval to phase window
            t0 = max(t_prev, t_start)
            t1 = min(t, t_end)

            dt = t1 - t0
            if dt <= 0:
                prev_sample = sample
                continue

            # Use previous power sample (zero-order hold)
            E_system += (P_sys_prev * dt) / 3600.0
            E_fpga += (P_fpga_prev * dt) / 3600.0
            E_total += (P_tot_prev * dt) / 3600.0

            prev_sample = sample

        # Store results
        self.energy_total_phase[p][ph] = E_total
        self.energy_system_phase[p][ph] = E_system
        self.energy_fpga_phase[p][ph] = E_fpga
        self.time_phase[p][ph] = float(msg.duration_s)

        self.get_logger().info(
            f"[PhaseResult] p={p}, ph={ph}, "
            f"E_total={E_total:.4f}Wh, "
            f"E_sys={E_system:.4f}Wh, "
            f"E_fpga={E_fpga:.4f}Wh, "
            f"t={msg.duration_s:.2f}s"
        )

    # ---------------- Battery aggregation ----------------

    def try_publish_battery(self, source, msg):
        if self.system is None or self.fpga is None:
            return

        # Voltage assumed identical source rail
        V = self.system.x

        # Total load
        I = self.system.y + self.fpga.y
        P = self.system.z + self.fpga.z

        # Energy integration
        now = self.now_ros_seconds()

        if self.last_time is not None:
            dt = now - self.last_time

            if dt > 0.0:
                # --- Power split ---
                P_total = P
                P_system = self.system.z
                P_fpga = self.fpga.z

                self.power_samples.append((now, P_system, P_fpga, P_total))

                cutoff = now - self.power_sample_keep_s
                while self.power_samples and self.power_samples[0][0] < cutoff:
                    self.power_samples.pop(0)

                # --- Energy increments (Wh) ---
                E_total = (P_total * dt) / 3600.0
                E_system = (P_system * dt) / 3600.0
                E_fpga = (P_fpga * dt) / 3600.0

                # --- Episode totals ---
                self.episode_energy_total += E_total
                self.episode_energy_system += E_system
                self.episode_energy_fpga += E_fpga
                self.episode_has_data = True

                # # --- Phase-based logging ---
                # if (
                #     self.phase_active and
                #     self.current_pickup is not None and
                #     self.current_phase is not None and
                #     0 <= self.current_pickup < 3 and
                #     0 <= self.current_phase < 4
                # ):
                #     p = self.current_pickup
                #     ph = self.current_phase

                #     self.energy_total_phase[p][ph] += E_total
                #     self.energy_system_phase[p][ph] += E_system
                #     self.energy_fpga_phase[p][ph] += E_fpga

                #     # time in seconds
                #     self.time_phase[p][ph] += dt

        self.last_time = now

        # Log CSV
        #self.write_csv(source, msg, energy_inc)

        # Battery status estimation for dashboard
        percent = self.voltage_to_percentage(V)
        runtime_min = self.estimate_runtime_minutes(V, I)

        # self.get_logger().info(
        #     f"[battery] V={V:.2f}V  I={I:.2f}A  "
        #     f"P={P:.1f}W  {percent:.0f}%  "
        #     f"E_episode={self.episode_energy_total:.3f}Wh  "
        #     f"pickup={self.current_pickup}, phase={self.current_phase}, active={self.phase_active} "
        #     f"state={STATE_NAMES.get(self.current_state, 'UNKNOWN')}  "
        #     f"runtime={runtime_min:.0f} min"
        # )

        # Publish battery status
        out = Vector3()
        out.x = P              # total power [W]
        out.y = percent        # SOC [%]
        out.z = runtime_min    # remaining time [min]
        self.battery_pub.publish(out)

    def reset_episode_data(self):
        def zero_3x4():
            return [[0.0 for _ in range(4)] for _ in range(3)]

        self.energy_total_phase = zero_3x4()
        self.energy_fpga_phase = zero_3x4()
        self.energy_system_phase = zero_3x4()
        self.time_phase = zero_3x4()

        self.episode_energy_total = 0.0
        self.episode_energy_fpga = 0.0
        self.episode_energy_system = 0.0
        self.episode_start_time = self.now_ros_seconds()

        self.power_samples = []
        self.last_time = None
        self.episode_has_data = False

    # Cleanup on shutdown
    def destroy_node(self):
        if hasattr(self, "file") and not self.file.closed:
            if self.episode_has_data:
                self.write_episode_row()
            self.file.close()

        super().destroy_node()

def main():
    rclpy.init()
    node = PowerLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
