"""
Side-by-side Python SNN vs. FPGA SNN comparison launch.

Both networks consume the same /snn/input and the same /reward/dopamine.
Only the Python SNN drives /cmd_vel/snn (and so the robot). The FPGA decoder's
outputs are remapped under /snn/fpga/* so cmd_arbiter and dopamine_reward_node
see only the Python side.

Run it:
  ros2 launch my_ros2_bringup taskbot_comparison.launch.py
  # or with an explicit seed file:
  ros2 launch my_ros2_bringup taskbot_comparison.launch.py \
      initial_weights:=/path/to/seed.mem

Initial weights:
  Default seed is /opt/robot_ws/src/ros2/weights_logs/weights_current.mem —
  the same file the standalone python_snn flow writes to. So right after a
  normal taskbot.launch.py training run, this launch is ready to go with no
  manual file staging. The launch banner reports the resolved seed path and
  loudly warns if the file is missing.

When weights are saved (both sides, per episode):
  Trigger: img_recog publishes /episode_complete on successful drop-off.
  - weights_logger → /save_weights service → Python SNN writes
      python_snn/weights_current.mem
      python_snn/episode_logs/weights_ep_NNNN.mem
  - uart_node → CMD_STOP → FPGA returns weights → uart_node writes
      fpga_snn/weights_current.mem
      fpga_snn/episode_logs/weights_ep_NNNN.mem
  Both sides run in parallel; episode numbering is per-side (independent
  counters). Mid-episode the FPGA does not expose a snapshot command, so
  on-disk comparison is only meaningful at episode boundaries.

Per-tick decision logging:
  snn_comparator writes a single CSV (one row per Python /snn/winner event)
  to ~/.ros/snn_comparison_logs/snn_comparison_<timestamp>.csv — see
  snn_comparator.py for the schema.

Caveat:
  The dopamine reward is computed from the Python winner because
  dopamine_reward_node subscribes to /snn/winner only. The FPGA receives the
  same /reward/dopamine, so its R-STDP updates are technically off-policy
  whenever the two networks disagree on the winner. This is unavoidable when
  one body is driving live; expect the weights to drift apart faster than the
  decisions do.

Known fragility (existing, not introduced here):
  episode_complete_callback in uart_node returns early if state != READY.
  If /episode_complete fires while a SPIKE round-trip is in flight, the FPGA
  weight save for that episode is skipped. Look for "Ignoring episode
  complete" in /uart/error if some episodes are missing on the FPGA side.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


DEFAULT_INITIAL_WEIGHTS = '/opt/robot_ws/src/ros2/weights_logs/weights_current.mem'
PYTHON_WEIGHTS_DIR = '/opt/robot_ws/src/ros2/weights_logs/python_snn'
FPGA_WEIGHTS_FILE = '/opt/robot_ws/src/ros2/weights_logs/fpga_snn/weights_current.mem'
POWER_LOG_DIR = '/opt/robot_ws/src/ros2/power_monitor/analysis/csv_logs/Comparison'


def _check_seed(context, *_args, **_kwargs):
    """Print a clear status line about the seed weights file at launch."""
    seed = LaunchConfiguration('initial_weights').perform(context)
    if not seed:
        return [LogInfo(msg='[comparison] No initial_weights set — both SNNs will use their own internal init (NOT directly comparable).')]
    if not os.path.isfile(seed):
        return [LogInfo(
            msg=(
                f'[comparison] WARNING: initial_weights file not found: {seed}\n'
                '[comparison] Each SNN will fall back to its own internal init, '
                'so tick-zero weights will differ. Stage the file or pass '
                'initial_weights:=/path/to/seed.mem to make the comparison meaningful.'
            )
        )]
    return [LogInfo(msg=f'[comparison] Both SNNs will seed from: {seed}')]


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('my_ros2_bringup'), 'config', 'params.yaml'
    )
    with open(params_file) as f:
        all_params = yaml.safe_load(f)

    def p(node_name):
        return all_params.get(node_name, {}).get('ros__parameters', {})

    camera_config = PathJoinSubstitution(
        [FindPackageShare('robot_camera_config'), 'config', 'c922.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument('motor_control', default_value='true'),
        DeclareLaunchArgument(
            'initial_weights',
            default_value=DEFAULT_INITIAL_WEIGHTS,
            description=(
                'Path to the .mem file both SNNs load on startup so they begin '
                'from identical weights. Default is the file the standalone '
                'python_snn flow writes to, so a normal training run leaves '
                'the comparison ready to launch.'
            ),
        ),
        OpaqueFunction(function=_check_seed),
        LogInfo(msg=f'[comparison] Python SNN saves under: {PYTHON_WEIGHTS_DIR}'),
        LogInfo(msg=f'[comparison] FPGA   SNN saves under: {os.path.dirname(FPGA_WEIGHTS_FILE)}'),
        LogInfo(msg='[comparison] Per-tick decisions CSV: ~/.ros/snn_comparison_logs/'),
        LogInfo(msg=f'[comparison] Per-episode power CSV: {POWER_LOG_DIR}'),

        # ── Camera + TF ───────────────────────────────────────────────────────
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='c922_camera',
            namespace='camera',
            parameters=[
                camera_config,
                {
                    'camera_name': 'c922',
                    'camera_info_url':
                        'file:///opt/robot_ws/install/robot_camera_config/share/robot_camera_config/config/c922_camera_info.yaml',
                },
            ],
            remappings=[('image_raw', 'image_raw')],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf',
            arguments=[
                '--x', '0.10', '--y', '0.0', '--z', '0.09',
                '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0',
                '--frame-id', 'base_link', '--child-frame-id', 'camera_link',
            ],
            output='screen',
        ),

        # ── Sensing ───────────────────────────────────────────────────────────
        Node(
            package='distance_sensor',
            executable='proximity_adapter_node',
            name='proximity_adapter_node',
            output='screen',
        ),
        Node(
            package='distance_sensor',
            executable='distance_sensor_node',
            name='distance_sensor_node',
            output='screen',
        ),
        Node(
            package='proximity_stop',
            executable='proximity_stop_node',
            name='proximity_stop',
            output='screen',
            parameters=[p('proximity_stop')],
        ),

        # ── Vision / encoding ─────────────────────────────────────────────────
        Node(
            package='opencv_nodes',
            executable='img_kp_grid',
            name='img_kp_grid',
            output='screen',
            parameters=[p('img_kp_grid')],
        ),
        Node(
            package='opencv_nodes',
            executable='img_recog',
            name='img_recog',
            output='screen',
            parameters=[p('img_recog')],
        ),
        Node(
            package='encoding_node',
            executable='encoding_node',
            name='encoding_node',
            output='screen',
            parameters=[p('encoding_node')],
        ),

        # ── Python SNN — drives /cmd_vel/snn and is the dopamine reference ────
        Node(
            package='python_snn_node',
            executable='snn_node',
            name='python_snn_node',
            output='screen',
            parameters=[
                p('python_snn_node'),
                {
                    'weights_base_dir': PYTHON_WEIGHTS_DIR,
                    'initial_weights_file': LaunchConfiguration('initial_weights'),
                },
            ],
        ),

        # ── FPGA SNN — outputs remapped to /snn/fpga/* for comparison only ────
        Node(
            package='uart',
            executable='uart_node',
            name='uart_bridge_node',
            output='screen',
            parameters=[
                p('uart_bridge_node'),
                {
                    'save_weights_file': FPGA_WEIGHTS_FILE,
                    'initial_weights_file': LaunchConfiguration('initial_weights'),
                },
            ],
        ),
        Node(
            package='fpga_action_decoder',
            executable='fpga_action_decoder_node',
            name='fpga_action_decoder_node',
            output='screen',
            parameters=[{
                'forward_speed': 0.05,
                'turn_speed': 0.05,
                # Send the FPGA's twist somewhere cmd_arbiter does not watch.
                'cmd_vel_topic': '/cmd_vel/snn_fpga',
            }],
            remappings=[
                ('/snn/winner', '/snn/fpga/winner'),
                ('/snn/decision', '/snn/fpga/decision'),
            ],
        ),

        # ── Episode archiver ──────────────────────────────────────────────────
        # On /episode_complete (from img_recog), two things happen in parallel:
        #   - weights_logger calls /save_weights → Python writes its weights to
        #     python_snn/{weights_current.mem, episode_logs/weights_ep_NNNN.mem}
        #   - uart_node sends CMD_STOP → FPGA returns its weights → uart_node
        #     writes them to fpga_snn/{weights_current.mem, episode_logs/...}
        # Both sides finish independently. img_recog resets when either side
        # publishes /episode_reset.
        Node(
            package='python_snn_node',
            executable='weights_logger',
            name='weights_logger',
            output='screen',
        ),

        # ── Reward + comparator ───────────────────────────────────────────────
        Node(
            package='dopamine_reward_node',
            executable='dopamine_reward_node',
            name='dopamine_reward_node',
            output='screen',
        ),
        Node(
            package='python_snn_node',
            executable='snn_comparator',
            name='snn_comparator',
            output='screen',
            parameters=[p('snn_comparator')],
        ),

        # ── Actuation ─────────────────────────────────────────────────────────
        Node(
            package='motor_control',
            executable='motor_control_node',
            name='motor_control_node',
            output='screen',
            parameters=[p('motor_control_node')],
            condition=IfCondition(LaunchConfiguration('motor_control')),
        ),
        Node(
            package='motor_control',
            executable='gripper_node',
            name='gripper_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('motor_control')),
        ),
        Node(
            package='grab_node',
            executable='grab_node',
            name='grab_node',
            parameters=[p('grab_node')],
            output='screen',
        ),
        Node(
            package='grab_node',
            executable='prox_node',
            name='prox_node',
            parameters=[p('prox_node')],
            output='screen',
        ),
        Node(
            package='cmd_arbiter',
            executable='cmd_arbiter',
            name='cmd_arbiter',
            output='screen',
            parameters=[p('cmd_arbiter')],
        ),

        # ── Task + telemetry ──────────────────────────────────────────────────
        Node(
            package='task_manager',
            executable='task_manager',
            name='task_manager',
            output='screen',
        ),
        Node(
            package='power_monitor',
            executable='system_power_node',
            name='system_power_node',
            output='screen',
        ),
        Node(
            package='power_monitor',
            executable='fpga_power_node',
            name='fpga_power_node',
            output='screen',
        ),
        Node(
            package='power_monitor',
            executable='power_logger',
            name='power_logger',
            output='screen',
            parameters=[
                p('power_logger'),
                {'log_dir': POWER_LOG_DIR},
            ],
        ),
    ])
