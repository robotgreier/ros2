import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from taskbot_interfaces.srv import SaveWeights
from task_manager_interfaces.srv import SetTaskState


class WeightsLogger(Node):
    def __init__(self):
        super().__init__('weights_logger')

        self.episode_counter = 1
        self.busy = False

        self.save_weights_client = self.create_client(SaveWeights, 'save_weights')

        self.episode_sub = self.create_subscription(
            Empty,
            'episode_complete',
            self.episode_complete_callback,
            10
        )

        self.set_state_client = self.create_client(SetTaskState, '/task/set_state')

        self.get_logger().info("weights_logger started")

        self.get_logger().info("Waiting briefly for services...")
        self.save_weights_client.wait_for_service(timeout_sec=2.0)
        self.set_state_client.wait_for_service(timeout_sec=2.0)

        self.get_logger().info(f"save_weights ready: {self.save_weights_client.service_is_ready()}")
        self.get_logger().info(f"set_state ready: {self.set_state_client.service_is_ready()}")

        self.episode_reset_pub = self.create_publisher(
            Empty,
            '/episode_reset',
            10
        )

    def episode_complete_callback(self, msg):
        if self.busy:
            self.get_logger().warn("Already processing an episode completion, ignoring duplicate signal")
            return

        if not self.save_weights_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("save_weights service is not available after waiting")
            return

        self.busy = True

        # 1. Save episode archive
        filename = f"weights_ep_{self.episode_counter:04d}.mem"

        req = SaveWeights.Request()
        req.filename = filename

        self.get_logger().info(f"Saving episode weights: {filename}")
        future = self.save_weights_client.call_async(req)
        future.add_done_callback(self.on_save_weights_done)

        req = SaveWeights.Request()
        req.filename = filename

        self.get_logger().info(f"Requesting weight save to {filename}")
        future = self.save_weights_client.call_async(req)
        future.add_done_callback(self.on_save_weights_done)

    def on_save_weights_done(self, future):
        try:
            response = future.result()

            if response.success:
                self.get_logger().info(f"Episode weight save successful: {response.message}")

                # ---- NEW: update current weights file ----
                req_current = SaveWeights.Request()
                req_current.filename = "../config/weights_current.mem"

                self.get_logger().info("Updating weights_current.mem")
                future_current = self.save_weights_client.call_async(req_current)
                future_current.add_done_callback(self.on_update_current_done)

            else:
                self.get_logger().error(f"Weight save failed: {response.message}")
                self.busy = False

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.busy = False

    def on_update_current_done(self, future):
        try:
            response = future.result()

            if response.success:
                self.get_logger().info("weights_current.mem updated")

                # Continue with state reset
                if not self.set_state_client.wait_for_service(timeout_sec=2.0):
                    self.get_logger().error("set_state service not available")
                    self.busy = False
                    return

                req = SetTaskState.Request()
                req.new_state = 0
                req.requester = "weights_logger"

                self.get_logger().info("Requesting state change to SEARCH_ITEM")
                future_state = self.set_state_client.call_async(req)
                future_state.add_done_callback(self.on_set_state_done)

            else:
                self.get_logger().error(f"Failed to update current weights: {response.message}")
                self.busy = False

        except Exception as e:
            self.get_logger().error(f"Error updating current weights: {e}")
            self.busy = False

    def on_set_state_done(self, future):
        try:
            response = future.result()

            if response.success:
                self.get_logger().info("State transition to SEARCH_ITEM successful")
                self.episode_reset_pub.publish(Empty())
                self.get_logger().info("Published /episode_reset")
                self.episode_counter += 1
            else:
                self.get_logger().error(f"State change failed: {response.message}")

        except Exception as e:
            self.get_logger().error(f"State service call failed: {e}")

        self.busy = False

def main(args=None):
    rclpy.init(args=args)
    node = WeightsLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()