// Proximity sensor node. Will only run on Pi. 
//
// Must install on Pi:
// sudo apt update
// sudo apt install -y pigpio libpigpio-dev
// colcon build --packages-select hc_sr04_driver
//
// Then start daemon:
// sudo systemctl enable pigpiod
// sudo systemctl start pigpiod
//
// Check if it is running:
// systemctl status pigpiod


#include <chrono>
#include <cmath>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

// pigpio C API
#include <pigpio.h>

using namespace std::chrono_literals;

class HcSr04LaserScanNode : public rclcpp::Node
{
public:
  HcSr04LaserScanNode()
  : Node("hc_sr04_laserscan_node")
  {
    // Parameters (pins can be set later)
    this->declare_parameter<int>("trigger_pin", 23);  // BCM numbering
    this->declare_parameter<int>("echo_pin", 24);     // BCM numbering

    this->declare_parameter<std::string>("topic", "/ultrasonic/front/scan");
    this->declare_parameter<std::string>("frame_id", "ultrasonic_front_link");

    this->declare_parameter<double>("rate_hz", 10.0);
    this->declare_parameter<double>("range_min", 0.02);
    this->declare_parameter<double>("range_max", 4.0);

    // Speed of sound (m/s). ~343 at 20C; you can tune later if desired.
    this->declare_parameter<double>("speed_of_sound", 343.0);

    // Timeout waiting for echo pulse (microseconds)
    this->declare_parameter<int>("echo_timeout_us", 30000); // ~5m round-trip, conservative

    // Simple median filter window (odd number: 1,3,5...). 1 = no filtering
    this->declare_parameter<int>("median_window", 3);

    // Read params
    trigger_pin_ = this->get_parameter("trigger_pin").as_int();
    echo_pin_    = this->get_parameter("echo_pin").as_int();
    topic_       = this->get_parameter("topic").as_string();
    frame_id_    = this->get_parameter("frame_id").as_string();

    rate_hz_     = this->get_parameter("rate_hz").as_double();
    range_min_   = this->get_parameter("range_min").as_double();
    range_max_   = this->get_parameter("range_max").as_double();
    c_           = this->get_parameter("speed_of_sound").as_double();
    echo_timeout_us_ = this->get_parameter("echo_timeout_us").as_int();
    median_window_   = this->get_parameter("median_window").as_int();

    if (median_window_ < 1) median_window_ = 1;
    if (median_window_ % 2 == 0) median_window_ += 1;

    pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(topic_, 10);

    // Init pigpio
    // Note: pigpio can work via daemon or direct init; gpioInitialise() works direct.
    int rc = gpioInitialise();
    if (rc < 0) {
      throw std::runtime_error("pigpio gpioInitialise() failed. Is pigpiod running or are permissions correct?");
    }

    // Configure pins
    gpioSetMode(trigger_pin_, PI_OUTPUT);
    gpioSetMode(echo_pin_, PI_INPUT);

    gpioWrite(trigger_pin_, 0);
    gpioSetPullUpDown(echo_pin_, PI_PUD_DOWN);

    // Set up alert (edge) callback for echo pin
    gpioSetAlertFuncEx(echo_pin_, &HcSr04LaserScanNode::echo_alert_trampoline, this);

    // Timer for publishing at rate_hz
    auto period = std::chrono::duration<double>(1.0 / std::max(1e-6, rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&HcSr04LaserScanNode::on_timer, this)
    );

    RCLCPP_INFO(this->get_logger(),
      "HC-SR04 LaserScan node started. trigger_pin=%d echo_pin=%d topic=%s frame_id=%s rate=%.2fHz",
      trigger_pin_, echo_pin_, topic_.c_str(), frame_id_.c_str(), rate_hz_);
  }

  ~HcSr04LaserScanNode() override
  {
    // Remove callback and terminate pigpio
    gpioSetAlertFuncEx(echo_pin_, nullptr, nullptr);
    gpioTerminate();
  }

private:
  // pigpio alert callback trampoline
  static void echo_alert_trampoline(int gpio, int level, uint32_t tick, void* userdata)
  {
    auto* self = static_cast<HcSr04LaserScanNode*>(userdata);
    self->on_echo_edge(gpio, level, tick);
  }

  void on_echo_edge(int /*gpio*/, int level, uint32_t tick)
  {
    // level: 0 falling, 1 rising, 2 watchdog timeout
    if (level == 1) {
      // Rising edge: start timing
      echo_rise_tick_.store(tick, std::memory_order_relaxed);
      have_rise_.store(true, std::memory_order_relaxed);
    } else if (level == 0) {
      // Falling edge: compute pulse width
      if (!have_rise_.load(std::memory_order_relaxed)) return;

      uint32_t rise = echo_rise_tick_.load(std::memory_order_relaxed);
      uint32_t fall = tick;

      // pigpio tick wraps ~72 minutes; gpioTickDiff handles wrap.
      uint32_t pulse_us = gpioTickDiff(rise, fall);

      {
        std::lock_guard<std::mutex> lock(meas_mutex_);
        last_pulse_us_ = pulse_us;
        last_meas_time_ = this->now();
        have_measurement_ = true;
      }
      have_rise_.store(false, std::memory_order_relaxed);
    }
  }

  void trigger_ping()
  {
    // HC-SR04 trigger pulse: 10us HIGH
    gpioWrite(trigger_pin_, 0);
    gpioDelay(2);      // settle
    gpioWrite(trigger_pin_, 1);
    gpioDelay(10);
    gpioWrite(trigger_pin_, 0);
  }

  static double median(std::vector<double>& v)
  {
    // v size is small; simple nth_element is fine
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
  }

  void on_timer()
  {
    // Trigger a ping
    trigger_ping();

    // Wait briefly for measurement (non-blocking-ish but bounded)
    // We’ll poll a few times up to echo_timeout_us_ total.
    auto start = this->now();
    bool got = false;
    uint32_t pulse_us = 0;

    while (((this->now() - start).nanoseconds() / 1000) < echo_timeout_us_) {
      {
        std::lock_guard<std::mutex> lock(meas_mutex_);
        if (have_measurement_) {
          pulse_us = last_pulse_us_;
          have_measurement_ = false; // consume it
          got = true;
          break;
        }
      }
      // sleep a tiny bit to reduce CPU churn
      std::this_thread::sleep_for(200us);
    }

    double distance_m = std::numeric_limits<double>::infinity();

    if (got) {
      double t = static_cast<double>(pulse_us) * 1e-6; // seconds
      distance_m = (t * c_) / 2.0;

      // Clamp to valid range; if out-of-range, treat as invalid
      if (!(std::isfinite(distance_m)) || distance_m < range_min_ || distance_m > range_max_) {
        distance_m = std::numeric_limits<double>::infinity();
      }
    }

    // Median filter (optional)
    if (median_window_ > 1) {
      if (std::isfinite(distance_m)) {
        recent_.push_back(distance_m);
        if ((int)recent_.size() > median_window_) recent_.erase(recent_.begin());
      }
      if ((int)recent_.size() == median_window_) {
        auto tmp = recent_;
        distance_m = median(tmp);
      }
    }

    // Publish LaserScan with a single range sample
    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp = this->now();
    scan.header.frame_id = frame_id_;

    scan.angle_min = 0.0;
    scan.angle_max = 0.0;
    scan.angle_increment = 0.0;

    scan.time_increment = 0.0;
    scan.scan_time = 1.0 / std::max(1e-6, rate_hz_);

    scan.range_min = range_min_;
    scan.range_max = range_max_;

    scan.ranges.resize(1);
    scan.ranges[0] = static_cast<float>(distance_m);

    pub_->publish(scan);
  }

private:
  int trigger_pin_;
  int echo_pin_;
  std::string topic_;
  std::string frame_id_;

  double rate_hz_;
  double range_min_;
  double range_max_;
  double c_;
  int echo_timeout_us_;
  int median_window_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Edge timing state
  std::atomic<uint32_t> echo_rise_tick_{0};
  std::atomic<bool> have_rise_{false};

  // Measurement handoff
  std::mutex meas_mutex_;
  bool have_measurement_{false};
  uint32_t last_pulse_us_{0};
  rclcpp::Time last_meas_time_{0, 0, RCL_ROS_TIME};

  std::vector<double> recent_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<HcSr04LaserScanNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    fprintf(stderr, "Fatal: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
