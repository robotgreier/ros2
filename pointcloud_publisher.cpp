void MonocularSlamNode::publishPointCloud_()
{
    if (!m_SLAM) return;

    // --- Debug: is SLAM even running? ---
    RCLCPP_INFO(this->get_logger(), "publishPointCloud_()");

    // --- Get Atlas ---
    ORB_SLAM3::Atlas* atlas = m_SLAM->GetAtlas();
    RCLCPP_INFO(this->get_logger(), "Atlas: %p", atlas);
    if (!atlas) return;

    // --- Get current map ---
    ORB_SLAM3::Map* map = atlas->GetCurrentMap();
    RCLCPP_INFO(this->get_logger(), "Map: %p", map);
    if (!map) return;

    // --- Get all map points ---
    std::vector<ORB_SLAM3::MapPoint*> mps = map->GetAllMapPoints();
    RCLCPP_INFO(this->get_logger(), "MapPoints: %zu", mps.size());
    if (mps.empty()) return;

    // --- Collect XYZ points ---
    std::vector<std::array<float,3>> points;
    points.reserve(std::min<std::size_t>(mps.size(), pcloud_max_points_));

    std::size_t count = 0;
    const std::size_t step = std::max<std::size_t>(1, pcloud_decimation_);

    for (std::size_t i = 0; i < mps.size(); i += step) {
        ORB_SLAM3::MapPoint* pMP = mps[i];
        if (!pMP) continue;
        if (pMP->isBad()) continue;

        // --- Eigen version of GetWorldPos() ---
        Eigen::Vector3f pos = pMP->GetWorldPos();
        const float x = pos.x();
        const float y = pos.y();
        const float z = pos.z();

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            continue;

        points.push_back({x, y, z});
        if (++count >= pcloud_max_points_) break;
    }

    RCLCPP_INFO(this->get_logger(), "Publishing %zu points", points.size());
    if (points.empty()) return;

    // --- Build PointCloud2 ---
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = this->get_clock()->now();
    cloud_msg.header.frame_id = pcloud_frame_id_;  // "odom" recommended

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (const auto& p : points) {
        *iter_x = p[0]; ++iter_x;
        *iter_y = p[1]; ++iter_y;
        *iter_z = p[2]; ++iter_z;
    }

    pcloud_pub_->publish(cloud_msg);
}
