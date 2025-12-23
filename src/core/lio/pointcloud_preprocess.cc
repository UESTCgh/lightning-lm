#include "pointcloud_preprocess.h"
#include <execution>

#include <glog/logging.h>

namespace lightning {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const sensor_msgs::PointCloud2 ::ConstPtr &msg, PointCloudType::Ptr &pcl_out) {
    switch (lidar_type_) {
        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                                   PointCloudType::Ptr &pcl_out) {
    cloud_out_.clear();
    cloud_full_.clear();

    if (!msg) {
        pcl_out->clear();
        return;
    }

    const std::size_t plsize = static_cast<std::size_t>(msg->point_num);
    const std::size_t ptsz   = msg->points.size();

    constexpr std::size_t kMaxPoints = 2'000'000;
    if (plsize < 2 || plsize != ptsz || plsize > kMaxPoints) {
        LOG(ERROR) << "Bad Livox msg: point_num=" << msg->point_num
                   << " points.size=" << ptsz;
        pcl_out->clear();
        return;
    }

    cloud_out_.points.clear();
    cloud_out_.points.reserve(plsize);
    cloud_full_.resize(plsize);
    std::vector<uint8_t> is_valid_pt(plsize, 0);

    // index 用 size_t，避免符号/溢出问题
    std::vector<std::size_t> index(plsize - 1);
    for (std::size_t i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从 1 开始
    }

    // 用 par 比 par_unseq 更稳（少很多奇怪 UB）
    std::for_each(std::execution::par, index.begin(), index.end(),
                  [&](const std::size_t i) {
        const auto &pt = msg->points[i];

        if ((pt.line < static_cast<uint32_t>(num_scans_)) &&
            (((pt.tag & 0x30) == 0x10) || ((pt.tag & 0x30) == 0x00))) {

            if (point_filter_num_ > 0 &&
                (i % static_cast<std::size_t>(point_filter_num_) == 0)) {

                cloud_full_[i].x = pt.x;
                cloud_full_[i].y = pt.y;
                cloud_full_[i].z = pt.z;
                cloud_full_[i].intensity = pt.reflectivity;

                cloud_full_[i].time = pt.offset_time / 1000000.0;  // ms

                const auto &prev = cloud_full_[i - 1];
                const double dx = cloud_full_[i].x - prev.x;
                const double dy = cloud_full_[i].y - prev.y;
                const double dz = cloud_full_[i].z - prev.z;

                const double r2 = cloud_full_[i].x * cloud_full_[i].x +
                                  cloud_full_[i].y * cloud_full_[i].y +
                                  cloud_full_[i].z * cloud_full_[i].z;

                if ((std::abs(dx) > 1e-7 || std::abs(dy) > 1e-7 || std::abs(dz) > 1e-7) &&
                    (r2 > blind_ * blind_)) {
                    is_valid_pt[i] = 1;
                }
            }
        }
    });

    for (std::size_t i = 1; i < plsize; ++i) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }

    cloud_out_.width = static_cast<uint32_t>(cloud_out_.points.size());
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
    *pcl_out = cloud_out_;
}


void PointCloudPreprocess::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) {
            continue;
        }

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) {
            continue;
        }

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.time = pl_orig.points[i].t / 1e6;  // curvature unit: ms

        cloud_out_.points.push_back(added_pt);
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.time = pl_orig.points[i].time * time_scale_;  // curvature unit: ms

        if (!given_offset_time_) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.time = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.time;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.time = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.time = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.time < time_last[layer]) {
                added_pt.time += 360.0 / omega_l;
            }

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.time;
        }

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

}  // namespace lightning
