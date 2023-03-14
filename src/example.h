#pragma once

#include <Eigen/Core>

#include <vector>

namespace ece {

// Point-point distances

template <bool USE_EIGEN = true>
Eigen::VectorXf compute_point_point_distances_gpu(
    const Eigen::MatrixXf& V,
    const std::vector<std::array<int, 2>>& point_pairs);

Eigen::VectorXf compute_point_point_distances_cpu(
    const Eigen::MatrixXf& V,
    const std::vector<std::array<int, 2>>& point_pairs);

// Line-line distances

template <bool USE_EIGEN = true>
Eigen::VectorXf compute_line_line_distances_gpu(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& E,
    const std::vector<std::array<int, 2>>& line_pairs);

Eigen::VectorXf compute_line_line_distances_cpu(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& E,
    const std::vector<std::array<int, 2>>& line_pairs);

// Lambda functions

Eigen::VectorXf apply_function_on_gpu(const Eigen::VectorXf& x);

} // namespace ece