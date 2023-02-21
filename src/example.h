#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Core>

namespace ece {

// Point-point distances

Eigen::VectorXf compute_point_point_distances_gpu(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2);

Eigen::VectorXf compute_point_point_distances_gpu_no_eigen(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2);

Eigen::VectorXf compute_point_point_distances_cpu(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2);

// Line-line distances

Eigen::VectorXf compute_line_line_distances_gpu(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1);

Eigen::VectorXf compute_line_line_distances_gpu_no_eigen(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1);

Eigen::VectorXf compute_line_line_distances_cpu(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1);

} // namespace ece