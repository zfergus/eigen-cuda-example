#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Core>

Eigen::VectorXf
compute_distances_gpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2);

Eigen::VectorXf
compute_distances_cpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2);