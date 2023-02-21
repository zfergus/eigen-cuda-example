#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Core>

namespace ece {

struct Vec3D {
    float x;
    float y;
    float z;
};

Eigen::VectorXf
compute_distances_gpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2);

Eigen::VectorXf compute_distances_gpu_no_eigen(
    const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2);

Eigen::VectorXf
compute_distances_cpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2);

} // namespace ece