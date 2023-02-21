#include "config.hpp"

#include "example.h"

#include <Eigen/Geometry>

namespace ece {

// Eigen-based

__host__ __device__ float
point_point_distance(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
{
    return (p1 - p2).squaredNorm();
}

__host__ __device__ float line_line_distance(
    const Eigen::Vector3f& ea0,
    const Eigen::Vector3f& ea1,
    const Eigen::Vector3f& eb0,
    const Eigen::Vector3f& eb1)
{
    const Eigen::Vector3f normal = (ea1 - ea0).cross(eb1 - eb0);
    const float line_to_line = (eb0 - ea0).dot(normal);
    return line_to_line * line_to_line / normal.squaredNorm();
}

template <typename Vector>
__global__ void compute_point_point_distances_cuda(
    const Vector* const p1, const Vector* const p2, float* const out, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = point_point_distance(p1[idx], p2[idx]);
    }
}

template <typename Vector>
Eigen::VectorXf compute_point_point_distances_gpu_common(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2)
{
    Vector *d_p1, *d_p2;
    float* d_out;
    assert(p1.rows() == 3 && p2.rows() == 3);
    assert(p1.cols() == p2.cols());
    const size_t N = p1.cols();

    cudaMalloc(&d_p1, N * sizeof(Vector));
    cudaMalloc(&d_p2, N * sizeof(Vector));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_p1, p1.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, p2.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);

    compute_point_point_distances_cuda<<<(N + 255) / 256, 256>>>(
        d_p1, d_p2, d_out, N);

    Eigen::VectorXf out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_out);

    return out;
}

Eigen::VectorXf compute_point_point_distances_gpu(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2)
{
    return compute_point_point_distances_gpu_common<Eigen::Vector3f>(p1, p2);
}

template <typename Vector>
__global__ void compute_line_line_distances_cuda(
    const Vector* const ea0,
    const Vector* const ea1,
    const Vector* const eb0,
    const Vector* const eb1,
    float* const out,
    size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = line_line_distance(ea0[idx], ea1[idx], eb0[idx], eb1[idx]);
    }
}

template <typename Vector>
Eigen::VectorXf compute_line_line_distances_gpu_common(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1)
{
    Vector *d_ea0, *d_ea1, *d_eb0, *d_eb1;
    float* d_out;
    assert(ea0.cols() == ea1.cols());
    assert(ea0.cols() == eb0.cols());
    assert(ea0.cols() == eb1.cols());
    const size_t N = ea0.cols();

    cudaMalloc(&d_ea0, N * sizeof(Vector));
    cudaMalloc(&d_ea1, N * sizeof(Vector));
    cudaMalloc(&d_eb0, N * sizeof(Vector));
    cudaMalloc(&d_eb1, N * sizeof(Vector));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_ea0, ea0.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ea1, ea1.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eb0, eb0.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eb1, eb1.data(), N * sizeof(Vector), cudaMemcpyHostToDevice);

    compute_line_line_distances_cuda<<<(N + 255) / 256, 256>>>(
        d_ea0, d_ea1, d_eb0, d_eb1, d_out, N);

    Eigen::VectorXf out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_ea0);
    cudaFree(d_ea1);
    cudaFree(d_eb0);
    cudaFree(d_eb1);
    cudaFree(d_out);

    return out;
}

Eigen::VectorXf compute_line_line_distances_gpu(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1)
{
    return compute_line_line_distances_gpu_common<Eigen::Vector3f>(
        ea0, ea1, eb0, eb1);
}

// Baseline CUDA implementation

struct Vec3D {
    float x;
    float y;
    float z;
};

__host__ __device__ float point_point_distance(const Vec3D& p1, const Vec3D& p2)
{
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)
        + (p1.z - p2.z) * (p1.z - p2.z);
}

__host__ __device__ float line_line_distance(
    const Vec3D& ea0, const Vec3D& ea1, const Vec3D& eb0, const Vec3D& eb1)
{
    const float t0 = ea0.x - ea1.x;
    const float t1 = eb0.y - eb1.y;
    const float t2 = ea0.y - ea1.y;
    const float t3 = eb0.x - eb1.x;
    const float t4 = t0 * t1 - t2 * t3;
    const float t5 = eb0.z - eb1.z;
    const float t6 = ea0.z - ea1.z;
    const float t7 = -t1 * t6 + t2 * t5;
    const float t8 = t4 * (ea0.z - eb0.z) + t7 * (ea0.x - eb0.x)
        - (ea0.y - eb0.y) * (t0 * t5 - t3 * t6);
    const float t9 = t4 * t4;
    const float t10 = t7 * t7;
    const float t11 = t0 * t5 - t3 * t6;
    return t8 * t8 / (t9 + t10 + t11 * t11);
}

Eigen::VectorXf compute_point_point_distances_gpu_no_eigen(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2)
{
    return compute_point_point_distances_gpu_common<Vec3D>(p1, p2);
}

Eigen::VectorXf compute_line_line_distances_gpu_no_eigen(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1)
{
    return compute_line_line_distances_gpu_common<Vec3D>(ea0, ea1, eb0, eb1);
}

// Baseline CPU implementation

Eigen::VectorXf compute_point_point_distances_cpu(
    const Eigen::MatrixXf& p1, const Eigen::MatrixXf& p2)
{
    Eigen::VectorXf out(p1.cols());
    for (int i = 0; i < p1.cols(); i++) {
        out(i) = point_point_distance(p1.col(i), p2.col(i));
    }
    return out;
}

Eigen::VectorXf compute_line_line_distances_cpu(
    const Eigen::MatrixXf& ea0,
    const Eigen::MatrixXf& ea1,
    const Eigen::MatrixXf& eb0,
    const Eigen::MatrixXf& eb1)
{
    Eigen::VectorXf out(ea0.cols());
    for (int i = 0; i < ea0.cols(); i++) {
        out(i) =
            line_line_distance(ea0.col(i), ea1.col(i), eb0.col(i), eb1.col(i));
    }
    return out;
}

} // namespace ece