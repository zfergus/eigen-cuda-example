#include "config.hpp"

#include "example.h"

namespace ece {

// Eigen-based

__host__ __device__ float
point_point_distance(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
{
    return (p1 - p2).norm();
}

__global__ void compute_distances_cuda(
    Eigen::Vector3f* v1, Eigen::Vector3f* v2, float* out, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = point_point_distance(v1[idx], v2[idx]);
    }
}

Eigen::VectorXf
compute_distances_gpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2)
{
    Eigen::Vector3f* d_v1;
    Eigen::Vector3f* d_v2;
    float* d_out;
    assert(v1.cols() == v2.cols());
    const size_t N = v1.cols();

    cudaMalloc(&d_v1, N * sizeof(Eigen::Vector3f));
    cudaMalloc(&d_v2, N * sizeof(Eigen::Vector3f));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(
        d_v1, v1.data(), N * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_v2, v2.data(), N * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);

    compute_distances_cuda<<<(N + 255) / 256, 256>>>(d_v1, d_v2, d_out, N);

    Eigen::VectorXf out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_out);

    return out;
}

// Baseline CUDA implementation

struct Vec3D {
    float x;
    float y;
    float z;
};

__host__ __device__ float point_point_distance(const Vec3D& p1, const Vec3D& p2)
{
    return sqrt(
        (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)
        + (p1.z - p2.z) * (p1.z - p2.z));
}

__global__ void
compute_distances_cuda(Vec3D* v1, Vec3D* v2, float* out, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = point_point_distance(v1[idx], v2[idx]);
    }
}

Eigen::VectorXf compute_distances_gpu_no_eigen(
    const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2)
{
    Vec3D* d_v1;
    Vec3D* d_v2;
    float* d_out;
    assert(v1.cols() == v2.cols());
    const size_t N = v1.cols();

    cudaMalloc(&d_v1, N * sizeof(Vec3D));
    cudaMalloc(&d_v2, N * sizeof(Vec3D));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_v1, v1.data(), N * sizeof(Vec3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2.data(), N * sizeof(Vec3D), cudaMemcpyHostToDevice);

    compute_distances_cuda<<<(N + 255) / 256, 256>>>(d_v1, d_v2, d_out, N);

    Eigen::VectorXf out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_out);

    return out;
}

// Baseline CPU implementation

Eigen::VectorXf
compute_distances_cpu(const Eigen::MatrixXf& v1, const Eigen::MatrixXf& v2)
{
    Eigen::VectorXf out(v1.cols());
    for (int i = 0; i < v1.cols(); i++) {
        out(i) = point_point_distance(v1.col(i), v2.col(i));
    }
    return out;
}

} // namespace ece