#include "config.hpp"

#include "example.h"

#include <Eigen/Geometry>

#include <spdlog/spdlog.h>

#define cudaErrorCheck(ans)                                                    \
    {                                                                          \
        ece::throw_on_cuda_error((ans), __FILE__, __LINE__);                   \
    }

namespace ece {

void throw_on_cuda_error(
    const cudaError_t code, const char* file, const int line)
{
    if (code != cudaSuccess) {
        const std::string message =
            fmt::format("{}({}): {}", file, line, cudaGetErrorString(code));
        spdlog::error(message);
        throw std::runtime_error(message);
    }
}

//==============================================================================
// Point-point example
//==============================================================================

template <typename DerivedP0, typename DerivedP1>
__host__ __device__ inline auto point_point_distance(
    const Eigen::MatrixBase<DerivedP0>& p1,
    const Eigen::MatrixBase<DerivedP1>& p2)
{
    return (p1 - p2).squaredNorm();
}

__host__ __device__ inline float point_point_distance(
    const float p1_x,
    const float p1_y,
    const float p1_z,
    const float p2_x,
    const float p2_y,
    const float p2_z)
{
    const float dx = p1_x - p2_x;
    const float dy = p1_y - p2_y;
    const float dz = p1_z - p2_z;
    return dx * dx + dy * dy + dz * dz;
}

template <bool USE_EIGEN = true>
__global__ void compute_point_point_distances_kernel(
    const float* __restrict__ Vx,
    const float* __restrict__ Vy,
    const float* __restrict__ Vz,
    const size_t __restrict__ n_points,
    const int* __restrict__ pairs,
    const size_t __restrict__ n_pairs,
    float* const out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n_pairs; i += stride) {
        const int vi = pairs[2 * i];
        const int vj = pairs[2 * i + 1];
        if constexpr (USE_EIGEN) {
            out[i] = point_point_distance(
                Eigen::Vector3f(Vx[vi], Vy[vi], Vz[vi]),
                Eigen::Vector3f(Vx[vj], Vy[vj], Vz[vj]));
        } else {
            out[i] = point_point_distance(
                Vx[vi], Vy[vi], Vz[vi], Vx[vj], Vy[vj], Vz[vj]);
        }
    }
}

template <bool USE_EIGEN>
Eigen::VectorXf compute_point_point_distances_gpu(
    const Eigen::MatrixXf& V,
    const std::vector<std::array<int, 2>>& point_pairs)
{
    assert(V.cols() == 3);
    const size_t n_points = V.rows();
    const size_t n_pairs = point_pairs.size();

    float* d_V;
    cudaErrorCheck(cudaMalloc(&d_V, V.size() * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(
        d_V, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));
    const float* const d_Vx = d_V;
    const float* const d_Vy = d_Vx + n_points;
    const float* const d_Vz = d_Vy + n_points;

    int* d_pairs;
    cudaErrorCheck(cudaMalloc(&d_pairs, 2 * n_pairs * sizeof(int)));
    cudaErrorCheck(cudaMemcpy(
        d_pairs, point_pairs.data(), 2 * n_pairs * sizeof(int),
        cudaMemcpyHostToDevice));

    float* d_out;
    cudaErrorCheck(cudaMalloc(&d_out, n_pairs * sizeof(float)));

    const int block_size = 256;
    const int num_blocks = (n_pairs + block_size - 1) / block_size;
    compute_point_point_distances_kernel<USE_EIGEN><<<num_blocks, block_size>>>(
        d_Vx, d_Vy, d_Vz, n_points, d_pairs, n_pairs, d_out);
    cudaErrorCheck(cudaPeekAtLastError());
    cudaErrorCheck(cudaDeviceSynchronize());

    Eigen::VectorXf out(n_pairs);
    cudaMemcpy(
        out.data(), d_out, n_pairs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_V);
    cudaFree(d_pairs);
    cudaFree(d_out);

    return out;
}

//==============================================================================
// Line-line example
//==============================================================================

__host__ __device__ inline float line_line_distance(
    const Eigen::Vector3f& ea0,
    const Eigen::Vector3f& ea1,
    const Eigen::Vector3f& eb0,
    const Eigen::Vector3f& eb1)
{
    const Eigen::Vector3f normal = (ea1 - ea0).cross(eb1 - eb0);
    const float line_to_line = (eb0 - ea0).dot(normal);
    assert(normal.squaredNorm() != 0);
    return line_to_line * line_to_line / normal.squaredNorm();
}

__host__ __device__ inline float line_line_distance(
    const float ea0_x,
    const float ea0_y,
    const float ea0_z,
    const float ea1_x,
    const float ea1_y,
    const float ea1_z,
    const float eb0_x,
    const float eb0_y,
    const float eb0_z,
    const float eb1_x,
    const float eb1_y,
    const float eb1_z)
{
    const float t0 = ea0_x - ea1_x;
    const float t1 = eb0_y - eb1_y;
    const float t2 = ea0_y - ea1_y;
    const float t3 = eb0_x - eb1_x;
    const float t4 = t0 * t1 - t2 * t3;
    const float t5 = eb0_z - eb1_z;
    const float t6 = ea0_z - ea1_z;
    const float t7 = -t1 * t6 + t2 * t5;
    const float t8 = t4 * (ea0_z - eb0_z) + t7 * (ea0_x - eb0_x)
        - (ea0_y - eb0_y) * (t0 * t5 - t3 * t6);
    const float t9 = t4 * t4;
    const float t10 = t7 * t7;
    const float t11 = t0 * t5 - t3 * t6;
    return t8 * t8 / (t9 + t10 + t11 * t11);
}

template <bool USE_EIGEN = true>
__global__ void compute_line_line_distances_kernel(
    const float* __restrict__ Vx,
    const float* __restrict__ Vy,
    const float* __restrict__ Vz,
    const size_t __restrict__ n_points,
    const int* __restrict__ E0,
    const int* __restrict__ E1,
    const size_t __restrict__ n_edges,
    const int* __restrict__ pairs,
    const size_t __restrict__ n_pairs,
    float* const out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n_pairs; i += stride) {
        const int ei = pairs[2 * i];
        const int ej = pairs[2 * i + 1];
        const int ea0 = E0[ei], ea1 = E1[ei];
        const int eb0 = E0[ej], eb1 = E1[ej];

        if constexpr (USE_EIGEN) {
            out[i] = line_line_distance(
                Eigen::Vector3f(Vx[ea0], Vy[ea0], Vz[ea0]),
                Eigen::Vector3f(Vx[ea1], Vy[ea1], Vz[ea1]),
                Eigen::Vector3f(Vx[eb0], Vy[eb0], Vz[eb0]),
                Eigen::Vector3f(Vx[eb1], Vy[eb1], Vz[eb1]));
        } else {
            out[i] = line_line_distance(
                Vx[ea0], Vy[ea0], Vz[ea0], //
                Vx[ea1], Vy[ea1], Vz[ea1], //
                Vx[eb0], Vy[eb0], Vz[eb0], //
                Vx[eb1], Vy[eb1], Vz[eb1]);
        }
    }
}

template <bool USE_EIGEN>
Eigen::VectorXf compute_line_line_distances_gpu(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& E,
    const std::vector<std::array<int, 2>>& line_pairs)
{
    assert(V.cols() == 3);
    const size_t n_points = V.rows();
    const size_t n_edges = E.rows();
    const size_t n_pairs = line_pairs.size();

    float* d_V;
    cudaErrorCheck(cudaMalloc(&d_V, V.size() * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(
        d_V, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));
    const float* const d_Vx = d_V;
    const float* const d_Vy = d_Vx + n_points;
    const float* const d_Vz = d_Vy + n_points;

    int* d_E;
    cudaErrorCheck(cudaMalloc(&d_E, E.size() * sizeof(int)));
    cudaErrorCheck(cudaMemcpy(
        d_E, E.data(), E.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int* const d_E0 = d_E;
    const int* const d_E1 = d_E + n_edges;

    int* d_pairs;
    cudaErrorCheck(cudaMalloc(&d_pairs, 2 * n_pairs * sizeof(int)));
    cudaErrorCheck(cudaMemcpy(
        d_pairs, line_pairs.data(), 2 * n_pairs * sizeof(int),
        cudaMemcpyHostToDevice));

    float* d_out;
    cudaErrorCheck(cudaMalloc(&d_out, n_pairs * sizeof(float)));

    const int block_size = 256;
    const int num_blocks = (n_pairs + block_size - 1) / block_size;
    compute_line_line_distances_kernel<USE_EIGEN><<<num_blocks, block_size>>>(
        d_Vx, d_Vy, d_Vz, n_points, d_E0, d_E1, n_edges, d_pairs, n_pairs,
        d_out);
    cudaErrorCheck(cudaPeekAtLastError());
    cudaErrorCheck(cudaDeviceSynchronize());

    Eigen::VectorXf out(n_pairs);
    cudaMemcpy(
        out.data(), d_out, n_pairs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_pairs);
    cudaFree(d_out);

    return out;
}

//==============================================================================
// Baseline CPU implementation
//==============================================================================

Eigen::VectorXf compute_point_point_distances_cpu(
    const Eigen::MatrixXf& V,
    const std::vector<std::array<int, 2>>& point_pairs)
{
    Eigen::VectorXf out(point_pairs.size());
    int pi = 0;
    for (const auto& [vi, vj] : point_pairs) {
        out[pi++] = point_point_distance(V.row(vi), V.row(vj));
    }
    assert(pi == point_pairs.size());
    return out;
}

Eigen::VectorXf compute_line_line_distances_cpu(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& E,
    const std::vector<std::array<int, 2>>& line_pairs)
{
    Eigen::VectorXf out(line_pairs.size());
    int li = 0;
    for (const auto& [ei, ej] : line_pairs) {
        out[li++] = line_line_distance(
            V.row(E(ei, 0)), V.row(E(ei, 1)), V.row(E(ej, 0)), V.row(E(ej, 1)));
    }
    assert(li == line_pairs.size());
    return out;
}

//==============================================================================
// Template instantiations
//==============================================================================

template Eigen::VectorXf compute_point_point_distances_gpu<false>(
    const Eigen::MatrixXf&, const std::vector<std::array<int, 2>>&);
template Eigen::VectorXf compute_point_point_distances_gpu<true>(
    const Eigen::MatrixXf&, const std::vector<std::array<int, 2>>&);

template Eigen::VectorXf compute_line_line_distances_gpu<false>(
    const Eigen::MatrixXf&,
    const Eigen::MatrixXi&,
    const std::vector<std::array<int, 2>>&);
template Eigen::VectorXf compute_line_line_distances_gpu<true>(
    const Eigen::MatrixXf&,
    const Eigen::MatrixXi&,
    const std::vector<std::array<int, 2>>&);

} // namespace ece