#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <example.h>

#include <iostream>

TEST_CASE("Test Point-Point Distances", "[point-point]")
{
    const int N = GENERATE(10, 100'000);
    const int iters = 100;

    for (int i = 0; i < iters; ++i) {
        const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N, 3);

        std::vector<std::array<int, 2>> point_pairs(N);
        for (auto& pair : point_pairs) {
            for (auto& i : pair) {
                i = std::rand() % N;
            }
        }
        const Eigen::VectorXf r_cpu =
            ece::compute_point_point_distances_cpu(V, point_pairs);
        const Eigen::VectorXf r_gpu =
            ece::compute_point_point_distances_gpu</*USE_EIGEN=*/true>(
                V, point_pairs);
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_point_point_distances_gpu</*USE_EIGEN=*/false>(
                V, point_pairs);

        if (N <= 10
            && (!r_cpu.isApprox(r_gpu) || !r_cpu.isApprox(r_gpu_no_eigen))) {
            std::cout << "r_cpu         : " << r_cpu.transpose() << std::endl;
            std::cout << "r_gpu         : " << r_gpu.transpose() << std::endl;
            std::cout << "r_gpu_no_eigen: " << r_gpu_no_eigen.transpose()
                      << std::endl;
        }

        CHECK(r_cpu.isApprox(r_gpu));
        // CHECK(r_cpu.isApprox(r_gpu_no_eigen));
    }
}

TEST_CASE("Test Line-Line Distances", "[line-line]")
{
    const int N = GENERATE(10, 100'000);
    const int iters = 100;

    for (int i = 0; i < iters; ++i) {
        const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N, 3);
        const Eigen::MatrixXi E = random_edges(N, N);
        REQUIRE(E.minCoeff() >= 0);
        REQUIRE(E.maxCoeff() < N);

        std::vector<std::array<int, 2>> line_pairs(N);
        for (auto& pair : line_pairs) {
            pair[0] = std::rand() % N;
            while ((pair[1] = std::rand() % N) == pair[0]) { }
        }

        const Eigen::VectorXf r_cpu =
            ece::compute_line_line_distances_cpu(V, E, line_pairs);
        const Eigen::VectorXf r_gpu =
            ece::compute_line_line_distances_gpu</*USE_EIGEN=*/true>(
                V, E, line_pairs);
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_line_line_distances_gpu</*USE_EIGEN=*/false>(
                V, E, line_pairs);

        if (N <= 10
            && (!r_cpu.isApprox(r_gpu) || !r_cpu.isApprox(r_gpu_no_eigen))) {
            std::cout << "r_cpu         : " << r_cpu.transpose() << std::endl;
            std::cout << "r_gpu         : " << r_gpu.transpose() << std::endl;
            std::cout << "r_gpu_no_eigen: " << r_gpu_no_eigen.transpose()
                      << std::endl;
        }

        CHECK(r_cpu.isApprox(r_gpu));
        CHECK(r_cpu.isApprox(r_gpu_no_eigen));
    }
}