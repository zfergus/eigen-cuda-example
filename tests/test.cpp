#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <example.h>

#include <iostream>

TEST_CASE("Test Point-Point Distances", "[point-point]")
{
    const int N = GENERATE(10, 100000);
    const int iters = 100;

    for (int i = 0; i < iters; ++i) {
        Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(3, N);
        Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(3, N);

        const Eigen::VectorXf r_cpu =
            ece::compute_point_point_distances_cpu(m1, m2);
        const Eigen::VectorXf r_gpu =
            ece::compute_point_point_distances_gpu(m1, m2);
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_point_point_distances_gpu_no_eigen(m1, m2);

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

TEST_CASE("Test Line-Line Distances", "[line-line]")
{
    const int N = GENERATE(10, 100000);
    const int iters = 100;

    for (int i = 0; i < iters; ++i) {
        Eigen::MatrixXf ea0s = Eigen::MatrixXf::Random(3, N);
        Eigen::MatrixXf ea1s = Eigen::MatrixXf::Random(3, N);
        Eigen::MatrixXf eb0s = Eigen::MatrixXf::Random(3, N);
        Eigen::MatrixXf eb1s = Eigen::MatrixXf::Random(3, N);

        const Eigen::VectorXf r_cpu =
            ece::compute_line_line_distances_cpu(ea0s, ea1s, eb0s, eb1s);
        const Eigen::VectorXf r_gpu =
            ece::compute_line_line_distances_gpu(ea0s, ea1s, eb0s, eb1s);
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_line_line_distances_gpu_no_eigen(
                ea0s, ea1s, eb0s, eb1s);

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