#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <example.h>

TEST_CASE("Test Eigen CUDA", "")
{
    for (int i = 0; i < 100; ++i) {
        Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(3, 100000);
        Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(3, 100000);

        const Eigen::VectorXf r_cpu = ece::compute_distances_cpu(m1, m2);
        const Eigen::VectorXf r_gpu = ece::compute_distances_gpu(m1, m2);
        const Eigen::VectorXf r_gpu_no_eigen =
            ece::compute_distances_gpu_no_eigen(m1, m2);

        CHECK(r_cpu.isApprox(r_gpu));
        CHECK(r_cpu.isApprox(r_gpu_no_eigen));
    }
}