#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <example.h>

TEST_CASE("Benchmark Eigen CUDA", "[!benchmark]")
{
    Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(3, 1000000);
    Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(3, 1000000);

    BENCHMARK("CPU") { return ece::compute_distances_cpu(m1, m2); };
    BENCHMARK("GPU") { return ece::compute_distances_gpu(m1, m2); };
    BENCHMARK("GPU_NO_EIGEN")
    {
        return ece::compute_distances_gpu_no_eigen(m1, m2);
    };
}