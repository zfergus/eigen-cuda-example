#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <example.h>

TEST_CASE("Benchmark Eigen CUDA", "[!benchmark]")
{
    Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(3, 10000000);
    Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(3, 10000000);

    BENCHMARK("CPU") { return compute_distances_cpu(m1, m2); };

    BENCHMARK("GPU") { return compute_distances_gpu(m1, m2); };
}