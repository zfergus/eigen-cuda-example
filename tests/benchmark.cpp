#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <example.h>

TEST_CASE("Benchmark Point-Point Distances", "[!benchmark]")
{
    Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(3, 1000000);
    Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(3, 1000000);

    BENCHMARK("CPU") { return ece::compute_point_point_distances_cpu(m1, m2); };
    BENCHMARK("GPU") { return ece::compute_point_point_distances_gpu(m1, m2); };
    BENCHMARK("GPU_NO_EIGEN")
    {
        return ece::compute_point_point_distances_gpu_no_eigen(m1, m2);
    };
}

TEST_CASE("Benchmark Line-Line Distances", "[!benchmark]")
{
    const int N = 1'000'000;
    const Eigen::MatrixXf ea0s = Eigen::MatrixXf::Random(3, N);
    const Eigen::MatrixXf ea1s = Eigen::MatrixXf::Random(3, N);
    const Eigen::MatrixXf eb0s = Eigen::MatrixXf::Random(3, N);
    const Eigen::MatrixXf eb1s = Eigen::MatrixXf::Random(3, N);

    BENCHMARK("CPU")
    {
        return ece::compute_line_line_distances_cpu(ea0s, ea1s, eb0s, eb1s);
    };
    BENCHMARK("GPU")
    {
        return ece::compute_line_line_distances_gpu(ea0s, ea1s, eb0s, eb1s);
    };
    BENCHMARK("GPU_NO_EIGEN")
    {
        return ece::compute_line_line_distances_gpu_no_eigen(
            ea0s, ea1s, eb0s, eb1s);
    };
}