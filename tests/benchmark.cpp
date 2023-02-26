#include "utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <example.h>

TEST_CASE("Benchmark Point-Point Distances", "[!benchmark]")
{
    constexpr int N_POINTS = 1'000'000;
    constexpr int N_PAIRS = 1'000'000;
    const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N_POINTS, 3);

    std::vector<std::array<int, 2>> point_pairs(N_PAIRS);
    for (auto& pair : point_pairs) {
        for (auto& i : pair) {
            i = std::rand() % N_POINTS;
        }
    }

    BENCHMARK("CPU")
    {
        return ece::compute_point_point_distances_cpu(V, point_pairs);
    };
    BENCHMARK("GPU")
    {
        return ece::compute_point_point_distances_gpu</*USE_EIGEN=*/true>(
            V, point_pairs);
    };
    BENCHMARK("GPU_NO_EIGEN")
    {
        return ece::compute_point_point_distances_gpu</*USE_EIGEN=*/false>(
            V, point_pairs);
    };
}

TEST_CASE("Benchmark Line-Line Distances", "[!benchmark]")
{
    constexpr int N_POINTS = 1'000'000;
    constexpr int N_EDGES = 1'000'000;
    constexpr int N_PAIRS = 1'000'000;

    const Eigen::MatrixXf V = Eigen::MatrixXf::Random(N_POINTS, 3);
    const Eigen::MatrixXi E = random_edges(N_POINTS, N_EDGES);
    REQUIRE(E.minCoeff() >= 0);
    REQUIRE(E.maxCoeff() < N_POINTS);

    std::vector<std::array<int, 2>> line_pairs(N_PAIRS);
    for (auto& pair : line_pairs) {
        for (auto& i : pair) {
            i = std::rand() % N_EDGES;
        }
    }

    BENCHMARK("CPU")
    {
        return ece::compute_line_line_distances_cpu(V, E, line_pairs);
    };
    BENCHMARK("GPU")
    {
        return ece::compute_line_line_distances_gpu</*USE_EIGEN=*/true>(
            V, E, line_pairs);
    };
    BENCHMARK("GPU_NO_EIGEN")
    {
        return ece::compute_line_line_distances_gpu</*USE_EIGEN=*/false>(
            V, E, line_pairs);
    };
}