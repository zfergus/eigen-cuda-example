#include "utils.hpp"

#include <unordered_set>

struct HashEdge {
    size_t operator()(const std::array<int, 2>& e) const noexcept
    {
        size_t h = (size_t(e[0]) << 32) + size_t(e[0]);
        h *= 1231231557ull; // "random" uneven integer
        h ^= (h >> 32);
        return h;
    }
};

Eigen::MatrixXi random_edges(const size_t n_points, const size_t n_edges)
{
    Eigen::MatrixXi E(n_edges, 2);
    std::unordered_set<std::array<int, 2>, HashEdge> unordered_edges;

    for (int i = 0; i < n_edges; ++i) {
        bool is_edge_unique;
        do {
            E(i, 0) = std::rand() % n_points;
            while ((E(i, 1) = std::rand() % n_points) == E(i, 0)) { }

            is_edge_unique = unordered_edges.find({ { E(i, 0), E(i, 1) } })
                == unordered_edges.end();
        } while (!is_edge_unique);
        unordered_edges.insert(std::array<int, 2> { { E(i, 0), E(i, 1) } });
        unordered_edges.insert(std::array<int, 2> { { E(i, 1), E(i, 0) } });
    }

    return E;
}