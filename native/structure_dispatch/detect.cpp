#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

// Candidate partition inference, not a symmetry or distance certificate.
// All scratch storage is allocated before the matching loops.
py::dict detect(py::array_t<uint8_t, py::array::c_style> input,
                double seconds, double minimum_confidence) {
    if (!std::isfinite(seconds) || seconds < 0 || !std::isfinite(minimum_confidence)
        || minimum_confidence < 0 || minimum_confidence > 1)
        throw std::invalid_argument("Invalid detector budget or confidence");
    if (input.ndim() != 2) throw std::invalid_argument("Expected a binary matrix");
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    const int64_t m64 = input.shape(0), n64 = input.shape(1);
    std::string status = "no_route", matching = "none";
    uint64_t updates = 0, relaxations = 0, workspace = 0;
    double confidence = 0, upper_confidence = 0;
    std::vector<int> order;
    auto finish = [&]() {
        py::dict result;
        result["status"] = status; result["order"] = order;
        result["confidence"] = confidence; result["upper_confidence"] = upper_confidence;
        result["elapsed_seconds"] = std::chrono::duration<double>(Clock::now()-start).count();
        result["workspace_bytes"] = workspace; result["score_updates"] = updates;
        result["relaxations"] = relaxations; result["matching"] = matching;
        return result;
    };
    auto stop = [&]() {
        if (Clock::now() >= deadline) { status = "deadline"; return true; }
        if (updates > 4000000 || relaxations > 64000000) { status = "work_cap"; return true; }
        return false;
    };
    if (stop()) return finish();
    if (n64 < 4 || n64 % 2 || n64 > 4096 || m64 < 2 || m64 > 4096) {
        status = "shape_cap"; return finish();
    }
    const int n = static_cast<int>(n64), m = static_cast<int>(m64);
    std::vector<int> offsets(m+1, 0), indices;
    indices.reserve(static_cast<size_t>(m)*std::min(n, 64));
    workspace = offsets.capacity()*sizeof(int) + indices.capacity()*sizeof(int);
    auto a = input.unchecked<2>();
    for (int row=0; row<m; ++row) {
        if (stop()) return finish();
        int degree = 0;
        for (int q=0; q<n; ++q) {
            if (a(row,q) > 1) throw std::invalid_argument("Matrix must be binary");
            if (a(row,q)) {
                if (++degree > 64) { status = "degree_cap"; return finish(); }
                indices.push_back(q);
            }
        }
        offsets[row+1] = static_cast<int>(indices.size());
    }
    const int edges = offsets[m-1];
    if (!edges) { status = "empty"; return finish(); }
    std::vector<uint16_t> scores(static_cast<size_t>(n)*n, 0);
    workspace += scores.capacity()*sizeof(uint16_t);
    for (int row=0; row<m-1; ++row) {
        if (stop()) return finish();
        for (int i=offsets[row]; i<offsets[row+1]; ++i) {
            auto *score = scores.data() + static_cast<size_t>(indices[i])*n;
            for (int j=offsets[row+1]; j<offsets[row+2]; ++j) {
                ++score[indices[j]]; ++updates;
            }
        }
    }
    std::vector<int> permutation(n), targets(n, 0);
    workspace += (permutation.capacity()+targets.capacity())*sizeof(int);
    bool bijective = true;
    uint64_t maximum = 0;
    for (int q=0; q<n; ++q) {
        if (stop()) return finish();
        const auto *row = scores.data() + static_cast<size_t>(q)*n;
        int target = static_cast<int>(std::max_element(row, row+n)-row);
        permutation[q] = target; maximum += row[target];
        if (++targets[target] > 1) bijective = false;
    }
    upper_confidence = static_cast<double>(maximum)/edges;
    if (upper_confidence < minimum_confidence) { status = "low_confidence"; return finish(); }
    if (bijective) {
        matching = "row_maxima";
    } else {
        matching = "assignment";
        // Shortest augmenting paths for maximum overlap; negative scores are costs.
        std::vector<int> u(n+1, 0), v(n+1, 0), p(n+1, 0), way(n+1, 0), distance(n+1);
        std::vector<uint8_t> used(n+1);
        workspace += (u.capacity()+v.capacity()+p.capacity()+way.capacity()+distance.capacity())*sizeof(int)
            + used.capacity();
        for (int i=1; i<=n; ++i) {
            p[0] = i;
            std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
            std::fill(used.begin(), used.end(), 0);
            int j0 = 0;
            do {
                if (stop()) return finish();
                used[j0] = 1;
                const int i0 = p[j0];
                int delta = std::numeric_limits<int>::max(), j1 = 0;
                const auto *row = scores.data()+static_cast<size_t>(i0-1)*n;
                for (int j=1; j<=n; ++j) {
                    if (!used[j]) {
                        int cost = -static_cast<int>(row[j-1])-u[i0]-v[j];
                        if (cost < distance[j]) { distance[j] = cost; way[j] = j0; }
                        if (distance[j] < delta) { delta = distance[j]; j1 = j; }
                    }
                }
                relaxations += n;
                for (int j=0; j<=n; ++j) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else if (j) distance[j] -= delta;
                }
                j0 = j1;
            } while (p[j0] != 0);
            do { int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0);
        }
        for (int j=1; j<=n; ++j) permutation[p[j]-1]=j-1;
    }
    if (stop()) return finish();
    uint64_t assigned = 0;
    for (int q=0; q<n; ++q) assigned += scores[static_cast<size_t>(q)*n+permutation[q]];
    confidence = static_cast<double>(assigned)/edges;
    if (confidence < minimum_confidence) { status = "low_confidence"; return finish(); }
    std::fill(targets.begin(), targets.end(), 0);
    order.reserve(n);
    workspace += order.capacity()*sizeof(int);
    int cycles = 0;
    for (int first=0; first<n; ++first) {
        if (targets[first]) continue;
        int q=first, length=0;
        do { targets[q]=1; order.push_back(q); ++length; q=permutation[q]; } while (!targets[q]);
        if (++cycles > 2 || length != n/2) {
            order.clear(); status="not_two_cycles"; return finish();
        }
    }
    if (stop()) { order.clear(); return finish(); }
    status = "accepted";
    return finish();
}

PYBIND11_MODULE(structure_dispatch_native, m) {
    m.def("detect", &detect, py::arg("matrix").noconvert(), py::arg("seconds"),
          py::arg("minimum_confidence")=0.5);
}
