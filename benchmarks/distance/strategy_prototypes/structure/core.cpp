#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace py = pybind11;
using U64 = uint64_t;

namespace structure_candidate {
class Search {
    int n, m, k, sw, vw;
    std::vector<U64> columns, logicals, basis_s, basis_v, syndrome, relation;
    std::vector<std::vector<int>> neighbors;
    std::vector<uint8_t> occupied, discovered;
    std::vector<int> frontier;
public:
    Search(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> h,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> l) {
        if (h.ndim() != 2 || l.ndim() != 2 || h.shape(1) != l.shape(1) || h.shape(1) <= 0)
            throw std::invalid_argument("expected compatible two-dimensional matrices");
        m = h.shape(0); n = h.shape(1); k = l.shape(0);
        sw = (m + 63) / 64; vw = (n + 63) / 64;
        columns.resize(size_t(n) * sw); logicals.resize(size_t(k) * vw);
        auto hh = h.unchecked<2>(); auto ll = l.unchecked<2>();
        std::vector<std::vector<int>> checks(m), incidence(n);
        for (int r = 0; r < m; ++r) for (int c = 0; c < n; ++c) {
            if (hh(r, c) > 1) throw std::invalid_argument("non-binary check");
            if (hh(r, c)) {
                columns[size_t(c)*sw+r/64] |= U64(1) << (r%64);
                checks[r].push_back(c); incidence[c].push_back(r);
            }
        }
        for (int r = 0; r < k; ++r) for (int c = 0; c < n; ++c) {
            if (ll(r, c) > 1) throw std::invalid_argument("non-binary logical");
            if (ll(r, c)) logicals[size_t(r)*vw+c/64] |= U64(1) << (c%64);
        }
        neighbors.resize(n);
        std::vector<uint8_t> seen(n);
        for (int c = 0; c < n; ++c) {
            std::fill(seen.begin(), seen.end(), 0); seen[c] = 1;
            for (int r : incidence[c]) for (int q : checks[r]) if (!seen[q]) {
                seen[q] = 1; neighbors[c].push_back(q);
            }
        }
        basis_s.resize(size_t(m) * sw); basis_v.resize(size_t(m) * vw);
        syndrome.resize(sw); relation.resize(vw); occupied.resize(m); discovered.resize(n);
        frontier.reserve(n);
    }

    py::dict run(double seconds, U64 seed, py::function emit) {
        if (!std::isfinite(seconds) || seconds < 0) throw std::invalid_argument("invalid seconds");
        using Clock = std::chrono::steady_clock;
        auto started = Clock::now(); auto deadline = started + std::chrono::duration<double>(seconds);
        U64 attempts = 0, inserted = 0, dependencies = 0, nontrivial = 0, components = 0;
        int best = n + 1;
        std::mt19937_64 rng(seed);
        {
            py::gil_scoped_release release;
            while (k && Clock::now() < deadline && best > 1) {
                int cap = std::max(1, int((size_t(n) * (1 + attempts % 4)) / 4));
                ++attempts;
                std::fill(occupied.begin(), occupied.end(), 0);
                std::fill(discovered.begin(), discovered.end(), 0);
                frontier.clear();
                for (int added = 0; added < cap; ++added) {
                    // Bound cancellation latency to one insertion, including elimination.
                    if (Clock::now() >= deadline) break;
                    if (frontier.empty()) {
                        int start = rng() % n;
                        while (discovered[start]) start = (start + 1) % n;
                        frontier.push_back(start); discovered[start] = 1; ++components;
                    }
                    size_t pick = rng() % frontier.size();
                    int q = frontier[pick]; frontier[pick] = frontier.back(); frontier.pop_back();
                    for (int a : neighbors[q]) if (!discovered[a]) {
                        discovered[a] = 1; frontier.push_back(a);
                    }
                    ++inserted;
                    for (int w = 0; w < sw; ++w) syndrome[w] = columns[size_t(q)*sw+w];
                    std::fill(relation.begin(), relation.end(), 0);
                    relation[q/64] = U64(1) << (q%64);
                    bool independent = false;
                    for (int w = 0; w < sw && !independent; ++w) while (syndrome[w]) {
                        int pivot = 64*w + __builtin_ctzll(syndrome[w]);
                        if (!occupied[pivot]) {
                            occupied[pivot] = 1;
                            std::copy(syndrome.begin(), syndrome.end(), basis_s.begin()+size_t(pivot)*sw);
                            std::copy(relation.begin(), relation.end(), basis_v.begin()+size_t(pivot)*vw);
                            independent = true; break;
                        }
                        const U64* bs = basis_s.data()+size_t(pivot)*sw;
                        const U64* bv = basis_v.data()+size_t(pivot)*vw;
                        for (int j = w; j < sw; ++j) syndrome[j] ^= bs[j];
                        for (int j = 0; j < vw; ++j) relation[j] ^= bv[j];
                    }
                    // A newly stored pivot terminates elimination of this column.
                    if (independent) continue;
                    ++dependencies;
                    int weight = 0;
                    for (U64 word : relation) weight += __builtin_popcountll(word);
                    if (weight >= best) continue;
                    bool logical = false;
                    for (int r = 0; r < k && !logical; ++r) {
                        U64 parity = 0;
                        for (int w = 0; w < vw; ++w) parity ^= relation[w] & logicals[size_t(r)*vw+w];
                        logical = __builtin_parityll(parity);
                    }
                    if (!logical) continue;
                    ++nontrivial; best = weight;
                    std::vector<int> support; support.reserve(weight);
                    for (int w = 0; w < vw; ++w) {
                        U64 bits = relation[w];
                        while (bits) { support.push_back(64*w+__builtin_ctzll(bits)); bits &= bits-1; }
                    }
                    py::gil_scoped_acquire acquire;
                    emit(weight, support);
                }
            }
        }
        py::dict out;
        out["attempts"] = attempts; out["columns"] = inserted;
        out["dependencies"] = dependencies; out["improvements"] = nontrivial;
        out["components_started"] = components;
        out["best_weight"] = best <= n ? py::cast(best) : py::none();
        out["elapsed_seconds"] = std::chrono::duration<double>(Clock::now()-started).count();
        out["region_cap_cycle"] = "n/4,n/2,3n/4,n";
        return out;
    }
};
} // namespace structure_candidate

PYBIND11_MODULE(_strategy_structure, m) {
    py::class_<structure_candidate::Search>(m, "Search")
        .def(py::init<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>())
        .def("run", &structure_candidate::Search::run, py::arg("seconds"), py::arg("seed"), py::arg("emit"));
}
