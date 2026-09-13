#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
constexpr size_t MAX_WORDS = 64;
constexpr size_t WIDTH = 32;
constexpr size_t LIGHT = 64;
struct Candidate {
    std::array<uint64_t, MAX_WORDS> bits{};
    int weight = 0, last = -1;
    bool logical = false;
};
class Session {
    size_t n, nw, tw, words, rows;
    std::vector<uint64_t> original, matrix;
    std::vector<size_t> columns, order;
    std::mt19937_64 rng;
    std::array<Candidate, WIDTH * 2> beam{}, next{};
    size_t beam_size = 0;
    int best;
    uint64_t trials = 0, scored = 0;
    std::vector<std::pair<int, std::vector<int>>> events;
    bool tagged(const uint64_t *v) const {
        uint64_t tag = 0;
        for (size_t j = nw; j < words; ++j) tag |= v[j];
        return tag != 0;
    }
    int weight(const uint64_t *v) const {
        int w = 0;
        for (size_t j = 0; j < nw; ++j) w += __builtin_popcountll(v[j]);
        return w;
    }
    void observe(const uint64_t *v, int w, bool logical) {
        ++scored;
        if (!logical || w >= best) return;
        best = w;
        std::vector<int> support;
        support.reserve(w);
        for (size_t j = 0; j < n; ++j)
            if ((v[j / 64] >> (j % 64)) & 1) support.push_back(static_cast<int>(j));
        events.emplace_back(w, std::move(support));
    }
    // Separate logical and trivial beams prevent short stabilizers swallowing all slots.
    void retain(const Candidate &c, size_t &logical_count, size_t &trivial_count) {
        size_t &count = c.logical ? logical_count : trivial_count;
        const size_t offset = c.logical ? 0 : WIDTH;
        if (!c.weight) return;
        size_t pos = count;
        if (count == WIDTH) {
            pos = WIDTH - 1;
            if (next[offset + pos].weight <= c.weight) return;
        } else ++count;
        while (pos && next[offset + pos - 1].weight > c.weight) {
            next[offset + pos] = next[offset + pos - 1];
            --pos;
        }
        next[offset + pos] = c;
    }
public:
    Session(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> packed,
            size_t n_bits, size_t tag_bits, uint64_t seed)
        : n(n_bits), nw((n + 63) / 64), tw((tag_bits + 63) / 64),
          words(nw + tw), rng(seed), best(static_cast<int>(n + 1)) {
        auto a = packed.request();
        if (a.ndim != 2 || static_cast<size_t>(a.shape[1]) != words || !n || !tw || words > MAX_WORDS)
            throw std::invalid_argument("bad packed dimensions or n+tag word count >64");
        rows = static_cast<size_t>(a.shape[0]);
        auto data = static_cast<uint64_t *>(a.ptr);
        original.assign(data, data + rows * words);
        matrix.resize(original.size());
        columns.resize(n);
        std::iota(columns.begin(), columns.end(), 0);
        order.resize(rows);
        events.reserve(32);
    }
    py::dict advance() {
        events.clear();
        matrix = original;
        std::shuffle(columns.begin(), columns.end(), rng);
        size_t rank = 0;
        for (size_t col : columns) {
            size_t pivot = rank;
            while (pivot < rows && !((matrix[pivot * words + col / 64] >> (col % 64)) & 1)) ++pivot;
            if (pivot == rows) continue;
            for (size_t j = 0; j < words; ++j) std::swap(matrix[rank * words + j], matrix[pivot * words + j]);
            const auto *p = &matrix[rank * words];
            for (size_t r = 0; r < rows; ++r) {
                if (r == rank || !((matrix[r * words + col / 64] >> (col % 64)) & 1)) continue;
                auto *v = &matrix[r * words];
                for (size_t j = 0; j < words; ++j) v[j] ^= p[j];
            }
            if (++rank == rows) break;
        }
        std::iota(order.begin(), order.end(), 0);
        for (size_t r = 0; r < rank; ++r) {
            const auto *v = &matrix[r * words];
            observe(v, weight(v), tagged(v));
        }
        std::sort(order.begin(), order.begin() + rank, [&](size_t a, size_t b) {
            return weight(&matrix[a * words]) < weight(&matrix[b * words]);
        });
        // Up to48 logical and16 trivial light rows; unused quotas filled from remaining rows.
        std::array<size_t, LIGHT> selected{};
        size_t used = 0, lc = 0, tc = 0;
        for (size_t i = 0; i < rank && used < LIGHT; ++i) {
            bool logical = tagged(&matrix[order[i] * words]);
            if ((logical && lc < 48) || (!logical && tc < 16)) {
                selected[used++] = order[i];
                logical ? ++lc : ++tc;
            }
        }
        for (size_t i = 0; i < rank && used < LIGHT; ++i)
            if (std::find(selected.begin(), selected.begin() + used, order[i]) == selected.begin() + used)
                selected[used++] = order[i];
        beam_size = 0;
        lc = tc = 0;
        Candidate c;
        for (size_t i = 0; i < used; ++i) {
            const auto *v = &matrix[selected[i] * words];
            std::copy(v, v + words, c.bits.begin());
            c.weight = weight(v); c.logical = tagged(v); c.last = static_cast<int>(i);
            retain(c, lc, tc);
        }
        for (size_t i = 0; i < lc; ++i) beam[beam_size++] = next[i];
        for (size_t i = 0; i < tc; ++i) beam[beam_size++] = next[WIDTH + i];
        for (int depth = 2; depth <= 4; ++depth) {
            lc = tc = 0;
            for (size_t bi = 0; bi < beam_size; ++bi) {
                const auto &b = beam[bi];
                for (size_t j = b.last + 1; j < used; ++j) {
                    const auto *v = &matrix[selected[j] * words];
                    for (size_t w = 0; w < words; ++w) c.bits[w] = b.bits[w] ^ v[w];
                    c.weight = weight(c.bits.data()); c.logical = tagged(c.bits.data());
                    c.last = static_cast<int>(j);
                    observe(c.bits.data(), c.weight, c.logical);
                    retain(c, lc, tc);
                }
            }
            beam_size = 0;
            for (size_t i = 0; i < lc; ++i) beam[beam_size++] = next[i];
            for (size_t i = 0; i < tc; ++i) beam[beam_size++] = next[WIDTH + i];
        }
        ++trials;
        py::dict result;
        result["events"] = events;
        result["trials"] = trials;
        result["scored"] = scored;
        result["rank"] = rank;
        result["workspace_bytes"] = (original.capacity() + matrix.capacity()) * 8
            + (columns.capacity() + order.capacity()) * sizeof(size_t) + sizeof(beam) + sizeof(next);
        return result;
    }
};
PYBIND11_MODULE(reduced_space_native, m) {
    py::class_<Session>(m, "Session")
        .def(py::init<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>, size_t, size_t, uint64_t>())
        .def("advance", &Session::advance);
}
