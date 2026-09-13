#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

// Every move adds a supplied stabilizer. Byte state and sparse incidence lists
// let a flip update all affected row gains without scanning packed dense rows.
class Descent {
    int n_, m_, words_;
    std::vector<int> offsets_, qubits_, column_offsets_, column_rows_;
    std::vector<uint8_t> state_, best_;
    std::vector<int> delta_, last_used_;
    std::vector<uint64_t> history_;
    std::vector<int> history_weights_;
    uint64_t rng_ = 0;

    uint64_t random() {
        uint64_t z = (rng_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    void rebuild() {
        for (int row = 0; row < m_; ++row) {
            int value = offsets_[row + 1] - offsets_[row];
            for (int j = offsets_[row]; j < offsets_[row + 1]; ++j)
                value -= 2 * state_[qubits_[j]];
            delta_[row] = value;
        }
    }
    int flip(int row) {
        const int change = delta_[row];
        for (int j = offsets_[row]; j < offsets_[row + 1]; ++j) {
            const int q = qubits_[j];
            const int adjustment = state_[q] ? 2 : -2;
            state_[q] ^= 1;
            for (int k = column_offsets_[q]; k < column_offsets_[q + 1]; ++k)
                delta_[column_rows_[k]] += adjustment;
        }
        return change;
    }
    void check(int weight) const {
        if (weight != std::count(state_.begin(), state_.end(), uint8_t{1}))
            throw std::runtime_error("incremental weight invariant failed");
        for (int row = 0; row < m_; ++row) {
            int expected = offsets_[row + 1] - offsets_[row];
            for (int j = offsets_[row]; j < offsets_[row + 1]; ++j)
                expected -= 2 * state_[qubits_[j]];
            if (delta_[row] != expected)
                throw std::runtime_error("incremental gain invariant failed");
        }
    }

public:
    explicit Descent(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> own) {
        if (own.ndim() != 2 || own.shape(1) <= 0 ||
            own.shape(0) > std::numeric_limits<int>::max() ||
            own.shape(1) > std::numeric_limits<int>::max())
            throw std::invalid_argument("expected a nonempty-width matrix");
        m_ = own.shape(0); n_ = own.shape(1); words_ = (n_ + 63) / 64;
        auto view = own.unchecked<2>();
        offsets_.push_back(0);
        column_offsets_.resize(n_ + 1, 0);
        for (int row = 0; row < m_; ++row) {
            for (int q = 0; q < n_; ++q) {
                if (view(row, q) > 1) throw std::invalid_argument("nonbinary matrix");
                if (view(row, q)) { qubits_.push_back(q); ++column_offsets_[q + 1]; }
            }
            offsets_.push_back(qubits_.size());
        }
        for (int q = 0; q < n_; ++q) column_offsets_[q + 1] += column_offsets_[q];
        column_rows_.resize(qubits_.size());
        auto positions = column_offsets_;
        for (int row = 0; row < m_; ++row)
            for (int j = offsets_[row]; j < offsets_[row + 1]; ++j)
                column_rows_[positions[qubits_[j]]++] = row;
        state_.resize(n_); best_.resize(n_); delta_.resize(m_); last_used_.resize(m_);
        history_.resize(static_cast<size_t>(n_) * words_);
        history_weights_.resize(n_);
    }

    py::dict run(const std::vector<int>& support, double seconds, uint64_t seed,
                 int max_steps, bool debug) {
        if (!std::isfinite(seconds) || seconds < 0 || max_steps < 0)
            throw std::invalid_argument("invalid time or step limit");
        std::fill(state_.begin(), state_.end(), 0);
        for (int q : support) {
            if (q < 0 || q >= n_ || state_[q]) throw std::invalid_argument("invalid support");
            state_[q] = 1;
        }
        best_ = state_;
        int weight = support.size(), best_weight = weight, count = 0;
        int steps = 0, restarts = 0, uphill = 0, neutral = 0, stagnant = 0;
        rng_ = seed;
        std::fill(last_used_.begin(), last_used_.end(), -1000);
        rebuild();
        const auto start = Clock::now();
        const auto deadline = start + std::chrono::duration<double>(seconds);
        {
            py::gil_scoped_release release;
            while (m_ && Clock::now() < deadline && (!max_steps || steps < max_steps)) {
                int chosen = -1, smallest = std::numeric_limits<int>::max();
                uint64_t ties = 0;
                for (int row = 0; row < m_; ++row) {
                    if (offsets_[row] == offsets_[row + 1]) continue;
                    // Tabu tenure prevents immediate undo/cycles; an improvement
                    // of the incumbent is always allowed (aspiration criterion).
                    if (steps - last_used_[row] <= 7 && weight + delta_[row] >= best_weight)
                        continue;
                    if (delta_[row] < smallest) {
                        chosen = row; smallest = delta_[row]; ties = 1;
                    } else if (delta_[row] == smallest && random() % ++ties == 0) {
                        chosen = row;
                    }
                }
                if (chosen < 0 || stagnant >= 64 || weight + smallest > best_weight + 8) {
                    state_ = best_; weight = best_weight; rebuild();
                    std::fill(last_used_.begin(), last_used_.end(), -1000);
                    // Perturb a few rows, then resume greedy tabu descent.
                    const int kicks = 1 + random() % 3;
                    for (int k = 0; k < kicks; ++k) weight += flip(random() % m_);
                    ++restarts; stagnant = 0;
                } else {
                    weight += flip(chosen);
                    uphill += smallest > 0; neutral += smallest == 0;
                    last_used_[chosen] = steps;
                    ++stagnant;
                }
                ++steps;
                if (debug) check(weight);
                if (weight < best_weight) {
                    best_weight = weight; best_ = state_; stagnant = 0;
                    auto* stored = history_.data() + static_cast<size_t>(count) * words_;
                    std::fill(stored, stored + words_, 0);
                    for (int q = 0; q < n_; ++q)
                        if (state_[q]) stored[q / 64] |= uint64_t{1} << (q % 64);
                    history_weights_[count++] = weight;
                }
            }
        }
        py::list improvements;
        for (int i = 0; i < count; ++i) {
            std::vector<int> found;
            const auto* stored = history_.data() + static_cast<size_t>(i) * words_;
            for (int q = 0; q < n_; ++q)
                if ((stored[q / 64] >> (q % 64)) & 1) found.push_back(q);
            improvements.append(py::make_tuple(history_weights_[i], found));
        }
        py::dict result;
        result["improvements"] = improvements;
        result["best_weight"] = best_weight;
        result["steps"] = steps;
        result["restarts"] = restarts;
        result["uphill_moves"] = uphill;
        result["neutral_moves"] = neutral;
        result["native_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }
};

PYBIND11_MODULE(_stabilizer_descent, module) {
    py::class_<Descent>(module, "Descent")
        .def(py::init<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>())
        .def("run", &Descent::run, py::arg("support"), py::arg("seconds"), py::arg("seed"),
             py::arg("max_steps") = 0, py::arg("debug") = false);
}
