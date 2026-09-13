#include "ris/ris.hpp"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace ris_guided {

void Matrix::Delete::operator()(uint64_t* p) const {
    ::operator delete[](p, std::align_val_t(64));
}

Matrix::Matrix(int rows, int columns) : rows_(rows), columns_(columns), words_(0) {
    if (rows < 0 || columns < 1 || columns == std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid matrix dimensions");
    words_ = columns / 64 + (columns % 64 != 0);
    if (std::size_t(rows) > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t) / words_)
        throw std::length_error("matrix is too large");
    data_.reset(static_cast<uint64_t*>(::operator new[](std::max(bytes(), std::size_t(8)), std::align_val_t(64))));
    std::memset(data_.get(), 0, bytes());
}

Matrix::Matrix(const Matrix& other) : Matrix(other.rows_, other.columns_) {
    std::memcpy(data_.get(), other.data_.get(), bytes());
}

std::size_t Matrix::bytes() const { return std::size_t(rows_) * words_ * sizeof(uint64_t); }
bool Matrix::bit(int r, int c) const { return (row(r)[c / 64] >> (c % 64)) & 1; }
void Matrix::set(int r, int c) { row(r)[c / 64] |= uint64_t(1) << (c % 64); }

namespace detail {

inline void xor_row(uint64_t* __restrict dst, const uint64_t* __restrict src, int words) {
    for (int w = 0; w < words; ++w) dst[w] ^= src[w];
}

bool odd_dot(const uint64_t* a, const uint64_t* b, int words) {
    uint64_t acc = 0;
    for (int w = 0; w < words; ++w) acc ^= a[w] & b[w];
    return __builtin_parityll(acc);
}

struct Reduced {
    Matrix matrix;
    std::vector<int> pivots;
};

Reduced reduce(Matrix matrix) {
    std::vector<int> pivots;
    const int words = matrix.words();
    for (int col = 0, r = 0; col < matrix.columns() && r < matrix.rows(); ++col) {
        int pivot = r;
        while (pivot < matrix.rows() && !matrix.bit(pivot, col)) ++pivot;
        if (pivot == matrix.rows()) continue;
        if (pivot != r)
            for (int w = 0; w < words; ++w) std::swap(matrix.row(pivot)[w], matrix.row(r)[w]);
        for (int i = 0; i < matrix.rows(); ++i)
            if (i != r && matrix.bit(i, col)) xor_row(matrix.row(i), matrix.row(r), words);
        pivots.push_back(col);
        ++r;
    }
    return {std::move(matrix), std::move(pivots)};
}

Matrix nullspace(const Matrix& h) {
    auto reduced = reduce(Matrix(h));
    const auto& pivots = reduced.pivots;
    Matrix result(h.columns() - int(pivots.size()), h.columns());
    std::size_t p = 0;
    int row = 0;
    for (int col = 0; col < h.columns(); ++col) {
        if (p < pivots.size() && pivots[p] == col) { ++p; continue; }
        result.set(row, col);
        for (std::size_t i = 0; i < pivots.size(); ++i)
            if (reduced.matrix.bit(int(i), col)) result.set(row, pivots[i]);
        ++row;
    }
    return result;
}

Matrix quotient(const Matrix& own, const Matrix& opposite) {
    const int n = own.columns(), words = own.words();
    // Echelon rows indexed by leading column allow linear independence checks
    // without repeatedly rebuilding a reduced basis during preparation.
    Matrix echelon(n, n);
    std::vector<uint8_t> occupied(n, 0);
    std::vector<uint64_t> scratch(words);
    auto insert = [&](const uint64_t* row) {
        std::copy_n(row, words, scratch.data());
        for (int c = 0; c < n; ++c) {
            if (!((scratch[c / 64] >> (c % 64)) & 1)) continue;
            if (occupied[c]) xor_row(scratch.data(), echelon.row(c), words);
            else {
                occupied[c] = 1;
                std::copy_n(scratch.data(), words, echelon.row(c));
                return true;
            }
        }
        return false;
    };
    for (int i = 0; i < opposite.rows(); ++i) insert(opposite.row(i));
    auto candidates = nullspace(own);
    std::vector<int> selected;
    for (int i = 0; i < candidates.rows(); ++i)
        if (insert(candidates.row(i))) selected.push_back(i);
    Matrix result(int(selected.size()), n);
    for (std::size_t i = 0; i < selected.size(); ++i)
        std::copy_n(candidates.row(selected[i]), words, result.row(int(i)));
    return result;
}

// Same xoshiro256** and unbiased Fisher-Yates stream as the baseline, permitting
// exact same-trial comparisons independently of implementation speed.
struct Random {
    uint64_t state[4];
    explicit Random(uint64_t seed) {
        for (auto& value : state) {
            uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            value = z ^ (z >> 31);
        }
    }
    static uint64_t rotate(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        uint64_t result = rotate(state[1] * 5, 7) * 9;
        uint64_t t = state[1] << 17;
        state[2] ^= state[0]; state[3] ^= state[1];
        state[1] ^= state[2]; state[0] ^= state[3];
        state[2] ^= t; state[3] = rotate(state[3], 45);
        return result;
    }
    uint64_t bounded(uint64_t bound) {
        uint64_t threshold = (-bound) % bound, r;
        do { r = next(); } while (r < threshold);
        return r % bound;
    }
    void shuffle(std::vector<int>& permutation) {
        for (int i = int(permutation.size()) - 1; i > 0; --i) {
            std::swap(permutation[i], permutation[bounded(uint64_t(i + 1))]);
        }
    }
};

struct Worker {
    const Prepared& prepared;
    Matrix work, witnesses, table, elites;
    std::vector<int> elite_pivots, elite_nonpivots;
    int elite_fitness[4], elite_count = 0;
    uint64_t accepted_children = 0, immigrants = 0;
    std::vector<uint64_t*> rows;
    std::vector<int> permutation, weights, indices, event_weights, pivots, nonpivots;
    std::vector<uint8_t> is_pivot;
    std::vector<uint64_t> event_trials;
    Random random;
    int pairs, best, block_size, event_count = 0, delivered = 0;
    bool masked;
    uint64_t restart_interval, reductions = 0, proposals = 0, exchanges = 0;
    int exchange_proposals;
    uint64_t completed = 0, requested = 0;

    Worker(const Prepared& p, uint64_t seed, int pair_depth, bool use_mask, int block = 1, uint64_t restart = 0, int proposals_per_sample = 8)
        : prepared(p), work(p.kernel.rows(), p.kernel.columns()),
          witnesses(p.kernel.columns(), p.kernel.columns()), table(1 << block, p.kernel.columns()),
          elites(4 * p.kernel.rows(), p.kernel.columns()),
          elite_pivots(4 * p.kernel.rows()), elite_nonpivots(4 * (p.kernel.columns() - p.kernel.rows())),
          rows(p.kernel.rows()),
          permutation(p.kernel.columns()), weights(p.kernel.rows()), indices(p.kernel.rows()),
          event_weights(p.kernel.columns()), pivots(p.kernel.rows()),
          nonpivots(p.kernel.columns() - p.kernel.rows()), is_pivot(p.kernel.columns()), event_trials(p.kernel.columns()), random(seed),
          pairs(std::min(pair_depth, p.kernel.rows())), best(p.kernel.columns() + 1), block_size(block), masked(use_mask),
          restart_interval(restart), exchange_proposals(proposals_per_sample) {
        std::iota(permutation.begin(), permutation.end(), 0);
    }

    bool nontrivial(const uint64_t* a, const uint64_t* b = nullptr) const {
        const auto& logicals = prepared.logicals;
        for (int i = 0; i < logicals.rows(); ++i) {
            uint64_t acc = 0;
            const auto* l = logicals.row(i);
            if (b) for (int w = 0; w < work.words(); ++w) acc ^= (a[w] ^ b[w]) & l[w];
            else for (int w = 0; w < work.words(); ++w) acc ^= a[w] & l[w];
            if (__builtin_parityll(acc)) return true;
        }
        return false;
    }

    void retain(int weight, const uint64_t* a, const uint64_t* b = nullptr) {
        assert(weight > 0 && weight < best && event_count < work.columns());
        best = weight;
        event_weights[event_count] = weight;
        event_trials[event_count] = completed + 1;
        auto* destination = witnesses.row(event_count++);
        if (b) for (int w = 0; w < work.words(); ++w) destination[w] = a[w] ^ b[w];
        else std::copy_n(a, work.words(), destination);
    }

    static unsigned panel_index(const uint64_t* row, const int* columns, int size) {
        // An explicit fallthrough avoids vectorizing a tiny, irregular gather
        // into shifts, inserts, and horizontal reductions. Keep row XORs SIMD.
        auto bit = [&](int j) { return unsigned((row[columns[j] / 64] >> (columns[j] % 64)) & 1) << j; };
        unsigned result = 0;
        switch (size) {
            case 8: result |= bit(7); [[fallthrough]];
            case 7: result |= bit(6); [[fallthrough]];
            case 6: result |= bit(5); [[fallthrough]];
            case 5: result |= bit(4); [[fallthrough]];
            case 4: result |= bit(3); [[fallthrough]];
            case 3: result |= bit(2); [[fallthrough]];
            case 2: result |= bit(1); [[fallthrough]];
            case 1: result |= bit(0); [[fallthrough]];
            default: return result;
        }
    }

    void reduce_blocked() {
        const int count = work.rows(), words = work.words();
        int r = 0, next_column = 0;
        while (r < count) {
            int columns[8], size = 0;
            while (size < block_size && r + size < count && next_column < work.columns()) {
                int col = permutation[next_column++], word = col / 64;
                uint64_t bit = uint64_t(1) << (col % 64);
                int pivot = r + size;
                unsigned panel_bits = 0;
                for (int j = 0; j < size; ++j)
                    panel_bits |= unsigned((rows[r + j][word] & bit) != 0) << j;
                // Remaining rows have deferred panel elimination. Evaluate the
                // prospective pivot bit without writing each candidate row.
                for (; pivot < count; ++pivot) {
                    bool value = (rows[pivot][word] & bit) != 0;
                    if (panel_bits) value ^= __builtin_parity(panel_index(rows[pivot], columns, size) & panel_bits);
                    if (value) break;
                }
                if (pivot == count) continue;
                std::swap(rows[r + size], rows[pivot]);
                auto* chosen = rows[r + size];
                for (int j = 0; j < size; ++j)
                    if ((chosen[columns[j] / 64] >> (columns[j] % 64)) & 1)
                        xor_row(chosen, rows[r + j], words);
                for (int j = 0; j < size; ++j)
                    if (rows[r + j][word] & bit) xor_row(rows[r + j], chosen, words);
                pivots[r + size] = col;
                columns[size++] = col;
            }
            assert(size > 0);
            // At most 2^block_size rows; a six-pivot table at n=1000 is 8 KiB.
            for (int mask = 1; mask < (1 << size); ++mask) {
                const auto* previous = table.row(mask & (mask - 1));
                const auto* source = rows[r + __builtin_ctz(unsigned(mask))];
                auto* destination = table.row(mask);
                for (int w = 0; w < words; ++w) destination[w] = previous[w] ^ source[w];
            }
            for (int i = 0; i < count; ++i) {
                if (i >= r && i < r + size) continue;
                auto* row = rows[i];
                unsigned mask = panel_index(row, columns, size);
                if (mask) xor_row(row, table.row(int(mask)), words);
            }
            r += size;
        }
    }

    void fresh_basis() {
        const int count = work.rows(), words = work.words();
        random.shuffle(permutation);
        std::memcpy(work.row(0), prepared.kernel.row(0), work.bytes());
        for (int i = 0; i < count; ++i) rows[i] = work.row(i);
        int r = 0;
        if (block_size > 1) reduce_blocked();
        else for (int col : permutation) {
            int word = col / 64;
            uint64_t bit = uint64_t(1) << (col % 64);
            int pivot = r;
            while (pivot < count && !(rows[pivot][word] & bit)) ++pivot;
            if (pivot == count) continue;
            std::swap(rows[r], rows[pivot]);
            pivots[r] = col;
            const auto* src = rows[r];
            for (int i = 0; i < count; ++i) {
                if (i == r) continue;
                auto* dst = rows[i];
                if (masked) {
                    uint64_t mask = uint64_t(0) - uint64_t((dst[word] & bit) != 0);
                    for (int w = 0; w < words; ++w) dst[w] ^= src[w] & mask;
                } else if (dst[word] & bit) xor_row(dst, src, words);
            }
            if (++r == count) break;
        }
        assert(block_size > 1 || r == count);  // Prepared kernel has full rank.
        ++reductions;
        {  // Guided search always needs the nonpivot complement.
            std::fill(is_pivot.begin(), is_pivot.end(), 0);
            for (int col : pivots) is_pivot[col] = 1;
            int next = 0;
            for (int col = 0; col < work.columns(); ++col)
                if (!is_pivot[col]) nonpivots[next++] = col;
        }
    }

    void exchange_basis() {
        // Uniform row/nonpivot proposals, including rejected zero entries,
        // give symmetric transitions. Selecting only eligible rows would bias
        // the walk by the number of ones in the proposed column.
        for (int step = 0; step < exchange_proposals; ++step) {
            int r = int(random.bounded(rows.size()));
            int j = int(random.bounded(nonpivots.size()));
            int col = nonpivots[j], word = col / 64;
            uint64_t bit = uint64_t(1) << (col % 64);
            ++proposals;
            if (!(rows[r][word] & bit)) continue;
            for (int i = 0; i < work.rows(); ++i)
                if (i != r && (rows[i][word] & bit)) xor_row(rows[i], rows[r], work.words());
            std::swap(pivots[r], nonpivots[j]);
            ++exchanges;
        }
    }

    void save_elite(int slot, int fitness) {
        for (int i = 0; i < work.rows(); ++i)
            std::copy_n(rows[i], work.words(), elites.row(slot * work.rows() + i));
        std::copy(pivots.begin(), pivots.end(), elite_pivots.begin() + slot * pivots.size());
        std::copy(nonpivots.begin(), nonpivots.end(), elite_nonpivots.begin() + slot * nonpivots.size());
        elite_fitness[slot] = fitness;
    }

    void load_elite(int slot) {
        for (int i = 0; i < work.rows(); ++i) {
            rows[i] = work.row(i);
            std::copy_n(elites.row(slot * work.rows() + i), work.words(), rows[i]);
        }
        std::copy_n(elite_pivots.begin() + slot * pivots.size(), pivots.size(), pivots.begin());
        std::copy_n(elite_nonpivots.begin() + slot * nonpivots.size(), nonpivots.size(), nonpivots.begin());
    }

    void trial() {
        const int count = work.rows(), words = work.words();
        const bool immigrant = elite_count < 4 || completed % 64 == 0 || nonpivots.empty();
        int slot;
        if (immigrant) {
            fresh_basis();
            ++immigrants;
            if (elite_count < 4) slot = elite_count++;
            else slot = int(std::max_element(elite_fitness, elite_fitness + 4) - elite_fitness);
        } else {
            slot = int(random.bounded(4));
            load_elite(slot);
            exchange_basis();
        }
        int fitness = work.columns() + 1;
        for (int i = 0; i < count; ++i) {
            int weight = 0;
            for (int w = 0; w < words; ++w) weight += __builtin_popcountll(rows[i][w]);
            weights[i] = weight;
            if (weight > 0 && weight < fitness && nontrivial(rows[i])) {
                fitness = weight;
                if (weight < best) retain(weight, rows[i]);
            }
        }
        if (pairs > 1) {
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + pairs, indices.end(),
                              [&](int a, int b) { return weights[a] < weights[b]; });
            for (int i = 0; i < pairs; ++i) for (int j = i + 1; j < pairs; ++j) {
                const auto* a = rows[indices[i]];
                const auto* b = rows[indices[j]];
                int weight = 0;
                for (int w = 0; w < words; ++w) weight += __builtin_popcountll(a[w] ^ b[w]);
                if (weight > 0 && weight < fitness && nontrivial(a, b)) {
                    fitness = weight;
                    if (weight < best) retain(weight, a, b);
                }
            }
        }
        if (immigrant || fitness <= elite_fitness[slot]) {
            save_elite(slot, fitness);
            if (!immigrant) ++accepted_children;
        }
        ++completed;
    }
    void run() { for (uint64_t i = 0; i < requested; ++i) trial(); }
    std::size_t bytes() const {
        return work.bytes() + witnesses.bytes() + table.bytes() + elites.bytes() +
               (elite_pivots.size() + elite_nonpivots.size()) * sizeof(int) + rows.size() * sizeof(uint64_t*) +
               (permutation.size() + weights.size() + indices.size() + event_weights.size() + pivots.size() + nonpivots.size()) * sizeof(int) +
               event_trials.size() * sizeof(uint64_t) + is_pivot.size();
    }
};

}  // namespace detail

using detail::nullspace;
using detail::odd_dot;
using detail::quotient;
using detail::Worker;

Prepared::Prepared(const Matrix& own, const Matrix& opposite)
    : kernel(0, own.columns()), logicals(0, own.columns()) {
    if (own.columns() != opposite.columns()) throw std::invalid_argument("matrix widths differ");
    for (int i = 0; i < own.rows(); ++i) for (int j = 0; j < opposite.rows(); ++j)
        if (odd_dot(own.row(i), opposite.row(j), own.words()))
            throw std::invalid_argument("checks do not commute");
    kernel = nullspace(opposite);
    logicals = quotient(own, opposite);
}

struct Session::Impl {
    std::shared_ptr<const Prepared> prepared;
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::thread> pool;
    std::mutex mutex, calls;
    std::condition_variable start, done;
    uint64_t generation = 0;
    int pending = 0;
    int ready = 0;
    bool stopping = false;

    Impl(std::shared_ptr<const Prepared> p, int threads, uint64_t seed, int pairs, bool masked, int block_size, uint64_t restart_interval, int exchange_proposals)
        : prepared(std::move(p)) {
        if (!prepared || threads < 1 || pairs < 0 || block_size < 1 || block_size > 8 || exchange_proposals < 1)
            throw std::invalid_argument("invalid session options");
        for (int t = 0; t < threads; ++t)
            workers.push_back(std::make_unique<Worker>(*prepared, seed + 0x9e3779b97f4a7c15ULL * uint64_t(t + 1),
                                                      pairs, masked, block_size, restart_interval, exchange_proposals));
        try {
            if (threads > 1) for (int t = 0; t < threads; ++t) pool.emplace_back([this, t] {
                uint64_t seen = 0;
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                done.notify_one();
                for (;;) {
                    start.wait(lock, [&] { return stopping || seen != generation; });
                    if (stopping) return;
                    seen = generation;
                    lock.unlock();
                    workers[t]->run();
                    lock.lock();
                    if (--pending == 0) done.notify_one();
                }
            });
            if (threads > 1) {
                std::unique_lock<std::mutex> lock(mutex);
                done.wait(lock, [&] { return ready == threads; });
            }
        } catch (...) { shutdown(); throw; }
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        start.notify_all();
        for (auto& thread : pool) thread.join();
    }
    ~Impl() { shutdown(); }
};

Session::Session(std::shared_ptr<const Prepared> p, int threads, uint64_t seed, int pairs, bool masked, int block_size, uint64_t restart_interval, int exchange_proposals)
    : impl_(std::make_unique<Impl>(std::move(p), threads, seed, pairs, masked, block_size, restart_interval, exchange_proposals)) {}
Session::~Session() = default;

Batch Session::advance(uint64_t trials) {
    auto& state = *impl_;
    std::unique_lock<std::mutex> call_lock(state.calls, std::try_to_lock);
    if (!call_lock) throw std::logic_error("overlapping calls on a session");
    Batch result;
    result.best_weight = state.prepared->kernel.columns() + 1;
    for (const auto& worker : state.workers) result.best_weight = std::min(result.best_weight, worker->best);
    if (trials == 0 || !state.prepared->applicable()) {
        for (const auto& worker : state.workers) {
            result.reductions += worker->reductions;
            result.proposals += worker->proposals;
            result.exchanges += worker->exchanges;
            result.accepted_children += worker->accepted_children;
            result.immigrants += worker->immigrants;
        }
        return result;
    }
    const auto count = state.workers.size();
    for (std::size_t t = 0; t < count; ++t)
        state.workers[t]->requested = trials / count + (t < trials % count);
    if (count == 1) state.workers[0]->run();
    else {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.pending = int(count);
        ++state.generation;
        state.start.notify_all();
        state.done.wait(lock, [&] { return state.pending == 0; });
    }
    result.trials = trials;
    for (std::size_t t = 0; t < count; ++t) {
        auto& worker = *state.workers[t];
        result.best_weight = std::min(result.best_weight, worker.best);
        result.reductions += worker.reductions;
        result.proposals += worker.proposals;
        result.exchanges += worker.exchanges;
        result.accepted_children += worker.accepted_children;
        result.immigrants += worker.immigrants;
        for (int e = worker.delivered; e < worker.event_count; ++e) {
            Improvement improvement{int(t), worker.event_weights[e], worker.event_trials[e], {}};
            improvement.support.reserve(improvement.weight);
            const auto* row = worker.witnesses.row(e);
            for (int w = 0; w < worker.work.words(); ++w) {
                auto bits = row[w];
                while (bits) {
                    improvement.support.push_back(w * 64 + __builtin_ctzll(bits));
                    bits &= bits - 1;
                }
            }
            result.improvements.push_back(std::move(improvement));
        }
    }
    for (auto& worker : state.workers) worker->delivered = worker->event_count;
    return result;
}

std::size_t Session::workspace_bytes() const {
    std::size_t size = 0;
    for (const auto& worker : impl_->workers) size += worker->bytes();
    return size;
}

}  // namespace ris
