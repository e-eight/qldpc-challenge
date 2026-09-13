#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ris_guided {

// Dense rows are tightly packed; only the start of the allocation is aligned.
class Matrix {
public:
    Matrix(int rows, int columns);
    Matrix(const Matrix& other);
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    int rows() const { return rows_; }
    int columns() const { return columns_; }
    int words() const { return words_; }
    std::size_t bytes() const;
    uint64_t* row(int i) { return data_.get() + std::size_t(i) * words_; }
    const uint64_t* row(int i) const { return data_.get() + std::size_t(i) * words_; }
    bool bit(int r, int c) const;
    void set(int r, int c);
private:
    struct Delete { void operator()(uint64_t* p) const; };
    int rows_, columns_, words_;
    std::unique_ptr<uint64_t[], Delete> data_;
};

struct Prepared {
    Matrix kernel;
    Matrix logicals;
    // X search: own = H_X, opposite = H_Z. Swap for Z search.
    Prepared(const Matrix& own, const Matrix& opposite);
    bool applicable() const { return logicals.rows() != 0; }
};

struct Improvement {
    int worker;
    int weight;
    uint64_t trial;  // Cumulative trials for this worker, independent of batches.
    std::vector<int> support;
};

struct Batch {
    uint64_t trials = 0;  // Scored bases in this batch; correlated in incremental mode.
    uint64_t reductions = 0, proposals = 0, exchanges = 0;  // Cumulative session counters.
    uint64_t accepted_children = 0, immigrants = 0;
    int best_weight = 0;  // n + 1 means no logical has been found.
    std::vector<Improvement> improvements;
};

// Owns independent RNG streams, scratch, and persistent native workers.
// No allocations occur inside a trial. Calls must not overlap on one session.
class Session {
public:
    Session(std::shared_ptr<const Prepared> prepared, int threads, uint64_t seed,
            int pair_depth = 8, bool masked = false, int block_size = 1,
            uint64_t restart_interval = 0, int exchange_proposals = 8);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Batch advance(uint64_t trials);
    std::size_t workspace_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ris
