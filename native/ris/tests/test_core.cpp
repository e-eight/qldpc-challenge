// Include the implementation to audit its private trial kernel without exposing
// test hooks in the public library interface.
#include "../src/ris.cpp"
#include <atomic>
#include <cstdlib>
#include <iostream>

namespace {
thread_local bool forbid_allocations = false;
void check(bool condition) {
    if (!condition) std::abort();
}
}

void* operator new(std::size_t size) {
    check(!forbid_allocations);
    if (auto* p = std::malloc(std::max(size, std::size_t(1)))) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t size, std::align_val_t align) {
    check(!forbid_allocations);
    void* p = nullptr;
    if (posix_memalign(&p, std::size_t(align), std::max(size, std::size_t(1))) != 0) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t size, std::align_val_t align) { return ::operator new(size, align); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

int main() {
    // Deterministic random full-rank checks with redundant and zero rows,
    // including columns that can never participate in an exchange.
    for (int n : {3, 63, 64, 65, 127, 128, 129}) {
        ris::Matrix own(0, n), opposite(n / 2 + 2, n);
        ris::detail::Random random(84);
        for (int r = 0; r < n / 2; ++r) {
            opposite.set(r, r);
            for (int c = n / 2; c < n - 1; ++c)
                if (random.next() & 1) opposite.set(r, c);
        }
        opposite.set(0, n / 2);
        std::copy_n(opposite.row(0), opposite.words(), opposite.row(n / 2));
        ris::Prepared prepared(own, opposite);
        auto expected = ris::detail::reduce(ris::Matrix(prepared.kernel));
        for (int block : {1, 4, 6, 8}) {
            ris::Worker worker(prepared, 42, 8, false, block, 17, 8);
            for (int sample = 0; sample < 80; ++sample) {
                forbid_allocations = true;
                worker.trial();
                forbid_allocations = false;
                for (int i = 0; i < worker.work.rows(); ++i)
                    for (int j = 0; j < worker.work.rows(); ++j)
                        check(bool((worker.rows[i][worker.pivots[j] / 64] >> (worker.pivots[j] % 64)) & 1) == (i == j));
                auto actual = ris::detail::reduce(ris::Matrix(worker.work));
                check(actual.pivots == expected.pivots);
                check(std::memcmp(actual.matrix.row(0), expected.matrix.row(0), actual.matrix.bytes()) == 0);
            }
            check(worker.reductions == 5 && worker.proposals == 75 * 8);
            check(worker.exchanges > 0 && worker.exchanges <= worker.proposals);
        }
    }
    for (int n : {3, 63, 64, 65, 127, 128, 129}) {
        ris::Matrix own(n - 1, n), opposite(0, n);
        for (int i = 0; i < n - 1; ++i) { own.set(i, i); own.set(i, i + 1); }
        auto prepared = std::make_shared<ris::Prepared>(own, opposite);
        check(prepared->logicals.rows() == 1);
        for (int block : {1, 4, 6, 8}) for (bool masked : {false, true}) {
            ris::Worker worker(*prepared, 123, 8, masked, block);
            forbid_allocations = true;
            for (int i = 0; i < 40; ++i) worker.trial();
            forbid_allocations = false;
            // The known repetition-code Z logical is a single-qubit operator.
            check(worker.best == 1 && worker.completed == 40);
            check(worker.event_count == 1 && worker.event_weights[0] == 1);
        }
        ris::Session session(prepared, 4, 123);
        check(session.advance(7).trials == 7);
        check(session.advance(0).best_weight == 1);
    }
    for (int checks : {0, 17}) {
        ris::Matrix own(0, 65), opposite(checks, 65);
        for (int i = 0; i < checks; ++i) opposite.set(i, i);
        ris::Prepared prepared(own, opposite);
        ris::Worker worker(prepared, 81, 8, false, 6, 7, 8);
        forbid_allocations = true;
        for (int i = 0; i < 20; ++i) worker.trial();
        forbid_allocations = false;
        check(worker.exchanges == 0);
        check(worker.reductions == (checks ? 3 : 20));
        check(worker.proposals == (checks ? 17 * 8 : 0));
    }
    std::cout << "Core checks and allocation-free trial checks passed\n";
}
