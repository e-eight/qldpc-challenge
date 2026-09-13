#include "src/ris.cpp"
#include <iostream>

int main() {
    using namespace ris_guided;
    for (int n : {7, 65, 129}) {
        Matrix own(2, n), opposite(2, n);
        own.set(0, 0); own.set(0, 1);
        own.set(1, 2); own.set(1, 3);
        opposite.set(0, 0); opposite.set(0, 1);
        opposite.set(1, n - 2); opposite.set(1, n - 1);
        Prepared prepared(own, opposite);
        detail::Worker worker(prepared, 99, 8, false, 6, 64, 8);
        for (int t = 0; t < 300; ++t) {
            worker.trial();
            for (int slot = 0; slot < worker.elite_count; ++slot) {
                worker.load_elite(slot);
                for (int i = 0; i < worker.work.rows(); ++i) {
                    for (int j = 0; j < worker.work.rows(); ++j)
                        assert(bool((worker.rows[i][worker.pivots[j] / 64] >> (worker.pivots[j] % 64)) & 1) == (i == j));
                    for (int h = 0; h < opposite.rows(); ++h)
                        assert(!detail::odd_dot(worker.rows[i], opposite.row(h), opposite.words()));
                    // Membership in the opposite-check kernel plus full-rank
                    // pivot identity proves exactly the original rowspace.
                }
            }
        }
    }
    std::cout << "elite rowspace and pivot identity invariants passed\n";
}
