# Matrix structure prototype

This finite candidate generator recovers a possible hidden coordinate translation
from supplied check matrices, completes check-row orbits, and transfers opposite
logical-basis combinations when an exact sector exchange is verified. It receives
no construction metadata, parent matrices, target weights or reference witnesses.

For consecutive supplied rows, form the integer overlap matrix
`score[j,k] = sum_t H[t,j] H[t+1,k]`. Maximum-weight bipartite assignment chooses
the column permutation mapping earlier incidences to later incidences. In a
circulant check family with deleted rows, surviving adjacent parent rows support
the true translation, while gaps introduce noisy evidence. Recovery is repeated
using each sector. A two-block translation in the supplied coordinates is retained
as a final fallback. Each distinct permutation extends every supplied own-sector
row for at most `n` steps, deduplicating proposals. Every candidate is checked for
zero syndrome and nontrivial parity against supplied logical detectors.

The recovered permutation is **not presumed to be a code automorphism**. That
would exclude precisely the missing-parent checks this method seeks. Counters
report permutation cycle lengths and the overlap of translated and supplied rows.
All distinct accepted orbit candidates are emitted. Exact sector transfers emit
strict weight improvements, using singles followed by all combinations for at
most 16 logical basis rows and pairs otherwise, as in common initialization.

Sector-exchange proposals match identical column signatures after leaving the
source row order unchanged or reversing it. Both directions of the resulting
permutation must exchange the actual check-row sets exactly. This permits
transferring opposite-sector logicals while retaining syndrome and logical-parity
validation. Duplicate signature groups use deterministic matching; ambiguous
assignments may fail verification even when a valid exchange exists.

The recovery uses **check-row ordering**. With a unique assignment optimum it is
equivariant under arbitrary qubit relabeling, but arbitrary row permutations can
destroy its evidence. It is not a general graph canonicalizer or parent-code
reconstructor. The method does not generate lighter vectors from exact symmetry
alone: sector transfer reuses a weight already reachable in the other sector.
Completing a fully supplied cyclic check family usually produces only stabilizers.

Configuration is fixed: two inferred translations, one layout fallback, two
possible sector-exchange proposals, at most `n` orbit steps, and a 4096-column
cap. Matching stores a dense integer `n*n` score matrix and uses SciPy assignment;
its call cannot be interrupted mid-operation. Deadline checks occur around
analysis and every proposal. Analysis, combination enumeration and callbacks are
charged to the provided deadline. The adapter may finish its finite pass early;
unused time is not filled with another search. The seed is intentionally unused.

Run synthetic unit tests with the benchmark environment and
`PYTHONPATH=benchmarks/distance`. Corpus evaluations must use the parent harness
which independently validates and persists every exported witness through the
kit. Emissions are checked upper-bound witnesses, not exact-distance certificates
or full candidate-gate passes.
