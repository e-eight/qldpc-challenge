# Exploratory routing of inferred blocks

This separate follow-up combines the frozen matrix-structure detector with the
existing single-block circulant search. It does not modify either implementation
and is not an additional method in the frozen main comparison.

For each sector's supplied row ordering, adjacent-row incidence matching proposes
a column translation. If its permutation has exactly two cycles of length n/2,
their columns provide two candidate blocks. The adapter reorders both matrices
into these blocks and calls the existing benchmark_native.PreparedSearch with
circulant=True and explicit block_size=n/2. Only the original coordinate-layout
recognizer is bypassed. The native core constructs restricted kernels of the
actual opposite check matrix and tests actual logical parity. Every returned
support is mapped back to the input coordinates, checked again, and exported.

The block partition need not be an exact code automorphism. Its practical value
is measured only by valid nontrivial witnesses in its restricted kernel spaces.
Two equal permutation cycles are a routing proposal, not proof of a hidden
bicycle construction. Recovery still uses check-row ordering: qubit relabelings
can preserve it, while independent row shuffles can destroy it. Different cycle
starting points, orientations, or exchange of the two blocks do not create extra
partition proposals. At most two distinct routes are prepared, one from each
sector's inferred translation.

Timing includes matching, matrix reordering, native preparation, search, support
mapping and callback delivery. Search uses the unchanged native single-block
algorithm, eight pair candidates, one thread, and batches targeting 50 ms with
cap4096. Both block kernels alternate as in the existing baseline. Every native
batch return is exported, including duplicates, ties and a late final result;
the parent uses actual callback timestamps for deadline credit. No logical
initialization occurs inside this adapter. No-route returns may give time back
to the parent's guided fallback.

The isolated study compares original-layout `circulant` and `routed` on three
original hard inputs plus their six frozen column/row controls, 2 seconds/code,
seed1240. Its routed wrapper explicitly uses candidate_search.Search with
method="matrix-structure" and an overridden adapter mapping; its recorded method
is "routed". This reuses identical common initialization and guided fallback.
The original study files and main source snapshots remain unchanged. The new
output directory should have a unique name, such as route-2s, under the parent
candidate-study-v2 result directory. Total search allocation is 36 seconds.

Corpus runs belong exclusively to this persisted driver or another parent
harness, never ad hoc calls. Each exported support is independently validated
and saved through the kit. Results are witnessed upper bounds; no full candidate
gate, exact-distance certificate, or generalized structure recognition is claimed.
