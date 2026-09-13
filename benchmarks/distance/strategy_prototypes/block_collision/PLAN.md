# Frozen CPU block/collision experiment

User requests two experiments: remove the restrictive orbit ansatz and implement
proper information-set collision search. Leave GPUs alone. Cluster search remains
second tier; inspect sparse-check degrees without adding a third search campaign.

Corpus: the same affine inputs 684-10-101, 684-12-73 and 684-8-85. Evaluation
targets are 90 (the previous validated witness), 70 and 81. The first target was
101 before the preceding study. Targets/references/names never enter Search.
The single-block split is construction metadata (two equal coordinate blocks).

Five methods: unchanged guided; full-pairs, block-pairs, full-stern, block-stern.
The four factorial methods use the same new C++ systematic-basis engine. Each
trial randomizes information-set columns, row-reduces packed rows, and scores all
singles and pairs. Stern additionally splits the systematic rows randomly into
two lists, builds all two-row combinations in each half, and joins matching
projections on l nonpivot coordinates. Every match represents a four-row word;
logical parity filters stabilizers. This is a fixed p=4 Stern-style search, not
Dumer partial elimination, BJMM, or an asymptotically tuned implementation.

l = min(nonpivot columns, 16, floor(log2(left pair count))). No target-driven
parameter choice. Hash buckets, pair entries, elimination scratch and row tags
are preallocated. Packed words omit coordinates identically zero in the supplied
space. Sessions persist across batches. Only strict per-session improvements
are exported, as in the other native engines; all exports are saved through the
research kit. No per-trial duplicate-witness flood. Full and block methods have
no guided fallback, so improvements can be attributed directly.

Main grid: 3 cases x 5 methods x 3 seeds x (2s,10s) = 90 configurations and 540
allocated search seconds. Seeds 1700–1702 and 1710–1712. Equal X/Z split,
common logical initialization, cold preparation included, one CPU, serial timed
runs. Block sessions receive round-robin 10ms slices; each native call is one
trial. Keep all late exports but exclude them from deadline results. The original
90 bound is an established reference, not a newly measured orbit baseline.

Separate deterministic check: enumerate every full-block kernel of dimension <=20
on the three inputs, with a 2s per-sector cap. Larger spaces are explicitly
skipped. Exactness is only within enumerated blocks. This check is separate from
the randomized factorial comparison and does not seed it.

Continuation criterion: a new lower witnessed bound on any input, or 3/3 recovery
of a sibling target at 2s while guided recovers it at most once at 10s. Merely
recovering the already inexpensive affine-90 witness is not a breakthrough. If
neither holds, stop without parameter sweeps or additional compute. Report all
factorial effects and exact-space misses, not only winning cells.

Freeze source and binary hashes before corpus search; test first on synthetic
codes. Preserve all previous measured sources/binaries/results. No verify/,
codes/, GPU, CI, commits or publication changes. Use trusted algebra for witness
validity and the kit for packaging; no full-code exact-distance claims.
