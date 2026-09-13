# Bounded physical-weight polynomial experiment

Frozen before corpus searches. Use only the verified 682-182-76 (lift 341)
and 664-170-18 (lift 83) inputs. No GPU, new submissions, or verifier edits.

Construct the COMPLETE opposite-check kernel by field nullspaces and CRT
lifting. Reverse check coefficients to respect the binary syndrome convention.
Verify every lifted generator and full binary rank. Retain all logical detector
bits, including k > 64. No assumption that Hamming weight adds across factors.
This differs from the earlier reduced-space RIS/beam: full kernel coverage,
and exhaustive physical-weight updates within groups of field coefficients.

Each field-nullspace generator contributes d binary coefficients. Enumerate
all 2^d updates when d <= 10; split larger degrees into consecutive chunks of
10 (so the degree-82 factor is NOT an exact factor update). Precompute packed
XOR tables, then shuffled block coordinate descent with bounded restarts.
Keep the lowest-weight nontrivial update, allow equal-weight random ties.
Restart after two non-improving passes or eight passes, alternating seed-pool
perturbations and uniform full-kernel restarts. Initial seeds are only the
common logical initialization, never reference witnesses.

Compare three methods: unchanged guided; CRT-group optimizer; same optimizer
with shuffled ordinary binary kernel rows and IDENTICAL group sizes. Both
optimizer methods pay identical algebraic setup, deliberately making the
binary method a grouping ablation rather than an optimized binary baseline.
Neither optimizer uses guided fallback. Report initialization separately.

Use previous diagnostic factors and lift lengths as supplied reusable
structural metadata. Verify irreducibility and their product in each timed
run; charge field reduction, binary verification, table building and common
initialization. Discovery/factorization in the prior diagnostic is excluded
and must be disclosed. Every side starts cold, no setup caches across runs.

Grid: 2 and 10 seconds/code, equally split X/Z, seeds 2000,2001,2002.
2 cases x 3 methods x 3 seeds x 2 budgets = 36 configurations, 216 allocated
CPU seconds. CPU 0, one numerical thread, shuffled serial measurements;
trusted witness validation and kit persistence between measurements.
Cooperative stop granularity is setup or one group update; late exports do
not receive budget credit. Unit-test exhaustive updates, nontriviality above
64 bits, padding and invalid inputs, and CRT span against binary kernels.
Archive sources/binaries, raw events, independently audit witnesses and timing.
No tuning or expanded grid unless results show a substantial advantage.

Mathematical background: Wang/Pryadko, Distance bounds for generalized bicycle
codes, sections II.1 and III.1, https://arxiv.org/html/2203.17216 . The heuristic
here is our experiment, not a claim of a published optimal algorithm.
