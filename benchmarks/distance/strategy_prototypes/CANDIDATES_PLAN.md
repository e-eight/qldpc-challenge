# Three next-step candidate generators: frozen evaluation plan

User requested parallel implementation/testing of matrix structure recovery,
reduced algebraic-space search, and decoder-based logical search. Three agents
implemented isolated candidates and cross-reviewed algebra/persistence. Corpus
search starts only after synthetic tests and source freeze. Read-only structural
analysis was permitted during development; no corpus witness was used for tuning.

## Candidates and shared output policy

- `matrix-structure`: recover candidate column translations from adjacent check-row
  incidence, complete missing orbits, and verify sector exchanges. Transfer dual
  representatives under verified exchange. This uses row-order clues and is not
  arbitrary code-equivalence recovery. Finite pass, seed-independent.
- `reduced-space`: verified coupled polynomial space and separately labeled
  single-block cyclic spaces; packed C++ random bases and pruned up-to-four-row
  combinations. Four coupled branch visits per one visit to each single branch;
  visits are not equal runtime allocations. Config is in its frozen adapter.
- `decoder`: augmented nonzero logical syndrome, ldpc BP+OSD, 100 product-sum BP
  iterations, parallel schedule/one OMP thread, OSD_CS order 1, random nonzero
  logical combinations and column permutations. Uniform base p=0.05 with fixed
  per-coordinate log-odds jitter ±0.25, selected on synthetic toys only.
- Controls: unchanged incremental and guided native kernels; existing single-block
  circulant core. The latter exports each 50-ms-target batch winner (all returns
  retained), rather than the prior 5-ms observation cadence. Native incremental
  and guided batches target 2 ms/cap64; core parameters unchanged.
- Every method starts with the SAME native preparation and logical initialization:
  singles, all combinations if k<=16 or pairs otherwise, eight-class seed pool.
  Shared bound is retained externally, not injected into search state.
- A finite/inapplicable candidate gives remaining time to a fresh guided session.
  Report candidate-only outputs separately from this portfolio result and separate
  transfers, inferred orbit recovery, layout proposals, coupled and single-block
  reduced-space outputs. Applicability flags alone are not successes.

## Grid and timing

All runs use one CPU worker pinned to CPU0, numerical libraries limited to one
thread, timed configurations serial, method order shuffled within case/seed.
Development/analysis can run on CPU7; no competing timed corpus search. Each code
budget splits equally between X and Z. Charge preparation, common initialization,
analysis, conversions/session creation, search and witness delivery. Exclude imports,
synthetic-only warmup, matrix loading and independent validation/persistence.
No method gets code names, construction metadata, reference witnesses, or target
weights; never stop on a reference target.

1. Seven current original cases, six methods, seeds1200--1202, 10s/code:
   126 configurations, at most21min search.
2. Three hard cases with a fixed random common qubit permutation, and with that
   permutation plus independent X/Z check-row shuffles. Six variants, six methods,
   seed1200, 10s/code: 36 configurations, at most6min. Permutation RNG seed1230;
   actual permutations and correctly transformed evaluation references archived.
   Row operations beyond reordering are outside this diagnostic.
3. Three original hard cases, six methods, seeds1210--1211, 60s/code:
   36 configurations, at most36min. Independent runs, not prefixes of the short sweep.

Total:198 configurations and at most63min of search, plus validation. Three short
seeds and two long seeds support screening only; relabel controls are single-seed
mechanism diagnostics. DistQLDPC and new exact-solving experiments are outside scope.

## Evidence and decision criteria

Every emitted support is logged with an observed timestamp and stage; all distinct
per-side/run supports are independently checked and saved through kit
make_submission/save_submission. Save failure is a hard error. Preserve late
results with no deadline credit. Existing n>700 fixture size-cap exemption only.
No verifier/leaderboard changes, exact-distance claims, full-gate passes or publication.

Report medians, per-seed bounds, target hits, candidate-vs-initialization gains,
branch-specific improvements, fallback time, setup cost, CPU time and memory where
available. Compare with both native controls and the existing circulant method:
rediscovering its known weight48 on regression690 is not a new algorithmic gain.
Use random relabelings to distinguish coordinate assumptions from code properties.
Inspect original-vs-relabel sensitivity rather than claiming permutation invariance
from one example. Source/binary versions, matrices, transformations and raw supports
must accompany complete-grid and witness-identity audits.
