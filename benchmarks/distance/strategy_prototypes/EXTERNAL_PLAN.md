# Refreshed external-tool comparison

Freeze this protocol before corpus evaluation. No structural candidate generators,
sector transfers, new search policies, or evaluation-seed tuning are included.

- Seven existing cases: board700, board682, regression690, Tanner432, Mitten975,
  toric1000 and bicycle960, identified by the prior reference-corpus matrix hashes.
- Four methods: incremental native RIS, guided native RIS, pinned dist-m4ri CLI,
  and pinned upstream QDistEvol. External commits are those in sources.json.
- Three fresh seeds per method/case/budget: 1000--1002 at 10 seconds/code and
  1010--1012 at 60 seconds/code. This is 168 configurations, with a maximum of
  98 minutes of timed search. Three seeds support screening, not reliability claims.
- One worker pinned CPU0; serial timed runs with shuffled method order per case
  and seed. Complete validation/saving before the next timed search. Disable
  numerical library/Numba nested threads. No competing timed search jobs.
- Equal X/Z half-budgets. Charge matrix preparation, common initialization,
  backend session setup, CLI input/startup/export and witness delivery. Exclude
  Python imports, synthetic-toy JIT warmup, matrix loading and independent validation.
- Identical common initialization for all methods: single logical representatives,
  all their nonzero combinations for k <= 16 or all pairs otherwise. Retain its
  best witness and the same eight-class pool. It supplies a common output bound;
  it does not alter an engine's internal incumbent, permutations or population.
- Preserve prior native parameters (block6, restart64, exchange8, pair8; guided
  elite4 and immigrant64). Native batches target 2 ms and cap at 64 bases.
- QDistEvol: population 100, ten offspring/parent, upstream defaults for mutation,
  pivot swapping and CSS representation; allow 10,000 generations. Observe the
  existing permMinRowsK function, passing its result back unchanged. Retain each
  strict best-weight improvement's first returned representative, as in the older
  observation adapter. On deadline, retain the final result before unwinding.
- M4RI: unchanged host-tuned binary, method=1 (random window), threads=1, wmin=0,
  dW=0, large iteration cap. Reserve 50 ms for final export; judge delivery against
  the actual common deadline. Record every returned codeword, including ties.
  No internal M4RI discovery timestamp or unobserved time-to-target is inferred.
- No target stopping for any method; reference supports/weights never enter search.
  Preserve stronger existing per-side reference supports when updating bicycle's
  code-level target to 60 from the saved preparation diagnostic.
- Keep every exported support, stage and observed timestamp. Save distinct supports
  via make_submission/save_submission; a failed save stops the study. Retain late
  witnesses without deadline credit. The only schema exemption is the existing
  n>700 fixture cap exemption.
- Check observation semantics on synthetic inputs before timing. Archive source,
  executable hashes, versions, exact commands and matrices. Audit complete grids,
  witness identity, initialization equality and deadlines after evaluation.

Report median best delivered weight, target hits, paired outcomes and method-only
search progress beyond the common initialization. Show per-side results so the
code-level minimum does not hide differences. Native scored bases and external
iterations are not equivalent units of work. Interface overhead is intentional;
this is not a pure elimination-kernel benchmark or an exact-distance certificate.
