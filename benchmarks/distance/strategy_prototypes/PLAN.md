# Strategy pilot: frozen before evaluation

This experiment compares three new native search strategies with fresh six-pivot
RIS, incremental RIS (restart 64, proposals 8), and the existing circulant pass.
It evaluates distance upper-bound search on existing fixtures, not new codes.

- Cases: board-700-222-28, board-682-172-79, regression-690-182,
  tanner-432_8_33, mitten-975-195, toric-1000, fresh-bb-960.
- Five evaluation seeds, 500–504; one worker pinned to CPU 0.
- Two seconds per code, split equally between X and Z; no target stopping.
- Methods execute serially in a shuffled order within each case/seed. Parallel
  development is finished before measurements. Parallel validation runs only
  between measurements and is awaited before the next search.
- Matrix preparation is outside the search clock and recorded separately.
  All randomized search initialization, RIS seeding, refinement, and witness
  delivery are inside the clock. Reference supports are never search inputs.
- Every emitted witness is retained, independently checked with the trusted
  algebra, and packaged through make_submission/save_submission. Late witnesses
  remain evidence but receive no deadline credit. Above-cap fixtures retain the
  existing schema-size-cap-only exemption; no claim of full-gate acceptance.
- Primary outcomes: per-code median best timely weight and frozen-target hits.
  Paired outcomes are descriptive, not independent confidence estimates.
- This is a screening experiment. Promising candidates may receive a separate
  longer-budget check with new seeds; that selection will be reported explicitly.

## Hypotheses

1. Guided: retaining four promising bases will improve useful candidates per
   second despite basis-copy and fitness-evaluation costs. Fitness is the lightest
   nontrivial row or pair; one fresh immigrant every 64 scored bases.
2. Descent: sparse stabilizer moves can cheaply shorten independently generated
   RIS witnesses. Alternate 128-basis RIS seeds with up to 5 ms local tabu search.
3. Structure: connected-region column ordering exposes useful dependencies more
   frequently than unstructured sampling. Region caps cycle n/4, n/2, 3n/4, n.
   This is a heuristic basis-dependency search, not exhaustive cluster enumeration.

All three configurations are fixed before evaluation. The 72-qubit 0.1-second
seed-499 smoke run checks integration and persistence only; it is excluded from
comparative results. Each prototype has separate source/build files so the
previous native engine and benchmark results remain reproducible.
