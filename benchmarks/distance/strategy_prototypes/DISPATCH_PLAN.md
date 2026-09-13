# Bounded dispatcher: frozen regression plan

The user authorized implementing the bounded structure-to-search dispatcher.
Prior measured implementations and their evidence remain unchanged. A new C++17
detector and Python scheduler use the existing native restricted-kernel and guided
search cores. No decoder or new reduced-space search is added.

## Policy frozen before witness search

- Common preparation and logical initialization are identical to the candidate
  study and charged inside each sector budget.
- Detector allowance: min(50 ms, 2% of sector budget, remaining time). Try own
  check-row adjacency, then opposite adjacency, accepting the first completed
  confident two-equal-cycle partition. Overall assigned adjacent-row overlap must
  reach 0.5 of source incidence. This threshold was chosen on synthetic fixtures
  before a matrix-only inspection of the corpus.
- C++ inference builds uint16 overlap counts, rejects low upper-bound confidence,
  uses independent row maxima when bijective and bounded assignment otherwise.
  Limits: even n<=4096, 2<=rows<=4096, row weight<=64, 4 million score updates,
  64 million matching relaxations. Cooperative clock checks in scan/scoring and
  matching loops. There is no full orbit enumeration or SciPy assignment call.
- Accepted partitions prepare the existing full individual-block kernels. A
  pilot gets min(100 ms, 10% of remaining budget). Continue that session only if
  it improves common initialization; otherwise yield to guided immediately.
- A continued route targets 80% of remaining time, reserving 20% for guided.
  Kernel construction and a final batch are noninterruptible, so this is a target
  reserve, not hard real-time scheduling. Export every returned batch winner,
  including late, tied and non-improving results. Batch target 50 ms, cap 4096,
  same pairs 8/one-worker core as the control.
- Unrecognized or over-cap inputs go directly to guided. No case names,
  construction metadata, target weights or reference supports enter search.

The matrix-only pre-search diagnostic is retained and source-hashed. It runs no
witness search and did not change policy. New dtype/shape guards and synthetic
tests were completed before the timed source freeze.

## Matched grid

Four methods: incremental, guided, existing circulant with fallback, dispatch.
One worker pinned CPU0; numerical libraries single-threaded. Timed configurations
serial, method order shuffled per case/seed. Equal X/Z budget split. Preparation,
initialization, detection, conversion, kernel setup, search and delivery charged;
imports, matrix loading, and independent validation/persistence excluded.

1. Seven original codes plus six fresh qubit/row relabelings: 13 inputs × four
   methods × seeds 1400–1402 × 2 seconds/code = 156 configurations, 312 budget seconds.
2. Seven original codes × four methods × seeds 1410–1411 × 10 seconds/code
   = 56 configurations, 560 budget seconds. Independent runs, not short-run prefixes.

Total 212 configurations, 872 allocated search seconds, plus validation and native
overruns. Relabelings use RNG 1460, distinct from the previous study's 1230. Each
hard original has a qubit-only permutation and that same permutation plus
independent check-row permutations. Actual transforms and inverse-mapped reference
supports are archived. These are new layouts, not held-out code families.

Report final medians, per-seed bounds, paired outcomes, routed-only outputs,
decisions, detection latency/cap overrun/workspace, route time and guided time.
Do not attribute common-initialization or fallback improvements to the route.
Report regressions as well as gains; no zero-regression guarantee is assumed.

## Evidence

Every export gets a parent timestamp and raw support log; late output gets no
deadline credit. All distinct-per-side/run supports go through the existing
validate_and_stage -> kit make_submission/save_submission path; save failures
are fatal. Only the established n>700 fixture size-cap exemption applies.
Complete-grid, identical-initialization, source/binary/archive, matrix/transform,
stage/decision and saved-witness-identity audits must pass. No verifier or codes/
edits, new-code discovery claim, full submission gate, commit, push or CI changes.
