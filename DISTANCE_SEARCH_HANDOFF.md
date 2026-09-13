# Distance-search handoff — 2026-09-13

**Status: research stopped; useful engineering improvements, no general scaling breakthrough.** The full sweep of all 593 submitted CSS codes (7–700 qubits) is complete and audited. The new native RIS core roughly doubles fixed-work throughput on the initial corpus; guided search is the strongest general strategy in the final sweep, with incremental RIS providing useful complementary wins. Nothing has been integrated into the production checker. This branch is an archive for reuse or extraction into a standalone repository, **not a proposed merge**.

**Recommendation:** retain the native six-pivot core, cheap common logical initialization, guided and incremental policies, and the existing applicable circulant search. Before integration, consolidate the guided prototype into the core and repeat the strategy disagreements across seeds. Preserve independent witness validation and the existing gate's refutation safeguards. Do not raise the size cap on this evidence: the study does not estimate the gate's false-acceptance rate, and several hard 700-qubit inputs remain far from their submitted bounds.

## Final corpus comparison

“Below” means a strictly lighter validated logical than the actual submitted JSON distance claim; “matches” means equal weight; “above” means the search did not recover that claim. These are **witnessed upper bounds**, not exact distances. A match proves neither optimality nor reliable refutation of inflated claims. The old C++ row also receives the new common initialization, so it is a controlled engine comparison rather than a replay of the production gate.

| Seconds/code | Method | Below claim | Matches claim | Above claim | Lower / equal / higher than old C++ |
|---:|---|---:|---:|---:|---|
| 30 | Old C++ RIS | 5 | 535 | 53 | 0 / 593 / 0 |
| 30 | New fresh RIS | 5 | 544 | 44 | 37 / 548 / 8 |
| 30 | Incremental RIS | 6 | 550 | 37 | 47 / 544 / 2 |
| 30 | Guided | 9 | 555 | 29 | 50 / 542 / 1 |
| 30 | dist-m4ri | 5 | 533 | 55 | 22 / 546 / 25 |
| 60 | Old C++ RIS | 5 | 541 | 47 | 0 / 593 / 0 |
| 60 | New fresh RIS | 5 | 547 | 41 | 32 / 553 / 8 |
| 60 | Incremental RIS | 6 | 555 | 32 | 41 / 551 / 1 |
| 60 | Guided | 9 | 555 | 29 | 43 / 549 / 1 |
| 60 | dist-m4ri | 5 | 541 | 47 | 16 / 555 / 22 |

At 60 seconds, guided recovers or beats 564 claims versus 546 for old C++: useful, but mostly incremental on an already easy majority of the corpus. Per-code values, family breakdowns and source hashes are in the [compact full-suite evidence](benchmarks/distance/handoff/evidence/full-suite).

### What each method does

- **Old C++ RIS (`cpp`)** repeatedly reduces a kernel basis using randomized information sets, then tests single rows and pairs among eight light rows. The benchmark exposes the repository's existing native core through `benchmark_native.PreparedSearch`.
- **Fresh RIS (`fresh`)** implements the same basic candidate policy in a standalone C++17 core with thin pybind11 bindings. Persistent scratch storage and six-pivot XOR tables reduce elimination work; the initial fixed-work comparison reproduced the old core's best weights and supports on all 33 inputs.
- **Incremental RIS (`incremental`)** reuses a systematic basis through pivot exchanges instead of rebuilding it for every sample. The evaluated configuration makes eight exchange proposals per sample and restarts with fresh six-pivot reduction every 64 bases; these samples are correlated.
- **Guided (`guided`)** maintains four parent bases, mutates them with pivot exchanges, and favors bases yielding lighter nontrivial logicals. Tie moves and a random immigrant every 64 scored bases preserve some exploration; it shares native RIS machinery but is not a faithful reproduction of QDistEvol.
- **dist-m4ri (`m4ri`)** is the external pinned method-1 search, independently implemented using M4RI. An observer-only patch exports strict improvements with receipt timestamps; its I/O cost is included, and fixed-work tests confirm unchanged final outputs against the unmodified binary.

**Common initialization:** before each engine, score matrix-derived canonical logical representatives: singles, then all nonzero representative combinations for k ≤ 16, otherwise pairs. Retain up to eight distinct logical-class seeds. This finds cheap candidates already latent in preparation; it does not enumerate their stabilizer cosets. Its timing and outputs count equally for every method, but its incumbent is not injected into the engine's fitness or stopping policy. Submitted witnesses never enter search.

### How much did the bounds improve?

The best over five separate 60-second runs beats 10 submissions, matches 557, and misses 26. Four improvements are at least 20%; most submissions are unchanged. This oracle union costs up to **300 seconds/code**, not 60.

| Submission identifier | Actual claim | Best sweep witness | Reduction | Common initialization |
|---|---:|---:|---:|---:|
| 684-12-81 | 81 | 63 | 22.2% | 63 |
| 684-8-85 | 81 | 63 | 22.2% | 63 |
| 684-12-77 | 77 | 61 | 20.8% | 69 |
| 400-12-50 | 50 | 40 | 20.0% | 40 |
| 396-10-39 | 39 | 33 | 15.4% | 33 |
| 672-8-36 | 36 | 34 | 5.6% | 134 |
| 684-14-78 | 78 | 75 | 3.8% | 108 |
| 684-20-72 | 72 | 70 | 2.8% | 120 |
| 360-8-48 | 48 | 47 | 2.1% | 70 |
| 600-8-96 | 96 | 95 | 1.0% | 116 |

Five claims were already beaten by initialization; four final winning bounds are entirely initialization results. Guided improves 684-12-77 further, from 69 to 61, and uniquely supplies four other below-claim sweep results. Incremental uniquely supplies 360-8-48's 47. These are not novelty claims across all earlier work: the structural experiment already found **54** on 684-8-85, better than this sweep's 63. Filenames can contain stale distances; use the actual JSON claim.

### Overlap and portfolio

At 60 seconds, guided and incremental tie in weight on 565 codes; guided is lighter on 17 and incremental on 11. Both recover 558 submitted targets, guided alone six, and incremental alone three. Their union recovers 567, the same coverage as all five general engines, but costs 120 seconds/code.

The disagreements can be substantial: incremental finds **32 versus guided's 72** on 576-8-32, and 47 versus 50 on 360-8-48. Guided finds **70 versus incremental's 88** on 684-20-72, and 61 versus 69 on 684-12-77. Guided-only target recoveries are 288-8-35, 630-6-36, 672-18-30, 672-8-36, 684-14-78 and 684-20-72; incremental-only recoveries are 360-8-48, 576-8-32 and 682-140-86.

A retrospective **30 seconds guided + 30 seconds incremental** portfolio recovers 566 targets versus guided's 564 at 60 seconds. Against guided alone it gives lower/equal/higher weights on 7/582/4 codes. This uses recorded checkpoints, not a separately executed scheduler; one seed cannot establish an optimal allocation. The [distribution analysis](benchmarks/distance/handoff/evidence/distribution.json) retains all differences and target sets.

The existing circulant-specific pass, separately allowed two seconds/code, recovers five more targets missed by the general union: 390-68-28, 502-102-50, 666-150-76-b, 666-150-76 and 674-170-76. It does not tighten another submitted claim. Combined coverage is 572/593; the general sweep still returns weights 71–75 on four 700-qubit weight-28 claims. Particular structure-aware searches can do much better on recognized layouts, but that does not make general search reliable there.

## Strongest engineering and structural findings

**Native throughput:** the six-pivot core's initial fixed-work median speedup was 1.96× across 33 inputs, with 30 improved and three toric controls 3–5% slower. Repeated one-worker speedups were 2.38× on Board700, 2.24× on Tanner432, 1.98× on Mitten975 and 2.62× on bicycle960; toric1000 was 0.96×. Four-worker throughput scaled about 3.2–3.8×. These timings exclude preparation and independent validation, and are not a universal end-to-end 2× claim.

The core packs binary rows into 64-bit words, preallocates per-worker matrices and scratch, swaps row pointers, and performs table-based panel elimination. Compiler-generated SIMD XOR/popcount works well; explicitly unrolling the six-bit table index removed an expensive generated gather/reduction sequence. At n=1000 the six-pivot table is 8 KiB and a 600-row matrix is 75 KiB; immutable and working copies are separate. These fit different cache levels, so reducing passes matters more than indiscriminately eliminating branches. There is no handwritten ISA dispatch. The plain allocation-free rewrite alone was usually slightly slower than the old code.

**Family-specific success:** affine single-half kernels have dimensions 20–40 instead of full dimensions 346–348. Searching these spaces lowered the prior study references on 684-12-73 from 70 to **51**, and on 684-8-85 from 81 to **54**; another affine case reached its already improved reference 90. Systematic single/pair search found 51 and 54 in roughly 13–23 ms of the relevant sector; small buffers were 8–12 KiB. The winning ingredient was the restricted space, not the collision join. Selected exact component enumerations independently establish minima within those spaces; **none establishes the full code's distance**, since mixed-half logicals remain possible. Saved [structural witness documents](benchmarks/distance/results/block-collision-study/witness-index.json) identify the source, sector and support.

**Structure recovery/routing:** a bounded detector inferred useful block partitions under hidden column permutations and routed the existing block kernel to weight 28 on Board700 and 48 on Regression690. It failed under independent row shuffles. A conservative competing-session allocator retained those wins and tied guided where partitions were rejected, but showed no new best hard reference bound and no observed preference switches after its pilots. This is a selective engineering candidate, not general code-equivalence recovery.

**External comparison:** a separate seven-case, three-seed refresh at 10 and 60 seconds ran 168 configurations with common initialization. Guided beat/tied/lost to dist-m4ri in 25/17/0 of 42 matched outcomes, and to QDistEvol in 29/13/0; versus incremental it was 11/24/7. External parameters were not swept. QDistEvol was therefore omitted from the expensive final corpus run. Easy toric1000 and bicycle960 bounds were already found by initialization; difficult Board700 still had median 73 versus target 28 at 60 seconds. See the [external audit](benchmarks/distance/handoff/evidence/studies/external-refresh/combined-audit.json).

## What could be put into use

1. Extract or consolidate [native/ris](native/ris/README.md) and the [guided policy](benchmarks/distance/strategy_prototypes/guided/README.md). The guided implementation currently duplicates the core; expose a stable policy selection API before treating it as a maintained package. Native preparation owns its data, releases the GIL, and persistent sessions retain RNG/worker state. Cancellation is between bounded batches, not a hard native wall-clock deadline.
2. Add [common initialization](benchmarks/distance/initialized_search.py) and the new native backend to the research checker first. Keep guided and incremental available; repeat the 28 final weight disagreements across seeds before fixing a portfolio default.
3. Preserve the trusted Python syndrome and non-stabilizer checks on every proposed witness. The production paths in `research/kit/surrogate.py` and `verify/gate_changed.py` are unchanged. Do not replace the whole verification gate with guided or interpret its correlated samples using independent-RIS confidence formulas.
4. Keep the existing applicable circulant pass. Consider the bounded partition detector only with fallback and held-out relabeling/family tests. Further broad mathematical experiments are shelved pending a concrete new hypothesis; GPUs were deliberately not pursued.

Issue [#1016](https://github.com/unitaryfoundation/qldpc-challenge/issues/1016) is the appropriate discussion: this provides useful implementation and target-recovery evidence toward its budget question. Its proposed gate-equivalent, repeated-seed false-acceptance calibration remains outstanding. An aggregate recovery percentage on this mostly easy corpus cannot substitute for that study.

## Protocol, validation and retained evidence

The final run used one seed (2100), five general engines × 593 codes, plus 593 separate circulant configurations: **3,558 configurations**. Each general code budget splits equally between X and Z, with no target stopping. The 30-second checkpoint is the first 15 seconds of **each** sector of the 60-second run, not both sector results delivered within its first 30 wall seconds.

Preparation, common initialization, setup, candidate scoring and witness delivery count against the budget. Imports, matrix loading and independent validation/packaging do not. Four single-thread jobs ran concurrently on VM CPUs 0,2,4,6, sharing last-level cache. The VM reports AMD EPYC 9554P, 32 KiB L1 data and 1 MiB L2 per exposed physical core, and 32 MiB shared L3. This is a throughput experiment under VM contention, not isolated-core latency. General allocation totals 49h25m of CPU-worker budgets; parallel completion took an overnight run, including a recovered interruption.

The final independent audit checked **94,944 raw observations and 85,898 kit-saved witness documents**, input/source/binary hashes, common initialization and checkpoint accounting. Zero events were late at 60 seconds; 305 after the 30-second checkpoint were correctly excluded there. Four interrupted attempts were preserved and excluded from completed statistics. No full candidate gate or leaderboard update was performed; all 27 trusted verifier files remained unchanged.

The handoff recheck passed 62 native/guided/harness Python tests, a fresh standalone CMake core test, and 17,906 witness checks over 7,116 archived sector records. Earlier validation includes 51 Python native tests, allocation-free trial checks, ASAN/UBSAN, portable wheel import outside the repository, and standalone CMake tests from a source distribution. Guided has Python reproducibility/witness tests and native parent-basis invariants under sanitizers. The full-suite harness passed 14 tests including live observer timing, unmodified dist-m4ri equivalence, deadline accounting, failure persistence and a synthetic end-to-end audit.

The branch retains sources, build recipes, tests and **compact measured evidence**, not generated binaries, dependency caches or bulky repeated matrix documents. [Checkpoint witnesses](benchmarks/distance/handoff/evidence/full-suite/witnesses.jsonl) preserve both sectors for every method/case at both budgets, plus initialization and original result hashes. [The compact auditor](benchmarks/distance/handoff/audit_evidence.py) checks these against the committed submissions using the unchanged trusted GF(2) routines and reproduces the table counts. It cannot replay the original timing audit from this reduced archive; the original audit result is retained with that limitation. No witness search runs during extraction or checking. Local raw data remain untouched at handoff time; disk cleanup is a separate step.

Historical prototype READMEs/plans describe their own study stages and sometimes refer to full local run trees omitted from this source archive. This file supersedes their “next steps.” Historical aggregate reports are retained as text and JSON under [study evidence](benchmarks/distance/handoff/evidence/studies); their paths describe the original run, not a promise that all raw artifacts are in Git. Re-running an experiment creates fresh output and fresh source/binary hashes; do not expect rebuilt binaries to match the recorded hashes.

## Build and run

For the standalone core, use [its README](native/ris/README.md); no external search package is needed. For the shared benchmark harness, from a fresh checkout on Linux with a C/C++ toolchain, make, autotools prerequisites and Python 3.12:

```sh
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python -r benchmarks/distance/requirements.txt
RIS_NATIVE=1 .venv-benchmark/bin/python benchmarks/distance/bootstrap.py
(cd native/ris && RIS_NATIVE=1 ../../.venv-benchmark/bin/python setup.py build_ext --inplace)
(cd benchmarks/distance/strategy_prototypes/guided && ../../../../.venv-benchmark/bin/python setup.py build_ext --inplace)
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m full_suite.build_observer
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m pytest --import-mode=importlib -q benchmarks/distance/test_ris_native.py benchmarks/distance/strategy_prototypes/guided/test_guided.py benchmarks/distance/full_suite/test_suite.py
.venv-benchmark/bin/python benchmarks/distance/handoff/audit_evidence.py
```

Host tuning is for local measurement; omit `RIS_NATIVE=1` for a portable core build. The observer builder refuses an existing destination. Dependencies are pinned in [requirements.txt](benchmarks/distance/requirements.txt) and [sources.json](benchmarks/distance/sources.json); the native package alone has a much smaller dependency surface. The full environment pin list includes dependencies of optional historical methods.

To deliberately repeat the long full sweep, use an unused output directory:

```sh
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m full_suite.prepare new-full-suite
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -u -m full_suite.study new-full-suite
```

Use a durable process supervisor for long runs; the initial foreground tool session was interrupted and the continuation completed under systemd. The [driver documentation](benchmarks/distance/full_suite/README.md) explains resume and evidence handling. The fixed harness uses CPUs 0,2,4,6; adapt and record affinity on other machines before preparing the run. The full sweep loads `codes/*.json` directly and does not need the older external fixture corpus.

For earlier studies, build the relevant prototype's `setup.py` first, then use its `study_*.py` entry point and plan. Generate the initial fixture corpus with `benchmarks/distance/corpus.py`. The later structural chain uses `strategy_prototypes/large_structure/prepare_corpus.py`, `block_collision/prepare.py`, then `component_search/prepare.py`; these reconstruct matrix files from committed inputs and retained structural reference witnesses. Component preparation also creates the relabeled corpus used by the algebra diagnostics and polynomial experiment. Existing output directories may be refused deliberately. Some historical reporting scripts require the newly generated full run and local witness packaging, not just this compact archive.

## Unsuccessful or limited approaches — why we stopped

These are negative results for the tested implementations, spaces and budgets, not impossibility results for their algorithm families. Source for every prototype is retained under [strategy_prototypes](benchmarks/distance/strategy_prototypes), with orchestration and audit scripts under [benchmarks/distance](benchmarks/distance).

| Attempt | Result and failure mechanism | Disposition |
|---|---|---|
| Plain allocation-free C++ rewrite / masked XOR | Usually slightly slower than old C++; removing allocations or branches alone did not reduce the dominant elimination work. Six-pivot tables and efficient index extraction produced the real gain. | Keep the measured six-pivot implementation. |
| Stabilizer tabu/local refinement | Initially improved Tanner recovery and replayed Tanner/bicycle seeds, but consumed time on high-check-weight codes and weakened Mitten. Common initialization explained some apparent large-code wins; the subsequent hybrid did not justify a general default. | Optional selective refinement, not the primary checker. |
| Connected-region / cluster-guided elimination | Worse weights than incremental in 26/35 matched outcomes. Toric recovery was already available from cheap initialization. | Shelved; no evidence for a general cluster advantage. |
| BP+OSD logical-constraint decoder | Won none of 21 short paired runs versus guided. On two hard inputs a trial took about 12–14 seconds, often past the sector deadline; BP exhausted its iterations and OSD supplied outputs. Longer runs still failed to improve the common hard-case initialization. | Amortized setup and lighter logical detectors remain untested possibilities, not demonstrated gains. |
| Reduced polynomial/coupled beam spaces | Coupled branch improved initialization in 0/30 hard sectors and consumed 94–97% of branch time. Membership checks found important stored witnesses outside proposed spaces; some in-space targets were still missed. Useful results came from full single-block branches. | Both coverage and search quality were inadequate; do not just increase its budget. |
| Automatic block recovery | Useful for hidden columns, unsuccessful with independently shuffled check rows. Unproductive detection could consume enough budget to worsen general results. Bounded rejection and fallback helped. | Selective routing candidate only. |
| Competing guided/structured allocation | Retained useful 28/48 family wins; no improvement to best hard references. No preference switched after pilots, so the data do not demonstrate a benefit from ongoing reassessment. | Optional engineering, not a breakthrough. |
| Fixed p=4 Stern 2+2 collision join | Within affine blocks, systematic single/pair search and collision search both reached 90/51/54 across seeds and budgets. The join did not lower final bounds further. | Retain restricted-space insight; this does not evaluate fully tuned Stern/Dumer/BJMM implementations. |
| Component-restricted enumeration | Exact selected affine spaces were useful, but no broad gain followed. Generic detection failed on relevant affine/product layouts; dimension caps excluded several larger spaces; tested individual lifted blocks had no nontrivial logicals. Small product spaces exhausted at 48/56 while initialization already supplied 22. | Metadata-specific tool. Block combinations and raised caps were not comprehensively explored. |
| Exact binary decomposition / cut-rank diagnostics | Five hard whole kernels were coordinate-indecomposable; surface controls split. Best of eight tested balanced-cut orders had coupling 136–170 on hard cases versus 6–7 within surface components. This disfavors those trellis orders, not all possible orders. | Useful diagnostic, no new distance bound. |
| Cyclic factorization / CRT structure | Verified polynomial module structure on lift-341 and lift-83 codes. CRT components overlap in physical coordinates, so Hamming weight is not additive across factors. Low algebraic dimensions did not create independent low-weight pieces. | Structure information alone did not solve the weight search. |
| Full-kernel polynomial weight descent (last experiment) | Completed the missing kernel coverage and compared CRT coefficient groups with shuffled binary-basis groups using the same fast optimizer. At 10 seconds, guided returned 80/81/80 on the 682 case versus CRT 99/99/99, and 18/18/18 on the 664 case versus CRT 82/82/86. Binary groups were also much worse than guided. About 159 million updates/s did not overcome poor moves/local minima; 4.3 MiB tables and ~0.15 s/sector setup were not the main explanation. | Shelved. Increasing budget from 2 to 10 seconds did not rescue it; arbitrary polynomial methods are not ruled out. |

The last experiment covered complete kernels of dimensions 432 and 417; degree-82 factors were chunked into coefficient groups of at most ten bits. Thus it removed the earlier missing-space objection but did not test exhaustive optimization over a whole large field factor. The specialized block engine also has implementation limits (n ≤ 2048, basis dimension ≤ 512, at most 64 logical detectors); its wrapper compresses detectors on the restricted space. It must not be used generically by truncating a high-k logical basis.
