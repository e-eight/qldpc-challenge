# CPU distance-search study (#1016)

**Final status:** research is stopped and the 593-code sweep is complete. Start
with the [distance-search handoff](../../DISTANCE_SEARCH_HANDOFF.md) for final
results, recommendations, retained evidence and reproduction instructions. This
branch is an experiment archive, not a proposed production merge. The plans and
stage reports below describe earlier phases and are superseded by that handoff.

The decision is which CPU search, or combination of searches, should serve the
repository's distance refutation gate. Finding a light nontrivial logical is the
success criterion. Failure to find one is never an exact-distance certificate.

See [initial results and recommendations](RESULTS.md) for the completed
33-case pilot and one-worker diagnostics. The full repeated-seed study remains
to be run; a frozen reference corpus is included under `results/reference-corpus`.

## Comparisons

* The current C++ RIS, with its normal eight-row pair search.
* The same C++ kernel with pairs disabled, separating pair-search benefit from
  implementation speed.
* The current multithreaded `dist-m4ri` random-window search.
* QDistEvol: independent evolutionary populations sharing a fixed CPU budget.
* The verifier's NumPy RIS, as a completeness and portability baseline.
* The current C++ circulant-block search, on matrices that it recognizes.

No verifier, gate, eligibility limit, or distance algorithm is changed by this
study. Benchmark adapters may expose existing native primitives and observe
intermediate results; any such difference from the public entry point is
reported with the results.

## Corpus and selection

The 33-case corpus combines the cases in issue #1016 with small controls, difficult
current submissions, published quantum Tanner and bicycle examples, and fresh
fixed-seed constructions. It includes both non-abelian families behind the four
large reference bars: ZSZ-LP and mitten. Those families both have rate 1/5, so
larger low-rate bicycle/toric and fresh hypergraph-product cases are included too.

Fresh examples are selected without running a distance search. Their random
seeds, constructors, and parameters are fixed before comparing methods. The
corpus records matrix hashes, dimensions, GF(2) ranks, check weights, provenance,
and reference-witness status. A paper's distance label is initially an
unverified target, not a validated answer. Toric examples are scaling
controls, not evidence of typical qLDPC search difficulty.

The staged 666-qubit example in the issue is excluded until its matrices are
public and the discrepancy between its claimed witness weight and support list
is resolved. This exclusion is recorded rather than replaced by a guessed code.
The public failed 690-qubit candidate in `notes/682-172-79.md` supplies a separate
regression: its historical claim was 77 and the note retains a weight-28 logical.

## Resource and timing contract

The primary comparison uses four CPU workers total, including subprocesses and
nested native pools. A one-worker comparison separates algorithm differences
from scaling. Methods run sequentially on the same host; different methods do
not compete for CPU. BLAS, OpenMP, and Numba internal pools are limited to one
thread unless explicitly allocated as the method's workers. Where supported,
CPU affinity is fixed and recorded. On heterogeneous CPUs the lack of an
equivalent core allocation must be reported.

Each code gets an equal time allocation for X and Z search. Report per-side
results as well as the minimum over both sides. Imports/JIT and process startup,
matrix/logical-basis preparation, search, and witness validation/persistence are
measured separately. The end-to-end cost includes all applicable stages; warm
search throughput is also useful but must not be labeled end-to-end latency.
An observation arriving after a deadline does not count as success by that
deadline. Batch boundaries can conservatively delay observation of a witness;
report that granularity and any overrun.

Both sides use the same code-level stopping target: the minimum of supplied
witness weights, paper targets, and analytic targets. A newly frozen witness
does not relax a smaller paper target. Reference recovery and paper-target
recovery remain distinguishable in the output.

QDistEvol's generations must remain intact within each worker. Four independent
populations are explicitly different from parallel evaluation of one population.
Keep population and mutation settings fixed across comparisons. Random seeds
are recorded, but identical seed integers do not imply identical random trials
across implementations or thread counts.

## Outcomes

1. For a held-out, validated reference witness of weight w, measure the fraction
   of runs independently returning a valid logical of weight at most w. This is
   equivalent to refuting a hypothetical claim greater than w; it does not
   require knowing the exact distance.
2. Separately measure actual refutations of the submission's current claim.
3. Record best weight, time to target, completed trials, wall and CPU time,
   preparation cost, and deadline overshoot. Derive trials/second from those
   records. Per-run peak memory measurement remains follow-up work.
4. Report each code/family/size band; do not hide failures behind one aggregate
   speedup. Give binomial uncertainty for repeated-seed success rates and retain
   timed-out runs as censored observations.

Reference witnesses are used only for scoring and validation, never as initial
search candidates. Comparisons use a frozen reference set. New lighter
witnesses are retained and reported separately rather than silently changing
the target during the experiment. Where no reference witness has yet been
validated, report bounds and throughput without claiming a success rate against
ground truth.

The full study uses 20 seeds and 1/10/60-second checkpoints, with longer runs on
unresolved cases. A shorter pilot first validates adapters, persistence,
deadline behavior, and estimated cost. Pilot results cannot establish a
population-wide missed-refutation rate or justify raising the code-size cap.

## Evidence and reproducibility

Pin source revisions and dependencies. Preserve raw run records and every
returned improving witness before scoring. Validate witnesses using the
repository's existing GF(2) routines; package retained candidates through
`research/kit/submit.py`. A failed witness save is a hard error. Raw benchmark
evidence is stored separately from candidate staging so it can accompany a
reviewable report. There are no automatic leaderboard submissions.

Inputs above the current 700-qubit cap are deliberately ineligible benchmark
fixtures. Their retained candidate documents are allowed only that specific
schema violation; malformed metadata and all invalid logicals are hard errors.
Witness validation in this harness is not a pass through the full candidate gate.

Sources:

* https://github.com/unitaryfoundation/qldpc-challenge/issues/1016
* https://arxiv.org/abs/2603.22532 (RIS/QDistEvol benchmarks)
* https://github.com/QEC-pages/dist-m4ri
* https://github.com/m-webster/codeDistancePYPI
* https://arxiv.org/abs/2607.27644 (ZSZ-LP)
* https://github.com/a7b/yarn (mitten matrices)

## Running the study

From the repository root, prepare an isolated Python 3.12 environment:

```sh
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python -r benchmarks/distance/requirements.txt
.venv-benchmark/bin/python benchmarks/distance/bootstrap.py
.venv-benchmark/bin/python benchmarks/distance/corpus.py
.venv-benchmark/bin/python -m pytest -q benchmarks/distance/test_benchmark.py
```

The bootstrap downloads pinned public sources and compiles them under the
benchmark cache. It does not install system libraries or change the production
environment. The corpus step performs structure/reference checks, without a
distance search or screening fresh inputs by distance.

Run a small target-recovery pilot, then export portable evidence:

```sh
.venv-benchmark/bin/python benchmarks/distance/run.py \
  --m4ri benchmarks/distance/cache/deps/dist-m4ri/src/dist_m4ri \
  --output benchmarks/distance/runs/pilot \
  --methods cpp cpp-no-pairs m4ri qdistevol numpy cpp-circulant \
  --threads 4 --seconds 10 --seeds 1 --seed-start 11
.venv-benchmark/bin/python benchmarks/distance/report.py \
  benchmarks/distance/runs/pilot benchmarks/distance/results/pilot
```

`--seconds` is the total per-code search budget; half goes to each side.
`--no-target-stop` instead spends the full budget looking for tighter bounds.
Use `--cases` to select manifest IDs and `--seed-start` to reserve independent
evaluation seeds. On Linux, `--cpus` fixes the affinity inherited by workers;
choose one physical core per worker and avoid SMT siblings. Four processes each
running one native thread count as four workers, not sixteen.

After a reference phase, freeze witnesses into a new corpus and reserve fresh
seeds for evaluation. Late witnesses can improve the reference set even though
they did not count as timely successes in the original run:

```sh
.venv-benchmark/bin/python benchmarks/distance/freeze_references.py \
  --corpus benchmarks/distance/cache/corpus \
  --runs benchmarks/distance/runs/pilot \
  --output benchmarks/distance/cache/evaluation-corpus
.venv-benchmark/bin/python benchmarks/distance/run.py \
  --corpus benchmarks/distance/cache/evaluation-corpus \
  --m4ri benchmarks/distance/cache/deps/dist-m4ri/src/dist_m4ri \
  --output benchmarks/distance/runs/evaluation-4-workers \
  --methods cpp m4ri qdistevol cpp-circulant \
  --threads 4 --seconds 60 --seeds 20 --seed-start 100
```

Run the same frozen corpus with `--threads 1` for the scaling comparison.
The included `results/reference-corpus` can also be supplied directly to
`--corpus`, using held-out seeds. Repeat the command at `--seconds 1` and
`--seconds 10` for the shorter budget comparisons.
The commands above start substantial local computation; choose a host and time
budget appropriate to the study. They do not schedule unattended background jobs.

The initial local pilot uses Apple M4 Pro hardware (10 performance and 4
efficiency cores, 48 GiB RAM). macOS does not expose Linux-style affinity through
this harness, so core equivalence is uncontrolled. Its M4RI library uses the
release's default `-O2`; dist-m4ri and the repository kernel use `-O3`. The
reproducible bootstrap uses `-O3` for both the library and executable. Record
these build settings with any comparison; repeat on the intended CI CPU before
making a production performance claim.

## Standalone native rewrite

The experimental C++17 library and Python binding live in `native/ris`. They can
be extracted independently of this repository. No production verifier is changed.
The `ris`, `ris-masked`, `ris-block4`, and `ris-block6` method names select its
conditional, masked, and table-based elimination variants. All retain the same
RIS candidate policy and eight-row pair search.

A native-only environment needs NumPy, pybind11, setuptools, pytest, jsonschema,
SciPy, and threadpoolctl; the QDistEvol dependencies are optional for these runs.
Build both extensions with the same CPU tuning:

```sh
RIS_NATIVE=1 .venv-benchmark/bin/python benchmarks/distance/setup_native.py \
  build_ext --force --build-lib benchmarks/distance/build \
  --build-temp benchmarks/distance/build/temp
cd native/ris
RIS_NATIVE=1 ../../.venv-benchmark/bin/python setup.py build_ext --inplace --force
cd ../..
.venv-benchmark/bin/python -m pytest -q benchmarks/distance/test_ris_native.py
```

`RIS_NATIVE=1` opts into instructions supported by the build CPU. Omit it for
portable builds; do not compare a tuned rewrite against an untuned baseline.

For an identical-work experiment, each method receives the same seeds and trial
counts. The exporter rejects any mismatch in best weight or support:

```sh
.venv-benchmark/bin/python benchmarks/distance/compare_native.py \
  --output benchmarks/distance/runs/native-fixed \
  --methods cpp ris ris-block6 --trials 1024 --repeats 3 \
  --threads 1 --cpus 0
.venv-benchmark/bin/python benchmarks/distance/report_native.py \
  benchmarks/distance/runs/native-fixed benchmarks/distance/results/native-fixed
```

For target recovery under equal time budgets, use the ordinary runner:

```sh
.venv-benchmark/bin/python benchmarks/distance/run.py \
  --corpus benchmarks/distance/results/reference-corpus \
  --output benchmarks/distance/runs/native-timed \
  --methods cpp ris-block6 --threads 4 --cpus 0 2 4 6 \
  --seconds 2 --seeds 3 --seed-start 100
```

Choose CPU IDs from the actual host topology. A VM's reported topology does not
prove that its vCPUs map to exclusive host physical cores. The rewrite persists
worker state across observation batches; the original adapter reseeds each
batch, so equal-time experiments need repeated seeds and are not identical-trial
comparisons. New-engine preparation and worker startup are recorded in each
side's `setup_seconds`, outside the warm search budget. Both adapters also incur
the runner's shared preparation, which must be included in an end-to-end study.

### Incremental information-set experiment

`ris-incremental` reuses the systematic basis through random pivot exchanges,
with a fresh six-pivot reduction every 64 scored bases. Eight exchange proposals
precede each intervening scan. Set `--restart-interval` and
`--exchange-proposals` to explore other schedules. The default remains fresh RIS;
correlated incremental samples must not be counted as independent trials.
Every returned worker improvement follows the same validation and persistence
path as the baseline.

Build dist-m4ri with `RIS_NATIVE=1` when building the native engines with host
optimization. Its public CLI exports witnesses only at exit. The current adapter
reserves up to 50 ms per side from the native timeout for startup/export and
records actual process exit against the full side deadline. Late exports remain
in the evidence but receive no timely credit.

Run the frozen six-code, ten-seed comparison at 1/10 seconds and 1/4 workers:

```sh
.venv-benchmark/bin/python benchmarks/distance/study_incremental.py
```

This uses seeds 300–309, separate from pilot seeds 200–201. It compares `cpp`,
`ris-block6`, `ris-incremental`, and pinned `m4ri`, serially with fixed CPU affinity.
It exports each completed configuration and generates a cross-study summary.
An existing complete raw study is checked and exported; a partial study fails
closed rather than overwriting evidence.

The timed C++ adapter hashes `(study seed, batch index)` into a batch seed using
BLAKE2b. The earlier additive mapping reused batch streams between adjacent study
seeds and must not be used for independent-seed recovery intervals. The study
runner detects legacy controls, reruns only those controls, and retains the old
and replacement evidence separately. `repair_incremental_control.py` can also
perform that correction directly. Corrected one-worker controls in the retained
study were measured separately; their reports explicitly disclose the timing
separation. The standalone RIS and dist-m4ri streams were unaffected.
## Strategy prototypes and initialization baseline

The [strategy study](results/strategy-study/README.md) compares guided elite bases,
stabilizer refinement, and connected-region column ordering against the existing
native controls. The [isolated prototypes](strategy_prototypes/README.md) have
their own builds and tests; `study_strategies.py` runs their shared comparison.

The main practical finding is that canonical logical-basis witnesses already
give weight 60 on the fresh 960-qubit bicycle case and weight 20 on toric-1000.
Future comparisons should include these cheap initial bounds. The study separates
warm-search results from this initialization issue and includes exact seed replays
to identify improvements attributable to stabilizer refinement.

The follow-up [initialized study](results/initialized-study/README.md) charges
preparation and common logical initialization to 10- and 60-second budgets.
Selective refinement tied guided search's final code-level weight in all 27
matched comparisons. A separate matrix-only cyclic-shift pass recovered the
hard 700-qubit case's weight-28 witness in about 45 ms, but depends on its visible
coordinate layout. Finite polynomial-division proposals did not improve the
hard cases. The report includes relabeling controls, exact sector-symmetry checks,
audited witnesses, and directions for broader algebraic and decoder-based search.

The [refreshed external comparison](results/external-refresh/README.md) measures
incremental, guided, dist-m4ri and QDistEvol on those seven cases at 10 and 60
seconds/code, with three fresh seeds, one worker and common initialization inside
the deadline. Across 42 matched outcomes, guided beats dist-m4ri 25 times and
QDistEvol 29 times, tying the rest. Dist-m4ri matches guided's Tanner median at
60 seconds. Both native methods still miss the hard reference targets. The
168-run grid and all 7,142 saved witness documents passed their evidence audits.

The [three-candidate follow-up](results/candidate-study-v2/README.md) tests matrix
structure recovery, reduced algebraic-space search, and augmented-syndrome BP+OSD
against native controls and the existing circulant search. Its 198 frozen main
configurations include 10/60-second budgets and independent relabeling controls.
A separate 18-run routing diagnostic restores weight 28 on column-relabeled
Board700 and weight 48 on column-relabeled Regression690 at two seconds/code.
Row shuffling still defeats recovery. The coupled-space search is uncompetitive
as configured, and exact membership checks identify both coverage and search
limitations. All 21,263 saved witness documents passed their audits; 33 synthetic
unit tests passed. The report gives candidate/fallback attribution, decoder costs,
source snapshots, and next experiments.

The [bounded dispatcher](results/dispatch-study/README.md) adds a native C++
partition detector, a short restricted-search pilot, and a guided time reserve.
Across 212 configurations at 2/10 seconds, it preserves the 700/690 structural
gains under fresh column permutations and ties guided in all 29 matched outcomes
with rejected partitions. Detection stays below 11 ms in this study. Its current
pilot still over-allocates time on Board682, so it remains an explicit experimental
choice. The [API and build guide](../../native/structure_dispatch/README.md),
29 passing tests, and audits of 13,519 saved documents accompany the results.

The [competing-session allocation study](results/allocation-study/README.md) adds resumable guided and structured pilots with conservative 80/20 allocation. Across 180 configurations, the new policy produced 18 lower weights, 42 ties and no higher weights against guided; against the previous dispatcher it won four and lost three paired outcomes. It preserves the 700/690 structural gains and matches guided on the 682-qubit case. No preference switches occurred after the pilots, so repeated reassessment has no demonstrated benefit yet. All 46 tests passed and 14,532 saved witness documents were audited. See the [API and reproduction notes](results/allocation-study/IMPLEMENTATION.md).

The [bounded large-code experiment](results/large-structure-study/README.md) uses
three submitted 684–700-qubit codes and construction details from their original
PRs. Product-block and torus-strip searches only match the controls. On the
affine 684-10-101 input, subgroup restrictions find weight 90 in all three
2-second runs; guided and dispatcher controls finish at 108, 109 and 110 with
20 seconds. A fixed ten-dimensional restriction reproduces 90 by exhaustive
enumeration in about 27 ms including setup, but misses both sibling-code targets.
The expansion stops there. The report preserves the new witness, 54 main
configurations, three deterministic checks, 46 passing tests, and the supplemental
audit needed to reconcile harness/engine clock origins in one torus control.

The [full-block/collision follow-up](results/block-collision-study/README.md)
removes the orbit constraint and compares full/block kernels with ordinary pair
scoring and a native four-row Stern join. Both block methods return weights
90, 51 and 54 in all three seeds at both 2 and 10 seconds/code. The sibling bounds
improve from reference values 70 and 81; refreshed guided returns 72 and 63.
Collision matching adds no final-bound gain over block pairs on this corpus.
Matrix diagnostics localize the sibling winners to 171-qubit components with
19-dimensional kernels, making component enumeration the next useful probe.
The study includes 93 configurations, 18 passing synthetic tests, six portable
sector witnesses, and audits of all 4,613 saved documents. These remain witnessed
upper bounds; neither full-code exactness nor thousand-qubit scaling is established.

The [component enumeration and cross-family probe](results/component-study/README.md)
exhausts the affine siblings' dimension-19 components in about 60 ms/code,
confirming restricted minima 51 and 54. Construction-assisted search retains those
gains, while the existing automatic detector misses their partitions. Across four
unrelated large inputs, neither component variant improves guided at 2/10 seconds.
Coverage limits matter: dimension-85/91/140 spaces exceed the fixed 64 cutoff,
and individual lifted-product blocks contain no logicals. The retained small
balanced-product spaces have exact minima 48/56, above guided's 22. Joint row/column
permutations produce no automatic-search gain. The frozen continuation criterion
fails; retain the affine specialization and pause broad expansion of this
configuration. All 122 configurations and 6,724 saved documents were audited;
27 tests passed. These results do not rule out larger spaces or different partitions.

The [algebraic diagnostic](results/algebra-diagnostic/README.md) tests exact
coordinate decompositions, binary rank separators and cyclic module structure
without running a distance search. Both full kernels of each of the five non-surface inputs are
coordinate-indecomposable; the only hidden groups in documented restrictions
are forced-zero coordinates already compacted by the native engine. Tested
balanced cuts have coupling ranks 136–170 on the hard inputs, while the local
surface components have small sweep widths 7–8. Exact polynomial descriptions
are verified for the 682-qubit bicycle and the 664-qubit lifted-product input.
These algebraic factors overlap on physical coordinates and are not independent
minimum-weight searches. Eleven tests and the independent certificate audit pass;
all 1,174 prior artifacts remain unchanged. The report gives complete profiles,
rank/factor certificates and the limits of the resulting search opportunities.
