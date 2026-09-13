# Search-strategy prototypes

Three isolated experiments use the existing packed RIS core or independent native
search code. They are benchmark candidates, not changes to the production verifier
or to the established `native/ris` API.

| Candidate | Search idea | Main question |
|---|---|---|
| [guided](guided/README.md) | Four elite bases, local exchanges, periodic fresh immigrants | Does selecting promising bases repay fitness/copy costs? |
| [descent](descent/README.md) | Independent RIS seeds plus sparse stabilizer tabu moves | Can valid witnesses be shortened cheaply within their logical class? |
| [structure](structure/README.md) | Connected-region column elimination | Does local column ordering expose useful low-weight dependencies? |

Each directory has a separate extension, adapter, and mathematical unit tests.
The common protocol is `prepare(own, opposite).run(seconds, seed, emit)`, where
`emit(weight, support)` hands each exported witness to the benchmark for immediate
recording and subsequent trusted validation/persistence. Search routines receive
no reference witnesses. Matrix preparation is measured separately from search.

Build each extension with the benchmark environment, from its own directory:

```sh
../../../../.venv-benchmark/bin/python setup.py build_ext --inplace
```

Run the [predeclared pilot](PLAN.md) from the repository root:

```sh
.venv-benchmark/bin/python benchmarks/distance/study_strategies.py \
  --output benchmarks/distance/results/strategy-study/screen-2s
.venv-benchmark/bin/python benchmarks/distance/report_strategies.py \
  benchmarks/distance/results/strategy-study/screen-2s
```

The runner requires a new output directory and never overwrites an earlier study.
Timed methods run serially on one pinned CPU; validation uses separate processes
between searches. Every distinct-per-side/run exported support is packaged with
the research kit. This can take substantially longer than search, especially for
descent's many independent seed-session improvement histories.

The source archives pin the measured code; subsequent prototype development must
use a new result directory. Neither ordinary best weights nor repeated-seed
agreement establishes exact distance or full-gate acceptance.
