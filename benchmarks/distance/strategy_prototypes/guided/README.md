# Guided basis search prototype

This isolated prototype copies the native packed RIS implementation, renames its C++ namespace and extension, and replaces the independent/incremental walk policy with four-parent basis hill climbing. It does not modify the production native core or verifier.

Frozen configuration: four parents, eight uniform row/nonpivot exchange proposals per offspring, a fresh random immigrant every 64 scored bases, six-pivot fresh elimination, and pairs among the eight lightest basis rows. The first four bases initialize the parents. An immigrant replaces the parent with the worst fitness, even if the immigrant is worse, to inject diversity. Other offspring choose a uniformly random parent, copy it, mutate, and replace that parent when fitness is no worse. Fitness is the lightest **nontrivial logical** found among the scored rows and pairs; a basis without such a candidate has fitness n+1. Stabilizer weight alone does not guide selection. Equal-fitness offspring allow movement across plateaus.

The current implementation still scores unchanged offspring when all proposals reject. Parents are independent slots, without deduplication or crossover. The algorithm is intentionally biased, and scored bases are not independent trials. This is a small guided-search experiment, not a reproduction of QDistEvol.

The archive costs four additional dense reduced bases per worker, plus pivot metadata. Offspring require one basis copy and accepted children another, and computing fitness requires more logical-class checks than incumbent-only scoring. This tests whether exploitation repays that overhead and reduced exploration; it makes no claim that throughput or target recovery must improve.

Build from this directory:

```sh
../../../../.venv-benchmark/bin/python setup.py build_ext --inplace
```

The build uses `-O3 -march=native` and a unique `ris_guided_native` module. `adapter.prepare(own, opposite)` performs matrix preparation once; `.run(seconds, seed, emit)` includes session allocation/random initialization in its budget. It uses one worker and adaptive batches targeting 2 ms, capped at 64 bases. Every strict improvement is passed to `emit(weight, support)`, including any returned after the deadline; the harness owns deadline credit and durable witness persistence. In-batch discovery timestamps are unavailable. One reduction can exceed a deadline; there is no native cancellation.

`Prepared.session(seed)` also exposes deterministic fixed-work runs for toy tests. The copied native constructor retains its general parameter surface for convenience, but only the frozen adapter configuration is evaluated. Search policy is guided regardless of the inherited `restart_interval` argument.

Validation:

- Five Python tests: witness syndrome and non-stabilizer checks, strict improvement ordering, and exact event/counter reproducibility across batch partitions at n=7,65,129; Steane minimum 3; no-logical and zero-budget behavior.
- C++ test checks the pivot identity and opposite-check kernel membership of every parent after each of 300 trials at n=7,65,129. Together these establish full rank and preservation of the intended kernel rowspace.
- C++ invariant test passed with AddressSanitizer and UndefinedBehaviorSanitizer; LeakSanitizer disabled in this environment.
- No corpus searches were run independently; the shared harness is responsible for preserving and independently validating their witnesses.

```sh
../../../../.venv-benchmark/bin/python -m pytest test_guided.py -q
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -Iinclude test_invariants.cpp -o /tmp/guided-invariants
ASAN_OPTIONS=detect_leaks=0 /tmp/guided-invariants
```
