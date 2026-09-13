# Stabilizer descent prototype

This isolated candidate tests whether a short local search can turn an ordinary
RIS witness into a better representative of the same logical class. It changes
neither the trusted verifier nor the reusable native RIS engine.

For every move, `word ^= own[row]`. Commuting checks therefore preserve zero
syndrome, and adding an own stabilizer preserves the logical class. The adapter
only starts from cold incremental-RIS witnesses obtained within its run budget;
it never initializes from a known/reference witness.

The C++ implementation stores sparse row/qubit incidence lists and maintains
the exact weight change of every stabilizer flip. Flipping a qubit updates just
the gains of rows incident on that qubit. The search chooses a smallest-gain row,
breaking ties randomly, with a seven-step tabu tenure (overridden by an incumbent
improvement). It permits plateau and uphill moves within weight eight of the
incumbent. After 64 unsuccessful steps or exhausting admissible moves it returns
to the incumbent, flips one to three random rows, and resumes descent. All state,
gain, tabu and packed improvement buffers are allocated before the move loop.
This is a heuristic, not an exact coset minimizer.

## Frozen first-pass configuration

Each cycle creates an independently seeded RIS session and scores 128 bases,
using six-pivot elimination, a fresh reduction every 64 bases, and eight exchange
proposals. It then spends up to 5 ms refining that cycle's best witness. Thus
independent cycles can explore different logical classes. RIS work and session
creation are included in the total elapsed budget. Preparation packs/validates
the checks, constructs the kernel and logical quotient through existing native
RIS, and builds the descent incidence lists outside the warm search timer.

The fixed 128-base seeding call may cross a deadline; every returned improvement
is still emitted, and the parent harness must retain it without giving late
delivery credit. Local descent checks its deadline before every step and exports
its complete strict-improvement history at the end of the <=5 ms slice.

`prepare(own, opposite).run(seconds, seed, emit)` returns JSON-serializable
counters. `descent_improvements` counts improvements of each local seed;
`global_descent_improvements` counts those that also improve the run's global
incumbent. All seed-session and local strict improvements are emitted, including
those heavier than a different session's best. The parent harness is responsible
for independent witness validation and durable persistence of every emission.

Build from this directory with:

```
../../../../.venv-benchmark/bin/python setup.py build_ext --inplace
```

The prototype is host-optimized (`-O3 -march=native`) like the measured RIS build.
Its toy tests cover incremental gain/weight consistency after every move,
syndrome and coset preservation, dependent rows, 64-bit storage boundaries,
strict improvement ordering, reproducibility with fixed steps, empty checks,
invalid supports, zero time, and cold adapter witness validity. They perform no
corpus witness searches. Run from the repository root:

```
.venv-benchmark/bin/python -m pytest -q benchmarks/distance/strategy_prototypes/descent/test_descent.py
```

The scientific comparison is the full cold hybrid against RIS with the same
elapsed budget. A local improvement alone is insufficient evidence of a useful
strategy: it must compensate for the RIS exploration time it consumes. The
fixed configuration above was selected before corpus evaluation.
