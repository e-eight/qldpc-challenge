# Native RIS experiment

A standalone C++17 random-information-set engine with thin pybind11 bindings.
The core does not depend on Python, NumPy, M4RI, or the qLDPC verifier. The Python
interface accepts NumPy binary matrices. This directory can be extracted into a
separate repository; repository-specific benchmark adapters live outside it.
The Python package has only NumPy as a runtime dependency.

For an X search, the engine seeks a small support in `ker(H_Z)` outside
`rowspace(H_X)`. Swap the matrices for Z. A returned support gives a distance
upper bound; failure to find a smaller support is not a distance certificate.

## Build

From this directory, using a Python environment with setuptools and pybind11:

```sh
python setup.py build_ext --inplace --force
```

The default build uses portable compiler settings. For a local CPU experiment:

```sh
RIS_NATIVE=1 python setup.py build_ext --inplace --force
```

This enables `-O3 -march=native`; such binaries may not run on other CPUs. Build
the baseline with the same flags when comparing speed. Current build scripts
target GCC/Clang on Linux or macOS. The compiler vectorizes the packed loops;
there is no handwritten ISA dispatch layer in this version.

The directory also supports ordinary `pip install .` / wheel builds using its
`pyproject.toml`. Source distributions include the C++ headers, core tests, and
Apache-2.0 license.

To build and test the C++ library without Python:

```sh
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

The core test replaces allocation functions and aborts if a trial allocates.
It covers ordinary, masked, and blocked elimination, word boundaries, and
worker lifecycle. Independent matrix and witness checks against the repository
verifier are in `benchmarks/distance/test_ris_native.py` in the enclosing repo.

## Python interface

`Prepared(own, opposite)` validates binary commuting checks, packs them, and
constructs a kernel basis and opposite logical representatives. Input dtypes
are `int8`, `uint8`, or `bool`; arbitrary NumPy strides are supported. Inputs are
copied. The `kernel` and `logicals` properties return copies for diagnostics.

`Session(prepared, threads=1, seed=0, pair_depth=8, masked=False, block_size=1,
restart_interval=0, exchange_proposals=8)`
owns the worker state. `advance(trials)` performs exactly that many total trials
across the workers and returns:

- `trials`: completed work in this batch (zero for codes with no logical qubits).
- `best_weight`: best weight across the session; `n + 1` if no logical was found.
- `improvements`: every new strict improvement per worker, with its support,
  weight, worker index, and cumulative worker trial number.
- `reductions`, `proposals`, `exchanges`: cumulative session counts of fresh
  reductions, proposed swaps, and accepted swaps, respectively.

The binding releases the GIL during preparation and search. Sessions retain
their prepared data. Overlapping calls to one session are rejected; independent
sessions may run concurrently. Worker RNG streams and best witnesses persist
across batches. The division of trial budgets between workers must match for
split and unsplit calls to visit identical trials. Batches divisible by the
worker count satisfy this condition.

The interface has no built-in hard wall-clock deadline. Use bounded batches and
record delivery time. The benchmark adapter targets approximately 50 ms batches,
retains late witnesses, and gives them no credit before their arrival. Cancellation
is currently between calls. Consumers must validate and save returned supports;
the repository adapters use the trusted GF(2) routines and the kit's
`make_submission` / `save_submission` path.

## Implementation

Each worker owns a tightly packed row-major matrix, row pointers, permutation,
weights, partial-sort scratch, RNG, and preallocated improvement storage. A trial
copies the immutable kernel into scratch and swaps row pointers during reduction.
Row width is `ceil(n/64)` words; padding bits remain zero. There are no allocations
or locks inside a trial. Native threads persist for the session lifetime; a
one-worker session runs directly on the caller's thread.

The initial search policy matches the existing C++ RIS: xoshiro256** with unbiased
Fisher-Yates column ordering, all reduced single rows, and pairs among the eight
lightest rows. `pair_depth=0` disables pairs. Trials on each worker have the same
seed mapping as the benchmark baseline. Persisting the state across batches is
different from the old time-budget adapter's reseeding on each batch.

Three elimination kernels are available:

- `block_size=1, masked=False`: conditional row XOR, the simple baseline.
- `block_size=1, masked=True`: masked row XOR even for an unset pivot bit.
- `block_size=2..8`: deferred panel elimination and a table of pivot-row XORs.
  Four- and six-pivot configurations are exposed in the benchmark runner.

Blocked elimination preserves the pivot visitation order and reduced basis. It
uses the standard table-based GF(2) elimination idea described in the
[M4RI algorithm documentation](https://malb.bitbucket.io/m4ri/brilliantrussian_8h.html),
with randomized pivot columns and independently implemented panel selection.
A six-pivot table uses 8 KiB at 1,000 columns. A 600-row kernel uses 75 KiB at that
width; original and working matrices are separate. `basis_bytes` and
`workspace_bytes` report matrix and scratch payloads, including witness storage;
they exclude allocator overhead, thread stacks, Python objects, and transient
preparation storage. They are not peak RSS measurements.

Witness storage reserves up to `n` strict improvements per worker, each with a
packed support. It is bounded but costs `n * ceil(n/64) * 8` bytes per worker.
This is deliberate evidence retention, and worth revisiting at much larger sizes.

With `restart_interval > 0`, each worker reuses its reduced basis between fresh
random-permutation restarts. Each intervening sample proposes `exchange_proposals`
uniform row/nonpivot-column swaps. A zero matrix entry rejects the proposal;
a one permits row XORs that exchange the pivot while preserving the row space.
Rejecting zero entries keeps the proposal symmetric. This does not establish
independence or rapid mixing: samples between restarts are correlated, and their
count must not be interpreted as independent RIS trials. A full-rank kernel with
no nonpivot columns falls back to fresh reductions. The default disables reuse.

The candidate scan remains single rows plus pairs of light rows. Benchmark
`ris-incremental` uses six-pivot restarts, initially every 64 samples with eight
proposals between scans; both settings are configurable. Family-specific search
remains a future experiment. The existing verifier remains responsible for
accepting witnesses.
