# Reduced polynomial spaces with bounded combination search

This prototype searches linear combinations of reduced generators, including a
coupled two-block family. The previous polynomial diagnostic scored individual
quotient generators only. This experiment changes both the search space and the
number of basis rows combined; it is not an isolated benchmark of either change.
It is a witness finder, with no completeness or lower-bound claim.

## Mathematical construction

Interpret the first supplied same-sector check as two length-L binary polynomials
`[a,b]`, in its supplied coordinate order. Set `f=gcd(a,b)`,
`h=gcd(f,x^L+1)`, and `g=(x^L+1)/h`. The three proposed spaces are the spans of
simultaneous cyclic rotations of:

- `reduced_both`: `[a/f,b/f]`;
- `reduced_single_left`: `[g,0]`;
- `reduced_single_right`: `[0,g]`.

For the standard circulant bicycle layout, the opposite sector has the
transpose/swapped check blocks. The quotient relation gives cancellation between
the two blocks, and `g` annihilates both original check polynomials modulo
`x^L+1`. These identities motivate the generators. We nevertheless test **every
proposed generator against the actual opposite matrix**, retaining only vectors
with zero syndrome. Consequently, every linear combination remains in that
actual kernel even if the input is a check-deletion child or has another layout.
The restricted spaces need not contain every original logical class or any
minimum-weight representative.

The polynomial motivation is Wang and Pryadko, Section III.1, especially
Equations (9), (11), and (12):
https://arxiv.org/html/2203.17216#S3.SS1 . Their complete decomposition also has a
Bezout-derived family. We do not implement that entire decomposition or claim
their exact distance identity for these three heuristic search spaces.

Logical nontriviality is checked using the supplied opposite logical detectors.
Each packed generator carries its full detector parity vector. Row elimination
and candidate XOR operations update that tag alongside the physical vector.
Only nonzero tags are exportable. Every export is checked again against the
actual syndrome and detector matrices; the common parent harness independently
validates and persists all exports through the repository's trusted path.

## Search and limits

Python performs exact polynomial arithmetic and row-independence reduction using
packed integers. An isolated C++17/pybind11 session then repeatedly constructs a
random information-set basis inside each restricted space. It scores basis rows
and runs a bounded beam search combining up to four rows. The shortlist has at
most 64 rows, preferring 48 logical rows and 16 trivial rows and filling unused
quotas. Separate beams retain at most 32 logical and 32 nonzero trivial states,
so short stabilizers cannot occupy every beam slot. This is a heuristic beam,
not exhaustive enumeration of all four-row combinations.

The frozen schedule is four two-block trials, one left-block trial, and one
right-block trial, skipping unavailable/trivial spaces. This is a **trial**
allocation, not a wall-time allocation: dimensions and runtimes differ. Counters
record time, trials, scored candidates, best weight, rank, and native workspace
separately for every branch. Emission stages identify the branch. Every branch's
strict improvement is exported even when another branch already found less.

Native loop buffers are allocated once; beam buffers have fixed capacity. The
combined physical and tag representation is limited to 64 uint64 words. There
are no threads, runtime compilation, unbounded lists, or external processes.
Reported `workspace_bytes` counts native matrices, index vectors, and beams;
it is not total RSS and excludes Python/NumPy storage and allocator overhead.
Python spaces and input matrices have dimensions bounded by the input size.

Timing starts on adapter entry and includes reduction, matrix checks, packing,
session creation, search, export, and post-export checks. One native trial is the
interrupt unit, so the parent must retain late exports without deadline credit.
The parent owns common initialization and charges it before calling this adapter.
There is no reference-weight stopping or target input.

The prototype requires visible two-block cyclic coordinates. It does not recover
hidden coordinates, use submission metadata, exploit X/Z witness transfer, or
invoke the independent structural prototype. Random relabelings are expected to
often produce `not_applicable`; valid accidental proposals remain allowed.
The single-block branches overlap the known family-specific approach and must
be reported separately from the novel coupled-space branch.

## Build and tests

From this directory:

```sh
../../../../.venv-benchmark/bin/python setup.py build_ext --inplace
```

From the repository root:

```sh
.venv-benchmark/bin/python -m pytest -q benchmarks/distance/strategy_prototypes/reduced_space_v2/test_reduced.py
```

The build uses `-O3 -march=native` and is machine-specific. No JIT warmup is
needed. Tests use synthetic matrices only: cancellation below an individual
quotient generator's weight, true syndrome/nontriviality, physical and logical
tag word boundaries, deterministic fixed-trial replay, invalid-layout safety,
zero budget, and storage limits. Corpus searches must run only through the
parent witness-preserving harness.
