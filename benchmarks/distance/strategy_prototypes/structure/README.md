# Connected-region column search prototype

This tests whether ordering qubits by randomly growing a connected region makes
low-weight logical operators easier to expose. It is a heuristic upper-bound
search, with no completeness or distance-certification claim.

Two qubits are neighbors if they share an opposite-type check. Starting from a
uniform random qubit, choose the next qubit uniformly from the discovered
frontier and insert its opposite-check syndrome column into a packed binary
elimination basis. A dependent insertion gives an exact zero-syndrome relation
among already inserted qubits. Score that relation and reject stabilizers by
testing parity against the independently tested native Prepared logical duals.
Every strict improving logical is emitted through the parent harness, which
validates and saves witnesses through the repository submission kit.

Region caps cycle through floor(n/4), floor(n/2), floor(3n/4), and n (each at
least one). Each attempt uses a fresh random seed qubit and frontier choices.
If an entire graph component is exhausted, choose another unused component.
This configuration is frozen before corpus evaluation. Full-size attempts give
a general column-order fallback; localized attempts test the structural idea.

The mathematical motivation is that every minimum nontrivial logical has a
connected representative in this graph: components of a zero-syndrome support
separately have zero syndrome, and not all can be stabilizers if their sum is
nontrivial. The prototype does not enumerate all connected supports, and its
emitted relation need not occupy the entire connected region or itself be
connected. Expansion can make the frontier rapidly lose spatial information.
It only scores the generated kernel-basis relations, not their combinations.
In particular, the existence of a short logical inside the region does not
guarantee this elimination basis will expose it.

Preparation constructs adjacency and reusable packed scratch storage. Search
has one worker, releases the GIL, checks its deadline before each insertion,
and allocates only for emitted improvements. Callback time counts toward the
budget. It returns JSON-compatible counters. An insertion may finish after
the deadline; parent scoring must censor late events while retaining them.

Build from this directory:

```
../../../../.venv-benchmark/bin/python setup.py build_ext --inplace
../../../../.venv-benchmark/bin/python -m pytest test_structure.py -q
```

The build explicitly uses the current host's instruction set. Portable ISA
dispatch is outside this prototype. Main API: `adapter.prepare(own, opposite)`
returns an object exposing `run(seconds, seed, emit)`.
