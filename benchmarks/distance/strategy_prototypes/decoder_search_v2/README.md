# Augmented-syndrome decoder candidate

`adapter.py` implements an independent candidate generator, with the same
six-argument `search(own, opposite, duals, seconds, seed, emit)` interface as the
other new experiments. The caller supplies prepared logical detectors and owns
trusted witness validation, persistence and common initialization.

For an X search, `own` is H_X, `opposite` is H_Z, and `duals` is a basis of Z
logical operators. Each trial chooses uniformly random nonzero coefficients in
that logical basis to form a detector l. It asks BP+OSD to solve

```
H_Z x = 0
  l x = 1
```

A successful result therefore has zero stabilizer syndrome and nonzero logical
parity. The construction for Z is identical with sectors exchanged. No error
sample, residual of two decodings, reference witness, target weight, or family
metadata enters this search. This follows the syndrome-decoder construction in
[Webster, Jacob and Higgott, Section 4.2.5](https://arxiv.org/html/2603.22532v1#S4.SS2.SSS5).

## Frozen screening settings

The native decoder is installed `ldpc==2.4.1`, `BpOsdDecoder`. The fixed settings
are product-sum BP, 100 maximum BP iterations, parallel message schedule,
`omp_thread_count=1`, and combination-sweep `OSD_CS` order 1. These BP/OSD settings
match `bposdDecode` in the local codeDistance source pinned at
`a4afe9c09bbf5790da9ecc05b65c5b62343979ad`.

Our sampling variant additionally randomizes column order and perturbs the
channel log odds independently each trial. A base error probability 0.05 gives
log odds log(19); independent uniform noise in [-0.25, 0.25] makes the per-qubit
probabilities approximately 0.039 to 0.063. All probabilities still favor zero.
The decoder consequently optimizes slightly perturbed costs, while the caller
scores the emitted vector by its actual Hamming weight. No corpus measurements
were used to choose these parameters.

The small perturbation is deliberate: with equal probabilities and parallel
messages, a toy Steane code could converge to its weight-seven logical without
running OSD. A column permutation alone cannot break this BP symmetry. Jitter
provides different message reliabilities and OSD tie orderings; it does not
promise to eliminate that failure mode. This is a controlled variant, not an
exact reproduction of the paper's benchmark configuration.

We do not add stabilizers to the logical detector in this first screen. Such
addition preserves the requested affine solution set, but can make the extra
check denser; we first test the simpler random-logical construction. Random
nonzero coefficients usually already produce a dense extra check, which may be
unfavorable for BP. The uniform coefficient distribution is over the supplied
independent logical basis, not over all physical representatives.

## Timing, output and limitations

The timer begins on entry to `search`, before binary-matrix conversion, augmented
matrix allocation, decoder construction, randomization and all decodes. Module
imports may occur before that clock. Optional `warmup()` performs one synthetic
three-qubit repetition-code decode; it does not read corpus data. The parent
harness charges native preparation and common initialization before passing the
remaining sector budget here. It must also establish numerical library thread
limits before imports and CPU affinity for a matched single-CPU comparison.

A trial contains at most 100 BP iterations plus finite order-one OSD work. The
native call has no wall-time interrupt: a final call can overrun the deadline.
Its returned valid support is still emitted, and the parent must retain it with
its actual timestamp and give late delivery no budget credit. Counters report
construction time, total and maximum decoder time, maximum whole-trial time,
trials, successful/failed decodes, BP convergence, unique exports, duplicates,
late exports, and best exported weight (including late output).

Every distinct valid decoded vector is exported immediately with stage
`decoder`, including ties, worse weights and a late final result. Exact duplicate
supports within a sector run are counted but do not produce additional exports.
The callback is synchronous and its exceptions propagate. Failed decodes have no
valid logical support to export and are counted. Runtime checks reject nonbinary
outputs and check both the selected augmented syndrome and supplied logical
parity. These are guardrails; the parent's trusted validator remains authoritative.

BP convergence is not optimality. OSD is used only when BP does not converge, as
implemented by the library. High-weight or failed outputs do not imply that a
code has large distance. This experiment yields witnessed upper bounds only.

## Tests

Run from the repository root:

```
.venv-benchmark/bin/python -m pytest -q benchmarks/distance/strategy_prototypes/decoder_search_v2/test_adapter.py
```

Six toy tests cover actual repetition/Steane decoding, qubit relabeling and
inverse permutation, invalid decode accounting, late-result preservation,
propagation of save failures, empty logical spaces, zero budget, and JSON-safe
counters. They do not run on or discard witnesses from the benchmark corpus.
