# Larger changes worth testing on the hard cases

This note separates local evidence, mathematical observations and untested
proposals. It does not claim an exact distance or a new validated code.

## Construction-derived candidate generation

The submitted 700-qubit matrix is a check-deletion child, according to its own
provenance in `codes/700-222-28.json`. Removing an independent X check leaves that
row commuting with every remaining Z check, but outside the remaining X rowspace.
It therefore supplies an X logical of the deleted row's weight. This is ordinary
CSS linear algebra; it does not require a new physical mechanism.

The matrix-only diagnostic `../orbit_completion.py` tries to reconstruct such
rows by shifting supplied checks within two equal coordinate blocks. It receives
no parent matrix, reference witness, target weight or construction metadata.
Every proposal must pass syndrome and logical-parity checks, and every emitted
witness is independently validated and saved by `../study_orbit_completion.py`.
Random qubit relabeling tests how much this relies on visible coordinate layout.
This is a specialized candidate generator; arbitrary code automorphism discovery
or recovery of a hidden parent is not implemented.

A second finite diagnostic, `../polynomial_completion.py`, takes the first
supplied check row as two polynomials a,b. For every simultaneous block rotation,
it proposes (a/f,b/f) for f = gcd(a,b) and f = gcd(a,b,x^L+1), then checks syndrome
and nontriviality on the actual matrices. This is motivated by the asymmetric
bicycle reduction in [Wang and Pryadko, Section III.1](https://arxiv.org/html/2203.17216#S3.SS1).
It does not claim to reproduce all logical families in that construction.

There is also a useful exact sector symmetry on four of the current inputs:
reversing all qubit columns maps the X check-row set to the Z check-row set for
board682, regression690, toric1000 and bicycle960. The direct matrix comparison
is recorded in `../results/initialized-study/symmetry-check.json`. Because this
permutation is an involution, it also maps Z checks back to X checks, and maps
logical witnesses between sectors. This can avoid duplicate X/Z search. It is
not used by the timed prototype; in particular, the measured bicycle Z-side
refinement gains are dominated by transferring the already available X witness.
The symmetry observation generates no new witness or distance claim by itself.

## Why a faster basis walk may still miss light logicals

For a systematic basis of the X zero-syndrome space, let I be its pivot columns.
The coefficients needed to express a logical x in this basis are exactly x on I.
Thus single-row and pair scoring can expose x only when its support intersects I
in at most two positions (and our pair search examines only eight light rows).
Changing the basis can help, but a target that needs many rows in most visited
bases is poorly served by additional near-identical samples.

The X kernel of board700 has dimension 456. A uniformly sampled 456-position set
would contain about 18 positions of a fixed weight-28 support on average. Actual
information sets are constrained by rank and guidance, and logicals have many
stabilizer representatives, so this observation is **not** a calibrated miss
probability or runtime estimate. It motivates changing candidate generation.

One engineering direction is higher-order information-set decoding: combine
more rows using partitioned lists and partial matching, rather than enumerating
every combination. This needs explicit list-size and cache budgets and is a
substantial algorithm change, not just enabling triples. Matching-bipartition
distance search and related enumeration methods are reviewed in
[Webster, Jacob and Higgott, Section 4](https://arxiv.org/html/2603.22532v1#S4).
Its value on our hard inputs remains unmeasured.

## Search for a nontrivial syndrome directly

Another independent experiment is to select a random Z logical l and decode the
augmented system H_Z x = 0, l dot x = 1, seeking small weight x. Any successful
solution is an X logical. A BP+OSD decoder can supply approximate solutions, with
randomized logical checks and coordinate permutations providing different trials.
This method is described in
[the same paper, Section 4.2.5](https://arxiv.org/html/2603.22532v1#S4.SS2.SSS5).
It is different from the repository's existing decoder-failure residual sampler.
Neither a decoder optimum nor a universal success probability is assumed.

## Priority after the initialized experiment

Use observed 2/10/30/60-second trajectories to decide whether further compute
buys useful progress. Keep cheap family-specific proposals when applicable.
If the guided/refinement combination stalls, prioritize an independent candidate
generator over further tuning of its small moves. SIMD, allocation reduction,
preparation sharing and skipping unchanged bases still matter, but should be
measured separately from changes to the search's ability to reach light logicals.
