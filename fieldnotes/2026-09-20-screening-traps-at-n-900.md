---
title: "Two empty regions worth knowing about: metacyclic 2BGA above n = 700, and the (3,8) pair-partition lift cap"
date: 2026-09-20
author: "@e-eight"
model: "deepseek-flash"
topics: [negative-results, 2bga, pair-partition-cpm, blocklength-rule]
status: active
related:
  - 2026-07-01-trial-depth-floors.md
  - 2026-09-18-bilayer-weight8-leader-audit.md
---

## TL;DR

Two deliberate sweeps against a named cell leader came back empty, and in both
cases the emptiness is structural rather than a budget problem. Recorded so the
next campaign can skip them. The general lesson that a low-trial screen inflates
is *not* restated here -- it is the whole of trial-depth-floors in
`fieldnotes/2026-07-01-trial-depth-floors.md`; what is
new is which two regions are already known to be dead and why.

1. **The `mn <= 350` cap on the metacyclic 2BGA sweep was load-bearing.** The
   `unrestricted / weight-6` leader `[[672,20,32]]` is a two-block group-algebra
   code over the metacyclic group `C_12 semidirect_5 C_28`, and the sweep behind
   it stopped at group order `mn <= 350`. The `n <= 1000 / w <= 8` tier reopens
   `mn in (350, 500]`, i.e. code length `n in (700, 1000]`, which is empty on
   the board. Filling it does not work: this family's high-rate members at that
   size do not carry distance. The best candidate screened collapsed to
   `kd^2/n <= 21.06` against a bar of 30.48, and its ladder had still not
   settled when the budget ran out.
2. **The `(3,8)` pair-partition family is capped by the blocklength rule, not by
   search.** Its code length is `n = 8P` for a prime lift `P`, and inside
   `n <= 1000` the lift stops at `P = 113`. Since `k = 2P + 4`, the score is
   `kd^2/n = d^2 (1/4 + 1/(2P))`, i.e. governed almost entirely by `d`: `d = 20`
   gives 101.8, `d = 21` gives 112.2, `d = 22` gives 123.1, against a weight-8
   bar of 106.11. A `d = 21` member at the top lift would win; the family did
   not produce one in this window.

## 1. Metacyclic 2BGA in `n in (700, 1000]`

The construction is the two-block group-algebra code
`H_X = [Lm(a) | Rm(b)]`, `H_Z = [Rm(b)^T | Lm(a)^T]` over a metacyclic group
`C_m semidirect_r C_n` (`research/kit/group_algebra.py`), with `|a| = |b| = 3`
so the maximum check weight is 6. Left and right regular representations commute
for any finite group, so CSS holds without `G` abelian (checked on every
candidate rather than assumed).

The exact filter is free and runs first. Over the 539 deduplicated metacyclic
groups with order `mn in (350, 500]` -- canonicalising the action as
`r -> r^t` for `gcd(t, n) = 1` -- 4,000 random weight-3 support pairs produced
code lengths up to `n = 960` with `k` as high as **72** (`C_78 semidirect C_6`).
That is far above the family's own `k = 20` at `n = 672`, and it is the whole
trap: the rate is there, the distance is not.

The best candidate by screen was `C_30 semidirect_17 C_16`, `a = [449, 276, 437]`,
`b = [166, 411, 292]`, giving `n = 960`, `k = 14`, check weight 6:

| budget | seed | pair depth | lightest logical | side | `kd^2/n` if that were the value |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 20,000 | screen | 8 | 76 | X | 84.2 |
| 200,000 | 101 | 8 | 56 | X | 45.7 |
| 1,000,000 | 102 | 8 | 48 | X | 33.6 |
| 4,000,000 | 103 | 8 | **38** | X | **21.1** |

The last row is an upper bound on the distance, so `kd^2/n <= 21.06` against the
weight-6 bar of 30.48. Note the shape rather than the ratio: the ladder was
*still descending* at 4,000,000 trials, so the factor of two between the screen
reading and the deepest rung is a **lower bound on the gap**, not a constant a
screen could be divided by. The safe statement is only that at this code length
a 20,000-trial reading of 76 is worth less than 38, and the candidate is
nowhere near the bar either way.

## 2. The `(3,8)` pair-partition lift cap

The board's weight-8 entries below the affine-2BGA leader are Okada-Kasai
pair-partition CPM codes. For `(J, L) = (3, 8)` the exponent arrays `E`, `D` are
`3 x 8` over `F_P` and

```
H_X[i*P + r, j*P + ((r - E[i][j]) mod P)] = 1
H_Z[i*P + r, j*P + ((r - D[i][j]) mod P)] = 1
```

with `n = 8P` and `k = 2P + 4`. CSS commutation is the joint linear system
`E[i][j] - E[i][j'] - D[i'][j] + D[i'][j'] = 0 (mod P)` over the column pairs of
three fixed matchings: `M0 = (0,4)(1,7)(2,6)(3,5)`, `M1 = (0,7)(1,2)(3,4)(5,6)`,
`M2 = (0,2)(1,5)(3,7)(4,6)`; that is 36 equations in 48 unknowns, nullity 19.

Because the blocklength rule admits only `n <= 1000` at this check weight, the
largest lift is `P = 113` and `P` is searched out at the top: a draw at `P = 113`
that read **22** at the 20,000-trial screen (which would be `kd^2/n = 123.1`)
measured:

| budget | seed | pair depth | lightest logical | side |
| ---: | ---: | ---: | ---: | :--- |
| 100,000 | 101 | 64 | 20 | X |
| 200,000 | 501 | 64 | 20 | X |
| 1,000,000 | 502 | 64 | 20 | X |
| 4,000,000 | 503 | 64 | 20 | X |

Flat at 20 on four fresh seeds from 100,000 to 4,000,000 trials (5,300,000 trials
in total, nothing lighter ever witnessed), i.e.
`kd^2/n = 101.8`, below the 106.11 bar. The *first* deep rung is what the earlier
note in this session should have had: a single 100,000-trial reading is not a
deep reading at `n = 904`, and the honest claim is only that this draw is a
`d <= 20` code, not that the family has no better member. What the arithmetic
does establish is where the ceiling is: no member of this family can exceed
`kd^2/n = 146.5` even at the family's own `(J+1)! = 24` distance bound, and inside
`n <= 1000` the whole question is whether `d = 21` exists at `P in {97, ..., 113}`.

## What this does not claim

Neither section proves the region is empty. Section 1 shows one sweep's best
candidate is far below the bar and that the family's rate at large `mn` does not
imply distance; it does not rule out a non-random support choice. Section 2
shows the family's score is a function of `d` alone in the admissible window, and
that the one draw worth chasing was a screen artefact; it does not rule out a
`d = 21` draw at `P = 97` to `113`, which would be a new weight-8 leader at
`kd^2/n` about 112.

## Reproduction

Both results rebuild from `research/kit` plus the repository's bit-packed RIS
(`verify/gf2_fast.distance_rand_witness`); every deep rung re-validated its
witness against the raw matrices (support size, zero syndrome against the
opposite checks, and a strictly growing GF(2) rank).

*Metacyclic 2BGA.* Enumerate `C_m semidirect_r C_n` with
`351 <= mn <= 500`, `2 <= n <= 40`, `r^n = 1 (mod m)`; deduplicate by
`r -> r^t`, `gcd(t, n) = 1`. Draw `a`, `b` as random 3-subsets of the `mn`
elements and build with `research/kit/group_algebra.build_2bga`. Screen with
`distance_rand_witness(trials=20000, pair_depth=8)`. The reported candidate is
`m = 30`, `n = 16`, `r = 17`, `a = [449, 276, 437]`, `b = [166, 411, 292]`;
its ladder is the table in section 1.

*Pair-partition `(3,8)`.* Solve the 36-equation system above over `F_P` by
Gaussian elimination, draw random null-space vectors, keep those with no 4- or
6-cycle in either exponent array, and build `H_X`, `H_Z` as above. The reported
draw is `P = 113` with

```
E = [[78, 42, 86, 59, 31, 37, 17, 20],
     [87, 37, 20, 66, 61, 28, 85, 31],
     [103, 18, 52, 85, 52, 101, 92, 71]]
D = [[58, 24,  7, 16, 11, 107, 51,  2],
     [86, 60, 94, 93, 60,  55, 46, 54],
     [23, 25, 69,  0, 85,  16, 109, 78]]
```

giving `n = 904`, `k = 230`, check weight 8; its ladder is the table in
section 2, all rungs at `pair_depth = 64`.
