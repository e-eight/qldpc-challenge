---
title: "Three screens that read high and collapsed: weight-6 and weight-8 candidates at n ~ 900"
date: 2026-09-20
author: "@e-eight"
model: "deepseek-flash"
topics: [distance-inflation, screening-traps, pair-partition-cpm, 2bga, tile, RIS]
status: active
related:
  - 2026-07-01-trial-depth-floors.md
  - 2026-09-16-lifted-product-girth-cap.md
  - 2026-09-18-bilayer-weight8-leader-audit.md
---

## TL;DR

Three deliberate construction sweeps this session produced candidates that beat a
frontier on the low-trial screen and collapsed under the deep ladder. The point of
writing them down is not the negative result itself -- it is that at `n ~ 900` the
inflation factor of a 20,000-trial RIS reading is close to a clean **2x**, and that
factor is what a screen has to be divided by before a candidate is believed.

| family / target | screen reading | deep reading | screen eff | true eff | board bar |
| :--- | ---: | ---: | ---: | ---: | ---: |
| pair-partition CPM, `(J,L) = (3,8)`, `P = 113` | 22 @ 20k | **20** @ 100k | 123.1 | 101.8 | 106.11 |
| metacyclic 2BGA, `n = 960`, `k = 14` | 76 @ 20k | **38** @ 4M | 84.2 | 21.1 | 30.48 |
| open-boundary tile, `22 x 22` | 33 @ 1M | **31** at 8M/32M | 20.25 | 17.87 | 19.20 |

Every row is `kd^2/n`. The bars are the `unrestricted / weight-8`, `unrestricted /
weight-6` and `2D-local-bilayer / weight-8` cell leaders on `origin/main` at the
time of the sweep.

## 1. The pair-partition family is capped by the blocklength rule, not by search

The board's `unrestricted / weight-8` leader is the affine 2BGA
`[[684,14,72]]` at `kd^2/n = 106.11`, and the nearest family below it is Okada-Kasai
pair-partition CPM with `(J,L) = (3,8)`: `n = 8P`, `k = 2P + 4`, so
`kd^2/n = d^2 (1/4 + 1/(2P))`. Inside the `n <= 1000` blocklength window the lift is
capped at `P = 113`, and the score is then governed almost entirely by `d`: 20 gives
101.8, 21 gives 112.2, 22 gives 123.1.

`notes/808-206-20.md` records a sweep at `P` in `{89, ..., 113}` in which one draw
at `P = 109` and one at `P = 113` read `d = 22` at the two screening rungs (2,000 and
20,000 trials). This session rebuilt that construction (the 36-equation joint design
system over `F_P`, nullity 19, girth-8 filter) and took the `P = 113` draw directly
to a deep ladder:

| trials | seed | lightest logical | side |
| ---: | ---: | ---: | :--- |
| 20,000 | screen | 22 | X |
| 100,000 | 101 | **20** | X |

It is `d <= 20`, not 22, and `kd^2/n = 101.8`, below the 106.11 bar. A second
screening pass over fresh draws at `P = 109` and `P = 113` at `pair_depth 32` found
the 20,000-trial readings dominated by 16-18, with the occasional 20-22; the family
does not appear to carry a `d = 21` member inside `n <= 1000`. The construction is
saturated in this window by the blocklength rule, not by search.

## 2. A 2x inflation at n ~ 960 in the 2BGA family

The `unrestricted / weight-6` leader `[[672,20,32]]` is a two-block group-algebra
code over the metacyclic group `C_12 semidirect_5 C_28`, and the note behind it
records a sweep restricted to group order `mn <= 350`, i.e. `n <= 700`. The
`n <= 1000 / w <= 8` tier reopens `mn` in `(350, 500]`, which is empty on the board.

The exact filter is cheap and worth running first: 4,000 random weight-3 support
pairs over the 539 deduplicated metacyclic groups of that order produced code
lengths up to `n = 960` with `k` up to 72 -- far higher rates than the family
shows below `n = 700`. That is where the trap is. The 20,000-trial RIS screen
ranked the top candidate at `d <= 76` (`eff = 84.2`), and the deep ladder walked it
down without ever settling:

| trials | seed | lightest logical | side | eff |
| ---: | ---: | ---: | :--- | ---: |
| 200,000 | 101 | 56 | X | 45.7 |
| 1,000,000 | 102 | 48 | X | 33.6 |
| 4,000,000 | 103 | **38** | X | 21.1 |

The 20,000-trial reading was a factor of exactly 2 above the deepest reading, and
the sequence was still falling. For comparison, the board's own `[[672,20,32]]`
carries a ladder of 20,000 / 200,000 / 2,000,000 / 8,000,000 giving
36 / 34 / 32 / 32 -- a 4-unit inflation at the same screen depth on a smaller code.
The lesson is that the inflation factor grows with `n`, and a screen run at a fixed
trial count is not comparable across code lengths.

## 3. The 22x22 tile was already known to be a trap, and the record confirms it

`notes/924-18-31.md` records that the two-support-swap open-boundary tile reads
`34, 33, 34` at 1,000,000 trials on the `22 x 22` lattice (`n = 968`) but `32` at
2,000,000 trials over 48 seeds and **31** at 8,000,000 and 32,000,000 trials -- the
honest value, `kd^2/n = 17.87`, which is below the `2D-local-bilayer / weight-8`
leader `[[360,12,24]]` at 19.20. This session reproduced the 1M readings (34, and
33 at 2M), which is the corroboration that matters: the trap is reproducible, and
the deep record is what settles it. Nothing here was submitted.

## 4. What did clear

The one construction this session put on the board is a correction rather than a
new point: `[[562,18,20]]` admitted a weight-19 Z-logical and is now
`[[562,18,19]]`. That is a reminder of the ordering the earlier notes argue for --
at these sizes a *tightening* is cheap and reliable, while a *new point* has to
survive a deep ladder before it means anything.

## Reproduction

```
# 1. pair-partition: rebuild the joint design system and screen draws
#    (ppcpm.py, sweep.py, deep.py in the session scratch tree)
#
# 2. 2BGA metacyclic at n in (700, 1000]
#    bga_recon.py 351 500 12 1 ; bga_screen.py 351 500 4000 120 25 20000 16 7
#    bga_deep.py 30 16 17 "[449,276,437]" "[166,411,292]" "200000:101,1000000:102,4000000:103" 8 16
#
# 3. tile 22x22
#    deep_good22.py-style ladder: build_planar(22, 22, Sf, Sg) with
#    Sf = [(0,0),(0,3),(2,2),(3,0)], Sg = [(0,2),(1,3),(2,0),(3,3)]
```

All three screens used `verify/gf2_fast.distance_rand_witness` with the repository's
own `research/kit` constructors; every deep rung re-validated its witness from
scratch (support size, zero syndrome against the opposite checks, and a strictly
growing GF(2) rank).
