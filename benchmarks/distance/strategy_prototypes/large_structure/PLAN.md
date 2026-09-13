# Large-code structural probe: bounded go/no-go experiment

The user requests a small largest-code corpus, construction/PR research, and a
stop unless a substantial advance appears. The submitted corpus has maximum
n=700 (593 local entries). Use three new hard cases, not the prior easy n~1000
initialization successes: 700-140-22 (ZSZ product), 700-6-32 (twisted torus), and
684-10-101 (affine group algebra). Their repository witnesses define recovery
targets 22,32,101; these are upper bounds, not independently certified exact
distances. PRs #368, #893 and #443 were read including all comment/review endpoints
(each has zero comments/reviews). PR #592/GALA is an excluded easy control.

Construction-assisted hypotheses, frozen before corpus witness search:
1. Product: search kernels on the four seed block pairs (0,1),(2,3),(0,2),(1,3)
   in the paper's five-block layout, reducing coordinate count from 700 to 280.
2. Twisted torus: enumerate short lattice vectors using the published lattice;
   build width-1..4 strips around the first six distinct directions, with at most
   60% of sites. Compute full zero-syndrome kernels inside each strip. This tests
   thick string representatives, not the previously failed shortest-path metric.
3. Affine: impose constant coefficients on left/right cyclic-subgroup orbits of
   orders 2,3,6,9,18,19 in the published group indexing. These are restricted
   ansatzes, not assumed quantum automorphisms. Actual check kernels and logical
   parity determine whether a space is useful. No claim of complete coverage.

Construction metadata is explicitly permitted here. No reference support, target,
case name or claimed distance enters Search; only matrices and structural specs.
The native wrapper builds ker(H_opposite E), expands into original coordinates,
and uses the existing RIS scoring core. All returned witnesses get retained by
the kit path. Common initialization is identical to controls; all preparation,
restriction construction, search and export count inside the budget. Up to 20%
of post-init budget for preparing spaces; structured search gets up to 80% of
post-init time, then native guided fallback. Inapplicable spaces yield immediately.
Round-robin space allocation; 20ms slices, 2ms batches, pairs8, one worker.

Controls: unchanged guided and competing dispatcher (race). Three seeds 1600–1602
at 2 seconds/code, then independent seeds 1610–1612 at 20 seconds/code.
3 cases x 3 methods x 3 seeds x 2 budgets =54 configurations,594 search seconds.
Timed configurations serial CPU0; numerical libraries single-threaded; equal X/Z
split. Validation/persistence between configurations. No source/policy tuning on
these runs, no growing the budget to salvage small gains.

Go criterion: actual structured (not initialization/fallback) target recovery in
all three 2s seeds on a hard case while BOTH controls recover it in at most one
of three 20s seeds; alternatively >=20% lower weight than both matched controls
in all three 2s seeds on at least two cases. This is a screening threshold, not a
statistical proof or a universal speedup. Report target recovery and censored
first-hit times. If neither criterion holds, stop this line after the audited
comparison; do not tune margins, grow seeds or add another heuristic.

No verify/, codes/, CI, publication or exact-distance claims. Preserve old measured
sources/binaries/results. Store all raw exported supports and stage every distinct
support through validate_and_stage -> make_submission/save_submission; save failure
is fatal. Only original in-cap board inputs are included.
