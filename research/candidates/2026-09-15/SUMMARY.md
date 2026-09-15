# Autoresearch campaign 2026-09-15 — staged candidates (NOT submitted)

Six parallel family sweeps, 1500 candidates each, screened with the kit funnel
(`research/kit/search.py`, `backend="auto"`, 3000 trials), then every
non-dominated survivor deep re-verified with `verify/gf2_fast` before being
trusted. Board baseline: merged `origin/main`, 593 entries.

## The screening → deep collapse (measured, not assumed)

Screening `d` is an upper bound and it was badly inflated at 3000 fast trials.
Deep re-verification on fresh seeds:

| candidate | screen d | 300k | 2M | 4M |
|---|---|---|---|---|
| LP [[684,6]] (Z_3⋊Z_112) | 68 | 42 | – | 40 |
| LP [[672,6]] (Z_3⋊Z_112) | 60 | 44 | 42 | **40** |
| LP [[648,4]] (Z_3⋊Z_108) | 54 | 46 | **40** | – |
| Kasai [[684,4]] (Aff F_19) | 111 | 83 | – | – |
| Kasai [[684,4]] (Aff F_19) | 110 | **88** | – | – |
| metacyclic [[648,4]] | 56 | 38 | 36 | – |

Nothing screened at 3000 trials survived unchanged; the metacyclic one fell to
36 and became *dominated* by the board's [[630,6,36]].

## Staged survivors (matrices in this directory, `.npz`)

1. **[[672,6,40]]** — lifted product on metacyclic `Z_3 ⋊ Z_112` (order 336),
   `support_a = {128,211,257}`, `support_b = {92,130,269}`, max check weight 6.
   Deep: lightest logical 40 (X) at 4M×2 fresh seeds.
   The board's maximum distance in the weight-6 cell is **36**
   ([[630,6,36]]), so this is non-dominated and a new weight-6 distance record.
   File `final_lp672.npz`; fingerprint `670ecbf0cc5275d4`.
2. **[[648,4,40]]** — lifted product on metacyclic `Z_3 ⋊ Z_108` (order 324),
   `support_a = {4,123,250}`, `support_b = {64,210,238}`, weight 6.
   Deep: 40 at 2M×2. Non-dominated (40 > 36). `final_lp648.npz`,
   fingerprint `d16f3f5fd4f0648d`.
3. **[[684,4,88]]** — two-block group algebra on the affine group
   `Aff(F_19) = Z_19 ⋊ Z_18`, `support_a = {234,269,233,249}`,
   `support_b = {28,205,318,53}`, max check weight 8.
   Deep: lightest 88 (X) at 1M×2. The board's maximum distance among
   weight ≤ 8 entries is **81** ([[684,12,81]]), so non-dominated.
   `kasai_1a1cc63eb8604a10.npz`.

## Unstaged but checked

- **[[562,15,15]] w11**, 2D-local bilayer (planar tile, `build_planar(17,17,
  Sf=[[0,0],[1,3],[3,3],[1,0],[2,3],[2,1]], Sg=[[3,0],[2,2],[0,2]])`,
  interaction radius 5.0). Non-dominated in the nearly-empty
  `weight-9plus × local-2d-bilayer` cell (only [[20,8,4]] there). Distance
  screened at 40k trials only — **not** deep-verified yet.
- The dihedral weight-6 sweep produced no non-dominated code at all.

## Honesty caveats

- Every distance here is a **witness-backed upper bound**; none is exact.
- The readings were **still descending** with budget (40 at 4M after 42 at 2M),
  so they are not converged. A submission must claim the lightest value found
  and expect the gate's deeper refutation; the margin over the board (40 vs 36,
  88 vs 81) is what makes them non-dominated.
- `make_submission`'s NumPy witness extraction is too slow at n ≈ 650–684 to
  package these inside the campaign budget; the raw `(H_X, H_Z)` are staged
  here instead, pending a witness extraction with the accelerated backend.
- Novelty vs the literature is unverified; "advances this board cell" only.
