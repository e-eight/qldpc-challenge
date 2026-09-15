# WIP CHECKPOINT — autoresearch 2026-09-15 (NOT a submission)

> **This branch must never be used as a pull-request head.**
> It is a working checkpoint of an unattended autoresearch campaign. It
> deliberately contains raw `.npz` matrices and sweep scripts, which a code
> submission must not (the repo accepts only `codes/<n>-<k>-<d>.json` plus
> `notes/<n>-<k>-<d>.md`). `research/candidates/` is gitignored by design; this
> directory is force-added only so the work is not lost.

## What this is

Six parallel family sweeps (dihedral 2BGA w6/w8, metacyclic 2BGA w6,
Kasai-affine w6/w8, non-abelian lifted product w6), 1500 candidates each,
screened with the `research/kit` funnel (`backend="auto"`, 3000 trials), then
every non-dominated survivor deep re-verified with `verify/gf2_fast` on fresh
seeds. Board baseline: merged `main` (593 entries).

`SUMMARY.md` has the full table, the staged survivors and the collapse data.

## Status (honest)

- Distances are **witness-backed upper bounds**, and they were **still
  descending** with search budget when the campaign stopped ([[672,6,40]] read
  44 → 42 → 40 at 300k → 2M → 4M). They are **not converged**, so they are not
  yet submission-grade: CI's distance gate runs an independent deep refutation
  and will refute an over-claim.
- The survivors were **not packaged**: `submit.make_submission`'s NumPy witness
  extraction exceeded ten minutes at n ≈ 672, so `codes/*.json` could not be
  written within budget.
- Novelty versus the literature is unverified. "Advances this board cell" only.

## Matrices

| file | code | group | supports |
|---|---|---|---|
| `final_lp672.npz` | [[672,6,40]] | metacyclic `Z_3 ⋊ Z_112` | a={128,211,257}, b={92,130,269} |
| `final_lp648.npz` | [[648,4,40]] | metacyclic `Z_3 ⋊ Z_108` | a={4,123,250}, b={64,210,238} |
| `kasai_1a1cc63eb8604a10.npz` | [[684,4,88]] | affine `Aff(F_19)` | a={234,269,233,249}, b={28,205,318,53} |

Each `.npz` stores `HX` and `HZ` as `int8` arrays.

## Scripts (`scripts/`)

`board_index.py` (board snapshot via the trusted verifier), `boardlib.py`
(dominance test), `qec_sweep.py` (the parallel family sweep),
`lp_deep.py` / `twobga_deep.py` / `kasai_deep.py` / `deep_final.py` /
`confirm.py` (deep RIS re-verification), `planar_*.py` (weight-9/10/11 planar
search), `frontier.py`, `stage.py`.

They carry the absolute paths of the machine that produced them; treat them as
a record of the method, not as drop-in tools.

## Path to a real submission

1. Deep-confirm the distance until it stops falling (100k+ trials, fresh seeds,
   both sides) and it still beats the cell.
2. Package: `./qldpc submit <code.npz> --authors @e-eight --family <family>
   --model "<model>" --note-file note.md` — it verifies, extracts the witness,
   writes `codes/<n>-<k>-<d>.json` + `notes/<n>-<k>-<d>.md`, and drafts the body.
3. Validate with `verify/validate_candidate.py`; stage the winner.
4. Open the PR from a **fresh** `submit-<n>-<k>-<d>` branch containing only
   those two files.
