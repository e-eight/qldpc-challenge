# Frozen component experiment

Two tasks: exhaust the dimension-19 components of the two affine siblings, then
measure whether block/component restrictions generalize. No GPU, cluster engine,
collision tuning, external-tool reruns, verifier edits or publication.

Inputs fixed before search: affine 684-12-73 and 684-8-85; non-affine cyclic
682-182-76, balanced product 700-140-22, pair-partition lifted product 664-170-18,
and local surface construction 676-4-13. Selection uses size and construction
variety, not trial outcomes. Reference updates 51/54 come from prior saved results.

Methods: unchanged guided; auto-components (full check-graph components plus the
existing native adjacent-row partition detector); metadata-components (full graph
components plus documented construction coordinate blocks). Automatic detection
is heuristic and may depend on row order. Metadata blocks: halves for affine and
cyclic, all pairs of five group-coordinate blocks for the balanced product, eight
83-coordinate CPM blocks for the lifted product, none for the surface control.
Restrict each proposal and split its check graph into connected components.
Deduplicate supports. Keep logically nontrivial kernels of dimension <=64. Exact
Gray traversal for dimension <=20; otherwise native systematic single/pair search.
Compress logical checks by independent columns of the basis/logical pairing so
high-k codes are supported without truncating logical information.

Preparation, common initialization, detection, component analysis and search all
count. Setup cap min(0.15s,15% sector budget), detection cap min(0.03s,3%). At most
32 retained spaces; deterministic coordinate order, one-pass analysis. Divide
remaining time equally between component search and a resumable guided fallback;
return unused component time immediately when all spaces are exhausted. One CPU,
serial measured runs, validation/kit saves completed before next run. Preserve
late exports but exclude them from deadline credit.

Main: six codes x three methods x three seeds x budgets(2,10)s =108 configurations,
648 allocated search seconds. Seeds1800-1802 /1810-1812. Relabel diagnostic: all six
codes, joint independently shuffled rows/columns, guided and auto, one seed1820,
2s =12 configs/24s. Transformed references remapped only for evaluation, never
search. Exact affine check: two inputs, metadata exact-only,4s/code,seed1830 =8s.
Enumerate ALL logically nontrivial components <=20, skip larger, no fallback.
Exact claims are restricted to exhausted supports, not full-code certificates.

Total122 configurations/680 allocated search seconds. No budget or parameter
sweeps. Main continuation criterion: on a non-affine input, a component method
returns weights at least10% below every guided10s result in ALL three2s seeds.
Report automatic versus metadata success separately and require evidence of
component-stage contribution, not fallback alone. If absent, retain specialized
methods and stop broad expansion. A new witnessed bound alone is reported but
is not evidence that automatic structure search generalizes.
