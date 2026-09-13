# Component search prototype

`adapter.Search(own, opposite, method, blocks=()).run(seconds, seed, emit)` uses
matrix-derived logical initialization and returns analysis/search counters. The
callback receives `(weight, support, stage)`; callers must preserve every emitted
witness. Use `study_components.py` for raw logging, trusted validation and kit
persistence. Save failures propagate.

- `guided`: unchanged guided baseline.
- `auto-components`: full check-graph components and the existing adjacent-row
  native detector's two-block proposal. No construction blocks are consulted.
- `metadata-components`: components of the supplied coordinate subsets, plus
  components of the full graph. Blocks are hypotheses, not certified symmetries.
- `exact-components`: metadata proposals, exhaustive eligible spaces only, no
  guided fallback. Intended for finite-space experiments.

The original opposite-check matrix restricted to a proposed support defines the
component graph. A native restricted kernel is computed in original coordinates.
Spaces with zero logical rank are skipped. Logical checks are compressed by
selecting independent columns of the basis/check pairing, preserving its entire
zero set on the restricted span. The search includes stabilizer directions and
filters trivial logical tags; it does not search only a chosen logical basis.

The frozen policy retains dimension <=64, enumerates dimension <=20, and uses
ordinary native single/pair trials otherwise. Analysis has a cooperative setup
cap and at most 32 retained sessions. Half of remaining time goes to components;
guided receives the rest, including time released by completed exact sessions.
All setup and exports are charged. The native libraries are reused unchanged.

This is a bounded experiment, not a general-purpose partition detector. In
particular the detector depends on row order, a proposed block may exclude the
shortest logical, and dimension >64 is an implementation policy rejection.
An exhausted connected component has an exact restricted minimum, not necessarily
the full-code distance.

See [PLAN.md](PLAN.md), [results](../../results/component-study/README.md), and
[reproduction](../../results/component-study/REPRODUCE.md). No CI or standalone
packaging is added in this experiment.
