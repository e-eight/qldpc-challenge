# Initialized guided/refinement experiment

Freeze the following policy before corpus timing. This is a new experiment;
earlier source snapshots and results remain unchanged.

- Same seven cases; original matrices and no supplied reference support as input.
- Incremental RIS, guided RIS, guided RIS plus selective stabilizer refinement.
- Preparation and initialization are inside the budget. Imports, matrix loading,
  independent verification and submission packaging are outside it.
- Common initialization: logical single representatives first; all nonzero
  representative combinations for k <= 16, otherwise all pairs. Retain the best
  returned bound and eight low-weight representatives of distinct logical classes.
- Guided parameters remain elite4/exchange8/immigrant64/pairs8/block6. Native
  kernels are unchanged. Both controls use 2 ms adaptive batches capped at 64 bases.
- Refinement uses at most the gradually earned 10% share of elapsed search time,
  in 5 ms native calls (small call overhead can overshoot). Refine each selected
  support once. New lighter representatives can earn another attempt. Candidate
  classes come from initialization and globally improving native search events.
  No restart, feedback into the guided population, or new seeding algorithm.
- Separate refinement RNG leaves the native basis-search random stream intact.
- Refresh bicycle960 reference from the saved preparation diagnostic (60).
- Main screen: 10 seconds/code, seeds 800--802, all seven cases, all three methods.
  Record prefixes at 0.1, 2 and 10 seconds/code. Each prefix uses t/2 per side.
- Longer budget diagnostic: 60 seconds/code, seeds 810--811, board700, board682,
  regression690; guided and guided-refine. Record 0.1/2/10/30/60 second prefixes.
  These seeds and parameters will not be retuned from screen results.
- One worker pinned CPU0, serial timed methods, shuffled method order per seed/case.
  Complete validation and saving before the next timed run. No competing timed jobs.
- Export every native improvement, every local improvement, initialization strict
  improvements and final seed pool. Save all distinct emitted supports via the
  kit; a failed save is a hard error. Retain late witnesses without deadline credit.
- Test toy algebra, logical-class tracking, deadline accounting and evidence
  retention before evaluation. Audit complete records, support identity, hashes
  and trusted-verifier integrity afterwards. No full gate or exact-distance claim.

The common initialization is retained at wrapper level; native search fitness
and internal incumbent are unchanged. Preparation currently duplicates some
matrix work and the wrapper allocates Python objects outside native trial loops.
This measures the implemented experimental interfaces; integration and setup
deduplication are later engineering tasks if the search policy warrants them.

Decision: keep the hybrid only if its improvements repay its time cost; use
long-run trajectories to distinguish slow progress from observed plateaus. Neither
failure at 60 seconds nor a small seed count proves that more compute cannot help.
