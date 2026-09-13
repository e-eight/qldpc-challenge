# Competing resumable searches: frozen policy and comparison

Continue the bounded dispatcher by comparing guided and routed progress directly.
Preserve every previously measured source, binary and result. This experiment
adds Python session scheduling; existing C++ search engines/detector are unchanged.

## Policy

Common preparation, initialization and bounded detection are identical to the
previous dispatcher and charged inside each sector's budget. A rejected partition
uses the unchanged guided control helper. An accepted partition runs a guided
pilot first and a route pilot second, each min(100 ms, 10% of post-detection
remaining time), including that engine's lazy setup. Both native sessions persist
across all later slices: RNG, population, batch counters and adaptive batch size.

After pilots, allocate epochs of at most 250 ms. Favor the route only if its best
witness strictly improves common initialization AND is at least 10% lighter than
the minimum of guided's best and common initialization. Otherwise favor guided.
Give the other engine 20% of the epoch first, then the favored engine 80%.
Reconsider after every epoch. This conservative margin is a hypothesis motivated
by the prior Board682 allocation failure; it is not learned from new results.
No policy changes after starting timed corpus searches. No target weights, names,
family metadata or reference supports enter the search. No target stopping.

Every returned witness is emitted, including ties and late results. Only results
delivered before the overall sector deadline may affect scheduling. Cooperative
native batch/setup overruns remain possible. Slices and decisions are archived;
per-engine work, exports and saved partitions are audited. Keeping both sessions
alive increases workspace; no additional search threads are used.

## Frozen grid

Use the same 13 matrices/layouts as dispatch-study, copied with manifest and actual
transforms. These are known families and layouts, not held-out validation.
Three methods: unchanged guided, unchanged dispatch, new race.

- 13 inputs x 3 methods x seeds 1500–1502 x 2 seconds/code = 117 configurations.
- Seven original inputs x 3 methods x seeds 1510–1512 x 10 seconds/code = 63 configurations.

Total 180 configurations, 864 allocated search seconds. Single search thread on
CPU0; configurations serial and method order shuffled; numerical libraries one
thread. Equal X/Z budget split. Independent budgets/seeds, not nested runs.
Matrix loading/imports and independent witness validation/persistence excluded;
preparation through export included. All supports use the existing
validate_and_stage -> make_submission/save_submission path; failures are fatal.
Only the existing large-fixture size-cap exemption applies. No trusted verifier,
leaderboard, CI, publication, or exact-distance claims.

Report every seed, medians, paired changes, original versus relabeled layouts,
route/guided active time, preference changes, actual route exports, overhead and
regressions. Include the Board682 gap and whether 700/690 structural gains survive.
A win here supports further evaluation, not a universal-default recommendation.
