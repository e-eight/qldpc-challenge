# Full submitted-code comparison

User authorized all five methods and an overnight run. Freeze all 593 JSON
submissions in codes/ as they exist at launch; one seed per code/method. Compare
cpp (existing core), fresh (six-pivot RIS), incremental (64-basis restart), guided
(frozen four-parent policy), and m4ri (pinned method 1). No GPU or verifier edits.

60 seconds/code/method, split into independent 30-second X/Z runs. Report the
30-second/code balanced checkpoint from each sector's first 15 seconds as well
as the final 60-second result. These are trajectory prefixes at equal sector
allocation, not a claim that both sectors ran within the first 30 wall seconds.
Include preparation, common initialization, setup and witness delivery. Exclude
matrix loading, imports, independent validation and kit packaging. No target
stopping. Preserve late deliveries but give them no earlier deadline credit.

All methods receive identical matrix-derived logical initialization. It is
reported separately and is not injected into engine fitness. The current C++
baseline uses the existing PreparedSearch wrapper with independent batch seeds;
it is an engine comparison, not a replay of the entire production gate. Native
engines use resumable sessions. Aim for 2 ms batches, cap 64 trials. Only strict
engine improvements are exported, plus common initialization/seed-pool exports.

dist-m4ri normally exports at exit. Build an isolated observer-only copy of its
pinned source: print/flush each strict improvement from codeword_add_maybe,
without changing candidates, RNG, retention, search parameters or stop decisions.
Parent receipt times determine checkpoint credit. Preserve original stdout,
stderr, command and final codeword file (including tied supports). Kit-package
every observer export; do not package every internally retained tied codeword.
Check fixed-work observer and original outputs agree on synthetic inputs.
The observer's I/O overhead is charged to dist-m4ri.

Also run the existing circulant detector/pass, separately, with 2 seconds/code
on every code (immediate return for inapplicable layouts). It is not included
in the five-way general-method table. Do not add new structure heuristics.

Four persistent single-job workers pinned to CPUs 0,2,4,6. Each worker performs
its own validation between searches on the same CPU, avoiding oversubscription.
Method jobs are shuffled deterministically; record CPU and timing. This is a
parallel throughput study with shared-cache/VM contention, not isolated-core
microbenchmarking. Single seed is screening, not calibrated miss probability.

Total main allocation 593*5*60 = 177900 seconds (49h25m CPU), ideal 12h21m wall
on four workers before validation and observer/setup overhead. Structure adds
at most 1186 CPU seconds. Expected wall time roughly 14–15 hours, measured pace
reported after launch. Full source/matrix/binary pins must remain unchanged.

Raw outputs are flushed immediately; result records are written before kit
validation and each distinct exported support is persisted by validate_and_stage.
A failed save or invalid witness is fatal: finish and preserve other in-flight
jobs, stop scheduling, retain raw evidence and an error report. Resume skips only
audited validated jobs, checks all input/source/binary hashes and creates a fresh
attempt directory for an interrupted job. No mutation of previous studies.

The unattended driver updates progress/ETA after each job and automatically
audits saved witnesses and writes the final comparison. It must never label an
incomplete run complete or a witnessed bound an exact distance. No publication
or production integration is part of this task.
