# Full-suite benchmark driver

This is an evaluation harness, not production integration. See PLAN.md for the
fixed protocol. It uses the existing native libraries, an isolated dist-m4ri
observer build, the trusted GF(2) checks and the research kit's witness saver.

From the repository root:

```
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m full_suite.build_observer
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m pytest benchmarks/distance/full_suite/test_suite.py -q
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m full_suite.prepare /tmp/new-full-suite
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -u -m full_suite.study /tmp/new-full-suite
```

The observer builder and preparer refuse existing destinations. Reuse the
already built observer rather than rebuilding the measured binary. For an
interrupted prepared run, the study accepts `--resume`; it checks source/binary
pins and completed-record hashes, skips validated jobs, and retains failed or
interrupted attempts in separate folders. Do not change measured sources or
binaries during the run. The 60-second general and two-second structure budgets
are fixed by this study, not per-method tuning parameters.

`progress.json` reports completion count, active cases/methods, saved documents
and an allocation-based ETA. `runs/<case>/<method>/attempt-*/` contains raw sector
JSONL events, original M4RI CLI files and each result. `completed.json` in a job
directory exists only after its witnesses were checked and kit-saved. A global
`completed.json` exists only after every job and the final independent audit
pass. Final outputs are `REPORT.md`, `summary.json`, `per-case.json`, `audit.json`
and `SHA256SUMS` (the live console log is excluded).

Failure handling stops scheduling new jobs and allows the other in-flight jobs
to finish and preserve their witnesses. `driver-error.json` and `progress.json`
record a failure. An interrupted raw export is never silently treated as a
completed measurement. Input reference witnesses are used only for packaging
the other sector after a failure, and never supplied to search.

The dist-m4ri patch only prints and flushes strict improvements on stderr.
Fixed-work tests compare its full final stdout and codeword sets with the
unmodified binary across three synthetic seeds. Separate tests check live
receipt timing, every engine's witnesses, balanced checkpoint accounting,
failure persistence, and the complete synthetic pipeline including final audit.

The corpus includes all codes/*.json at preparation; the additional n>700
synthetic/external benchmark fixtures from earlier studies are not submitted
codes and are outside this sweep.
