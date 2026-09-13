"""Supplement the frozen auditor for harness/engine clock-origin differences.

Only a failing slice-containment check may take this path. Require one common
feasible offset, bounded by outer elapsed minus inner elapsed. Recheck the full
control audit on an in-memory aligned copy. Raw event times, deadline summaries,
sources and benchmark records remain unchanged.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
import audit_large_structure as audit  # noqa: E402
from audit_candidates import require  # noqa: E402
from common import atomic_json  # noqa: E402

original_control = audit.control_worker
corrections = []


def feasible_offset(worker):
    counters = worker["counters"]
    upper = worker["search_seconds"] - counters["elapsed_seconds"]
    require(0 <= upper < 0.01, "Unbounded clock-origin discrepancy")
    intervals = [(0.0, upper)]
    for event in worker["events"]:
        engine = {"guided": "guided", "routed_single_block": "route"}.get(event["stage"])
        if engine is None:
            continue
        candidates = sorted(
            (max(a, event["seconds"] - s["end_seconds"]), min(b, event["seconds"] - s["start_seconds"]))
            for a, b in intervals
            for s in counters["slices"]
            if s["engine"] == engine
        )
        intervals = []
        for a, b in candidates:
            if a > b:
                continue
            if intervals and a <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
            else:
                intervals.append((a, b))
        require(intervals, "No common bounded clock offset explains all slices")
    return intervals[0]


def control(record, side):
    try:
        return original_control(record, side)
    except ValueError as error:
        if str(error) != "Export outside engine slice":
            raise
    require(record["method"] == "race", "Unexpected method for clock alignment")
    lower, upper = feasible_offset(record["workers"][side][0])
    offset = min(upper, lower + 1e-12)
    aligned = copy.deepcopy(record)
    for event in aligned["workers"][side][0]["events"]:
        event["seconds"] -= offset
    result = original_control(aligned, side)
    corrections.append(
        dict(case=record["case"], seed=record["seed"], side=side, feasible_offset=[lower, upper], audit_offset=offset)
    )
    return result


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    audit.control_worker = control
    audit.main(directory)
    atomic_json(
        directory / "clock-origin-audit.json",
        dict(
            status="passed",
            corrections=corrections,
            raw_times_and_deadline_results_unchanged=True,
            scope="Feasibility of a bounded common clock offset; not an estimate of the exact start offset",
        ),
    )
