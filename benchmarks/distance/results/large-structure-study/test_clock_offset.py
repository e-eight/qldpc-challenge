import audit_clock_offset as audit
import pytest


def worker(times):
    return dict(
        search_seconds=1.001,
        counters=dict(
            elapsed_seconds=1.0,
            slices=[dict(engine="route", start_seconds=0.3, end_seconds=0.4)],
        ),
        events=[dict(stage="routed_single_block", seconds=t) for t in times],
    )


def test_requires_one_consistent_offset():
    low, high = audit.feasible_offset(worker([0.300025, 0.400020]))
    assert low == pytest.approx(0.000020)
    assert high == pytest.approx(0.000025)


def test_cannot_hide_an_event_outside_any_feasible_slice():
    with pytest.raises(ValueError, match="No common bounded"):
        audit.feasible_offset(worker([0.300010, 0.400020]))
