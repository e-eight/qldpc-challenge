"""Check the deterministic enumeration on a fully enumerable synthetic code."""

import finite_probe as probe
import numpy as np


def test_gray_enumeration_covers_every_word(monkeypatch):
    monkeypatch.setattr(probe, "proposals", lambda spec, n: [(probe.LABEL, [[0, 4], [1, 5], [2, 6], [3, 7]])])
    rows = np.zeros((0, 8), dtype=np.int8)
    events = []
    result = probe.ExactSearch(rows, rows).run(1, 0, lambda *e: events.append(e))
    assert result["exhaustive"] and result["visited"] == 15
    assert result["nontrivial_visited"] == 15
    assert result["dimension"] == result["logical_rank"] == 4
    assert result["restricted_best"] == 2
    assert all(w == len(s) == 2 for w, s, label in events if label == probe.LABEL)


def test_dimension_cap(monkeypatch):
    monkeypatch.setattr(probe, "MAX_DIMENSION", 2)
    monkeypatch.setattr(probe, "proposals", lambda spec, n: [(probe.LABEL, [[0], [1], [2]])])
    rows = np.zeros((0, 4), dtype=np.int8)
    result = probe.ExactSearch(rows, rows).run(1, 0, lambda *e: None)
    assert result["status"] == "dimension_or_time_cap"
    assert not result["exhaustive"] and result["visited"] == 0
