"""Synthetic integration checks for the common candidate wrapper."""

import json

import pytest
from candidate_search import METHODS, Search, load_candidates
from run import gf2, np


def toy():
    own = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    opposite = np.array([[1, 1, 1, 1, 0]], dtype=np.uint8)
    return own, opposite


@pytest.mark.parametrize("method", METHODS)
def test_common_initialization_and_valid_outputs(method):
    own, opposite = toy()
    exported = []
    result = Search(own, opposite, method, load_candidates()).run(0.1, 201, lambda *args: exported.append(args))
    assert result["initialization"]["complete"]
    assert result["initial_best"] == 1
    json.dumps(result, allow_nan=False)
    assert exported
    for weight, support, stage in exported:
        word = np.zeros(own.shape[1], dtype=np.uint8)
        word[support] = 1
        assert weight == len(support) == int(word.sum())
        assert gf2.commutes(word, opposite) and not gf2.in_rowspace(word, own)
        assert stage


def test_finite_candidate_gets_guided_fallback():
    class Finite:
        @staticmethod
        def search(own, opposite, duals, seconds, seed, emit):
            emit(1, [4], "finite_test")
            return {"finite": True}

    own, opposite = toy()
    exported = []
    result = Search(own, opposite, "matrix-structure", {"matrix-structure": Finite}).run(
        0.08, 202, lambda *args: exported.append(args)
    )
    assert result["candidate"]["finite"]
    assert result["fallback"]["scored_bases"] > 0
    assert {e[2] for e in exported} >= {"finite_test", "guided"}


def test_candidate_callback_failure_propagates():
    class Finite:
        @staticmethod
        def search(own, opposite, duals, seconds, seed, emit):
            emit(1, [4], "finite_test")
            raise AssertionError("Must not continue after failed persistence")

    def emit(weight, support, stage):
        if stage == "finite_test":
            raise OSError("save failed")

    own, opposite = toy()
    with pytest.raises(OSError, match="save failed"):
        Search(own, opposite, "matrix-structure", {"matrix-structure": Finite}).run(0.08, 203, emit)
