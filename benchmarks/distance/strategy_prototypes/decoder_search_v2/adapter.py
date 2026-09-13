"""Seek CSS logicals by decoding a deliberately nonzero logical syndrome.

The caller owns trusted validation and durable witness persistence. This adapter
exports all distinct successful decoder outputs, including the final late one.
"""

import math
import time
from importlib.metadata import version

import numpy as np
from ldpc import BpOsdDecoder

CONFIG = {
    "max_iter": 100,
    "bp_method": "product_sum",
    "schedule": "parallel",
    "omp_thread_count": 1,
    "osd_method": "OSD_CS",
    "osd_order": 1,
}

CHANNEL_RATE = 0.05
LOG_ODDS_JITTER = 0.25


def _binary_matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a binary matrix")
    return np.asarray(array, dtype=np.uint8, order="C")


def random_detector(duals, rng):
    """Uniform nonzero coefficients in the supplied independent logical basis."""
    coefficients = rng.integers(0, 2, size=len(duals), dtype=np.uint8)
    while not coefficients.any():
        coefficients = rng.integers(0, 2, size=len(duals), dtype=np.uint8)
    return np.bitwise_xor.reduce(duals[coefficients.astype(bool)], axis=0)


def decode_trial(opposite, duals, augmented, syndrome, rng):
    """One bounded-work native call; return original-coordinate output and stats."""
    detector = random_detector(duals, rng)
    if not detector.any():
        raise ValueError("Supplied logical basis has a dependent nonzero combination")
    augmented[-1] = detector
    permutation = rng.permutation(opposite.shape[1])
    # Perturb costs mildly to break symmetric BP messages and OSD reliability ties.
    log_odds = math.log((1.0 - CHANNEL_RATE) / CHANNEL_RATE)
    priors = 1.0 / (1.0 + np.exp(log_odds + rng.uniform(-LOG_ODDS_JITTER, LOG_ODDS_JITTER, len(permutation))))
    construction_start = time.perf_counter()
    decoder = BpOsdDecoder(np.ascontiguousarray(augmented[:, permutation]), error_channel=priors.tolist(), **CONFIG)
    construction_seconds = time.perf_counter() - construction_start
    decode_start = time.perf_counter()
    permuted = np.asarray(decoder.decode(syndrome))
    decode_seconds = time.perf_counter() - decode_start
    if permuted.shape != (opposite.shape[1],) or not np.all((permuted == 0) | (permuted == 1)):
        raise RuntimeError("Decoder returned a malformed binary vector")
    candidate = np.empty(opposite.shape[1], dtype=np.uint8)
    candidate[permutation] = permuted
    valid = not ((opposite @ candidate) & 1).any() and bool((detector @ candidate) & 1)
    if valid and not ((duals @ candidate) & 1).any():
        raise RuntimeError("Successful augmented decode has no logical parity")
    return candidate, {
        "valid": bool(valid),
        "bp_converged": bool(decoder.converge),
        "bp_iterations": int(decoder.iter),
        "detector_weight": int(detector.sum()),
        "construction_seconds": construction_seconds,
        "decoder_seconds": decode_seconds,
    }


def search(own, opposite, duals, seconds, seed, emit):
    """Emit ``(weight, sorted_support, 'decoder')`` during a wall-clock budget.

    Imports are outside the clock. Matrix conversion, validation, decoder
    construction, decoding, candidate checks, and callbacks are inside it.
    Native decoding has no time interrupt: finish and export the final result
    before returning, so the caller can retain but exclude late output.
    """
    start = time.perf_counter()
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    deadline = start + seconds
    own = _binary_matrix(own, "own")
    opposite = _binary_matrix(opposite, "opposite")
    duals = _binary_matrix(duals, "duals")
    n = own.shape[1]
    if opposite.shape[1] != n or duals.shape[1] != n:
        raise ValueError("All matrices must have the same number of columns")
    counters = {
        "method": "augmented-syndrome-bp-osd",
        "parameters": {**CONFIG, "base_error_rate": CHANNEL_RATE, "log_odds_jitter": LOG_ODDS_JITTER},
        "ldpc_version": version("ldpc"),
        "random_logical_coefficients": "uniform nonzero",
        "random_column_permutation": True,
        "add_stabilizers_to_detector": False,
        "trials": 0,
        "successful_decodes": 0,
        "failed_decodes": 0,
        "bp_converged": 0,
        "bp_iterations": 0,
        "distinct_exports": 0,
        "duplicate_outputs": 0,
        "late_exports": 0,
        "max_trial_seconds": 0.0,
        "trial_seconds": 0.0,
        "decoder_seconds": 0.0,
        "max_decoder_seconds": 0.0,
        "construction_seconds": 0.0,
        "detector_weight_sum": 0,
        "best_weight": None,
    }
    rng = np.random.default_rng(seed)
    augmented = np.empty((len(opposite) + 1, n), dtype=np.uint8)
    augmented[:-1] = opposite
    syndrome = np.zeros(len(opposite) + 1, dtype=np.uint8)
    syndrome[-1] = 1
    seen = set()
    counters["setup_seconds"] = time.perf_counter() - start
    while len(duals) and time.perf_counter() < deadline:
        trial_start = time.perf_counter()
        candidate, result = decode_trial(opposite, duals, augmented, syndrome, rng)
        elapsed = time.perf_counter() - trial_start
        counters["decoder_seconds"] += result.get("decoder_seconds", 0.0)
        counters["max_decoder_seconds"] = max(counters["max_decoder_seconds"], result.get("decoder_seconds", 0.0))
        counters["construction_seconds"] += result.get("construction_seconds", 0.0)
        counters["trials"] += 1
        counters["trial_seconds"] += elapsed
        counters["max_trial_seconds"] = max(counters["max_trial_seconds"], elapsed)
        counters["bp_converged"] += int(result["bp_converged"])
        counters["bp_iterations"] += result["bp_iterations"]
        counters["detector_weight_sum"] += result["detector_weight"]
        if not result["valid"]:
            counters["failed_decodes"] += 1
            continue
        counters["successful_decodes"] += 1
        support = tuple(int(q) for q in np.flatnonzero(candidate))
        if support in seen:
            counters["duplicate_outputs"] += 1
            continue
        seen.add(support)
        # Callback exceptions propagate: failing to save a witness is fatal.
        emit(len(support), list(support), "decoder")
        counters["distinct_exports"] += 1
        counters["late_exports"] += int(time.perf_counter() > deadline)
        best = counters["best_weight"]
        counters["best_weight"] = len(support) if best is None else min(best, len(support))
    counters["elapsed_seconds"] = time.perf_counter() - start
    counters["status"] = "no_logicals" if not len(duals) else "deadline"
    return counters


def warmup():
    """Run one synthetic repetition-code decode; no corpus inputs or witnesses."""
    start = time.perf_counter()
    opposite = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    duals = np.array([[1, 0, 0]], dtype=np.uint8)
    augmented = np.vstack([opposite, np.zeros(3, dtype=np.uint8)])
    syndrome = np.array([0, 0, 1], dtype=np.uint8)
    candidate, result = decode_trial(opposite, duals, augmented, syndrome, np.random.default_rng(0))
    if not result["valid"] or candidate.sum() != 3:
        raise RuntimeError("Synthetic decoder warmup failed")
    return {"elapsed_seconds": time.perf_counter() - start, "synthetic_only": True, "ldpc_version": version("ldpc")}
