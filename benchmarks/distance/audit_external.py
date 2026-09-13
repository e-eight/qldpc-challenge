"""Audit shared initialization, external evidence, and refresh provenance."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from audit_strategies import main as audit_grid
from common import HERE, atomic_json
from external_search import QD_PARAMS
from report_strategies import checked_records
from run import read_codewords


def main(directory):
    audit_grid(directory)
    env, records = checked_records(directory)
    pins = json.loads((HERE / "sources.json").read_text())
    if any(commit != pins[name]["commit"] for name, commit in env["external_commits"].items()):
        raise ValueError("External commit pin mismatch")
    if env["threads"] != 1 or env["numba_threads"] != 1:
        raise ValueError("Unexpected worker/thread count")
    if any(pool["num_threads"] != 1 for pool in env["threadpools"]):
        raise ValueError("Numerical library not restricted to one thread")
    archived = {}
    for name in ("sources.tar.gz", "external-sources.tar.gz"):
        with tarfile.open(directory / name) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    archived[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if any(archived.get(path) != digest for path, digest in env["source_hashes"].items()):
        raise ValueError("Measured source differs from archived bytes")
    initializations = {}
    external_sides, exported_words = 0, 0
    for record in records:
        for side in ("X", "Z"):
            worker = record["workers"][side][0]
            counters = worker["counters"]
            initial = [
                (e["stage"], e["weight"], e["support"])
                for e in worker["events"]
                if e["stage"] in ("initialization", "initial_seed")
            ]
            key = (record["case"], side)
            value = (initial, counters["initialization"], counters["initial_best"])
            if value != initializations.setdefault(key, value):
                raise ValueError(f"Methods/seeds did not receive identical initialization: {key}")
            if not counters["initialization"]["complete"]:
                raise ValueError("Common initialization did not finish")
            if record["method"] == "qdistevol":
                if counters["parameters"] != QD_PARAMS or counters["status"] != "deadline":
                    raise ValueError("QDistEvol parameters changed or stopped before deadline")
                external_sides += 1
            elif record["method"] == "m4ri":
                folder = directory / record["case"] / f"m4ri-s{record['seed']}" / side
                command = json.loads((folder / "command.json").read_text())
                options = dict(arg.split("=", 1) for arg in command[1:])
                if any(options.get(k) != v for k, v in {"method": "1", "threads": "1", "wmin": "0", "dW": "0"}.items()):
                    raise ValueError("Unexpected M4RI command")
                words = read_codewords(folder / "codewords.txt")
                observed = [e["support"] for e in worker["events"] if e["stage"] == "search"]
                if words != observed:
                    raise ValueError("M4RI exported supports were not all retained in order")
                if counters["status"] != "completed" or counters["subprocess_cpu_seconds"] <= 0:
                    raise ValueError("Missing successful M4RI CPU accounting")
                exported_words += len(words)
                external_sides += 1
    atomic_json(
        directory / "external-audit.json",
        {
            "status": "passed",
            "configurations": len(records),
            "identical_complete_initializations": len(initializations),
            "external_sides_checked": external_sides,
            "m4ri_exported_words_checked": exported_words,
            "external_commits_match": True,
            "source_archive_bytes_match": True,
            "single_thread_libraries": True,
            "scope": "Complements audit.json; witnessed upper bounds, no full candidate gate",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
