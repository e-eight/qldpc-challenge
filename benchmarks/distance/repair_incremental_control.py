"""Rerun only legacy C++ controls and assemble a provenance-preserving study view."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import HERE, atomic_json
from report import main as export
from report_incremental import main as summarize

POLICY = "blake2b(study_seed,batch_index)"


def main(args):
    repaired = []
    for workers in (1, 4):
        for seconds in (1, 10):
            label = f"t{workers}-{seconds}s"
            original = args.raw / label
            environment = json.loads((original / "environment.json").read_text())
            if environment.get("cpp_batch_seed_policy") == POLICY:
                continue
            # Validate the whole original grid, retaining its raw legacy control
            # and suppressing invalid independent-seed intervals in its report.
            export(original, args.output / "legacy" / label)
            corrected = args.corrections / f"cpp-correction-{label}"
            if not corrected.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "run.py"),
                        "--corpus",
                        str(args.corpus),
                        "--output",
                        str(corrected),
                        "--cases",
                        *environment["cases"],
                        "--methods",
                        "cpp",
                        "--threads",
                        str(workers),
                        "--cpus",
                        *map(str, environment["affinity"]),
                        "--seconds",
                        str(seconds),
                        "--seeds",
                        str(environment["seeds"]),
                        "--seed-start",
                        str(environment["seed_start"]),
                        "--validation-workers",
                        "8",
                    ],
                    check=True,
                )
            export(corrected, args.output / "corrected-controls" / label)
            replacement = json.loads((corrected / "environment.json").read_text())
            for key in (
                "workers",
                "seconds_per_side",
                "seed_start",
                "seeds",
                "cases",
                "affinity",
                "corpus_manifest_sha256",
                "native_binary_sha256",
                "stop_at_target",
                "platform",
            ):
                if environment[key] != replacement[key]:
                    raise ValueError(f"Control mismatch for {key}: {label}")
            if replacement.get("cpp_batch_seed_policy") != POLICY:
                raise ValueError("Replacement control still has the legacy seed policy")
            merged = args.merged / label
            if not merged.exists():
                merged.mkdir(parents=True)
                shutil.copytree(original / "matrices", merged / "matrices")
                shutil.copyfile(original / "corpus.json", merged / "corpus.json")
                shutil.copytree(original / "source_snapshot", merged / "source_snapshot" / "original")
                shutil.copytree(corrected / "source_snapshot", merged / "source_snapshot" / "cpp_control")
                environment.update(
                    cpp_batch_seed_policy=POLICY,
                    cpp_control_repair={
                        "reason": "Legacy adjacent study seeds shared overlapping batch streams",
                        "mode": "C++ controls rerun separately; all other measurements retained unchanged",
                        "original_runner_sha256": environment["runner_sha256"],
                        "corrected_runner_sha256": replacement["runner_sha256"],
                        "corrected_environment": replacement,
                    },
                )
                atomic_json(merged / "environment.json", environment)
                for path in original.glob("*/*/result.json"):
                    record = json.loads(path.read_text())
                    relative = path.relative_to(original)
                    origin = "original"
                    if record["method"] == "cpp":
                        record = json.loads((corrected / relative).read_text())
                        origin = "cpp_control"
                    record["measurement_source"] = origin
                    atomic_json(merged / relative, record)
            # Both original and corrected evidence remain separately exported.
            # Only the reviewed display view is replaced here.
            export(merged, args.output / label)
            repaired.append(label)
    (args.output / "SEED_POLICY.md").write_text(
        "# C++ control correction\n\n"
        "The legacy timed adapter formed each batch seed as study_seed + batch_index * 1000003. "
        "The outer runner also advanced study seeds by 1000003, so adjacent study runs reused "
        "overlapping random streams. Individual witnesses remain valid, but the repeated counts "
        "were correlated and did not support independent-seed confidence intervals.\n\n"
        "The corrected adapter hashes the 64-bit study seed and 64-bit batch index together with "
        "BLAKE2b (8-byte digest, personalization qldpc-ris-batch). The C++ kernel is unchanged. "
        "A regression check covers adjacent seeds and both sides across 256 batches each.\n\n"
        f"Corrected views: {', '.join(repaired) or 'none needed'}. The affected C++ controls were "
        "rerun separately after the original study. Other method measurements were retained unchanged. "
        "This means those corrected controls were not interleaved with the other methods in time; "
        "account for possible VM drift when comparing their throughput or recovery. The four-worker "
        "runs used the corrected mapping from the outset. Incremental versus six-pivot RIS and "
        "dist-m4ri comparisons were unaffected by this issue.\n\n"
        "The legacy and corrected-controls directories retain both sets of raw observations. "
        "Merged results label measurement_source; source archives retain original and cpp_control "
        "snapshots, and environment.json records both runner hashes and the corrected environment. "
        "Search settings and targets were not tuned during this correction.\n"
    )
    summarize(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "cache" / "incremental-study")
    parser.add_argument("--corrections", type=Path, default=HERE / "cache" / "incremental-study-corrections")
    parser.add_argument("--merged", type=Path, default=HERE / "cache" / "incremental-study-reviewed")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "incremental-study")
    parser.add_argument("--corpus", type=Path, default=HERE / "results" / "reference-corpus")
    main(parser.parse_args())
