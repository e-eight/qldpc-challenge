"""Run and export the predeclared repeated-seed incremental RIS comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import HERE

CASES = [
    "board-700-222-28",
    "regression-690-182",
    "tanner-432_8_33",
    "mitten-975-195",
    "toric-1000",
    "fresh-bb-960",
]
METHODS = ["cpp", "ris-block6", "ris-incremental", "m4ri"]


def main(args):
    legacy_control = False
    for workers, cpus in ((1, [0]), (4, [0, 2, 4, 6])):
        for seconds in (1, 10):
            label = f"t{workers}-{seconds}s"
            raw = args.raw / label
            if raw.exists():
                # The exporter verifies the complete expected grid and all
                # witness validation statuses. Partial runs fail closed.
                print(f"Checking existing {raw}", flush=True)
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "run.py"),
                        "--corpus",
                        str(args.corpus),
                        "--output",
                        str(raw),
                        "--cases",
                        *CASES,
                        "--methods",
                        *METHODS,
                        "--threads",
                        str(workers),
                        "--cpus",
                        *map(str, cpus),
                        "--seconds",
                        str(seconds),
                        "--seeds",
                        str(args.seeds),
                        "--seed-start",
                        str(args.seed_start),
                        "--validation-workers",
                        "8",
                        "--restart-interval",
                        "64",
                        "--exchange-proposals",
                        "8",
                    ],
                    check=True,
                )
            subprocess.run([sys.executable, str(HERE / "report.py"), str(raw), str(args.output / label)], check=True)
            environment = json.loads((raw / "environment.json").read_text())
            legacy_control |= environment.get("cpp_batch_seed_policy") != "blake2b(study_seed,batch_index)"
    if legacy_control:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "repair_incremental_control.py"),
                "--raw",
                str(args.raw),
                "--output",
                str(args.output),
                "--corpus",
                str(args.corpus),
                "--corrections",
                str(args.raw.parent / (args.raw.name + "-corrections")),
                "--merged",
                str(args.raw.parent / (args.raw.name + "-reviewed")),
            ],
            check=True,
        )
    else:
        subprocess.run([sys.executable, str(HERE / "report_incremental.py"), str(args.output)], check=True)
    print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "results" / "reference-corpus")
    parser.add_argument("--raw", type=Path, default=HERE / "cache" / "incremental-study")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "incremental-study")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=300)
    main(parser.parse_args())
