"""Check prior frozen artifacts and measured sources without mutating them."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    reports = []
    pins = {}
    for name in ("large-structure-study", "block-collision-study", "component-study", "algebra-diagnostic"):
        folder = HERE.parent / name
        count = 0
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            assert digest(folder / relative) == expected, (name, relative)
            count += 1
        for env_path in folder.rglob("environment.json"):
            env = json.loads(env_path.read_text())
            for field in ("source_hashes", "binary_hashes"):
                for relative, expected in env.get(field, {}).items():
                    assert relative not in pins or pins[relative] == expected, relative
                    pins[relative] = expected
        reports.append(dict(study=name, artifacts=count))
    for relative, expected in pins.items():
        assert digest(ROOT / relative) == expected, relative
    (HERE / "preservation.json").write_text(
        json.dumps(
            dict(
                status="passed",
                studies=reports,
                prior_artifacts=sum(r["artifacts"] for r in reports),
                prior_source_and_binary_pins=len(pins),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
