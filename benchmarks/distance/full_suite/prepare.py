"""Freeze every submitted CSS code and the executable benchmark sources."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from run import benchmark_native, gf2, np

# isort: split
from common import HERE, ROOT, atomic_json, matrix_hash, sha256
from full_suite.adapter import CONFIG, METHODS
from full_suite.build_observer import DEST
from initialized_search import GUIDED
from qldpc_verify import _matrix
from study_strategies import ris_native


def snapshot(output):
    sources = set(HERE.glob("*.py")) | set(HERE.glob("*.cpp"))
    sources |= set((HERE / "full_suite").glob("*.py")) | set((HERE / "full_suite").glob("*.md"))
    for folder in (ROOT / "native/ris", HERE / "strategy_prototypes/guided"):
        sources |= {
            p
            for p in folder.rglob("*")
            if p.is_file()
            and "build" not in p.parts
            and "__pycache__" not in p.parts
            and (p.suffix in (".py", ".cpp", ".hpp", ".toml", ".txt", ".md") or p.name == "LICENSE")
        }
    for folder in (ROOT / "verify", ROOT / "research/kit"):
        sources |= set(folder.glob("*.py")) | set(folder.glob("*.cpp"))
    sources |= {
        p
        for p in DEST.iterdir()
        if p.suffix in (".c", ".h", ".patch", ".json", ".log") or p.name in ("makefile", "LICENSE", "COPYING")
    }
    sources |= {HERE / "sources.json", HERE / "cache/deps/build.json", ROOT / "schema/code.schema.json"}
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for p in sorted(sources):
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    binaries = [
        Path(benchmark_native.__file__),
        Path(ris_native.__file__),
        Path(GUIDED.native.__file__),
        DEST / "dist_m4ri",
        HERE / "cache/deps/dist-m4ri/src/dist_m4ri",
    ]
    binary_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in binaries}
    with tarfile.open(output / "binaries.tar.gz", "w:gz") as archive:
        for p in binaries:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    return hashes, binary_hashes


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "matrices").mkdir()
    (output / "inputs").mkdir()
    cases = []
    for path in sorted((ROOT / "codes").glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("code_type") != "CSS":
            raise ValueError(f"Unsupported input type: {path}")
        hx, hz = (_matrix(d["checks"][s], d["n"]) for s in ("X", "Z"))
        if np.any((hx @ hz.T) % 2) or d["n"] - gf2.rank(hx) - gf2.rank(hz) != d["k"]:
            raise ValueError(f"Invalid CSS/rank: {path}")
        reference = {s: d["distance"][s]["witness"] for s in ("X", "Z")}
        for side, own, opposite in (("X", hx, hz), ("Z", hz, hx)):
            word = np.zeros(d["n"], dtype=np.int8)
            word[reference[side]] = 1
            if not word.any() or not gf2.commutes(word, opposite) or gf2.in_rowspace(word, own):
                raise ValueError(f"Invalid input witness: {path}/{side}")
        filename = path.stem + ".npz"
        np.savez_compressed(output / "matrices" / filename, hx=hx, hz=hz)
        shutil.copyfile(path, output / "inputs" / path.name)
        cases.append(
            dict(
                id=path.stem,
                n=d["n"],
                k=d["k"],
                family=d.get("family") or "unspecified",
                file=filename,
                source=str(path.relative_to(ROOT)),
                source_sha256=sha256(path),
                matrix_sha256=matrix_hash(hx, hz),
                reference=reference,
                paper_target=d["distance"]["d"],
            )
        )
    atomic_json(output / "manifest.json", dict(cases=cases, scope="all codes/*.json at preparation"))
    sources, binaries = snapshot(output)
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    if any(p["num_threads"] != 1 for p in pools):
        raise RuntimeError("Numerical thread limit failed")
    atomic_json(
        output / "environment.json",
        dict(
            platform=platform.platform(),
            python=sys.version,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cases=len(cases),
            methods=list(METHODS),
            seconds_per_code=60,
            checkpoints=[30, 60],
            seed=2100,
            cpus=[0, 2, 4, 6],
            threads_per_search=1,
            parameters=CONFIG,
            source_hashes=sources,
            binary_hashes=binaries,
            threadpools=pools,
            timing="Each sector has its own clock; balanced checkpoints use seconds_per_code/2 per sector",
            observer="Only witness print/flush on upstream strict improvement; search decisions unchanged",
        ),
    )
    (output / "hardware.txt").write_bytes(subprocess.check_output(["lscpu"]))
    print(json.dumps(dict(status="prepared", cases=len(cases), output=str(output))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output.resolve())
