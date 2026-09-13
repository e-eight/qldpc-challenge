"""Build an isolated, observer-only dist-m4ri copy; never edit the pinned checkout."""

import json
import shutil
import subprocess

from common import HERE, ROOT, atomic_json, sha256

DEST = HERE / "full_suite/build/m4ri-observer"
MARKER = "QLDPC_BENCH_WITNESS"


def main():
    source = HERE / "cache/deps/dist-m4ri"
    expected = json.loads((HERE / "sources.json").read_text())["dist_m4ri"]["commit"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != expected or subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=source):
        raise RuntimeError("Pinned upstream source differs")
    DEST.mkdir(parents=True, exist_ok=False)
    original = {}
    for p in (source / "src").iterdir():
        if p.suffix in (".c", ".h") or p.name == "makefile":
            shutil.copyfile(p, DEST / p.name)
            original[p.name] = sha256(p)
    for name in ("LICENSE", "COPYING"):
        if (source / name).is_file():
            shutil.copyfile(source / name, DEST / name)
    p = DEST / "util_io.c"
    text = p.read_text()
    needle = "    if (weight < p->min_w) {\n      p->min_w = weight;"
    if text.count(needle) != 1:
        raise RuntimeError("Observer insertion site is ambiguous")
    replacement = """    if (weight < p->min_w) {
      /* Benchmark observation only; no search state or RNG is changed. */
      fprintf(stderr, "QLDPC_BENCH_WITNESS %d", weight);
      for (int qldpc_i = 0; qldpc_i < weight; ++qldpc_i)
        fprintf(stderr, " %d", arr[qldpc_i]);
      fprintf(stderr, "\\n");
      fflush(stderr);
      p->min_w = weight;"""
    p.write_text(text.replace(needle, replacement))
    diff = subprocess.run(["diff", "-u", str(source / "src/util_io.c"), str(p)], capture_output=True, check=False)
    (DEST / "observer.patch").write_bytes(diff.stdout)
    prefix = HERE / "cache/deps/m4ri-install"
    command = ["make", "-j4", "dist_m4ri", f"CC=cc -I{prefix}/include -L{prefix}/lib", "OPT=-O3 -march=native"]
    with (DEST / "build.log").open("w") as log:
        subprocess.run(command, cwd=DEST, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(
        DEST / "provenance.json",
        dict(
            commit=commit,
            upstream_files=original,
            command=command,
            binary_sha256=sha256(DEST / "dist_m4ri"),
            original_binary_sha256=sha256(source / "src/dist_m4ri"),
            observer_source=str(p.relative_to(ROOT)),
            marker=MARKER,
        ),
    )
    print(DEST / "dist_m4ri")


if __name__ == "__main__":
    main()
