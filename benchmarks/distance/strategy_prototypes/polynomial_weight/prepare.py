"""Non-search algebra preflight; no weight scoring or witness generation."""

import json
import time
from pathlib import Path

from run import np

# isort: split
from common import HERE
from strategy_prototypes.polynomial_weight.module import build


def main():
    structures = json.loads((Path(__file__).parent / "structure.json").read_text())
    for case, structure in structures.items():
        with np.load(HERE / "results/component-study/corpus" / (case + ".npz")) as d:
            for side, h in [("X", d["hz"]), ("Z", d["hx"])]:
                tick = time.perf_counter()
                basis, widths, groups = build(h, structure["length"], [int(f, 16) for f in structure["factors"]])
                print(
                    json.dumps(
                        dict(
                            case=case,
                            side=side,
                            seconds=time.perf_counter() - tick,
                            dimension=len(basis),
                            widths=widths,
                            groups=groups,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
