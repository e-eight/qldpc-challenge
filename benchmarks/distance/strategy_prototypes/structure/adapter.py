"""Connected-region column elimination; a heuristic upper-bound search."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "native" / "ris"))
import ris_native
from _strategy_structure import Search


def prepare(own, opposite):
    # Reuse the independently tested CSS validation and dual-logical preparation.
    prepared = ris_native.Prepared(own, opposite)
    return Search(np.ascontiguousarray(opposite, dtype=np.uint8), prepared.logicals)
