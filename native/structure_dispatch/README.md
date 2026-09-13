# Bounded native structure dispatch

`detect.cpp` provides a small pybind11 module that infers a two-block coordinate
partition from adjacent check-row incidence. It is a routing heuristic, not a
symmetry proof or distance certificate. Returned logical witnesses come from the
existing full single-block kernel, followed by native guided search.

Build from this directory:

```sh
../../.venv-benchmark/bin/python setup.py build_ext --inplace
```

The module accepts a contiguous uint8 binary matrix and exposes
`detect(matrix, seconds, minimum_confidence=0.5)`. It returns a status, coordinate
order, overlap confidence, time/work counters and a modeled scratch-storage
count. The Python wrapper handles corpus int8 input conversion inside its clock.
The detector itself has no SciPy dependency, runtime compilation, or worker threads.

Its score buffer uses uint16 counts; sparse row incidence is scanned once. An
upper bound on assignment quality rejects weak proposals cheaply. Bijective row
maxima avoid assignment entirely. Otherwise a shortest-augmenting-path assignment
uses preallocated buffers and checks its clock each augmentation step. No
allocations occur in the scoring or assignment loops.

Hard work/shape caps bound memory and interrupt granularity: n<=4096, rows<=4096,
row weight<=64, at most about 4 million score updates and 64 million relaxations
(one bounded loop chunk can cross a work cap). The dominant buffer is 2*n*n bytes,
32 MiB at the shape cap; it is roughly 1 MiB for the 700-qubit fixtures. This is a
temporary analysis allocation, separate from the packed search workspaces.
Reported workspace is a conservative sum of native buffer capacities, not RSS.

Clock limits are cooperative. A small allocation, one bounded loop chunk, and
Python return conversion can overrun. The wrapper records total detection time,
including conversion, and caps oversized shapes before copying input. Native
restricted-kernel preparation and final search batches are also noninterruptible.

The scheduler is [dispatch_search.py](../../benchmarks/distance/dispatch_search.py):

1. Prepare common logical initialization.
2. Spend at most a target min(50 ms, 2% of sector budget) on partition detection.
3. Try the existing single-block kernel for min(100 ms, 10% of remaining time).
4. Continue only if that pilot improves initialization; target a 20% time reserve
   for guided search. Otherwise give the remaining time to guided immediately.

`Search(own, opposite, method="dispatch").run(seconds, seed, emit)` has the existing
benchmark callback interface: `emit(weight, support, stage)`. Applications must
persist witnesses; callback failures propagate. The included benchmark driver
handles raw logs, independent trusted validation and kit persistence:

```sh
.venv-benchmark/bin/python benchmarks/distance/study_dispatch.py --output <new-output>
.venv-benchmark/bin/python benchmarks/distance/audit_dispatch.py <completed-output>
```

Run the new synthetic tests from the repository root:

```sh
PYTHONPATH=benchmarks/distance .venv-benchmark/bin/python -m pytest --import-mode=importlib -q benchmarks/distance/test_dispatch.py
```

The detector still depends on useful row order. High overlap and two equal cycles
only justify trying a partition; actual kernels and logical parity determine
witness validity. A helpful pilot does not guarantee that continued restricted
search beats guided at the final deadline. This prototype is not a packaged
standalone solver; its native search cores remain the existing repo extensions.
