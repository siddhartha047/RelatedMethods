# BMatching Python adapter

`bmatching_adapter.py` exposes the bundled maximum-weight b-matching solver to
Python. The original C++ implementation is rebuilt for Linux on first use and
stored under `/rcfs/scratch/dass304/EDSparse/bin/bmatching`.

The original solver uses a dense weight matrix, so `backend=auto` uses it only
for graphs with at most 512 nodes. Larger graphs use a sparse approximate
b-matching implementation. CUDA is used for edge ordering when it is available
and has enough free memory; degree constraints are enforced by a compiled CPU
kernel. Set `BMATCH_NUM_THREADS` or pass `--workers` to restrict CPU workers.

From the `Bmatching` directory, test the original C++ solver on Karate:

```bash
PY="/people/dass304/.conda/envs/py312/bin/python"
"$PY" -u -E -s main.py --dataset karate --backend auto --target-ratio 0.5
```

Test the sparse backend on Cora:

```bash
PY="/people/dass304/.conda/envs/py312/bin/python"
"$PY" -u -E -s main.py --dataset cora --backend scalable --target-ratio 0.5
```
