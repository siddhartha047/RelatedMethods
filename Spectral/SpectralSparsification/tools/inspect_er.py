#!/usr/bin/env python3
"""Inspect a huge ER NPZ in bounded memory."""

from __future__ import annotations

import argparse
import math
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from tgt_binary import _chunks, _shape  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-stats", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    with zipfile.ZipFile(args.artifact.expanduser().resolve(), "r") as values:
        shape = _shape(values, "resistance")
        if len(shape) != 1:
            raise ValueError(f"resistance must be one-dimensional, got {shape}")
        count = shape[0]
        print(f"artifact={args.artifact}")
        print(f"resistance_values={count}")
        first_values: list[float] = []
        finite = 0
        nonfinite = 0
        minimum = math.inf
        maximum = -math.inf
        total = 0.0
        for chunk in _chunks(values, "resistance"):
            array = np.asarray(chunk, dtype=np.float64)
            if len(first_values) < args.limit:
                needed = args.limit - len(first_values)
                first_values.extend(float(value) for value in array[:needed])
            if not args.no_stats:
                valid = np.isfinite(array)
                finite += int(valid.sum())
                nonfinite += int((~valid).sum())
                if valid.any():
                    selected = array[valid]
                    minimum = min(minimum, float(selected.min()))
                    maximum = max(maximum, float(selected.max()))
                    total += float(selected.sum(dtype=np.float64))
        print("first_values=" + ",".join(f"{value:.9g}" for value in first_values))
        if not args.no_stats:
            mean = total / finite if finite else math.nan
            print(f"finite={finite}")
            print(f"nonfinite={nonfinite}")
            print(f"minimum={minimum:.9g}")
            print(f"maximum={maximum:.9g}")
            print(f"mean={mean:.9g}")
            print(f"sum={total:.12g}")


if __name__ == "__main__":
    main()
