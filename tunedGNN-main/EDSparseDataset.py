"""tunedGNN adapter for EDSparse datasets and seeded data splits.

The batch runner sets ``EDSPARSE_DATASET_SEED`` for every run.  Loading data
through this module therefore gives tunedGNN the same graph and the exact same
train/validation/test membership as the corresponding EDSparse run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


TUNEDGNN_ROOT = Path(__file__).resolve().parent
SUPPORT_GRAPH_ROOT = Path(
    os.environ.get("SUPPORT_GRAPH_ROOT", TUNEDGNN_ROOT.parents[1])
).expanduser().resolve()
EDSPARSE_ROOT = Path(
    os.environ.get("EDSPARSE_PROJECT_ROOT", SUPPORT_GRAPH_ROOT / "EDSparse")
).expanduser().resolve()

if str(EDSPARSE_ROOT) not in sys.path:
    # Keep tunedGNN's entrypoint directory ahead of EDSparse.  Both projects
    # contain generic module names such as ``parse.py``; prepending EDSparse
    # would make tunedGNN import the wrong parser after loading this adapter.
    sys.path.append(str(EDSPARSE_ROOT))

from edsparse.data import (  # noqa: E402,F401
    CANONICAL_DATASETS,
    canonicalize_dataset_name,
    load_edsparse_bundle,
    load_pyg_data,
    method_scratch_dir,
    split_fingerprint,
)
from edsparse.data.connector import DEFAULT_SPLIT_PROTOCOL  # noqa: E402,F401


__all__ = [
    "CANONICAL_DATASETS",
    "DEFAULT_SPLIT_PROTOCOL",
    "canonicalize_dataset_name",
    "load_edsparse_bundle",
    "load_pyg_data",
    "method_scratch_dir",
    "split_fingerprint",
]
