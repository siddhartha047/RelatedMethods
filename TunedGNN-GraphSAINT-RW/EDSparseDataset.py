"""Dataset connector for the TunedGNN-GraphSAINT-RW baseline."""

import os
import sys
from pathlib import Path


SUPPORT_GRAPH_ROOT = Path(
    os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[2])
).resolve()
EDSPARSE_ROOT = Path(
    os.environ.get("EDSPARSE_PROJECT_ROOT", SUPPORT_GRAPH_ROOT / "EDSparse")
).resolve()
if str(EDSPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDSPARSE_ROOT))

from edsparse.data.connector import (  # noqa: E402,F401
    infer_embedding_dim,
    load_dense_tensors,
    load_edsparse_bundle,
    load_pyg_data,
    method_scratch_dir,
    select_pyg_split,
)

__all__ = [
    "infer_embedding_dim",
    "load_dense_tensors",
    "load_edsparse_bundle",
    "load_pyg_data",
    "method_scratch_dir",
    "select_pyg_split",
]
