"""Unified-LTH connector for EDSparse datasets and compatible splits."""

import os
import sys
from pathlib import Path

EDSPARSE_ROOT = Path(
    os.environ.get(
        "EDSPARSE_PROJECT_ROOT",
        "/people/dass304/dass304/Support Graph/EDSparse",
    )
).resolve()
if str(EDSPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDSPARSE_ROOT))

from edsparse.data.connector import (  # noqa: F401
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
