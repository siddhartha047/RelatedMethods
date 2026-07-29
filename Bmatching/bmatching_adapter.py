"""Python adapter for the bundled belief-propagation b-matching solver.

The 2010 C++ solver uses a dense ``n x n`` matrix.  This adapter therefore
provides two backends:

``legacy``
    Rebuild and run the original C++ implementation.  It is intended only for
    small graphs and is useful for reproducing the original algorithm.

``scalable``
    Run a sparse, deterministic approximation on the input edges.  Edge
    ordering can run on CUDA and the degree-constrained scan uses a compiled
    Numba kernel when Numba is installed.

``auto`` (the default)
    Use the C++ solver for small graphs and the scalable backend otherwise.

Both backends return positions in the supplied edge list, so the adapter can
be used directly by EDSparse's disjoint-layer decomposition.
"""

from __future__ import annotations

import fcntl
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOLVER_ROOT = HERE / "BMatchingSolver"
SOLVER_SOURCE = SOLVER_ROOT / "src"
DEFAULT_SCRATCH_ROOT = Path(
    os.environ.get("EDSPARSE_SCRATCH_ROOT", "/rcfs/scratch/dass304/EDSparse")
)
NUMBA_CACHE_ROOT = DEFAULT_SCRATCH_ROOT / "cache" / "numba" / "bmatching"
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(NUMBA_CACHE_ROOT),
)

try:
    import numba

    NUMBA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    numba.config.CACHE_DIR = str(NUMBA_CACHE_ROOT)
    njit = numba.njit
    prange = numba.prange
except ImportError:  # pragma: no cover - the Python fallback remains usable.
    njit = None
    prange = range


Backend = Literal["auto", "legacy", "scalable"]
Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class BMatchingResult:
    """Result and diagnostics from one b-matching selection."""

    selected_indices: torch.Tensor
    capacities: torch.Tensor
    requested_edges: int
    selected_edges: int
    backend: str
    ordering_device: str
    workers: int
    elapsed_seconds: float
    legacy_solver_ran: bool = False
    message: str = ""


def _available_cpu_count() -> int:
    candidates: list[int] = []
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        candidates.append(len(affinity))
    detected = os.cpu_count()
    if detected:
        candidates.append(int(detected))
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "LSB_DJOB_NUMPROC", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                candidates.append(max(1, int(value)))
            except ValueError:
                pass
    return max(1, min(candidates)) if candidates else 1


def _configure_workers(requested: int | None) -> int:
    available = _available_cpu_count()
    workers = available if requested is None else max(1, min(available, int(requested)))
    torch.set_num_threads(workers)
    if njit is not None:
        try:
            import numba

            numba.set_num_threads(min(workers, int(numba.config.NUMBA_NUM_THREADS)))
        except (RuntimeError, ValueError):
            pass
    return workers


if njit is not None:

    @njit(cache=True, nogil=True)
    def _greedy_bmatching_numba(src, dst, order, capacities, target):
        selected = np.empty(target, dtype=np.int64)
        used = np.zeros(capacities.shape[0], dtype=np.int64)
        count = 0
        for position in range(order.shape[0]):
            edge_id = int(order[position])
            u = int(src[edge_id])
            v = int(dst[edge_id])
            if used[u] >= capacities[u] or used[v] >= capacities[v]:
                continue
            selected[count] = edge_id
            used[u] += 1
            used[v] += 1
            count += 1
            if count == target:
                break
        return selected[:count], used

    @njit(cache=True, nogil=True, parallel=True)
    def _priority_buckets_numba(scores, low, high, bucket_count):
        buckets = np.empty(scores.shape[0], dtype=np.uint16)
        scale = float(bucket_count - 1) / max(float(high - low), 1e-20)
        for edge_id in prange(scores.shape[0]):
            bucket = int((float(scores[edge_id]) - low) * scale)
            bucket = max(0, min(bucket_count - 1, bucket))
            buckets[edge_id] = bucket_count - 1 - bucket
        return buckets

    @njit(cache=True, nogil=True)
    def _counting_order_numba(priority_buckets, bucket_count):
        counts = np.zeros(bucket_count, dtype=np.int64)
        for edge_id in range(priority_buckets.shape[0]):
            counts[int(priority_buckets[edge_id])] += 1
        offsets = np.empty(bucket_count, dtype=np.int64)
        position = 0
        for bucket in range(bucket_count):
            offsets[bucket] = position
            position += int(counts[bucket])
        cursors = offsets.copy()
        # int32 halves the dominant permutation memory and supports graphs up
        # to 2.1 billion edges, well beyond the datasets in this workspace.
        order = np.empty(priority_buckets.shape[0], dtype=np.int32)
        for edge_id in range(priority_buckets.shape[0]):
            bucket = int(priority_buckets[edge_id])
            order[cursors[bucket]] = edge_id
            cursors[bucket] += 1
        return order


def _greedy_bmatching_python(
    src: np.ndarray,
    dst: np.ndarray,
    order: np.ndarray,
    capacities: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    used = np.zeros(capacities.shape[0], dtype=np.int64)
    for raw_edge_id in order:
        edge_id = int(raw_edge_id)
        u = int(src[edge_id])
        v = int(dst[edge_id])
        if used[u] >= capacities[u] or used[v] >= capacities[v]:
            continue
        selected.append(edge_id)
        used[u] += 1
        used[v] += 1
        if len(selected) == target:
            break
    return np.asarray(selected, dtype=np.int64), used


def _greedy_bmatching(
    src: np.ndarray,
    dst: np.ndarray,
    order: np.ndarray,
    capacities: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        return _greedy_bmatching_numba(src, dst, order, capacities, target)
    return _greedy_bmatching_python(src, dst, order, capacities, target)


def _canonical_input(
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
    scores = torch.as_tensor(edge_scores, dtype=torch.float).detach().cpu().reshape(-1)
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if scores.numel() != edges.size(1):
        raise ValueError("edge_scores must have one value per edge")
    if int(num_nodes) < 1:
        raise ValueError("num_nodes must be positive")
    if edges.numel() == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty.astype(np.float32), None
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError("edge_index contains an out-of-range node")

    if bool((edges[0] == edges[1]).any()):
        raise ValueError("b-matching input must not contain self loops")

    # EDSparse already supplies canonical u < v edges in lexicographic order.
    # Keep this zero-copy fast path: constructing and sorting n*u+v IDs adds
    # multiple gigabytes of avoidable workspace on Reddit-sized graphs.
    canonical_orientation = bool((edges[0] < edges[1]).all())
    if canonical_orientation:
        if edges.size(1) <= 1:
            lexicographically_sorted = True
        else:
            source_increases = edges[0, 1:] > edges[0, :-1]
            same_source = edges[0, 1:] == edges[0, :-1]
            target_increases = edges[1, 1:] > edges[1, :-1]
            lexicographically_sorted = bool(
                (source_increases | (same_source & target_increases)).all()
            )
        if lexicographically_sorted:
            values = scores.contiguous().numpy().astype(np.float32, copy=False)
            if not bool(np.isfinite(values).all()):
                values = np.nan_to_num(
                    values,
                    copy=True,
                    nan=-np.finfo(np.float32).max,
                    posinf=np.finfo(np.float32).max,
                    neginf=-np.finfo(np.float32).max,
                )
            return (
                edges[0].contiguous().numpy().astype(np.int64, copy=False),
                edges[1].contiguous().numpy().astype(np.int64, copy=False),
                values,
                None,
            )

    lo = torch.minimum(edges[0], edges[1])
    hi = torch.maximum(edges[0], edges[1])
    keys = lo * int(num_nodes) + hi
    sorted_keys, input_positions = torch.sort(keys, stable=True)
    if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
        raise ValueError(
            "b-matching input must contain each undirected edge exactly once"
        )
    source = lo[input_positions].contiguous().numpy().astype(np.int64, copy=False)
    target = hi[input_positions].contiguous().numpy().astype(np.int64, copy=False)
    values = scores[input_positions].contiguous().numpy().astype(np.float32, copy=False)
    values = np.nan_to_num(
        values,
        nan=-np.finfo(np.float32).max,
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    )
    return (
        source,
        target,
        values,
        input_positions.contiguous().numpy().astype(np.int64, copy=False),
    )


def _resolve_ordering_device(device: Device, num_edges: int) -> str:
    normalized = str(device).lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("b-matching device must be auto, cpu, or cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("b-matching CUDA ordering requested but CUDA is unavailable")
    if normalized == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if normalized == "cuda":
        return "cuda"

    # argsort needs the score tensor plus a 64-bit permutation and workspace.
    # Keep a conservative margin so b-matching precomputation cannot starve
    # training on a GPU that is already occupied.
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return "cpu"
    estimated_bytes = max(1, int(num_edges)) * 24
    return "cuda" if estimated_bytes <= int(free_bytes * 0.25) else "cpu"


def _descending_order(scores: np.ndarray, device: str) -> np.ndarray:
    if device == "cuda":
        cuda_scores = torch.from_numpy(scores).to("cuda", non_blocking=False)
        order = torch.argsort(cuda_scores, descending=True, stable=True)
        result = order.cpu().numpy().astype(np.int64, copy=False)
        del cuda_scores, order
        return result
    exact_sort_limit = int(
        os.environ.get("BMATCH_EXACT_SORT_MAX_EDGES", "2000000")
    )
    if njit is not None and scores.shape[0] > max(0, exact_sort_limit):
        low = float(scores.min()) if scores.size else 0.0
        high = float(scores.max()) if scores.size else 1.0
        buckets = _priority_buckets_numba(scores, low, high, 4_096)
        return _counting_order_numba(buckets, 4_096)
    # lexsort gives deterministic edge-id tie breaking.
    return np.lexsort(
        (np.arange(scores.shape[0], dtype=np.int64), -scores)
    ).astype(np.int64, copy=False)


def _scalable_selection(
    src: np.ndarray,
    dst: np.ndarray,
    scores: np.ndarray,
    *,
    num_nodes: int,
    target_edges: int,
    ordering_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if target_edges == 0:
        return np.empty(0, dtype=np.int64), np.zeros(num_nodes, dtype=np.int64)

    degree = (
        np.bincount(src, minlength=int(num_nodes))
        + np.bincount(dst, minlength=int(num_nodes))
    ).astype(np.int64, copy=False)
    actual_ratio = float(target_edges) / max(1, src.shape[0])
    capacities = np.ceil(degree.astype(np.float64) * actual_ratio).astype(np.int64)
    capacities = np.minimum(degree, np.maximum(capacities, (degree > 0).astype(np.int64)))
    order = _descending_order(scores, ordering_device)

    selected, used = _greedy_bmatching(src, dst, order, capacities, target_edges)
    previous_count = -1
    while selected.shape[0] < target_edges and not np.array_equal(capacities, degree):
        missing = target_edges - int(selected.shape[0])
        expandable = capacities < degree
        expandable_count = int(expandable.sum())
        if expandable_count == 0:
            break
        increment = max(1, math.ceil((2 * missing) / expandable_count))
        capacities[expandable] = np.minimum(
            degree[expandable], capacities[expandable] + increment
        )
        previous_count = int(selected.shape[0])
        selected, used = _greedy_bmatching(
            src, dst, order, capacities, target_edges
        )
        if int(selected.shape[0]) == previous_count:
            # Jump directly to the original degrees after a stalled relaxation.
            capacities = degree.copy()
            selected, used = _greedy_bmatching(
                src, dst, order, capacities, target_edges
            )
            break

    if selected.shape[0] != target_edges:
        raise RuntimeError(
            f"b-matching selected {selected.shape[0]} of {target_edges} requested edges"
        )
    if np.any(used > capacities):
        raise RuntimeError("internal b-matching degree-capacity violation")
    return np.sort(selected), capacities


def _solver_binary(scratch_root: Path | None = None) -> Path:
    root = Path(scratch_root or DEFAULT_SCRATCH_ROOT).expanduser().resolve()
    return root / "bin" / "bmatching" / "BMatchingSolver"


def build_legacy_solver(
    *,
    scratch_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Compile the bundled C++ solver for the current Linux system."""

    sources = sorted(SOLVER_SOURCE.glob("*.cpp"))
    if not sources:
        raise FileNotFoundError(f"BMatching C++ sources not found under {SOLVER_SOURCE}")
    binary = _solver_binary(None if scratch_root is None else Path(scratch_root))
    binary.parent.mkdir(parents=True, exist_ok=True)
    lock_path = binary.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        newest_source = max(
            path.stat().st_mtime
            for path in list(SOLVER_SOURCE.glob("*.cpp"))
            + list(SOLVER_SOURCE.glob("*.h"))
        )
        if (
            not force
            and binary.is_file()
            and os.access(binary, os.X_OK)
            and binary.stat().st_mtime >= newest_source
        ):
            return binary

        compiler = os.environ.get("CXX") or shutil.which("g++")
        if not compiler:
            raise RuntimeError("g++ is required to rebuild the legacy BMatching solver")
        temporary = binary.with_name(f".{binary.name}.{os.getpid()}.tmp")
        command = [
            str(compiler),
            "-std=c++11",
            "-O3",
            "-DNDEBUG",
            "-pthread",
            *[str(path) for path in sources],
            "-o",
            str(temporary),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "failed to compile the BMatching solver:\n"
                + (completed.stderr or completed.stdout)
            )
        temporary.chmod(0o755)
        os.replace(temporary, binary)
    return binary


def _legacy_refinement(
    src: np.ndarray,
    dst: np.ndarray,
    scores: np.ndarray,
    scalable_selected: np.ndarray,
    *,
    num_nodes: int,
    workers: int,
    scratch_root: Path,
    max_iterations: int,
    timeout_seconds: float,
) -> np.ndarray:
    """Refine a feasible sparse solution with the original dense C++ solver."""

    seed_degrees = (
        np.bincount(src[scalable_selected], minlength=num_nodes)
        + np.bincount(dst[scalable_selected], minlength=num_nodes)
    ).astype(np.int64, copy=False)
    active_nodes = np.flatnonzero(seed_degrees > 0)
    if active_nodes.size < 2:
        return scalable_selected
    local_of_global = np.full(num_nodes, -1, dtype=np.int64)
    local_of_global[active_nodes] = np.arange(active_nodes.size, dtype=np.int64)
    active_degrees = seed_degrees[active_nodes]

    finite_scores = scores[np.isfinite(scores)]
    low = float(finite_scores.min()) if finite_scores.size else 0.0
    high = float(finite_scores.max()) if finite_scores.size else 1.0
    forbidden = low - max(1.0, abs(high - low) + 1.0) * 1_000_000.0
    matrix = np.full((active_nodes.size, active_nodes.size), forbidden, dtype=np.float64)
    np.fill_diagonal(matrix, forbidden * 2.0)
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_id in range(src.shape[0]):
        local_u = int(local_of_global[int(src[edge_id])])
        local_v = int(local_of_global[int(dst[edge_id])])
        if local_u < 0 or local_v < 0:
            continue
        # The belief-propagation solver is sensitive to tied optima.  Add a
        # deterministic, numerically insignificant perturbation so uniform
        # edge scores (for example Karate with one-hot features) still have a
        # well-defined preference order.
        score_scale = max(1.0, abs(high), abs(low))
        jitter = (
            ((edge_id * 2_654_435_761 + 1_013_904_223) & 0xFFFFFFFF)
            / float(0xFFFFFFFF)
        ) * score_scale * 1e-7
        weighted_score = float(scores[edge_id]) + jitter
        matrix[local_u, local_v] = weighted_score
        matrix[local_v, local_u] = weighted_score
        edge_lookup[(min(local_u, local_v), max(local_u, local_v))] = edge_id

    binary = build_legacy_solver(scratch_root=scratch_root)
    temporary_root = scratch_root / "tmp" / "bmatching"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="legacy_", dir=temporary_root) as directory:
        work = Path(directory)
        weights_path = work / "weights.txt"
        degrees_path = work / "degrees.txt"
        output_path = work / "solution.txt"
        np.savetxt(weights_path, matrix, fmt="%.17g", delimiter="\t")
        np.savetxt(degrees_path, active_degrees, fmt="%d")
        cache_size = min(
            int(active_nodes.size),
            max(2, int(active_degrees.max(initial=1)) + 2),
        )
        environment = os.environ.copy()
        environment["BMATCH_NUM_THREADS"] = str(workers)
        command = [
            str(binary),
            "--weights",
            str(weights_path),
            "--degrees",
            str(degrees_path),
            "--total",
            str(active_nodes.size),
            "--output_file",
            str(output_path),
            "--cacheSize",
            str(cache_size),
            "--max_iter",
            str(max(1, int(max_iterations))),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                "legacy BMatching solver failed: "
                + (completed.stderr or completed.stdout or f"exit {completed.returncode}")
            )
        chosen: set[int] = set()
        for line in output_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            u, v = int(fields[0]), int(fields[1])
            if u == v:
                continue
            edge_id = edge_lookup.get((min(u, v), max(u, v)))
            if edge_id is not None:
                chosen.add(edge_id)

    refined = np.asarray(sorted(chosen), dtype=np.int64)
    if refined.shape[0] != scalable_selected.shape[0]:
        raise RuntimeError(
            "legacy solver returned "
            f"{refined.shape[0]} valid undirected input edges for "
            f"{scalable_selected.shape[0]} requested; using sparse fallback"
        )
    refined_degrees = np.bincount(
        src[refined], minlength=num_nodes
    ) + np.bincount(dst[refined], minlength=num_nodes)
    if not np.array_equal(refined_degrees, seed_degrees):
        raise RuntimeError(
            "legacy solver did not preserve the requested b-degree sequence"
        )
    return refined


def solve_bmatching(
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    num_nodes: int,
    *,
    target_edges: int | None = None,
    target_ratio: float = 0.1,
    backend: Backend = "auto",
    device: Device = "auto",
    num_workers: int | None = None,
    legacy_max_nodes: int = 512,
    legacy_max_iterations: int = 1_000,
    legacy_timeout_seconds: float = 120.0,
    scratch_root: str | Path | None = None,
) -> BMatchingResult:
    """Select a maximum-weight approximate b-matching from sparse input edges."""

    started = time.perf_counter()
    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"auto", "legacy", "scalable"}:
        raise ValueError("b-matching backend must be auto, legacy, or scalable")
    src, dst, scores, input_positions = _canonical_input(
        edge_index, edge_scores, int(num_nodes)
    )
    num_edges = int(src.shape[0])
    if target_edges is None:
        if not 0.0 <= float(target_ratio) <= 1.0:
            raise ValueError("target_ratio must be in [0, 1]")
        requested_edges = min(
            num_edges, max(0, math.ceil(num_edges * float(target_ratio)))
        )
    else:
        requested_edges = min(num_edges, max(0, int(target_edges)))

    workers = _configure_workers(num_workers)
    ordering_device = _resolve_ordering_device(device, num_edges)
    scalable_selected, capacities = _scalable_selection(
        src,
        dst,
        scores,
        num_nodes=int(num_nodes),
        target_edges=requested_edges,
        ordering_device=ordering_device,
    )

    use_legacy = normalized_backend == "legacy" or (
        normalized_backend == "auto"
        and int(num_nodes) <= max(2, int(legacy_max_nodes))
    )
    selected = scalable_selected
    actual_backend = "scalable"
    legacy_solver_ran = False
    message = ""
    if use_legacy and requested_edges > 0:
        if int(num_nodes) > max(2, int(legacy_max_nodes)):
            raise ValueError(
                f"legacy backend is limited to {legacy_max_nodes} nodes; "
                "use --bmatch-backend scalable for this graph"
            )
        try:
            legacy_solver_ran = True
            selected = _legacy_refinement(
                src,
                dst,
                scores,
                scalable_selected,
                num_nodes=int(num_nodes),
                workers=workers,
                scratch_root=Path(scratch_root or DEFAULT_SCRATCH_ROOT)
                .expanduser()
                .resolve(),
                max_iterations=legacy_max_iterations,
                timeout_seconds=legacy_timeout_seconds,
            )
            actual_backend = "legacy"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if normalized_backend == "legacy":
                raise
            message = f"legacy refinement unavailable; used scalable fallback: {exc}"

    selected_input = (
        np.sort(selected)
        if input_positions is None
        else np.sort(input_positions[selected])
    )
    selected_tensor = torch.from_numpy(selected_input.copy()).long()
    return BMatchingResult(
        selected_indices=selected_tensor,
        capacities=torch.from_numpy(capacities.copy()).long(),
        requested_edges=requested_edges,
        selected_edges=int(selected_tensor.numel()),
        backend=actual_backend,
        ordering_device=ordering_device,
        workers=workers,
        elapsed_seconds=time.perf_counter() - started,
        legacy_solver_ran=legacy_solver_ran,
        message=message,
    )


__all__ = [
    "BMatchingResult",
    "build_legacy_solver",
    "solve_bmatching",
]
