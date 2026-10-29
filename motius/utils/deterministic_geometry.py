"""A scoped deterministic runtime for the frozen sparse maps in body geometry.

CUDA's default sparse matrix product can vary between identical calls even with
PyTorch deterministic algorithms enabled. For reproducibility audits, evaluate
constant sparse maps as deterministic CPU-expanded matrices followed by the same
dense matrix product on the caller device. Original-source comparisons must use
the same runtime policy; it does not promise the rounding of an unconstrained
CUDA sparse kernel. All neural parameters and original Python graphs are intact.
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _tracked_version(tensor):
    # Inference tensors have no version counter. Do not cache them: callers can
    # mutate their values or sparse structure inside inference_mode.
    try:
        return tensor._version
    except RuntimeError:
        return None


class DeterministicSparseMatmul(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(func)
        if (
            name in {"aten.mm.default", "aten._sparse_mm.default", "aten.mm.out"}
            and len(args) >= 2
        ):
            matrix, right = args[:2]
            if (
                isinstance(matrix, torch.Tensor)
                and matrix.is_cuda
                and matrix.layout in {torch.sparse_coo, torch.sparse_csr}
                and right.layout == torch.strided
            ):
                if matrix.requires_grad:
                    raise ValueError(
                        "This deterministic geometry mode requires a constant sparse matrix"
                    )
                if matrix.layout == torch.sparse_coo:
                    values = matrix._values()
                    indices = matrix._indices()
                    structure = (indices.data_ptr(), _tracked_version(indices))
                else:
                    values = matrix.values()
                    columns = matrix.col_indices()
                    rows = matrix.crow_indices()
                    structure = (
                        columns.data_ptr(),
                        _tracked_version(columns),
                        rows.data_ptr(),
                        _tracked_version(rows),
                    )
                # Accessors dispatched below autograd can return fresh views
                # with zero counters; the parent tracks ordinary sparse edits.
                matrix_version = _tracked_version(matrix)
                key = (
                    id(matrix),
                    matrix_version,
                    values.data_ptr(),
                    _tracked_version(values),
                    structure,
                    tuple(matrix.shape),
                    matrix.dtype,
                    matrix.device,
                )
                cacheable = (
                    matrix_version is not None
                    and _tracked_version(values) is not None
                    and all(item is not None for item in structure)
                )
                cached = self.cache.get(key) if cacheable else None
                dense = None if cached is None else cached[0]
                if dense is None:
                    dense = matrix.detach().cpu().to_dense().to(matrix.device)
                    # Keep source storage alive so allocator pointer reuse cannot
                    # make a different matrix hit an earlier cache entry.
                    if cacheable:
                        self.cache[key] = (dense, matrix, values)
                self.calls += 1
                return torch.mm(dense, right, **kwargs)
        return func(*args, **kwargs)


def reproducible_geometry_when_requested(function):
    """Apply the reproducibility policy to a Pipeline's complete video graph."""
    from functools import wraps

    @wraps(function)
    def run(*args, **kwargs):
        if kwargs.get("deterministic", False):
            with DeterministicSparseMatmul():
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    return run
