"""Constant sparse geometry must observe mutations under inference_mode too."""

import pytest
import torch
from motius.utils.deterministic_geometry import DeterministicSparseMatmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA geometry regression")
@pytest.mark.parametrize("layout", ["coo", "csr"])
@pytest.mark.parametrize("inference", [False, True])
def test_sparse_geometry_mutation_invalidates_dense_cache(layout, inference):
    context = torch.inference_mode if inference else torch.no_grad
    with context():
        dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device="cuda")
        sparse = dense.to_sparse() if layout == "coo" else dense.to_sparse_csr()
        right = torch.arange(12.0, device="cuda").reshape(3, 4)
        mode = DeterministicSparseMatmul()
        with mode:
            first = torch.mm(sparse, right)
        torch.testing.assert_close(first, dense @ right, rtol=0, atol=0)
        if layout == "coo":
            sparse._values().add_(1.0)
            sparse._indices()[1, 0] = 1
        else:
            sparse.values().add_(1.0)
            sparse.col_indices()[0] = 1
        expected = sparse.cpu().to_dense().cuda() @ right
        with mode:
            second = torch.mm(sparse, right)
        torch.testing.assert_close(second, expected, rtol=0, atol=0)
        assert mode.calls == 2
        if inference:
            assert not mode.cache
