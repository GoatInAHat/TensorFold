"""The adapter delta in plain torch, a row at a time: the definition the Triton kernels are tested against."""

from __future__ import annotations

import torch


def lowrank_add(out: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                ranks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """out[r] += scales[i] * (b[i, :, :k] @ (a[i, :k] @ x[r])) for every row r whose slot i is not 0 (k = ranks[i]);
    other rows are not touched. ``a`` [slots, R, K], ``b`` [slots, D, R]; sums in ``dtype``; out is updated in place."""

    for r in range(out.shape[0]):
        i, k = int(ids[r]), 0
        if i:
            k = int(ranks[i])
        if k == 0:
            continue
        t = a[i, :k].to(dtype) @ x[r].to(dtype)
        out[r] += (float(scales[i]) * (b[i, :, :k].to(dtype) @ t)).to(out.dtype)
    return out
