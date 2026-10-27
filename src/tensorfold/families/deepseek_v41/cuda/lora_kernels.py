"""Row-invariant Triton kernels for the adapter delta: a row's bits never depend on the other rows or on their adapters.

``delta[r] = scale[i] * B[i] @ (A[i] @ x[r])`` for the adapter slot ``i`` of row ``r``, summed in fp32 in a fixed order
over the input width and then over the rank, one program per row and tile. A row of slot 0 is not read or written, so a
stock row keeps its bits exactly. Slots, ranks and scales are read from device tables, and the slot of every row from a
device vector: a captured graph runs any mixture of adapters.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .lora_tables import TILE_J, Tables

TILE_K, TILE_N = 128, 128


@triton.jit(do_not_specialize=["xs", "IDS"])
def _project(X, xs, IDS, A, RANKS, T, K: tl.constexpr, R: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    """T[r, j] = sum_k A[slot, j, k] * x[r, k] for the rank tile of this program: fixed K order, fp32."""

    r = tl.program_id(0)
    jb = tl.program_id(1)
    ad = tl.load(IDS + r)
    rank = tl.load(RANKS + ad)
    j = jb * BJ + tl.arange(0, BJ)
    kk = tl.arange(0, BK)
    live = j < rank
    kend = tl.where(jb * BJ < rank, K, 0)
    acc = tl.zeros((BJ,), dtype=tl.float32)
    for k0 in range(0, kend, BK):
        ok = (k0 + kk) < K
        x = tl.load(X + r * xs + k0 + kk, mask=ok, other=0.0).to(tl.float32)
        a = tl.load(A + (ad * R + j[:, None]) * K + (k0 + kk)[None, :], mask=live[:, None] & ok[None, :], other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)
    tl.store(T + r * R + j, acc, mask=j < R)


@triton.jit(do_not_specialize=["os_", "IDS"])
def _apply(OUT, os_, IDS, B, T, RANKS, SCALES, D: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BJ: tl.constexpr):
    """out[r, n] += scale * sum_j B[slot, n, j] * T[r, j]: fixed rank order; a stock row's program does nothing."""

    r = tl.program_id(0)
    nb = tl.program_id(1)
    ad = tl.load(IDS + r)
    rank = tl.load(RANKS + ad)
    if rank > 0:
        nn = nb * BN + tl.arange(0, BN)
        jj = tl.arange(0, BJ)
        ok = nn < D
        acc = tl.zeros((BN,), dtype=tl.float32)
        for j0 in range(0, rank, BJ):
            live = (j0 + jj) < rank
            t = tl.load(T + r * R + j0 + jj, mask=live, other=0.0)
            b = tl.load(B + (ad * D + nn[:, None]) * R + (j0 + jj)[None, :], mask=ok[:, None] & live[None, :],
                        other=0.0)
            acc += tl.sum(b * t[None, :], axis=1)
        cur = tl.load(OUT + r * os_ + nn, mask=ok, other=0.0)
        tl.store(OUT + r * os_ + nn, cur + acc * tl.load(SCALES + ad), mask=ok)


def lowrank_add(out: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, tab: Tables) -> torch.Tensor:
    """out [n, D] fp32 += each row's adapter delta of x [n, K] (any float dtype, unit column stride); ``ids`` [n] int64
    holds each row's slot. Two launches, no host sync: safe inside a CUDA graph capture."""

    n = out.shape[0]
    if out.dtype != torch.float32 or out.dim() != 2 or out.shape[1] != tab.D or out.stride(1) != 1:
        raise ValueError(f"out must be a float32 [rows, {tab.D}] tensor with unit column stride")
    if x.dim() != 2 or x.shape != (n, tab.K) or x.stride(1) != 1 or x.dtype != tab.x_dtype:
        raise ValueError(f"x must be a {tab.x_dtype} [{n}, {tab.K}] tensor with unit column stride, got "
                         f"{x.dtype} {tuple(x.shape)}")
    if ids.dtype != torch.int64 or ids.dim() != 1 or ids.shape[0] != n or ids.stride(0) != 1:
        raise ValueError("ids must be a contiguous int64 vector with a slot a row")
    if n == 0:
        return out
    t = torch.empty((n, tab.R), dtype=torch.float32, device=out.device)
    # one tile shape for every table: how a row's sums are grouped never depends on the largest rank of any adapter
    _project[(n, triton.cdiv(tab.R, TILE_J))](x, x.stride(0), ids, tab.a, tab.ranks, t, K=tab.K, R=tab.R, BJ=TILE_J,
                                              BK=TILE_K, num_warps=4)
    _apply[(n, triton.cdiv(tab.D, TILE_N))](out, out.stride(0), ids, tab.b, t, tab.ranks, tab.scales, D=tab.D,
                                           R=tab.R, BN=TILE_N, BJ=TILE_J, num_warps=4)
    return out
