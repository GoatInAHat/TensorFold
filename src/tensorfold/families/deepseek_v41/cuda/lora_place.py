"""Where an adapter's factors live on a tensor-parallel rank, in numpy (no torch).

Both edited modules are row-parallel: a rank holds a slice of the input width and adds a partial sum to the layer's
existing cross-rank reduction, so the factor A is cut by columns and B is whole on every rank. The packed shared expert
needs one more step: its down projection reads the hidden state only after the expert kernels have scaled and rotated it
(``fold_rotated``), so A is rewritten once, at load, to read that form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lora_format import SHARED_W2, WO_B, Dims, Factors

HAD = 128                                 # the EXL3 rotation's block width


def hadamard() -> np.ndarray:
    """The 128 x 128 Walsh-Hadamard matrix in natural order divided by sqrt(128): symmetric, and its own inverse."""

    i = np.arange(HAD)
    both, par = i[:, None] & i[None, :], np.zeros((HAD, HAD), dtype=np.int64)
    for bit in range(7):
        par ^= (both >> bit) & 1
    return (1 - 2 * par) / math.sqrt(HAD)


def span(total: int, rank: int, world: int) -> tuple[int, int]:
    """The columns [lo, hi) of an input width that rank ``rank`` of ``world`` holds."""

    if world < 1 or not 0 <= rank < world or total % world or (total // world) % HAD:
        raise ValueError(f"input width {total} does not split into {world} ranks of whole {HAD}-wide blocks")
    per = total // world
    return rank * per, (rank + 1) * per


def fold_rotated(a: np.ndarray, suh: np.ndarray) -> np.ndarray:
    """A' with A' @ xd == A @ h, where xd = blockwise_hadamard(h * suh) is the form the expert kernels hand on: A' is
    the blockwise Hadamard of A / suh (computed in float64, stored float32)."""

    suh = np.asarray(suh, dtype=np.float64)
    if a.ndim != 2 or suh.shape != (a.shape[1],) or a.shape[1] % HAD:
        raise ValueError(f"cannot fold factors {a.shape} with a scale vector {suh.shape}")
    if not np.isfinite(suh).all() or not suh.all():
        raise ValueError("the expert's input scale vector holds zero or a value that is not finite")
    z = (a.astype(np.float64) / suh[None, :]).reshape(a.shape[0], -1, HAD) @ hadamard()
    return z.reshape(a.shape).astype(np.float32)


@dataclass(frozen=True)
class Placed:
    """One rank's share of a module: ``a`` [rank, local input], ``b`` [output, rank], delta = scale * b @ (a @ x)."""

    a: np.ndarray
    b: np.ndarray
    scale: float
    folded: bool                          # a reads the packed shared expert's rotated hidden, not the plain one

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])


def place(kind: str, factors: Factors, dims: Dims, rank: int, world: int, strength: float = 1.0,
          suh: np.ndarray | None = None) -> Placed:
    """This rank's slice of ``factors``. ``suh``: the packed shared expert's input scale vector for these columns."""

    if kind not in (WO_B, SHARED_W2):
        raise ValueError(f"no placement for {kind!r}")
    width, out = dims.shape(kind)
    if factors.a.shape[1] != width or factors.b.shape[0] != out:
        raise ValueError(f"factors {factors.a.shape} / {factors.b.shape} do not fit a [{out}, {width}] matrix")
    lo, hi = span(width, rank, world)
    a = np.ascontiguousarray(factors.a[:, lo:hi])
    if suh is not None:
        if kind != SHARED_W2:
            raise ValueError("only the packed shared expert reads a rotated input")
        a = fold_rotated(a, suh)
    return Placed(a, np.ascontiguousarray(factors.b), float(factors.scale) * float(strength), suh is not None)
