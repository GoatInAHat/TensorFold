"""One module's factors for every adapter slot of one layer, as the kernels read them (torch only, no Triton)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_format import PAD_J, TILE_J, padded_rank

__all__ = ["PAD_J", "TILE_J", "Tables", "padded_rank"]


@dataclass
class Tables:
    """Zero beyond a slot's rank; slot 0 (the stock model) is all zero and has rank 0."""

    a: torch.Tensor          # [slots, R, K] float32
    b: torch.Tensor          # [slots, D, R] float32
    ranks: torch.Tensor      # [slots] int32, 0 for the stock slot and for an adapter without this module
    scales: torch.Tensor     # [slots] float32
    folded: bool             # a reads the packed shared expert's rotated hidden (lora_place.fold_rotated)
    x_dtype: torch.dtype     # the dtype of the activations this module reads

    @property
    def slots(self) -> int:
        return int(self.a.shape[0])

    @property
    def R(self) -> int:
        return int(self.a.shape[1])

    @property
    def K(self) -> int:
        return int(self.a.shape[2])

    @property
    def D(self) -> int:
        return int(self.b.shape[1])

    @property
    def tile_j(self) -> int:
        return TILE_J

    @property
    def nbytes(self) -> int:
        return sum(int(t.numel() * t.element_size()) for t in (self.a, self.b, self.ranks, self.scales))

    @staticmethod
    def required_bytes(slots: int, rank: int, k: int, d: int) -> int:
        """Bytes for the padded fp32 factors, int32 ranks and fp32 scales, without allocating a tensor."""

        return int(slots) * (padded_rank(rank) * (int(k) + int(d)) + 2) * 4

    @classmethod
    def empty(cls, slots: int, rank: int, k: int, d: int, folded: bool, x_dtype: torch.dtype,
              device: str | torch.device = "cuda") -> Tables:
        """All-zero tables for ``slots`` slots of up to ``rank`` components (rounded up to PAD_J)."""

        r = padded_rank(rank)
        return cls(torch.zeros((slots, r, k), dtype=torch.float32, device=device),
                   torch.zeros((slots, d, r), dtype=torch.float32, device=device),
                   torch.zeros((slots,), dtype=torch.int32, device=device),
                   torch.zeros((slots,), dtype=torch.float32, device=device), folded, x_dtype)

    def put(self, slot: int, a, b, scale: float) -> None:
        """Slot ``slot``'s factors: a [rank, K], b [D, rank] (numpy or torch float32), delta = scale * b @ (a @ x)."""

        a, b = torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
        rank = int(a.shape[0])
        if slot < 1 or slot >= self.slots or rank > self.R or tuple(a.shape) != (rank, self.K) \
                or tuple(b.shape) != (self.D, rank):
            raise ValueError(f"slot {slot}: factors {tuple(a.shape)} / {tuple(b.shape)} do not fit tables "
                             f"[{self.slots}, {self.R}, {self.K}] / [{self.D}]")
        self.a[slot, :rank].copy_(a)
        self.b[slot, :, :rank].copy_(b)
        self.ranks[slot] = rank
        self.scales[slot] = float(scale)
