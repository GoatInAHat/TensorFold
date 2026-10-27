"""Adapter factors on one rank's device, and the hooks the model calls. Loaded once before warm-up, never written again.

Every edited (layer, module) has one set of tables with a slot per adapter; a row's slot comes from a device vector,
so one captured graph serves any mixture of stock and adapter rows. The hooks add to the module's fp32 partial sum
before the layer's existing cross-rank reduction: no collective is added, and a layer no adapter edits runs nothing.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .lora_format import SHARED_W2, WO_B, AdapterError, Dims, read_factors, read_header
from .lora_place import place
from .lora_registry import AdapterRegistry
from .lora_tables import Tables


class LayerTables:
    """The tables of one layer's edited modules (None: no adapter edits it)."""

    __slots__ = ("wo_b", "shared")

    def __init__(self) -> None:
        self.wo_b: Tables | None = None
        self.shared: Tables | None = None


class RowAdapters:
    """One forward's rows and the adapter slot of each: ``rows`` is an int64 device vector, one slot a row."""

    __slots__ = ("rt", "rows")

    def __init__(self, rt: LoraRuntime, rows: torch.Tensor) -> None:
        self.rt, self.rows = rt, rows

    def wo_b(self, lay: Any, u: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``out`` [n, d] fp32, the layer's wo_b partial for input ``u`` [n, local groups * o_rank] bf16, plus the
        rows' adapter deltas."""

        t = self.rt.layers.get(lay.idx)
        if t is not None and t.wo_b is not None:
            self.rt.add(out, u, self.rows, t.wo_b)
        return out

    def shared(self, lay: Any, x: torch.Tensor, out: torch.Tensor, packed: bool) -> torch.Tensor:
        """``out`` plus the shared expert's w2 delta. ``x``: the expert kernels' rotated hidden (fp16, ``packed``) or
        the plain hidden (bf16) of a shared expert outside the pack."""

        t = self.rt.layers.get(lay.idx)
        tab = None if t is None else t.shared
        if tab is not None:
            if tab.folded != packed:
                raise RuntimeError(f"layer {lay.idx}: the shared expert's factors were placed for the "
                                   f"{'plain' if packed else 'rotated'} hidden state")
            self.rt.add(out, x, self.rows, tab)
        return out


class LoraRuntime:
    """The tables of every edited layer on this rank. ``kernel`` is the function adding a delta (Triton by default)."""

    def __init__(self, registry: AdapterRegistry, layers: dict[int, LayerTables], device: str = "cuda",
                 kernel: Callable | None = None) -> None:
        self.registry, self.layers, self.device, self.kernel = registry, layers, device, kernel

    @property
    def nbytes(self) -> int:
        return sum(tab.nbytes for t in self.layers.values() for tab in (t.wo_b, t.shared) if tab is not None)

    def add(self, out: torch.Tensor, x: torch.Tensor, rows: torch.Tensor, tab: Tables) -> torch.Tensor:
        fn = self.kernel
        if fn is None:
            from .lora_kernels import lowrank_add as fn

            self.kernel = fn
        return fn(out, x, rows, tab)

    def uniform(self, slot: int, n: int) -> RowAdapters | None:
        """Rows [n] that all run adapter ``slot`` (a prompt chunk's); None for the stock slot: it runs no hook."""

        if not slot:
            return None
        return RowAdapters(self, torch.full((n,), int(slot), dtype=torch.int64, device=self.device))

    def bind(self, rows: torch.Tensor) -> RowAdapters:
        """Rows whose slots a graph's input vector holds (read at replay, not at capture)."""

        return RowAdapters(self, rows)

    def warm(self) -> int:
        """Build every kernel variant the model will launch (one launch each on zero rows), so no request compiles one.
        Returns the number of table shapes warmed."""

        seen = set()
        for t in self.layers.values():
            for tab in (t.wo_b, t.shared):
                if tab is None:
                    continue
                key = (tab.K, tab.D, tab.R, tab.x_dtype)
                if key in seen:
                    continue
                seen.add(key)
                out = torch.zeros((1, tab.D), dtype=torch.float32, device=self.device)
                x = torch.zeros((1, tab.K), dtype=tab.x_dtype, device=self.device)
                self.add(out, x, torch.zeros((1,), dtype=torch.int64, device=self.device), tab)
        if self.device != "cpu":
            torch.cuda.synchronize()
        return len(seen)

    def describe(self) -> dict[str, Any]:
        return {"adapters": self.registry.describe(), "layers": len(self.layers), "bytes": self.nbytes}

    @classmethod
    def load(cls, registry: AdapterRegistry, weights: Any, rank: int, world: int, *, device: str = "cuda",
             log: Callable[[str], None] = print, kernel: Callable | None = None,
             room: Callable[[], float | None] | None = None) -> LoraRuntime:
        """Read each adapter's factors, place this rank's share and upload it, one (layer, module) at a time (no
        second copy of an adapter in host memory). ``room()``: GiB the host can still give, None if unknown; tables
        that would not leave the floor that remains are refused before they are allocated."""

        t0 = time.perf_counter()
        dims = Dims.of(weights.cfg)
        wanted: dict[tuple[int, str], list] = {}
        for e in registry.entries:
            for m in e.spec.modules:
                wanted.setdefault((m.layer, m.kind), []).append((e, m))
        headers = {e.slot: read_header(e.spec.weights) for e in registry.entries}
        layers: dict[int, LayerTables] = {}
        total = dropped = 0
        for (layer, kind), items in sorted(wanted.items()):
            if layer >= len(weights.layers):
                raise AdapterError(f"layer {layer}: this engine holds {len(weights.layers)} layers")
            lay = weights.layers[layer]
            loaded = []
            for entry, module in items:
                f = read_factors(entry.spec, module, headers[entry.slot])
                if f is None:
                    dropped += 1
                else:
                    loaded.append((entry, f))
            if not loaded:
                continue
            packed = kind == SHARED_W2 and getattr(lay, "shared", None) is None
            suh = lay.experts.suh_d[lay.experts.count - 1].float().cpu().numpy() if packed else None
            width, out = dims.shape(kind)
            dtype = torch.float16 if packed else torch.bfloat16
            max_rank = max(f.rank for _, f in loaded)
            needed = Tables.required_bytes(registry.slots, max_rank, width // world, out)
            free = None if room is None else room()
            if free is not None and needed > free * 2**30:
                raise MemoryError(f"adapter factors for layer {layer} {kind} need {needed / 2**20:.0f} MiB, the "
                                  f"host has {free:.2f} GiB to give above its floor")
            tab = Tables.empty(registry.slots, max_rank, width // world, out, packed, dtype, device)
            for entry, f in loaded:
                p = place(kind, f, dims, rank, world, entry.strength, suh)
                tab.put(entry.slot, p.a, p.b, p.scale)
            total += tab.nbytes
            slot = layers.setdefault(layer, LayerTables())
            if kind == WO_B:
                slot.wo_b = tab
            else:
                slot.shared = tab
        rt = cls(registry, layers, device, kernel)
        names = ", ".join(f"{e.name} ({len(e.spec.modules)} modules)" for e in registry.entries)
        log(f"[tensorfold] rank {rank}: adapters {names}: {len(layers)} layers, {total / 2**20:.1f} MiB of factors"
            f"{f', {dropped} all-zero modules left out' if dropped else ''} in {time.perf_counter() - t0:.1f} s")
        return rt
