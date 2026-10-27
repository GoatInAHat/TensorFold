"""The adapters a server holds: fixed at start, each with an identity that names its effect on every cached row.

Slot 0 is the stock model and has no entry. An identity key is the same on every rank and in every process that loads
the same files at the same strength; it is what keeps a kept prompt, a spilled prompt or a graph input from meeting
another adapter's rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .lora_format import FORMAT, AdapterError, AdapterSpec, Dims, inspect_folder

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}")


class UnknownAdapter(LookupError):
    """A request named an adapter this server does not hold."""


@dataclass(frozen=True)
class AdapterRef:
    """What travels with a request: the slot to run and the identity the slot must have on every rank."""

    slot: int
    key: str


@dataclass(frozen=True)
class AdapterEntry:
    slot: int
    name: str
    key: str
    strength: float
    spec: AdapterSpec

    @property
    def ref(self) -> AdapterRef:
        return AdapterRef(self.slot, self.key)

    @property
    def model_id(self) -> str:
        return self.name


def adapter_key(spec: AdapterSpec, strength: float) -> str:
    """32 hex digits that change with the factors' bytes, the module set and every module's effective scale."""

    h = hashlib.sha256(f"{FORMAT}\0{spec.sha256}\0".encode())
    for m in spec.modules:
        h.update(f"{m.name} {m.rank} {float(m.scale * strength).hex()}\0".encode())
    return h.hexdigest()[:32]


def slot_of(obj: Any) -> int:
    """The registry slot of a stream or reference (0: stock)."""

    ref = obj if isinstance(obj, AdapterRef) or obj is None else getattr(obj, "adapter", None)
    return 0 if ref is None else int(ref.slot)


def key_of(obj: Any) -> str:
    """The identity key of a stream or reference ('' for stock)."""

    ref = obj if isinstance(obj, AdapterRef) or obj is None else getattr(obj, "adapter", None)
    return "" if ref is None else str(ref.key)


class AdapterRegistry:
    """Immutable after construction: entries, lookups and a digest every rank must agree on."""

    def __init__(self, entries: Sequence[AdapterEntry], fingerprint: str = "") -> None:
        self.entries = tuple(entries)
        self.fingerprint = str(fingerprint)
        for want, e in enumerate(self.entries, 1):
            if e.slot != want:
                raise ValueError(f"adapter {e.name!r} holds slot {e.slot}, expected {want}")
        self._by_name = {e.name: e for e in self.entries}
        if len(self._by_name) != len(self.entries):
            raise ValueError("two adapters share a name")

    @classmethod
    def load(cls, specs: Iterable[tuple[str, str, float]], cfg: Any, model_dir: str | Path,
             reserved: Iterable[str] = ()) -> AdapterRegistry:
        """Read and validate ``(name, folder, strength)`` specs against the base model. ``reserved``: model ids a
        name may not take (--name and the aliases)."""

        dims, taken, entries = Dims.of(cfg), {str(r) for r in reserved}, []
        config = Path(model_dir) / "config.json"
        sha = hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else ""
        for slot, (name, folder, strength) in enumerate(specs, 1):
            if not NAME.fullmatch(str(name)):
                raise AdapterError(f"adapter name {name!r}: letters, digits and . _ : / - only, 1 to 96 characters")
            if name in taken:
                raise AdapterError(f"adapter name {name!r} is already a model id of this server")
            strength = float(strength)
            if not math.isfinite(strength) or strength == 0.0:
                raise AdapterError(f"adapter {name!r}: strength must be a finite number other than 0")
            taken.add(name)
            spec = inspect_folder(folder, dims, config_sha=sha or None)
            entries.append(AdapterEntry(slot, name, adapter_key(spec, strength), strength, spec))
        return cls(entries, sha)

    # -- lookups ------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def slots(self) -> int:
        """Slots a table needs: the stock slot and one a adapter."""

        return len(self.entries) + 1

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def model_map(self) -> dict[str, str]:
        """Model id -> adapter name, for the ids this registry adds to /v1/models."""

        return {e.model_id: e.name for e in self.entries}

    def ref(self, name: str | None) -> AdapterRef | None:
        """The reference for an adapter name (None: stock); UnknownAdapter when no such adapter is held."""

        if name is None:
            return None
        entry = self._by_name.get(name)
        if entry is None:
            raise UnknownAdapter(f"no adapter named {name!r} (this server holds: {', '.join(self.names()) or 'none'})")
        return entry.ref

    def entry(self, slot: int) -> AdapterEntry:
        return self.entries[int(slot) - 1]

    def check(self, ref: AdapterRef | None) -> bool:
        """Whether this registry holds exactly the adapter ``ref`` names (stock always)."""

        if ref is None:
            return True
        return 1 <= ref.slot <= len(self.entries) and self.entries[ref.slot - 1].key == ref.key

    def pick(self, body: Mapping[str, Any], models: Mapping[str, str]) -> str | None:
        """The adapter name a request selects: its model id when that is an adapter's, else its "adapter" field.
        The two may not name different adapters; an unknown name is an error, never a quiet fall back to stock."""

        asked = body.get("model")
        by_model = models.get(asked) if isinstance(asked, str) else None
        field = body.get("adapter")
        if field is not None and (not isinstance(field, str) or field not in self._by_name):
            raise UnknownAdapter(f"adapter {field!r} is not held by this server (it holds: "
                                 f"{', '.join(self.names()) or 'none'})")
        if field is not None and by_model is not None and field != by_model:
            raise UnknownAdapter(f"model {asked!r} is adapter {by_model!r}, but the request's adapter is {field!r}")
        return field if field is not None else by_model

    # -- agreement ----------------------------------------------------------------------------------------------
    def digest(self) -> str:
        """One hash of the base config, every slot, name and key: equal on two ranks exactly when they hold the same."""

        body = json.dumps([self.fingerprint, [[e.slot, e.name, e.key] for e in self.entries]], separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def words(self) -> list[int]:
        """The digest as four 31-bit integers (what the ranks' start-up check gathers)."""

        raw = int(self.digest()[:32], 16)
        return [(raw >> (31 * i)) & 0x7FFFFFFF for i in range(4)]

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for e in self.entries:
            kinds: dict[str, int] = {}
            for m in e.spec.modules:
                kinds[m.kind] = kinds.get(m.kind, 0) + 1
            out.append({"slot": e.slot, "name": e.name, "key": e.key, "strength": e.strength,
                        "modules": kinds, "max_rank": max(m.rank for m in e.spec.modules),
                        "weights_sha256": e.spec.sha256})
        return out
