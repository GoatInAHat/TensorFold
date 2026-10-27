"""A LoRA adapter folder read and validated for DeepSeek-V4.1: PEFT files plus an optional manifest, no torch.

An adapter edits the output projection of attention (``wo_b``) and the shared expert's down projection (``w2``) of any
backbone layer, at any rank. Every other module, and anything that is not a plain LoRA pair, is refused by name: an edit
is served as the file says or not at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

FORMAT = "tensorfold-adapter-1"
CONFIG, WEIGHTS, MANIFEST = "adapter_config.json", "adapter_model.safetensors", "tensorfold_adapter.json"
WO_B, SHARED_W2 = "wo_b", "shared_w2"
KINDS = (WO_B, SHARED_W2)
MAX_RANK = int(os.environ.get("TF_DS_ADAPTER_MAX_RANK") or 256)       # a module's rank, before zero components go
MAX_HEADER = 64 << 20
DTYPES = {"F32": 4, "F16": 2, "BF16": 2}
TILE_J = 16     # the kernels' rank tile, fixed: no other adapter's rank can change how a row's sums are grouped
PAD_J = 4       # tables hold ranks in multiples of this (a 16-byte row of the fp32 factors)

_PREFIXES = ("base_model.model.model.", "base_model.model.", "model.")
_KIND_NAMES = ((WO_B, re.compile(r"layers\.(\d+)\.attn\.wo_b")),
               (SHARED_W2, re.compile(r"layers\.(\d+)\.ffn\.shared_experts\.w2")))
_LORA = re.compile(r"(?P<module>.+)\.lora_(?P<part>[AB])\.weight")
_REFUSED = ("PEFT writes this field when the adapter is DoRA, has a bias, trains other modules or replicates layers; "
            "none of those is served")


class AdapterError(ValueError):
    """The adapter folder cannot be served as written."""


@dataclass(frozen=True)
class Dims:
    """The base model's sizes an adapter's tensors must match."""

    n_layers: int
    dim: int
    inter: int
    o_in: int                                  # o_groups * o_rank: wo_b's input width

    @classmethod
    def of(cls, cfg: Any) -> Dims:
        return cls(int(cfg.n_layers), int(cfg.dim), int(cfg.inter), int(cfg.o_groups) * int(cfg.o_rank))

    def shape(self, kind: str) -> tuple[int, int]:
        """(input width, output width) of a module kind's full matrix."""

        return (self.o_in if kind == WO_B else self.inter), self.dim


@dataclass(frozen=True)
class ModuleSpec:
    """One edited module as the file stores it. ``scale`` is alpha / r (alpha / sqrt(r) with rsLoRA)."""

    layer: int
    kind: str
    rank: int
    alpha: float
    scale: float
    a_name: str
    b_name: str

    @property
    def name(self) -> str:
        return module_name(self.layer, self.kind)


@dataclass(frozen=True)
class Factors:
    """A module's factors in float32 with every all-zero rank component removed: delta = scale * B @ (A @ x)."""

    a: np.ndarray                              # [rank, in]
    b: np.ndarray                              # [out, rank]
    scale: float

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])


@dataclass(frozen=True)
class AdapterSpec:
    path: str
    weights: str
    sha256: str                                # of the weights file
    modules: tuple[ModuleSpec, ...]
    manifest: Mapping[str, Any] = field(default_factory=dict)     # the folder's tensorfold_adapter.json (read-only)

    def describe(self) -> dict[str, Any]:
        """The normalized manifest: what this adapter edits, derived from the files (a declared one must agree)."""

        return {"format": FORMAT, "weights_sha256": self.sha256,
                "modules": {m.name: {"rank": m.rank, "alpha": m.alpha, "scale": m.scale} for m in self.modules}}


def padded_rank(rank: int) -> int:
    """The rank a table holds for a largest rank of ``rank``."""

    return -(-max(int(rank), 1) // PAD_J) * PAD_J


def module_name(layer: int, kind: str) -> str:
    return f"layers.{layer}.attn.wo_b" if kind == WO_B else f"layers.{layer}.ffn.shared_experts.w2"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path: str | Path) -> tuple[dict[str, dict], int]:
    """(tensor entries, data start) of a safetensors file; every offset checked against the file's size."""

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        n = int.from_bytes(raw, "little") if len(raw) == 8 else 0
        if not 2 <= n <= MAX_HEADER or 8 + n > size:
            raise AdapterError(f"{path}: not a safetensors file (header length {n})")
        try:
            header = json.loads(f.read(n))
        except ValueError as exc:
            raise AdapterError(f"{path}: unreadable safetensors header ({exc})") from exc
    header.pop("__metadata__", None)
    end, spans = size - 8 - n, []
    for name, e in header.items():
        try:
            dtype, shape, (lo, hi) = e["dtype"], [int(v) for v in e["shape"]], [int(v) for v in e["data_offsets"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"{path}: tensor {name!r} has a malformed entry") from exc
        if dtype not in DTYPES:
            raise AdapterError(f"{path}: tensor {name!r} is {dtype}; adapters are served as F32, F16 or BF16")
        count = math.prod(shape)
        if min(shape, default=1) < 0 or not 0 <= lo <= hi <= end or hi - lo != count * DTYPES[dtype]:
            raise AdapterError(f"{path}: tensor {name!r} does not fit the file (shape {shape}, bytes {lo}:{hi})")
        spans.append((lo, hi, name))
    spans.sort()
    for (_, hi, a), (lo, _, b) in zip(spans, spans[1:]):
        if lo < hi:
            raise AdapterError(f"{path}: tensors {a!r} and {b!r} overlap")
    return header, 8 + n


def read_tensor(path: str | Path, entry: Mapping[str, Any], base: int) -> np.ndarray:
    """One tensor as float32 (BF16 and F16 widen exactly), read without the rest of the file."""

    lo, hi = entry["data_offsets"]
    raw = np.fromfile(path, dtype=np.uint8, count=hi - lo, offset=base + lo)
    if raw.size != hi - lo:
        raise AdapterError(f"{path}: short read")
    shape = [int(v) for v in entry["shape"]]
    if entry["dtype"] == "BF16":
        out = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    elif entry["dtype"] == "F16":
        out = raw.view("<f2").astype(np.float32)
    else:
        out = raw.view("<f4").copy()
    return out.reshape(shape)


def _canonical(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _classify(module: str) -> tuple[int, str]:
    for kind, rx in _KIND_NAMES:
        m = rx.fullmatch(module)
        if m:
            return int(m.group(1)), kind
    raise AdapterError(f"{module}: not a module this engine can edit (only attention wo_b and the shared expert's w2 "
                       "of a backbone layer; MTP, routed-expert, router, embedding, head and norm edits are refused)")


def _pattern(table: Any, module: str, what: str) -> Any:
    """PEFT's rank_pattern / alpha_pattern value for a module: the first key that matches it, else None."""

    for key, value in (table or {}).items():
        try:
            if re.fullmatch(rf"(?:.*\.)?(?:{key})", module) or re.fullmatch(rf"(?:.*\.)?(?:{key})",
                                                                            "base_model.model." + module):
                return value
        except re.error as exc:
            raise AdapterError(f"{CONFIG}: {what} key {key!r} is not a pattern ({exc})") from exc
    return None


def read_config(folder: Path) -> dict[str, Any]:
    try:
        cfg = json.loads((folder / CONFIG).read_text())
    except (OSError, ValueError) as exc:
        raise AdapterError(f"{folder}: {CONFIG} is missing or unreadable ({exc})") from exc
    if not isinstance(cfg, dict) or str(cfg.get("peft_type", "")).upper() != "LORA":
        raise AdapterError(f"{folder / CONFIG}: peft_type must be LORA")
    for key in ("use_dora", "fan_in_fan_out", "lora_bias", "layer_replication", "modules_to_save",
                "target_parameters", "trainable_token_indices"):
        if cfg.get(key):
            raise AdapterError(f"{folder / CONFIG}: {key} is set. {_REFUSED}")
    if str(cfg.get("bias", "none")) != "none":
        raise AdapterError(f"{folder / CONFIG}: bias is {cfg['bias']!r}. {_REFUSED}")
    for key in ("r", "lora_alpha"):
        if not isinstance(cfg.get(key), (int, float)) or isinstance(cfg.get(key), bool) or cfg[key] <= 0:
            raise AdapterError(f"{folder / CONFIG}: {key} must be a positive number")
    return cfg


def _declared(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AdapterError(f"{path}: unreadable ({exc})") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise AdapterError(f"{path}: format must be {FORMAT!r}")
    return manifest


def _check_manifest(manifest: Mapping[str, Any], folder: Path, sha: str, modules: tuple[ModuleSpec, ...],
                    dims: Dims, config_sha: str | None) -> None:
    """A declared manifest is a contract: the file hash, the base model and the exact module set must agree."""

    if not manifest:
        return
    files = manifest.get("files") or {}
    want = (files.get(WEIGHTS) or {}).get("sha256") if isinstance(files, dict) else None
    if want is not None and want != sha:
        raise AdapterError(f"{folder / MANIFEST}: {WEIGHTS} hashes to {sha}, the manifest says {want}")
    base = manifest.get("base") or {}
    for key, have in (("model_type", "deepseek_v41"), ("hidden_size", dims.dim), ("num_hidden_layers", dims.n_layers)):
        if key in base and base[key] != have:
            raise AdapterError(f"{folder / MANIFEST}: base {key} is {base[key]!r}, this model's is {have!r}")
    if base.get("config_sha256") not in (None, config_sha):
        raise AdapterError(f"{folder / MANIFEST}: it was made for another base config (config.json sha256 differs)")
    declared = manifest.get("modules")
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise AdapterError(f"{folder / MANIFEST}: modules must be an object")
    found = {m.name: m.rank for m in modules}
    given = {k: (v.get("rank") if isinstance(v, dict) else v) for k, v in declared.items()}
    if given != found:
        extra, lost = sorted(set(found) - set(given)), sorted(set(given) - set(found))
        wrong = sorted(k for k in set(found) & set(given) if found[k] != given[k])
        raise AdapterError(f"{folder / MANIFEST}: the module list and the tensors differ (not declared: {extra[:4]}, "
                           f"declared but absent: {lost[:4]}, other rank: {wrong[:4]})")


def inspect_folder(folder: str | Path, dims: Dims, *, config_sha: str | None = None) -> AdapterSpec:
    """Validate an adapter folder against the base model's sizes. Raises AdapterError naming what is wrong."""

    folder = Path(folder)
    if not folder.is_dir():
        raise AdapterError(f"{folder}: not a directory")
    cfg = read_config(folder)
    weights = folder / WEIGHTS
    if not weights.is_file():
        raise AdapterError(f"{folder}: {WEIGHTS} is missing")
    manifest = _declared(folder)
    header, _ = read_header(weights)
    pairs: dict[str, dict[str, str]] = {}
    for key in header:
        m = _LORA.fullmatch(key)
        if m is None:
            raise AdapterError(f"{key}: not a LoRA A or B weight ({WEIGHTS} may hold only lora_A.weight and "
                               "lora_B.weight pairs; DoRA magnitudes, biases, embeddings and full weights are refused)")
        module = _canonical(m["module"])
        if m["part"] in pairs.setdefault(module, {}):
            raise AdapterError(f"{module}: two {m['part']} tensors")
        pairs[module][m["part"]] = key
    if not pairs:
        raise AdapterError(f"{weights}: no LoRA tensors")
    modules = []
    for module, ab in sorted(pairs.items()):
        layer, kind = _classify(module)
        if layer >= dims.n_layers:
            raise AdapterError(f"{module}: layer {layer} of {dims.n_layers}")
        if set(ab) != {"A", "B"}:
            raise AdapterError(f"{module}: lora_{'B' if 'A' in ab else 'A'} is missing")
        (a_in, b_out), (a_shape, b_shape) = dims.shape(kind), (header[ab["A"]]["shape"], header[ab["B"]]["shape"])
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != a_in or b_shape[0] != b_out or \
                a_shape[0] != b_shape[1]:
            raise AdapterError(f"{module}: lora_A {a_shape} and lora_B {b_shape} do not fit a [{b_out}, {a_in}] matrix")
        rank = int(a_shape[0])
        if not 1 <= rank <= MAX_RANK:
            raise AdapterError(f"{module}: rank {rank} outside 1..{MAX_RANK} (TF_DS_ADAPTER_MAX_RANK)")
        want = _pattern(cfg.get("rank_pattern"), module, "rank_pattern")
        want = cfg["r"] if want is None else want
        if want != rank:
            raise AdapterError(f"{module}: the tensors have rank {rank}, {CONFIG} says {want}")
        alpha = _pattern(cfg.get("alpha_pattern"), module, "alpha_pattern")
        alpha = float(cfg["lora_alpha"] if alpha is None else alpha)
        scale = alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank
        modules.append(ModuleSpec(layer, kind, rank, alpha, scale, ab["A"], ab["B"]))
    modules.sort(key=lambda m: (m.layer, m.kind))
    sha = sha256_file(weights)
    spec = tuple(modules)
    _check_manifest(manifest, folder, sha, spec, dims, config_sha)
    return AdapterSpec(str(folder), str(weights), sha, spec, MappingProxyType(manifest))


def read_factors(spec: AdapterSpec, module: ModuleSpec, parsed: tuple[dict, int] | None = None) -> Factors | None:
    """The module's factors in float32, all-zero components dropped (their product is exactly zero); None when
    nothing is left. A value that is not finite is refused: it would reach every reply. ``parsed``: read_header's
    result for the weights file, when the caller reads several modules."""

    header, base = parsed or read_header(spec.weights)
    a = read_tensor(spec.weights, header[module.a_name], base)
    b = read_tensor(spec.weights, header[module.b_name], base)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise AdapterError(f"{module.name}: factors hold NaN or infinity")
    live = np.flatnonzero(a.any(axis=1) & b.any(axis=0))
    if live.size == 0:
        return None
    return Factors(np.ascontiguousarray(a[live]), np.ascontiguousarray(b[:, live]), module.scale)
