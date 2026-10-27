"""Adapter folders for the DeepSeek-V4.1 adapter tests: PEFT files written without torch, small model sizes."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# a small model: 4 layers, hidden 256, shared expert width 512, wo_b input 1024 (two ranks hold half of each)
CFG = SimpleNamespace(n_layers=4, dim=256, inter=512, o_groups=4, o_rank=256)
WO_B = "layers.{}.attn.wo_b"
SHARED = "layers.{}.ffn.shared_experts.w2"
PREFIX = "base_model.model."


def bf16_bytes(x: np.ndarray) -> bytes:
    """float32 -> bfloat16 bytes (round to nearest even)."""

    u = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()


def encode(x: np.ndarray, dtype: str) -> bytes:
    if dtype == "BF16":
        return bf16_bytes(x)
    return np.ascontiguousarray(x, dtype="<f2" if dtype == "F16" else "<f4").tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """``tensors``: name -> (dtype word, shape, raw bytes)."""

    header, off = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for _, _, raw in tensors.values():
            f.write(raw)


def factors(rng: np.random.Generator, out: int, inp: int, rank: int, scale: float = 0.05):
    return (rng.standard_normal((rank, inp)) * scale).astype(np.float32), \
        (rng.standard_normal((out, rank)) * scale).astype(np.float32)


def write_adapter(folder: Path, modules: dict[str, tuple[np.ndarray, np.ndarray]], *, dtype: str = "F32",
                  config: dict | None = None, manifest: dict | None = None, prefix: str = PREFIX,
                  extra: dict | None = None) -> Path:
    """An adapter folder: ``modules`` maps a module name (``layers.1.attn.wo_b``) to (A [r, in], B [out, r])."""

    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (a, b) in modules.items():
        tensors[f"{prefix}{name}.lora_A.weight"] = (dtype, list(a.shape), encode(a, dtype))
        tensors[f"{prefix}{name}.lora_B.weight"] = (dtype, list(b.shape), encode(b, dtype))
    for name, spec in (extra or {}).items():
        tensors[name] = spec
    write_safetensors(folder / "adapter_model.safetensors", tensors)
    ranks = {name: int(a.shape[0]) for name, (a, _) in modules.items()}
    rank = min(ranks.values(), default=1)                 # PEFT's default rank; the others come in a rank_pattern
    pattern = {re.escape(name): k for name, k in ranks.items() if k != rank}
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": rank, "bias": "none", "target_modules": ["wo_b", "w2"],
           "base_model_name_or_path": "base", **({"rank_pattern": pattern} if pattern else {}), **(config or {})}
    (folder / "adapter_config.json").write_text(json.dumps(cfg))
    if manifest is not None:
        (folder / "tensorfold_adapter.json").write_text(json.dumps(manifest))
    return folder


def standard(folder: Path, seed: int = 0, ranks: tuple[int, int] = (2, 3), dtype: str = "F32", **kw) -> Path:
    """Layer 1's wo_b at ranks[0] and layer 2's shared w2 at ranks[1]."""

    rng = np.random.default_rng(seed)
    wo_in = CFG.o_groups * CFG.o_rank
    mods = {WO_B.format(1): factors(rng, CFG.dim, wo_in, ranks[0]),
            SHARED.format(2): factors(rng, CFG.dim, CFG.inter, ranks[1])}
    return write_adapter(folder, mods, dtype=dtype, **kw)


def model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "hidden_size": CFG.dim}))
    return root


def full_config() -> dict:
    """A config.json the family's Cfg reads, at the small sizes of CFG."""

    return {"model_type": "deepseek_v41", "vocab_size": 1024, "hidden_size": CFG.dim, "moe_intermediate_size": CFG.inter,
            "num_hidden_layers": CFG.n_layers, "num_attention_heads": 4, "head_dim": 64, "qk_rope_head_dim": 16,
            "q_lora_rank": 64, "o_lora_rank": CFG.o_rank, "o_groups": CFG.o_groups, "sliding_window": 128,
            "rms_norm_eps": 1e-6, "n_routed_experts": 8, "num_experts_per_tok": 2, "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0, "compress_ratios": [0, 4, 128, 0], "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1], "index_n_heads": 4, "index_head_dim": 32, "index_topk": 16, "hc_mult": 4,
            "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rope_theta": 10000.0, "compress_rope_theta": 160000.0}
