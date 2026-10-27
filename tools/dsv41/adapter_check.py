"""Validate a LoRA adapter folder against a DeepSeek-V4.1 checkpoint without loading any weights, print what it edits and
what it costs a rank, and optionally write its tensorfold_adapter.json. Exit 1 with the reason when the server would refuse it.

  python3 adapter_check.py ADAPTER_DIR MODEL_DIR [--tp 2] [--slots 2] [--write-manifest] [--json]

``--slots`` is the number of adapters the server will hold (the tables keep a slot for each and one for the base model).
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensorfold.families.deepseek_v41.config import Cfg  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import lora_format as lf  # noqa: E402


def rank_bytes(spec: lf.AdapterSpec, dims: lf.Dims, world: int, slots: int) -> int:
    """Float32 table bytes one rank holds for this adapter alone: slots a module, the rank rounded up to fours."""

    total = 0
    for m in spec.modules:
        width, out = dims.shape(m.kind)
        total += (slots + 1) * lf.padded_rank(m.rank) * (width // world + out) * 4
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter")
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--write-manifest", action="store_true", help="write tensorfold_adapter.json from the files")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    model = pathlib.Path(args.model)
    cfg = Cfg.read(model)
    dims = lf.Dims.of(cfg)
    config = model / "config.json"
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    try:
        spec = lf.inspect_folder(args.adapter, dims, config_sha=sha)
        for module in spec.modules:                                  # a value that is not finite is refused at load
            lf.read_factors(spec, module)
    except lf.AdapterError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    report = spec.describe()
    report["base"] = {"model_type": "deepseek_v41", "hidden_size": dims.dim, "num_hidden_layers": dims.n_layers,
                      "config_sha256": sha}
    report["bytes_per_rank"] = rank_bytes(spec, dims, args.tp, args.slots)
    if args.write_manifest:
        manifest = {"format": lf.FORMAT, "base": report["base"], "files": {lf.WEIGHTS: {"sha256": spec.sha256}},
                    "modules": {m.name: {"rank": m.rank} for m in spec.modules},
                    "provenance": dict(spec.manifest.get("provenance") or {})}
        (pathlib.Path(args.adapter) / lf.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"ok: {len(spec.modules)} modules, weights sha256 {spec.sha256[:16]}")
    for m in spec.modules:
        print(f"  {m.name}: rank {m.rank}, alpha {m.alpha:g}, scale {m.scale:g}")
    print(f"float32 tables on a rank of {args.tp}, {args.slots} adapter(s) held: "
          f"{report['bytes_per_rank'] / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
