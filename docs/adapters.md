# LoRA adapters on DeepSeek-V4.1

`tensorfold serve --adapter NAME=FOLDER` serves one or more LoRA adapters beside the base model from one set of
weights. The EXL3 weights are not touched: one TP2 allocation, one shared KV pool, the same context, MTP, vision and
Engram. Each adapter is a model id of its own, and every request picks the base model or an adapter. Requests for
different adapters (and for the base model) decode in the same rounds, in the same CUDA graphs.

```bash
tensorfold serve DeepSeek-V4.1-Flash-EXL3 --parallel 4 --tp 2 --master <rank 0 address> \
    --adapter edit-a=/models/adapters/edit-a --adapter edit-b=/models/adapters/edit-b --adapter-strength edit-b=0.5
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -d '{"model": "edit-a", "messages": [{"role": "user", "content": "Hi"}]}'
curl http://127.0.0.1:8080/v1/chat/completions -d '{"model": "DeepSeek-V4.1-Flash-EXL3", "adapter": "edit-b", ...}'
```

## What an adapter may edit

An adapter edits the **output projection of attention** (`layers.N.attn.wo_b`) and the **shared expert's down
projection** (`layers.N.ffn.shared_experts.w2`) of backbone layers, at any rank, each module with its own rank and
scale. `delta = scale * B @ (A @ x)` is added to the module's fp32 partial sum before the cross-rank reduction the layer
already makes; no collective is added.

Everything else is refused when the server starts, by name: routed experts, the router, the shared expert's gate and up
projections, other attention projections, the compressor and indexer, embeddings, the head, norms, hyper-connection
parameters, Engram, the vision tower, and every MTP/DSpark block (`mtp.*`). A file that is not a plain, always-on LoRA
pair (DoRA magnitudes, biases, full weights, `modules_to_save`, `fan_in_fan_out`, layer replication, and PEFT's
activated LoRA, whose `alora_invocation_tokens` make it edit only after those tokens) is refused too. Nothing is
dropped quietly. The drafter always runs the base model, so an adapted reply is still the adapted target's own: drafts
are verified against it, and a weaker draft only costs acceptance.

## The folder

PEFT's two files, and optionally a manifest:

- `adapter_config.json`: `peft_type: LORA`, `r`, `lora_alpha`, and optionally `rank_pattern`, `alpha_pattern` and
  `use_rslora`. The scale of a module is `alpha / rank` (`alpha / sqrt(rank)` with rsLoRA).
- `adapter_model.safetensors`: `<prefix>layers.N.attn.wo_b.lora_A.weight` `[r, 8192]` and `lora_B.weight` `[5120, r]`, and
  the same for `layers.N.ffn.shared_experts.w2` (`[r, 2304]` and `[5120, r]`). F32, F16 and BF16 are read; the prefixes
  `base_model.model.`, `base_model.model.model.` and `model.` are accepted.
- `tensorfold_adapter.json` (optional) is a contract: `{"format": "tensorfold-adapter-1", "files": {"adapter_model.safetensors":
  {"sha256": ...}}, "base": {"model_type": ..., "hidden_size": ..., "num_hidden_layers": ..., "config_sha256": ...},
  "modules": {"layers.10.attn.wo_b": {"rank": 2}, ...}, "provenance": {...}}`. What it declares must match the files
  (the weights' hash, the base model, and exactly this set of modules at exactly these ranks), or the server does not
  start; a member of another type than the one shown (`files`, `base` or `provenance` not an object) is refused the same
  way. Without it the same checks run on what the files say; either way the normalized manifest is what the server
  holds.

At load, a rank component whose A row or B column is all zero is dropped (its product is exactly zero), and a module
with nothing left is left out; a value that is not finite is refused.

`python3 tools/dsv41/adapter_check.py FOLDER MODEL_DIR --tp 2 --slots 2` runs the same checks offline, with no weights
loaded: it prints the modules, the table memory a rank would hold, and exits 1 with the reason where the server would
refuse. `--write-manifest` writes `tensorfold_adapter.json` from the files (keeping an existing `provenance`).

## Tensor parallelism

Both modules are row-parallel: a rank holds its slice of the input width (`wo_b`: its attention groups; `w2`: its half
of the shared expert's intermediate width), so **A is cut by columns and B is whole on every rank**. The ranks' partial
deltas sum in the reduction the layer already makes. Each rank reads the files itself and keeps its share in fp32 tables.

The delta is `scale * B @ (A @ x)` summed in fp32 from the serialized factors (F16 and BF16 widen exactly). PEFT's own
forward rounds the rank-sized intermediate to the activation dtype; this one does not.

The shared expert is usually packed with the routed experts, and its down projection then reads the hidden state only
after the expert kernels have scaled it by the matrix's input vector and rotated it (a 128-wide Hadamard). At load, A is
rewritten once to read that form (`A' = H (A / suh)`, float64 arithmetic, stored float32), so no hidden state is
rebuilt at run time. The delta of a packed shared expert is therefore computed from the fp16 value the expert kernels
store (relative error at most 2^-11 per element); a shared expert outside the pack reads its bf16 hidden state as is.

On start every rank gathers a digest of the base config and each adapter's slot, name and identity key, and refuses to
serve if they differ. Slots follow the order of the `--adapter` flags, so every rank needs the same flags in the same
order: two adapters swapped is refused too. Each admission carries `[slot, key]` to the followers, and every step's
agreement check includes the slots of the live streams, so a rank that disagrees fails the step on every rank before any
collective.

## Graphs and rows

Slot 0 is the base model. A round's inputs carry one adapter slot a row (a sixth row of the graph's input buffer, read at
replay), padded rows copy their original's slot, and prompt chunks run one slot for all their rows. Two Triton kernels
add `scale * B @ (A @ x)` for a row's slot: one program per row and tile, fp32 sums in a fixed order, one tile shape for
every table. By construction a row's delta depends on its own input and its adapter's factors only: not on the other
rows, their adapters, the padding, the chunk it was prefilled in, or the largest rank of any other adapter. A row of
slot 0 is neither read nor written. The CPU tests check a row alone, reordered and padded against the whole batch, and
stock rows left untouched, under Triton's interpreter. The same on the compiled kernels, in prompt chunks, in captured
graphs and beside a larger rank is what the GPU tests (`tests/cuda/test_deepseek_v41_adapter_kernels.py`) and, on a
running server, `tools/dsv41/adapter_mix.py` check; they need a GPU. A layer no adapter edits runs nothing. The kernels
are built at start, before the graph warm-up.

## Caches

The KV pool is one pool, but a row computed under an adapter is that adapter's. Kept prompts (`TF_DS_KEEP`) and
spilled prompts (`TF_DS_SPILL_GIB`) carry the identity key of the adapter that made them: a prompt continues only a
kept or spilled prompt of the same key, and the spill's prefix hashes start with the key (the base model's hashes are
unchanged). The key changes with the weights file's bytes, the module set and every module's scale (including
`--adapter-strength`), never with the name. Adding this feature changes the engine identity hash of the spill
directory, so an existing spill directory is rebuilt once.

## Memory

Each rank holds, per edited module and layer, (held adapters + 1) slots of the largest rank any adapter has there
(rounded up to a multiple of 4), each `(local input width + output width) * 4` bytes per rank component. The tables are
made before the warm-up and counted by the allocator ceiling. Start refuses tables that would leave the host under
`TF_DS_MEM_FLOOR_GIB` plus one gibibyte of its available memory, before allocating them. `TF_DS_ADAPTER_MAX_RANK`
(default 256) bounds a module's rank.

## Limits

- `--parallel 2` or more is required (the single-stream engine has no per-row input). `TF_DS_KERNELS=0` and
  `TF_DS_EXACT_MM=1`, the test modes, are refused with `--adapter`.
- Adapters are fixed at start: no loading or unloading at run time.
- Logprobs, as under `--parallel` for the base model, are not served.
- Correctness here means the adapted model is the base weights plus the serialized factors, run as written. It does not
  claim that a checkpoint quantized again from merged weights would give the same activations; compare those separately.

## Checking a running server

Every adapter is a model id, so `python3 tools/bench_concurrent.py URL ADAPTER_ID --alone --serial` checks that an
adapter's concurrent replies equal their solo runs and drafted equals `"draft": false`, as for any model.
`python3 tools/dsv41/adapter_mix.py URL --models BASE,adapter-a,adapter-b --streams 4` checks the mixture: the ids rotate
over the streams, and each stream's token hash must be its own solo run's wherever it sits and whatever runs beside it.
To check that the flag changes nothing for the base model, compare the base model's hashes from a server with and
without `--adapter`.

## Tests

CPU: `tests/test_deepseek_v41_adapter_*.py` (folder validation and refusals, registry and identities, tensor-parallel
placement and the shared-expert fold in numpy, tables and hooks, the round columns, kept and spilled prompt scoping, the
admission ops, server and CLI selection, and the kernels under Triton's CPU interpreter). GPU: `tests/cuda/test_deepseek_v41_adapter_*.py`
(the compiled kernels, graph replay with changing slots, the fold against the real EXL3 expert kernels, and two ranks
over gloo).
