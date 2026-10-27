"""DeepSeek-V4.1 adapter folders: what is read, what is refused, and that nothing is dropped quietly (no torch, no GPU)."""

import json
import struct

import numpy as np
import pytest
from adapter_fakes import CFG, PREFIX, SHARED, WO_B, factors, standard, write_adapter, write_safetensors

from tensorfold.families.deepseek_v41.cuda import lora_format as lf

DIMS = lf.Dims.of(CFG)
WO_IN = CFG.o_groups * CFG.o_rank


def rng(seed=0):
    return np.random.default_rng(seed)


def test_a_standard_adapter_is_read_with_each_modules_rank_and_scale(tmp_path):
    spec = lf.inspect_folder(standard(tmp_path / "a", ranks=(2, 3), config={"lora_alpha": 8}), DIMS)
    assert [(m.layer, m.kind, m.rank) for m in spec.modules] == [(1, "wo_b", 2), (2, "shared_w2", 3)]
    assert [m.scale for m in spec.modules] == [8 / 2, 8 / 3]
    assert spec.describe()["modules"]["layers.2.ffn.shared_experts.w2"] == {"rank": 3, "alpha": 8.0, "scale": 8 / 3}


def test_rank_and_alpha_patterns_give_each_module_its_own_scale(tmp_path):
    r = rng()
    mods = {WO_B.format(0): factors(r, CFG.dim, WO_IN, 2), WO_B.format(3): factors(r, CFG.dim, WO_IN, 4)}
    cfg = {"r": 2, "lora_alpha": 2, "rank_pattern": {r"layers\.3\.attn\.wo_b": 4},
           "alpha_pattern": {r"layers\.3\.attn\.wo_b": 1}}
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", mods, config=cfg), DIMS)
    assert [(m.rank, m.alpha, m.scale) for m in spec.modules] == [(2, 2.0, 1.0), (4, 1.0, 0.25)]


def test_rslora_scales_by_the_square_root_of_the_rank(tmp_path):
    spec = lf.inspect_folder(standard(tmp_path / "a", ranks=(4, 4), config={"use_rslora": True, "lora_alpha": 4,
                                                                         "r": 4}), DIMS)
    assert [m.scale for m in spec.modules] == [2.0, 2.0]


@pytest.mark.parametrize("prefix", [PREFIX, "base_model.model.model.", "model.", ""])
def test_the_common_prefixes_name_the_same_module(tmp_path, prefix):
    mods = {WO_B.format(2): factors(rng(), CFG.dim, WO_IN, 2)}
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", mods, prefix=prefix), DIMS)
    assert [(m.layer, m.kind) for m in spec.modules] == [(2, "wo_b")]


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_every_served_dtype_widens_exactly(tmp_path, dtype):
    folder = standard(tmp_path / "a", dtype=dtype)
    spec = lf.inspect_folder(folder, DIMS)
    header, base = lf.read_header(spec.weights)
    for m in spec.modules:
        a = lf.read_tensor(spec.weights, header[m.a_name], base)
        assert a.dtype == np.float32 and a.shape == (m.rank, DIMS.shape(m.kind)[0])
    if dtype == "BF16":                                   # bf16 values are the high half of the float32
        raw = np.fromfile(spec.weights, dtype=np.uint8)
        lo, hi = header[spec.modules[0].a_name]["data_offsets"]
        want = (raw[base + lo:base + hi].view("<u2").astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(lf.read_tensor(spec.weights, header[spec.modules[0].a_name], base).ravel(), want)


def test_factors_drop_every_all_zero_component_and_nothing_else(tmp_path):
    a, b = factors(rng(1), CFG.dim, WO_IN, 4)
    a[1] = 0.0                                            # no input read: the component adds nothing
    b[:, 3] = 0.0                                         # no output written
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", {WO_B.format(1): (a, b)}), DIMS)
    f = lf.read_factors(spec, spec.modules[0])
    assert f.rank == 2
    np.testing.assert_array_equal(f.a, a[[0, 2]])
    np.testing.assert_array_equal(f.b, b[:, [0, 2]])
    np.testing.assert_allclose(f.b @ f.a, b @ a, rtol=0, atol=0)    # exactly the same product


def test_an_adapter_that_edits_nothing_has_no_factors_left(tmp_path):
    a, b = factors(rng(2), CFG.dim, WO_IN, 2)
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", {WO_B.format(0): (a, 0 * b)}), DIMS)
    assert lf.read_factors(spec, spec.modules[0]) is None


def test_factors_with_nan_or_infinity_are_refused(tmp_path):
    a, b = factors(rng(3), CFG.dim, WO_IN, 2)
    b[5, 1] = np.inf
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", {WO_B.format(0): (a, b)}), DIMS)
    with pytest.raises(lf.AdapterError, match="NaN or infinity"):
        lf.read_factors(spec, spec.modules[0])


def refuse(tmp_path, match, mods=None, **kw):
    mods = mods if mods is not None else {WO_B.format(1): factors(rng(), CFG.dim, WO_IN, 2)}
    folder = write_adapter(tmp_path / "bad", mods, **kw)
    with pytest.raises(lf.AdapterError, match=match):
        lf.inspect_folder(folder, DIMS)


@pytest.mark.parametrize("name,match", [
    ("mtp.0.attn.wo_b", "not a module this engine can edit"),               # the draft blocks
    ("layers.1.ffn.experts.7.w2", "not a module this engine can edit"),     # a routed expert
    ("layers.1.ffn.shared_experts.w1", "not a module this engine can edit"),
    ("layers.1.attn.wq_b", "not a module this engine can edit"),
    ("layers.1.attn.wo_a", "not a module this engine can edit"),
    ("embed", "not a module this engine can edit"),
    ("head", "not a module this engine can edit"),
    ("layers.1.ffn.gate", "not a module this engine can edit"),
])
def test_every_module_but_the_two_writers_is_refused_by_name(tmp_path, name, match):
    refuse(tmp_path, match, {name: factors(rng(), CFG.dim, WO_IN, 2)})


def test_a_layer_past_the_model_is_refused(tmp_path):
    refuse(tmp_path, "layer 4 of 4", {WO_B.format(4): factors(rng(), CFG.dim, WO_IN, 2)})


def test_matrices_of_the_wrong_size_are_refused(tmp_path):
    refuse(tmp_path, "do not fit", {WO_B.format(1): factors(rng(), CFG.dim, WO_IN + 128, 2)})
    refuse(tmp_path, "do not fit", {SHARED.format(1): factors(rng(), CFG.dim + 128, CFG.inter, 2)})
    a, b = factors(rng(), CFG.dim, WO_IN, 2)
    refuse(tmp_path, "do not fit", {WO_B.format(1): (a, b[:, :1])})                      # A and B disagree on rank


def test_tensors_that_are_not_a_plain_lora_pair_are_refused(tmp_path):
    r = rng()
    for extra in ("layers.1.attn.wo_b.lora_magnitude_vector", f"{PREFIX}layers.1.attn.wo_b.bias",
                  f"{PREFIX}layers.1.attn.wo_b.lora_embedding_A", f"{PREFIX}layers.1.attn.wo_b.weight"):
        refuse(tmp_path, "not a LoRA A or B weight",
               {WO_B.format(1): factors(r, CFG.dim, WO_IN, 2)}, extra={extra: ("F32", [4], b"\0" * 16)})


def test_a_lone_half_or_a_twice_named_module_is_refused(tmp_path):
    a, _ = factors(rng(), CFG.dim, WO_IN, 2)
    folder = tmp_path / "half"
    folder.mkdir()
    write_safetensors(folder / "adapter_model.safetensors",
                      {f"{PREFIX}{WO_B.format(1)}.lora_A.weight": ("F32", list(a.shape), a.tobytes())})
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2}))
    with pytest.raises(lf.AdapterError, match="lora_B is missing"):
        lf.inspect_folder(folder, DIMS)
    a, b = factors(rng(), CFG.dim, WO_IN, 2)
    raw = {"F32": a.tobytes()}
    both = {f"{PREFIX}{WO_B.format(1)}.lora_A.weight": ("F32", list(a.shape), raw["F32"]),
            f"{PREFIX}{WO_B.format(1)}.lora_B.weight": ("F32", list(b.shape), b.tobytes()),
            f"model.{WO_B.format(1)}.lora_A.weight": ("F32", list(a.shape), raw["F32"]),
            f"model.{WO_B.format(1)}.lora_B.weight": ("F32", list(b.shape), b.tobytes())}
    folder = tmp_path / "twice"
    folder.mkdir()
    write_safetensors(folder / "adapter_model.safetensors", both)
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2}))
    with pytest.raises(lf.AdapterError, match="two A tensors"):
        lf.inspect_folder(folder, DIMS)


@pytest.mark.parametrize("config,match", [
    ({"use_dora": True}, "use_dora"), ({"fan_in_fan_out": True}, "fan_in_fan_out"),
    ({"bias": "all"}, "bias"), ({"lora_bias": True}, "lora_bias"),
    ({"modules_to_save": ["head"]}, "modules_to_save"), ({"layer_replication": [[0, 2]]}, "layer_replication"),
    ({"target_parameters": ["x"]}, "target_parameters"), ({"peft_type": "IA3"}, "peft_type must be LORA"),
    ({"r": 3}, "rank 2, .* says 3"), ({"lora_alpha": 0}, "lora_alpha"), ({"r": 0}, "r must"),
])
def test_config_features_the_engine_does_not_run_are_refused(tmp_path, config, match):
    refuse(tmp_path, match, config=config)


def test_a_rank_pattern_that_disagrees_with_the_tensors_is_refused(tmp_path):
    refuse(tmp_path, "rank 2, .* says 5", config={"rank_pattern": {"wo_b": 5}})


def test_a_pattern_that_is_not_a_regex_is_refused(tmp_path):
    refuse(tmp_path, "not a pattern", config={"alpha_pattern": {"(": 1}})


def test_rank_over_the_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(lf, "MAX_RANK", 1)
    refuse(tmp_path, "outside 1..1", {WO_B.format(1): factors(rng(), CFG.dim, WO_IN, 2)})


def test_files_that_are_not_adapters_are_refused(tmp_path):
    with pytest.raises(lf.AdapterError, match="not a directory"):
        lf.inspect_folder(tmp_path / "nowhere", DIMS)
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(lf.AdapterError, match="adapter_config.json is missing"):
        lf.inspect_folder(folder, DIMS)
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2}))
    with pytest.raises(lf.AdapterError, match="adapter_model.safetensors is missing"):
        lf.inspect_folder(folder, DIMS)
    (folder / "adapter_model.safetensors").write_bytes(b"not a safetensors file at all")
    with pytest.raises(lf.AdapterError, match="not a safetensors file"):
        lf.inspect_folder(folder, DIMS)


def test_a_tensor_that_does_not_fit_its_file_is_refused(tmp_path):
    folder = standard(tmp_path / "a")
    path = folder / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes()[:-9])                      # truncated
    with pytest.raises(lf.AdapterError, match="does not fit the file"):
        lf.inspect_folder(folder, DIMS)
    write_safetensors(path, {"x.lora_A.weight": ("I8", [2], b"\0\0")})
    with pytest.raises(lf.AdapterError, match="served as F32, F16 or BF16"):
        lf.inspect_folder(folder, DIMS)


def test_overlapping_tensors_are_refused(tmp_path):
    folder = standard(tmp_path / "a")
    path = folder / "adapter_model.safetensors"
    size = struct.unpack("<Q", path.read_bytes()[:8])[0]
    head = json.loads(path.read_bytes()[8:8 + size])
    names = [k for k in head if k != "__metadata__"]
    head[names[1]]["data_offsets"] = list(head[names[0]]["data_offsets"])
    head[names[1]]["shape"] = list(head[names[0]]["shape"])
    raw = json.dumps(head).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + path.read_bytes()[8 + size:])
    with pytest.raises(lf.AdapterError, match="overlap"):
        lf.inspect_folder(folder, DIMS)


# -- the manifest ------------------------------------------------------------------------------------------------
def declared(folder, **over):
    spec = lf.inspect_folder(folder, DIMS)
    manifest = {"format": lf.FORMAT, "base": {"model_type": "deepseek_v41", "hidden_size": CFG.dim,
                                              "num_hidden_layers": CFG.n_layers},
                "files": {lf.WEIGHTS: {"sha256": spec.sha256}},
                "modules": {m.name: {"rank": m.rank} for m in spec.modules}, "provenance": {"anything": 1}}
    manifest.update(over)
    (folder / lf.MANIFEST).write_text(json.dumps(manifest))
    return manifest


def test_a_declared_manifest_that_agrees_is_accepted_and_kept(tmp_path):
    folder = standard(tmp_path / "a")
    manifest = declared(folder)
    held = lf.inspect_folder(folder, DIMS).manifest
    assert held == manifest
    with pytest.raises(TypeError):                                    # a registry's view of it cannot be edited
        held["format"] = "other"


@pytest.mark.parametrize("change,match", [
    ({"format": "other"}, "format must be"),
    ({"files": {lf.WEIGHTS: {"sha256": "0" * 64}}}, "hashes to"),
    ({"base": {"hidden_size": 4096}}, "base hidden_size"),
    ({"base": {"num_hidden_layers": 61}}, "base num_hidden_layers"),
    ({"base": {"model_type": "llama"}}, "base model_type"),
    ({"base": {"config_sha256": "0" * 64}}, "another base config"),
    ({"modules": {"layers.1.attn.wo_b": {"rank": 2}}}, "not declared"),
    ({"modules": {"layers.1.attn.wo_b": {"rank": 9}, "layers.2.ffn.shared_experts.w2": {"rank": 3}}}, "other rank"),
    ({"modules": {"layers.1.attn.wo_b": {"rank": 2}, "layers.2.ffn.shared_experts.w2": {"rank": 3},
                  "layers.3.attn.wo_b": {"rank": 1}}}, "declared but absent"),
    ({"modules": []}, "modules must be an object"),
])
def test_a_manifest_that_disagrees_with_the_files_refuses_the_adapter(tmp_path, change, match):
    folder = standard(tmp_path / "a")
    declared(folder, **change)
    with pytest.raises(lf.AdapterError, match=match):
        lf.inspect_folder(folder, DIMS, config_sha="1" * 64)


def test_a_matching_base_config_hash_is_accepted(tmp_path):
    folder = standard(tmp_path / "a")
    declared(folder, base={"config_sha256": "1" * 64})
    assert lf.inspect_folder(folder, DIMS, config_sha="1" * 64).manifest["base"]["config_sha256"] == "1" * 64


def test_an_unreadable_manifest_is_refused(tmp_path):
    folder = standard(tmp_path / "a")
    (folder / lf.MANIFEST).write_text("{")
    with pytest.raises(lf.AdapterError, match="unreadable"):
        lf.inspect_folder(folder, DIMS)
