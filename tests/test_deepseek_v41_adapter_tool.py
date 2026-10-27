"""tools/dsv41/adapter_check.py: the operator's offline check of an adapter folder against a checkpoint (no torch)."""

import importlib.util
import json
import pathlib
import struct

import pytest
from adapter_fakes import CFG, SHARED, WO_B, factors, full_config, standard, write_adapter

from tensorfold.families.deepseek_v41.cuda import lora_format as lf

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "dsv41" / "adapter_check.py"


def run(monkeypatch, capsys, *argv):
    spec = importlib.util.spec_from_file_location("adapter_check_under_test", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr("sys.argv", ["adapter_check.py", *map(str, argv)])
    code = tool.main()
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def model(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(full_config()))
    return tmp_path / "model"


def test_it_prints_the_modules_and_the_memory_a_rank_would_hold(tmp_path, model, monkeypatch, capsys):
    folder = standard(tmp_path / "a", ranks=(2, 3))
    code, out, _ = run(monkeypatch, capsys, folder, model, "--tp", "2", "--slots", "2")
    assert code == 0 and "ok: 2 modules" in out and "layers.1.attn.wo_b: rank 2" in out
    code, out, _ = run(monkeypatch, capsys, folder, model, "--json", "--tp", "2", "--slots", "1")
    report = json.loads(out)
    # slots+1 slots, rank rounded up to 4, float32: wo_b reads half of 1024 and writes 256, w2 half of 512 and writes 256
    assert report["bytes_per_rank"] == 2 * 4 * (1024 // 2 + 256) * 4 + 2 * 4 * (512 // 2 + 256) * 4
    assert set(report["modules"]) == {WO_B.format(1), SHARED.format(2)}


def test_a_written_manifest_is_accepted_by_the_server_and_its_provenance_is_kept(tmp_path, model, monkeypatch, capsys):
    folder = standard(tmp_path / "a", manifest={"format": lf.FORMAT, "provenance": {"made_by": "a tool"}})
    code, _, _ = run(monkeypatch, capsys, folder, model, "--write-manifest")
    assert code == 0
    written = json.loads((folder / lf.MANIFEST).read_text())
    assert written["provenance"] == {"made_by": "a tool"} and written["modules"][WO_B.format(1)] == {"rank": 2}
    assert lf.inspect_folder(folder, lf.Dims.of(CFG), config_sha=written["base"]["config_sha256"]).manifest == written
    assert run(monkeypatch, capsys, folder, model)[0] == 0                      # and it checks out again


def test_what_the_server_would_refuse_is_refused_with_the_reason_and_exit_one(tmp_path, model, monkeypatch, capsys):
    bad = write_adapter(tmp_path / "bad", {"mtp.0.attn.wo_b": factors(__import__("numpy").random.default_rng(0),
                                                                      CFG.dim, 1024, 2)})
    code, out, err = run(monkeypatch, capsys, bad, model)
    assert code == 1 and out == "" and "not a module this engine can edit" in err
    nan = write_adapter(tmp_path / "nan", {WO_B.format(1): (factors(__import__("numpy").random.default_rng(1),
                                                                    CFG.dim, 1024, 2)[0],
                                                            __import__("numpy").full((CFG.dim, 2), float("nan"),
                                                                                     dtype="float32"))})
    assert run(monkeypatch, capsys, nan, model)[0] == 1


@pytest.mark.parametrize("name,manifest,config,match", [
    ("base", {"format": lf.FORMAT, "base": "deepseek_v41"}, None, "base must be an object"),
    ("files", {"format": lf.FORMAT, "files": {lf.WEIGHTS: "abc"}}, None,
     "files.adapter_model.safetensors must be an object"),
    ("provenance", {"format": lf.FORMAT, "provenance": ["a"]}, None, "provenance must be an object"),
    ("activated", None, {"alora_invocation_tokens": [5, 6]}, "alora_invocation_tokens is set"),
])
def test_a_folder_of_the_wrong_shape_is_refused_with_the_reason_not_a_traceback(tmp_path, model, monkeypatch, capsys,
                                                                                name, manifest, config, match):
    folder = standard(tmp_path / name, manifest=manifest, config=config)
    before = sorted((p.name, p.read_bytes()) for p in folder.iterdir())
    code, out, err = run(monkeypatch, capsys, folder, model, "--write-manifest")
    assert code == 1 and out == "" and err.startswith("refused: ") and match in err
    assert sorted((p.name, p.read_bytes()) for p in folder.iterdir()) == before          # and nothing was written


def test_a_weights_header_that_is_not_an_object_is_refused_with_the_reason(tmp_path, model, monkeypatch, capsys):
    folder = standard(tmp_path / "a")
    (folder / lf.WEIGHTS).write_bytes(struct.pack("<Q", 2) + b"[]")
    code, out, err = run(monkeypatch, capsys, folder, model)
    assert code == 1 and out == "" and err.startswith("refused: ") and "header is not an object" in err
