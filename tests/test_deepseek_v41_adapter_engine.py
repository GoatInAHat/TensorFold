"""The engine's adapter entry points without a GPU: what start refuses, what a name resolves to, and what generate does
with an adapter when no concurrent scheduler is there to carry it."""

from types import SimpleNamespace

import pytest
from adapter_fakes import CFG, model_dir, standard

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import engine as eng  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import lora_registry as lr  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import lora_runtime  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import model as _model  # noqa: E402


def bare(tmp_path, adapters=None):
    e = eng.DsEngine.__new__(eng.DsEngine)
    e.w, e.rank, e.world, e.model_dir = SimpleNamespace(cfg=CFG, layers=[]), 0, 2, model_dir(tmp_path / "m")
    e.model, e.adapters, e.scheduler = SimpleNamespace(lora=None), adapters, None
    e.request = SimpleNamespace(stop_eos=True)
    return e


def specs(tmp_path):
    return [("alpha", str(standard(tmp_path / "a", 1)), 1.0)]


def test_start_refuses_adapters_where_a_rows_slot_cannot_reach_the_forward(tmp_path, monkeypatch):
    e = bare(tmp_path)
    with pytest.raises(ValueError, match="--parallel 2 or more"):
        e._load_adapters(specs(tmp_path), (), 1)
    monkeypatch.setattr(_model, "KERNELS", False)
    with pytest.raises(ValueError, match="kernel path"):
        e._load_adapters(specs(tmp_path), (), 4)
    monkeypatch.setattr(_model, "KERNELS", True)
    monkeypatch.setattr(_model, "EXACT_MM", True)
    with pytest.raises(ValueError, match="kernel path"):
        e._load_adapters(specs(tmp_path), (), 4)
    assert e.model.lora is None


def test_start_validates_loads_and_warms_and_hands_back_the_registry(tmp_path, monkeypatch):
    seen = {}

    class Runtime:
        def warm(self):
            seen["warmed"] = True

    def load(registry, weights, rank, world, room=None, **kw):
        seen.update(registry=registry, rank=rank, world=world, room=room)
        return Runtime()

    monkeypatch.setattr(_model, "KERNELS", True)
    monkeypatch.setattr(_model, "EXACT_MM", False)
    monkeypatch.setattr(lora_runtime.LoraRuntime, "load", staticmethod(load))
    e = bare(tmp_path)
    registry = e._load_adapters(specs(tmp_path), ("base",), 2)
    assert registry.names() == ["alpha"] and seen["registry"] is registry and seen["warmed"]
    assert (seen["rank"], seen["world"]) == (0, 2) and isinstance(e.model.lora, Runtime)
    assert seen["room"] is None or isinstance(seen["room"](), float)
    with pytest.raises(Exception, match="already a model id"):
        bare(tmp_path / "again")._load_adapters(specs(tmp_path / "again"), ("alpha",), 2)


def test_a_name_resolves_to_a_reference_and_a_missing_one_is_an_error(tmp_path):
    registry = lr.AdapterRegistry.load(specs(tmp_path), CFG, model_dir(tmp_path / "m"), ("base",))
    e = bare(tmp_path, registry)
    ref = e.adapter_ref("alpha")
    assert ref == registry.ref("alpha") and e.adapter_ref(None) is None
    assert e.adapter_ok(ref) and e.adapter_ok(None) and not e.adapter_ok(lr.AdapterRef(1, "0" * 32))
    with pytest.raises(lr.UnknownAdapter, match="no adapter named"):
        e.adapter_ref("beta")
    without = bare(tmp_path / "plain")
    assert without.adapter_ok(None) and not without.adapter_ok(ref)
    with pytest.raises(lr.UnknownAdapter, match="without --adapter"):
        without.adapter_ref("alpha")


def test_generate_hands_the_reference_to_the_scheduler_and_refuses_it_without_one(tmp_path):
    registry = lr.AdapterRegistry.load(specs(tmp_path), CFG, model_dir(tmp_path / "m"), ("base",))
    e = bare(tmp_path, registry)
    got = {}
    e.scheduler = SimpleNamespace(submit=lambda *a, **k: got.update(k) or {"ok": 1})
    e.tower = None
    assert e.generate([1, 2], 4, None, lambda new: None, adapter="alpha") == {"ok": 1}
    assert got["adapter"] == registry.ref("alpha")
    e.generate([1, 2], 4, None, lambda new: None)
    assert got["adapter"] is None                                      # the base model: the reference is None
    e.scheduler = None
    with pytest.raises(ValueError, match="concurrent engine"):
        e.generate([1, 2], 4, None, lambda new: None, adapter="alpha")
