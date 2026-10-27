"""Adapter factors as tables on a rank, and the hooks the model calls, on CPU tensors: each rank's share, the row-wise
sum over slots, the shared expert's two input forms, and that a stock row keeps its bits."""

from types import SimpleNamespace

import numpy as np
import pytest
from adapter_fakes import CFG, SHARED, WO_B, factors, model_dir, standard, write_adapter

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import lora_format as lf
from tensorfold.families.deepseek_v41.cuda import lora_place as lp
from tensorfold.families.deepseek_v41.cuda import lora_reference as ref
from tensorfold.families.deepseek_v41.cuda import lora_registry as lr
from tensorfold.families.deepseek_v41.cuda import lora_runtime as rt
from tensorfold.families.deepseek_v41.cuda.lora_tables import TILE_J, Tables, padded_rank

DIMS = lf.Dims.of(CFG)
WO_IN = DIMS.o_in


def kernel(out, x, ids, tab):
    return ref.lowrank_add(out, x, ids, tab.a, tab.b, tab.ranks, tab.scales, dtype=torch.float64)


def weights(packed=(2,), world=2, seed=0):
    """Layers whose shared expert is packed with the routed ones (has suh_d) or apart (``shared`` set)."""

    g = torch.Generator().manual_seed(seed)
    layers = []
    for i in range(CFG.n_layers):
        suh = (torch.rand((5, CFG.inter // world), generator=g) + 0.5) * torch.where(
            torch.rand((5, CFG.inter // world), generator=g) < 0.5, -1.0, 1.0)
        layers.append(SimpleNamespace(idx=i, shared=None if i in packed else ("w1", "w3", "w2"),
                                      experts=SimpleNamespace(count=5, suh_d=suh.to(torch.float16))))
    return SimpleNamespace(cfg=CFG, layers=layers)


@pytest.fixture
def registry(tmp_path):
    base = model_dir(tmp_path / "model")
    a = standard(tmp_path / "a", seed=1, ranks=(2, 3))
    rng = np.random.default_rng(2)
    b = write_adapter(tmp_path / "b", {WO_B.format(1): factors(rng, CFG.dim, WO_IN, 4),
                                       WO_B.format(3): factors(rng, CFG.dim, WO_IN, 1),
                                       SHARED.format(2): factors(rng, CFG.dim, CFG.inter, 2)})
    return lr.AdapterRegistry.load([("a", str(a), 1.0), ("b", str(b), 0.5)], CFG, base, ("model",))


def load(registry, rank=0, world=2, packed=(2,), **kw):
    return rt.LoraRuntime.load(registry, weights(packed, world), rank, world, device="cpu", log=lambda s: None,
                               kernel=kernel, **kw)


def file_factors(registry, slot, name):
    e = registry.entry(slot)
    m = next(m for m in e.spec.modules if m.name == name)
    return lf.read_factors(e.spec, m)


def test_tables_hold_ranks_in_fours_and_the_kernels_tile_is_one_constant():
    assert [padded_rank(r) for r in (0, 1, 2, 3, 4, 5, 16, 17, 33, 256)] == [4, 4, 4, 4, 4, 8, 16, 20, 36, 256]
    assert TILE_J == 16 and Tables.empty(2, 3, 128, 128, False, torch.float16, "cpu").tile_j == TILE_J
    assert Tables.empty(2, 200, 128, 128, False, torch.float16, "cpu").tile_j == TILE_J    # (not a function of the rank)


def test_tables_hold_exactly_what_was_put_and_zero_elsewhere():
    t = Tables.empty(3, 5, 256, 128, False, torch.bfloat16, "cpu")
    assert Tables.required_bytes(3, 5, 256, 128) == t.nbytes
    assert (t.slots, t.R, t.K, t.D, t.tile_j) == (3, 8, 256, 128, 16) and int(t.ranks.sum()) == 0
    a, b = np.ones((5, 256), np.float32), np.full((128, 5), 2.0, np.float32)
    t.put(2, a, b, 0.5)
    assert t.ranks.tolist() == [0, 0, 5] and t.scales.tolist() == [0.0, 0.0, 0.5]
    assert float(t.a[2, :5].sum()) == 5 * 256 and float(t.a[2, 5:].abs().sum()) == 0 and float(t.a[0].abs().sum()) == 0
    for bad in (0, 3, -1):
        with pytest.raises(ValueError):
            t.put(bad, a, b, 1.0)
    with pytest.raises(ValueError):
        t.put(1, np.ones((9, 256), np.float32), np.ones((128, 9), np.float32), 1.0)    # past the tables' rank
    with pytest.raises(ValueError):
        t.put(1, np.ones((5, 128), np.float32), b, 1.0)


@pytest.mark.parametrize("rank", [0, 1])
def test_each_rank_holds_its_cut_of_every_slot_and_the_stock_slot_holds_nothing(registry, rank):
    r = load(registry, rank)
    assert sorted(r.layers) == [1, 2, 3]
    wo1 = r.layers[1].wo_b
    assert wo1.slots == 3 and wo1.R == 4 and wo1.K == WO_IN // 2 and wo1.D == CFG.dim and not wo1.folded
    assert wo1.x_dtype == torch.bfloat16 and wo1.ranks.tolist() == [0, 2, 4]
    lo, hi = lp.span(WO_IN, rank, 2)
    fa, fb = file_factors(registry, 1, "layers.1.attn.wo_b"), file_factors(registry, 2, "layers.1.attn.wo_b")
    np.testing.assert_array_equal(wo1.a[1, :2].numpy(), fa.a[:, lo:hi])
    np.testing.assert_array_equal(wo1.b[1, :, :2].numpy(), fa.b)
    np.testing.assert_array_equal(wo1.a[2, :4].numpy(), fb.a[:, lo:hi])
    assert float(wo1.a[0].abs().sum()) == 0 and float(wo1.b[0].abs().sum()) == 0
    assert wo1.scales.tolist() == [0.0, fa.scale, fb.scale * 0.5]               # strength 0.5 on the second
    assert r.layers[3].wo_b.ranks.tolist() == [0, 0, 1] and r.layers[3].shared is None   # only adapter b edits layer 3
    assert r.layers[1].shared is None and r.nbytes > 0


def test_the_packed_shared_expert_gets_folded_factors_and_a_plain_one_does_not(registry):
    packed, apart = load(registry, 1, packed=(2,)), load(registry, 1, packed=())
    w = weights((2,), 2)
    tab = packed.layers[2].shared
    assert tab.folded and tab.x_dtype == torch.float16
    f = file_factors(registry, 1, "layers.2.ffn.shared_experts.w2")
    lo, hi = lp.span(CFG.inter, 1, 2)
    want = lp.fold_rotated(f.a[:, lo:hi], w.layers[2].experts.suh_d[4].float().numpy())    # the last expert: shared
    np.testing.assert_array_equal(tab.a[1, :3].numpy(), want)
    plain = apart.layers[2].shared
    assert not plain.folded and plain.x_dtype == torch.bfloat16
    np.testing.assert_array_equal(plain.a[1, :3].numpy(), f.a[:, lo:hi])


def test_the_hooks_add_each_rows_own_delta_and_a_stock_row_keeps_its_bits(registry):
    r = load(registry, 0)
    g = torch.Generator().manual_seed(3)
    lay = SimpleNamespace(idx=1)
    u = torch.randn((5, WO_IN // 2), generator=g).to(torch.bfloat16)
    out = torch.randn((5, CFG.dim), generator=g)
    rows = torch.tensor([0, 1, 2, 0, 1])
    got = r.bind(rows).wo_b(lay, u, out.clone())
    tab = r.layers[1].wo_b
    for i, slot in enumerate(rows.tolist()):
        if slot == 0:
            assert torch.equal(got[i], out[i])                      # not touched: the same bits
            continue
        k = int(tab.ranks[slot])
        t = tab.a[slot, :k].double() @ u[i].double()
        want = out[i].double() + float(tab.scales[slot]) * (tab.b[slot, :, :k].double() @ t)
        torch.testing.assert_close(got[i].double(), want, rtol=1e-6, atol=1e-6)
    assert not torch.equal(got[1], out[1]) and not torch.equal(got[2], out[2])


def test_the_hooks_run_nothing_for_a_layer_no_adapter_edits(registry):
    r = load(registry, 0)
    seen = []
    r.kernel = lambda *a: seen.append(a)
    out = torch.zeros((2, CFG.dim))
    rows = r.bind(torch.tensor([1, 2]))
    assert rows.wo_b(SimpleNamespace(idx=0), torch.zeros((2, WO_IN // 2), dtype=torch.bfloat16), out) is out
    assert rows.shared(SimpleNamespace(idx=0), torch.zeros((2, CFG.inter // 2)), out, packed=True) is out
    assert rows.wo_b(SimpleNamespace(idx=2), torch.zeros((2, WO_IN // 2), dtype=torch.bfloat16), out) is out   # no wo_b
    assert seen == []


def test_the_two_ranks_partials_add_up_to_the_merged_weight_result(registry):
    ranks = [load(registry, k) for k in (0, 1)]
    g = torch.Generator().manual_seed(4)
    lay = SimpleNamespace(idx=1)
    u = torch.randn((4, WO_IN), generator=g).to(torch.bfloat16)
    slots = torch.tensor([1, 2, 0, 2])
    total = torch.zeros((4, CFG.dim), dtype=torch.float64)
    for k, r in enumerate(ranks):
        half = u[:, k * WO_IN // 2:(k + 1) * WO_IN // 2].contiguous()
        total += r.bind(slots).wo_b(lay, half, torch.zeros((4, CFG.dim))).double()
    for i, slot in enumerate(slots.tolist()):
        if slot == 0:
            assert float(total[i].abs().sum()) == 0
            continue
        f = file_factors(registry, slot, "layers.1.attn.wo_b")
        merged = (f.b.astype(np.float64) @ f.a.astype(np.float64)) * f.scale * registry.entry(slot).strength
        np.testing.assert_allclose(total[i].numpy(), merged @ u[i].double().numpy(), rtol=1e-5, atol=1e-6)


def rotated(act, suh):
    z = (act.double() * suh.double()).reshape(act.shape[0], -1, 128) @ torch.from_numpy(lp.hadamard())
    return z.reshape(act.shape).to(torch.float16)


def test_both_shared_expert_forms_give_the_same_delta(registry):
    packed, apart = load(registry, 0, packed=(2,)), load(registry, 0, packed=())
    suh = weights((2,), 2).layers[2].experts.suh_d[4]
    g = torch.Generator().manual_seed(5)
    act = torch.randn((3, CFG.inter // 2), generator=g).to(torch.bfloat16)
    slots = torch.tensor([2, 0, 1])
    lay = SimpleNamespace(idx=2)
    plain = apart.bind(slots).shared(lay, act, torch.zeros((3, CFG.dim)), packed=False)
    folded = packed.bind(slots).shared(lay, rotated(act.float(), suh.float()), torch.zeros((3, CFG.dim)), packed=True)
    assert float(folded[1].abs().sum()) == 0 and float(plain[1].abs().sum()) == 0
    scale = plain.abs().max()
    assert (folded - plain).abs().max() <= 4e-3 * scale                # fp16 storage of the rotated hidden state
    assert float(plain[0].abs().sum()) > 0 and float(plain[2].abs().sum()) > 0


def test_a_shared_expert_read_in_the_wrong_form_is_an_error_not_a_wrong_delta(registry):
    r = load(registry, 0, packed=(2,))
    with pytest.raises(RuntimeError, match="placed for the rotated"):
        r.bind(torch.tensor([1])).shared(SimpleNamespace(idx=2), torch.zeros((1, CFG.inter // 2)),
                                         torch.zeros((1, CFG.dim)), packed=False)


def test_modules_whose_factors_are_all_zero_are_left_out(tmp_path):
    base = model_dir(tmp_path / "model")
    rng = np.random.default_rng(6)
    a, b = factors(rng, CFG.dim, WO_IN, 2)
    folder = write_adapter(tmp_path / "z", {WO_B.format(0): (a, 0 * b), WO_B.format(2): (a, b)})
    reg = lr.AdapterRegistry.load([("z", str(folder), 1.0)], CFG, base)
    logged = []
    r = rt.LoraRuntime.load(reg, weights(), 0, 2, device="cpu", log=logged.append, kernel=kernel)
    assert sorted(r.layers) == [2] and "1 all-zero modules left out" in logged[0]


def test_tables_that_would_not_fit_the_host_are_refused_before_they_are_allocated(registry, monkeypatch):
    with monkeypatch.context() as patch:
        def unexpected_allocation(*args, **kwargs):
            pytest.fail("factor tables allocated before the memory check")

        patch.setattr(Tables, "empty", unexpected_allocation)
        with pytest.raises(MemoryError, match="the host has 0.00 GiB"):
            load(registry, 0, room=lambda: 0.0)
    assert load(registry, 0, room=lambda: 64.0).nbytes > 0
    assert load(registry, 0, room=lambda: None).nbytes > 0               # unknown: no check


def test_an_engine_with_fewer_layers_than_the_adapter_edits_is_refused(registry):
    w = weights()
    w.layers = w.layers[:2]
    with pytest.raises(lf.AdapterError, match="holds 2 layers"):
        rt.LoraRuntime.load(registry, w, 0, 2, device="cpu", log=lambda s: None, kernel=kernel)


def test_uniform_rows_are_one_slot_and_stock_prompts_run_no_hook(registry):
    r = load(registry, 0)
    assert r.uniform(0, 7) is None
    rows = r.uniform(2, 5)
    assert rows.rows.dtype == torch.int64 and rows.rows.tolist() == [2] * 5 and rows.rt is r


def test_warm_runs_each_table_shape_once_on_zero_rows(registry):
    r = load(registry, 0)
    seen = []
    r.kernel = lambda out, x, ids, tab: seen.append((tuple(x.shape), x.dtype, tab.R, int(ids.sum()), float(out.sum())))
    n = r.warm()
    assert n == len(seen) == 2          # layers 1 and 3's wo_b share a shape (R=4); layer 2's shared w2 is packed (fp16)
    assert {(s[0][1], s[1], s[2]) for s in seen} == {(WO_IN // 2, torch.bfloat16, 4), (CFG.inter // 2, torch.float16, 4)}
    assert all(s[0][0] == 1 and s[3] == 0 and s[4] == 0 for s in seen)
