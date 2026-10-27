"""The hooks in the DeepSeek-V4.1 model code: wo_b's delta is computed from the very input wo_b consumed, and the shared
expert's from the rotated hidden state still in the expert scratch (packed) or the plain hidden state (apart). The model
functions run on CPU tensors with the GPU kernels and expert extension replaced by stand-ins that write known values."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F  # noqa: E402

from tensorfold.families.deepseek_v41.cuda import lora_reference as ref  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import lora_runtime as rt  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import model as M  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.lora_tables import Tables  # noqa: E402

D, K_IN, I, E, TOPK = 8, 6, 8, 5, 2
SLOTS = TOPK + 1
G = torch.Generator().manual_seed(0)


def kernel(out, x, ids, tab):
    return ref.lowrank_add(out, x, ids, tab.a, tab.b, tab.ranks, tab.scales, dtype=torch.float64)


def tables(k, x_dtype, folded, seed):
    g = torch.Generator().manual_seed(seed)
    t = Tables.empty(3, 3, k, D, folded, x_dtype, "cpu")
    t.put(1, torch.randn((2, k), generator=g), torch.randn((D, 2), generator=g), 0.5)
    t.put(2, torch.randn((3, k), generator=g), torch.randn((D, 3), generator=g), -1.0)
    return t


def runtime(idx, wo_b=None, shared=None):
    layer = rt.LayerTables()
    layer.wo_b, layer.shared = wo_b, shared
    return rt.LoraRuntime(None, {idx: layer}, "cpu", kernel)


def expected(tab, x, slots, out):
    want = out.clone().double()
    for i, s in enumerate(slots):
        if s:
            k = int(tab.ranks[s])
            want[i] += float(tab.scales[s]) * (tab.b[s, :, :k].double() @ (tab.a[s, :k].double() @ x[i].double()))
    return want


# -- wo_b ---------------------------------------------------------------------------------------------------------
class FakeGroup:
    """Exl3Group stand-in: writes known values into wo_a's output slices."""

    def __init__(self, values):
        self.values = values

    def rotated(self, xh, outs, rot=None):
        c = 0
        for o in outs:
            o.copy_(self.values[:, c:c + o.shape[1]])
            c += o.shape[1]
        return outs


def test_wo_ab_adds_the_delta_of_the_input_wo_b_consumed(monkeypatch):
    n = 4
    u_vals = torch.randn((n, K_IN), generator=G).to(torch.bfloat16)
    monkeypatch.setattr(M, "_group", lambda lay, key, layers: FakeGroup(u_vals))
    w = torch.randn((D, K_IN), generator=G).to(torch.bfloat16)
    lay = SimpleNamespace(idx=3, wo_a=[SimpleNamespace(n=3), SimpleNamespace(n=3)], wo_b=SimpleNamespace(weight=w))
    xh = [torch.zeros((n, 5), dtype=torch.float16)]
    base = M.wo_ab(lay, xh=xh, fold=False)
    assert torch.equal(base, u_vals.float() @ w.float().t())                      # without an adapter: as it was
    tab = tables(K_IN, torch.bfloat16, False, 1)
    slots = [0, 1, 2, 1]
    got = M.wo_ab(lay, xh=xh, fold=False, adapter=runtime(3, wo_b=tab).bind(torch.tensor(slots)))
    torch.testing.assert_close(got.double(), expected(tab, u_vals, slots, base), rtol=1e-5, atol=1e-5)
    assert torch.equal(got[0], base[0])                                           # the stock row: the same bits
    assert M.wo_ab(lay, xh=xh, fold=False, adapter=runtime(9, wo_b=tab).bind(torch.tensor(slots))).equal(base)   # layer 9: none


# -- the MoE ------------------------------------------------------------------------------------------------------
class FakeScratch:
    def __init__(self, ex, rows, slots, prompt=False):
        self.rows, self.slots, self.prompt = rows, slots, prompt
        self.xd = torch.zeros((rows * slots, I), dtype=torch.float16)


def fake_experts(written):
    """routed() leaves each (row, slot)'s down-projection input in the scratch, as the real one does."""

    def routed(x, pick, wts, ex, s, out, n, limit=None, act_mode=None, add=None, before_down=None):
        s.xd[:n * SLOTS] = torch.arange(n * SLOTS * I, dtype=torch.float16).view(n * SLOTS, I)
        written.append(s.xd[:n * SLOTS].clone())
        base = torch.zeros((n, D)) if add is None else add.clone()
        return base

    return SimpleNamespace(routed=routed, Scratch=FakeScratch, ACT_F32=1, EXACT_ROWS=64)


def fake_kernels():
    def route(logits, bias, topk, scale, shared_id, pick, wts):
        pick[:, :topk] = torch.arange(topk, dtype=torch.int32)
        pick[:, topk] = shared_id
        wts.fill_(1.0)

    return SimpleNamespace(on=lambda name: False, DECODE_ROWS=16, rowmm=lambda x, w: x.float() @ w.float().t(),
                           route=route)


def moe_model(monkeypatch, written):
    monkeypatch.setattr(M, "K", fake_kernels())
    monkeypatch.setattr(M, "exl3_experts", fake_experts(written))
    monkeypatch.setattr(M, "KERNELS", True)
    monkeypatch.setattr(M, "EXACT_MM", False)
    monkeypatch.setattr(M, "SHARED_OVERLAP", False)
    m = M.Model.__new__(M.Model)
    m.cfg, m.scratch = SimpleNamespace(topk=TOPK, route_scale=1.0, swiglu_limit=10.0), {}
    return m


def packed_layer(idx):
    return SimpleNamespace(idx=idx, shared=None, experts=SimpleNamespace(count=E), gate_w=torch.zeros((E, D)),
                           gate_b=torch.zeros((E,)), gate_b_vl=None)


def test_the_packed_shared_experts_delta_reads_its_slots_rotated_hidden_state_in_the_scratch(monkeypatch):
    written = []
    m = moe_model(monkeypatch, written)
    n = 5
    x = torch.randn((n, D), generator=G).to(torch.bfloat16)
    lay = packed_layer(2)
    base = m.moe(lay, x)
    assert torch.equal(base, torch.zeros((n, D)))                                 # no adapter: routed()'s result as is
    tab = tables(I, torch.float16, True, 2)
    slots = [2, 0, 1, 0, 2]
    got = m.moe(lay, x, adapter=runtime(2, shared=tab).bind(torch.tensor(slots)))
    shared_rows = written[-1].view(n, SLOTS, I)[:, SLOTS - 1]                    # the last slot of each row
    torch.testing.assert_close(got.double(), expected(tab, shared_rows, slots, base), rtol=1e-5, atol=1e-3)
    assert torch.equal(got[1], base[1]) and torch.equal(got[3], base[3])
    other = m.moe(packed_layer(7), x, adapter=runtime(2, shared=tab).bind(torch.tensor(slots)))
    assert torch.equal(other, base)                                               # a layer no adapter edits


def apart_layer(idx):
    w = [torch.randn((I, D), generator=G).to(torch.bfloat16), torch.randn((I, D), generator=G).to(torch.bfloat16),
         torch.randn((D, I), generator=G).to(torch.bfloat16)]
    lay = packed_layer(idx)
    lay.shared = tuple(SimpleNamespace(weight=t) for t in w)
    lay.experts = SimpleNamespace(count=E - 1)
    return lay, w


def test_a_shared_expert_apart_adds_the_delta_of_its_plain_hidden_state(monkeypatch):
    written = []
    m = moe_model(monkeypatch, written)
    n = 4
    x = torch.randn((n, D), generator=G).to(torch.bfloat16)
    lay, (w1, w3, w2) = apart_layer(1)
    base = m.moe(lay, x)
    h = (F.silu((x.float() @ w1.float().t()).to(torch.bfloat16).float().clamp(max=10.0))
         * (x.float() @ w3.float().t()).to(torch.bfloat16).float().clamp(-10.0, 10.0)).to(torch.bfloat16)
    torch.testing.assert_close(base, h.float() @ w2.float().t())                  # (routed() here adds only ``add``)
    tab = tables(I, torch.bfloat16, False, 3)
    slots = [1, 2, 0, 2]
    got = m.moe(lay, x, adapter=runtime(1, shared=tab).bind(torch.tensor(slots)))
    torch.testing.assert_close(got.double(), expected(tab, h, slots, base), rtol=1e-4, atol=1e-3)
    assert torch.equal(got[2], base[2])
