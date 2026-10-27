"""The packed shared expert's w2 adapter reads what the EXL3 expert kernels leave for the down projection. The factors are
folded at load to read that rotated, scaled fp16 hidden state (lora_place.fold_rotated); these tests run the real expert
kernels on synthetic layers and check both the premise (the scratch holds Hadamard(hidden * suh_d) for the shared slot,
decode and prompt paths) and the delta computed from it."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from test_exl3_experts import _hadamard, _layer, _picks, _rot

from tensorfold.cuda.exl3 import experts
from tensorfold.families.deepseek_v41.cuda import lora_format as lf
from tensorfold.families.deepseek_v41.cuda import lora_kernels as lk
from tensorfold.families.deepseek_v41.cuda import lora_place as lp
from tensorfold.families.deepseek_v41.cuda.lora_tables import Tables

D, I, E, TOPK = 512, 256, 10, 3                 # E routed experts, and the shared one last in the pack
SLOTS = TOPK + 1
CB = 2


def scratch_for(ex, rows):
    prompt = rows >= experts.EXACT_ROWS             # Model._moe_scratch: prompt chunks take a scratch of their own
    return experts.Scratch(ex, rows=rows if prompt else max(rows, experts.EXACT_ROWS), slots=SLOTS, prompt=prompt)


def hidden_of_the_shared_expert(x, mats):
    """The shared expert's SwiGLU hidden state [rows, I] in float64, from the decoded weights."""

    gate, up, _ = mats
    H = _hadamard(x.device)

    def lin(v, m):
        t, suh, svh = m
        return _rot(_rot(v * suh.double(), H) @ experts.dequant(t, CB).double(), H) * svh.double()

    out = []
    for r in range(x.shape[0]):
        gg, uu = lin(x[r].double(), gate[-1]), lin(x[r].double(), up[-1])
        out.append(gg * torch.sigmoid(gg) * uu)
    return torch.stack(out)


@pytest.mark.parametrize("rows", [1, 4, 16, 63, 64, 130])
def test_the_scratch_holds_the_rotated_scaled_hidden_state_of_the_shared_slot(rows):
    ex, mats = _layer(E + 1, D, I, [(4, 4, 4)] * (E + 1), CB, seed=3)
    g = torch.Generator().manual_seed(rows)
    x = torch.randn((rows, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, rows, TOPK, g, shared=True)
    assert int(sel[0, -1]) == E == ex.count - 1                       # the last pack entry is the shared slot's pick
    s = scratch_for(ex, rows)
    experts.routed(x, sel, w, ex, s, None, rows)
    xd = s.xd[:rows * SLOTS].view(rows, SLOTS, I)[:, SLOTS - 1]
    act = hidden_of_the_shared_expert(x, mats)
    want = _rot(act * ex.suh_d[E].double(), _hadamard(x.device))
    assert xd.dtype == torch.float16
    assert (xd.double() - want).abs().max().item() <= 2e-3 * want.abs().max().item()


@pytest.mark.parametrize("rows", [1, 6, 64, 100])
def test_the_folded_factors_add_the_delta_of_the_plain_hidden_state(rows):
    ex, mats = _layer(E + 1, D, I, [(4, 4, 4)] * (E + 1), CB, seed=4)
    g = torch.Generator().manual_seed(40 + rows)
    x = torch.randn((rows, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, rows, TOPK, g, shared=True)
    s = scratch_for(ex, rows)
    out = experts.routed(x, sel, w, ex, s, None, rows)
    xd = s.xd[:rows * SLOTS].view(rows, SLOTS, I)[:, SLOTS - 1]
    act = hidden_of_the_shared_expert(x, mats)

    rng = np.random.default_rng(5)
    dims = lf.Dims(n_layers=1, dim=D, inter=I, o_in=I)
    f = lf.Factors((rng.standard_normal((5, I)) * 0.1).astype(np.float32),
                   (rng.standard_normal((D, 5)) * 0.1).astype(np.float32), 0.5)
    placed = lp.place(lf.SHARED_W2, f, dims, 0, 1, strength=2.0, suh=ex.suh_d[E].float().cpu().numpy())
    assert placed.folded
    tab = Tables.empty(3, 5, I, D, True, torch.float16, "cuda")
    tab.put(2, placed.a, placed.b, placed.scale)
    ids = (torch.arange(rows, device="cuda") % 3 == 2).to(torch.int64) * 2          # every third row runs slot 2
    got = lk.lowrank_add(out.clone(), xd, ids, tab).double()

    delta = placed.scale * (act @ torch.from_numpy(f.a).double().cuda().T) @ torch.from_numpy(f.b).double().cuda().T
    want = out.double() + delta * (ids != 0).double()[:, None]
    scale = delta.abs().max().item()
    assert (got - want).abs().max().item() <= 4e-3 * scale                          # fp16 storage of the scratch
    keep = ids == 0
    assert torch.equal(got[keep].float(), out[keep])                                # the rows of the base model: its bits


def test_the_delta_is_the_same_for_a_row_in_a_small_window_and_in_a_prompt_chunk():
    """The scratch of a prompt chunk and of a decode window hold the same rotated hidden state for a row: fp16 apart."""

    ex, _ = _layer(E + 1, D, I, [(4, 4, 4)] * (E + 1), CB, seed=6)
    g = torch.Generator().manual_seed(7)
    rows = 128
    x = torch.randn((rows, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, rows, TOPK, g, shared=True)
    big = scratch_for(ex, rows)
    experts.routed(x, sel, w, ex, big, None, rows)
    small = scratch_for(ex, 8)
    experts.routed(x[:8].contiguous(), sel[:8].contiguous(), w[:8].contiguous(), ex, small, None, 8)
    a = big.xd[:rows * SLOTS].view(rows, SLOTS, I)[:8, SLOTS - 1].double()
    b = small.xd[:8 * SLOTS].view(8, SLOTS, I)[:, SLOTS - 1].double()
    assert (a - b).abs().max().item() <= 4e-3 * b.abs().max().item()
    assert math.isfinite(float(a.sum()))
