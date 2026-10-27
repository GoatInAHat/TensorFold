"""Checks of the adapter delta kernels against the plain-torch definition, run on a GPU or, for the logic, by Triton's CPU
interpreter (tests/test_deepseek_v41_adapter_kernels_cpu.py). Each check raises AssertionError."""

from __future__ import annotations

import torch

from tensorfold.families.deepseek_v41.cuda import lora_kernels as lk
from tensorfold.families.deepseek_v41.cuda import lora_reference as ref
from tensorfold.families.deepseek_v41.cuda.lora_tables import Tables


def make_tables(device, ranks, k, d, x_dtype, seed=0, scales=None, junk=False):
    """Tables of slots 1..len(ranks); a rank of 0 leaves the slot without this module. ``junk``: nonzero values past
    each slot's rank, which the kernels must never read."""

    g = torch.Generator().manual_seed(seed)
    t = Tables.empty(len(ranks) + 1, max(max(ranks), 1), k, d, False, x_dtype, device)
    for slot, r in enumerate(ranks, 1):
        if r:
            t.put(slot, torch.randn((r, k), generator=g) * 0.05, torch.randn((d, r), generator=g) * 0.05,
                  (scales or [1.0] * len(ranks))[slot - 1])
        if junk:
            t.a[slot, r:] = 7.0
            t.b[slot, :, r:] = 7.0
    return t


def rand(device, shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype).to(device)


def run(out, x, ids, tab):
    return lk.lowrank_add(out.clone(), x, ids, tab)


def want(out, x, ids, tab):
    return ref.lowrank_add(out.clone().double(), x, ids, tab.a, tab.b, tab.ranks, tab.scales, dtype=torch.float64)


def check_matches_the_definition(device, x_dtype, k=256, d=384, ranks=(1, 3, 5), n=6, seed=1):
    tab = make_tables(device, ranks, k, d, x_dtype, seed, scales=[1.0, 0.25, -2.0][:len(ranks)])
    x, out = rand(device, (n, k), x_dtype, seed + 1), rand(device, (n, d), torch.float32, seed + 2)
    ids = torch.tensor([(i % (len(ranks) + 1)) for i in range(n)], dtype=torch.int64, device=device)
    got, exp = run(out, x, ids, tab), want(out, x, ids, tab)
    err = (got.double() - exp).abs().max().item()
    assert err <= 2e-5 * max(1.0, exp.abs().max().item()), f"max error {err}"
    return err


def check_stock_rows_keep_their_bits(device, x_dtype, k=256, d=384):
    tab = make_tables(device, (2, 4), k, d, x_dtype, 3)
    x, out = rand(device, (5, k), x_dtype, 4), rand(device, (5, d), torch.float32, 5)
    ids = torch.tensor([0, 2, 0, 1, 0], dtype=torch.int64, device=device)
    got = run(out, x, ids, tab)
    for r in (0, 2, 4):
        assert torch.equal(got[r], out[r]), f"stock row {r} changed"
    assert not torch.equal(got[1], out[1]) and not torch.equal(got[3], out[3])


def check_a_row_does_not_depend_on_the_others(device, x_dtype, k=384, d=256):
    """Row by row, in a different order, beside other adapters and padded by copies: the same delta bits."""

    tab = make_tables(device, (2, 3, 1), k, d, x_dtype, 6)
    n = 7
    x, out = rand(device, (n, k), x_dtype, 7), rand(device, (n, d), torch.float32, 8)
    ids = torch.tensor([3, 1, 0, 2, 1, 3, 2], dtype=torch.int64, device=device)
    whole = run(out, x, ids, tab)
    for r in range(n):                                                   # a row alone
        alone = run(out[r:r + 1], x[r:r + 1].contiguous(), ids[r:r + 1].contiguous(), tab)
        assert torch.equal(alone[0], whole[r]), f"row {r} differs alone"
    perm = torch.tensor([6, 2, 4, 0, 5, 1, 3], device=device)           # another order
    again = run(out[perm], x[perm].contiguous(), ids[perm].contiguous(), tab)
    assert torch.equal(again, whole[perm]), "a reordered batch differs"
    pad = torch.cat([torch.arange(n, device=device), torch.full((9,), n - 1, device=device)])    # padded with copies
    padded = run(out[pad], x[pad].contiguous(), ids[pad].contiguous(), tab)
    assert torch.equal(padded[:n], whole) and torch.equal(padded[n:], whole[n - 1:].expand(9, -1)), "padding changed bits"


def check_nothing_past_a_slots_rank_is_read(device, x_dtype, k=256, d=256):
    clean = make_tables(device, (2, 5), k, d, x_dtype, 9)
    dirty = make_tables(device, (2, 5), k, d, x_dtype, 9, junk=True)
    x, out = rand(device, (4, k), x_dtype, 10), rand(device, (4, d), torch.float32, 11)
    ids = torch.tensor([1, 2, 1, 2], dtype=torch.int64, device=device)
    assert torch.equal(run(out, x, ids, clean), run(out, x, ids, dirty))


def check_a_strided_view_of_the_hidden_state_is_read_in_place(device, k=256, d=128, slots=7):
    """The packed shared expert's rotated hidden is one slot of [rows, slots, K] fp16."""

    tab = make_tables(device, (3,), k, d, torch.float16, 12)
    big = rand(device, (5, slots, k), torch.float16, 13)
    out = rand(device, (5, d), torch.float32, 14)
    ids = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int64, device=device)
    view = big[:, slots - 1]
    assert view.stride() == (slots * k, 1)
    assert torch.equal(run(out, view, ids, tab), run(out, view.contiguous(), ids, tab))


def check_a_slot_without_the_module_adds_nothing(device, x_dtype, k=256, d=256):
    tab = make_tables(device, (0, 3), k, d, x_dtype, 15)
    x, out = rand(device, (3, k), x_dtype, 16), rand(device, (3, d), torch.float32, 17)
    ids = torch.tensor([1, 2, 1], dtype=torch.int64, device=device)
    got = run(out, x, ids, tab)
    assert torch.equal(got[0], out[0]) and torch.equal(got[2], out[2]) and not torch.equal(got[1], out[1])


def check_refusals(device):
    tab = make_tables(device, (2,), 256, 128, torch.bfloat16, 18)
    x, out = rand(device, (2, 256), torch.bfloat16, 19), rand(device, (2, 128), torch.float32, 20)
    ids = torch.tensor([1, 0], dtype=torch.int64, device=device)
    for args in ((out.to(torch.float16), x, ids), (out, x.to(torch.float16), ids), (out, x, ids.int()),
                 (out, x[:, :128], ids), (out, x, ids[:1]), (out[:, :64], x, ids)):
        try:
            lk.lowrank_add(args[0], args[1], args[2], tab)
        except ValueError:
            continue
        raise AssertionError("a malformed call was accepted")


def run_all(device) -> dict:
    results = {}
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        name = str(dt).split(".")[1]
        results[f"definition_{name}"] = check_matches_the_definition(device, dt)
        check_stock_rows_keep_their_bits(device, dt)
        check_a_row_does_not_depend_on_the_others(device, dt)
        check_nothing_past_a_slots_rank_is_read(device, dt)
        check_a_slot_without_the_module_adds_nothing(device, dt)
    results["definition_wide_rank"] = check_matches_the_definition(device, torch.bfloat16, ranks=(17, 33), n=3, seed=21)
    check_a_strided_view_of_the_hidden_state_is_read_in_place(device)
    check_refusals(device)
    return results
