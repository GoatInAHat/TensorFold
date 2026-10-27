"""Two ranks on one GPU over gloo, as the Nemotron TP test runs them: each rank places its share of an adapter, adds its
partial delta with the compiled kernels, and the ranks' sum (the reduction the layer already makes) equals the unsharded
delta of the whole matrix. For wo_b and for the packed shared expert (folded factors on each rank's own scale vector)."""

import socket

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

ROWS = 5
DIMS = (4, 256, 512, 1024)                           # layers, dim, shared width, wo_b width
SLOTS_OF_ROWS = [0, 1, 2, 1, 0]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def problem(kind):
    """Both ranks and the test build the same numbers from the same seeds."""

    from tensorfold.families.deepseek_v41.cuda import lora_format as lf

    dims = lf.Dims(*DIMS)
    width, out = dims.shape(kind)
    rng = np.random.default_rng(11)
    mods = [lf.Factors((rng.standard_normal((r, width)) * 0.1).astype(np.float32),
                       (rng.standard_normal((out, r)) * 0.1).astype(np.float32), s) for r, s in ((2, 1.0), (5, 0.5))]
    x = rng.standard_normal((ROWS, width)).astype(np.float32)
    suh = (rng.uniform(0.5, 2.0, width) * np.where(rng.random(width) < 0.5, -1.0, 1.0)).astype(np.float16)
    return dims, mods, x, suh.astype(np.float64)


def _worker(rank: int, port: int, kind: str, out) -> None:
    import torch.distributed as dist

    from tensorfold.families.deepseek_v41.cuda import lora_kernels as lk
    from tensorfold.families.deepseek_v41.cuda import lora_place as lp
    from tensorfold.families.deepseek_v41.cuda.lora_tables import Tables

    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    dims, mods, x, suh = problem(kind)
    width, dim = dims.shape(kind)
    lo, hi = lp.span(width, rank, 2)
    packed = kind == "shared_w2"
    if packed:                                        # what the expert kernels leave for the down projection
        z = (x[:, lo:hi].astype(np.float64) * suh[lo:hi]).reshape(ROWS, -1, 128) @ lp.hadamard()
        local = torch.from_numpy(z.reshape(ROWS, hi - lo)).to(torch.float16).cuda()
    else:
        local = torch.from_numpy(x[:, lo:hi]).to(torch.bfloat16).cuda()
    tab = Tables.empty(3, 5, hi - lo, dim, packed, local.dtype, "cuda")
    for slot, f in enumerate(mods, 1):
        p = lp.place(kind, f, dims, rank, 2, suh=suh[lo:hi] if packed else None)
        tab.put(slot, p.a, p.b, p.scale)
    ids = torch.tensor(SLOTS_OF_ROWS, dtype=torch.int64, device="cuda")
    partial = lk.lowrank_add(torch.zeros((ROWS, dim), dtype=torch.float32, device="cuda"), local, ids, tab)
    total = partial.cpu()
    dist.all_reduce(total)                            # the reduction the layer's gather already makes
    out.put((rank, partial.cpu().numpy(), total.numpy()))
    dist.barrier()
    dist.destroy_process_group()


def run_pair(kind):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, kind, out)) for r in (0, 1)]
    for p in procs:
        p.start()
    got = {r: (part, total) for r, part, total in (out.get(timeout=600) for _ in procs)}
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    return got


def reference(kind):
    dims, mods, x, _ = problem(kind)
    bf16 = torch.from_numpy(x).to(torch.bfloat16).float().numpy().astype(np.float64)
    out = np.zeros((ROWS, dims.dim))
    for i, slot in enumerate(SLOTS_OF_ROWS):
        if slot:
            f = mods[slot - 1]
            out[i] = f.scale * (f.b.astype(np.float64) @ (f.a.astype(np.float64) @ (x if kind == "shared_w2" else bf16)[i]))
    return out


@pytest.mark.parametrize("kind", ["wo_b", "shared_w2"])
def test_two_ranks_partial_deltas_sum_to_the_whole_matrices_delta(kind):
    got = run_pair(kind)
    assert np.array_equal(got[0][1], got[1][1])                          # both ranks hold the same sum
    want = reference(kind)
    tol = 4e-3 if kind == "shared_w2" else 1e-4                         # fp16 storage of the rotated hidden state
    assert np.abs(got[0][1] - want).max() <= tol * np.abs(want).max()
    stock = [i for i, s in enumerate(SLOTS_OF_ROWS) if s == 0]
    assert all(not got[r][0][stock].any() for r in (0, 1))              # a stock row: no partial on either rank
