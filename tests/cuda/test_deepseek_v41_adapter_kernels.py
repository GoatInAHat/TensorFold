"""The adapter delta kernels compiled for the GPU: the plain-torch definition, independent rows (alone, reordered, padded),
production shapes and ranks, and a captured CUDA graph that replays any mixture of adapters from its input vector."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

import adapter_kernel_checks as checks

from tensorfold.families.deepseek_v41.cuda import lora_kernels as lk

DEV = "cuda"


def test_the_compiled_kernels_pass_every_check_of_the_cpu_interpreter_run():
    errors = checks.run_all(DEV)
    assert max(errors.values()) < 1e-4, errors


# one rank's shapes: wo_b reads 4096 columns (4 groups of 1024) and writes 5120; the shared w2 reads 1152
@pytest.mark.parametrize("k", [4096, 1152])
@pytest.mark.parametrize("ranks", [(1,), (2, 3), (16, 7), (17, 64), (33, 1, 128)])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16])
def test_production_shapes_and_ranks_match_the_definition(k, ranks, x_dtype):
    assert checks.check_matches_the_definition(DEV, x_dtype, k=k, d=5120, ranks=ranks, n=16, seed=len(ranks)) < 1e-4


def test_a_prompt_chunk_of_many_rows_matches_the_definition_and_keeps_every_row_independent():
    tab = checks.make_tables(DEV, (4, 9), 4096, 5120, torch.bfloat16, 30)
    n = 2048
    x, out = checks.rand(DEV, (n, 4096), torch.bfloat16, 31), checks.rand(DEV, (n, 5120), torch.float32, 32)
    ids = (torch.arange(n, device=DEV) % 3).to(torch.int64)
    whole = checks.run(out, x, ids, tab)
    sample = torch.tensor([0, 1, 2, 3, 700, 1023, 2047], device=DEV)
    exp = checks.want(out[sample], x[sample], ids[sample], tab)
    assert (whole[sample].double() - exp).abs().max().item() < 1e-4
    for r in sample.tolist():                                       # a decode-sized call gives the chunk's bits
        alone = checks.run(out[r:r + 1], x[r:r + 1].contiguous(), ids[r:r + 1].contiguous(), tab)
        assert torch.equal(alone[0], whole[r])
    stock = ids == 0
    assert torch.equal(whole[stock], out[stock])


@pytest.mark.parametrize("rows", [1, 3, 8, 16])
def test_a_captured_graph_replays_any_mixture_of_adapters_from_its_input_vector(rows):
    tab = checks.make_tables(DEV, (2, 5, 1), 1152, 5120, torch.float16, 40)
    x = checks.rand(DEV, (rows, 1152), torch.float16, 41)
    out0 = checks.rand(DEV, (rows, 5120), torch.float32, 42)
    ids = torch.zeros((rows,), dtype=torch.int64, device=DEV)           # the graph's input, like RoundDecoder's
    out = out0.clone()

    def body():
        out.copy_(out0)
        lk.lowrank_add(out, x, ids, tab)

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        body()                                                          # the warm-up the captures follow
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    g = torch.Generator().manual_seed(43)
    for _ in range(6):
        ids.copy_(torch.randint(0, 4, (rows,), generator=g).to(DEV))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, checks.run(out0, x, ids, tab)), ids.tolist()      # the eager call's bits
    ids.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, out0)                                       # all stock: nothing added


def test_another_adapters_larger_rank_does_not_change_a_rows_bits():
    """The same adapter in tables made wider by a second adapter gives the same bits: the kernels' tile is fixed and
    read only a slot's rank, so a server's other adapters never move an adapter's output."""

    k, d = 1152, 512
    narrow = checks.make_tables(DEV, (3,), k, d, torch.bfloat16, 50)
    wide = checks.make_tables(DEV, (3, 40), k, d, torch.bfloat16, 50)
    wide.put(1, narrow.a[1, :3], narrow.b[1, :, :3], float(narrow.scales[1]))
    x, out = checks.rand(DEV, (4, k), torch.bfloat16, 51), checks.rand(DEV, (4, d), torch.float32, 52)
    ids = torch.ones((4,), dtype=torch.int64, device=DEV)
    assert narrow.R < wide.R
    assert torch.equal(checks.run(out, x, ids, narrow), checks.run(out, x, ids, wide))
