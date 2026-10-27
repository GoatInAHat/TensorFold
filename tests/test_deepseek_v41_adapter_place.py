"""Where an adapter's factors live on a rank: the cut, the sum over ranks, and the shared expert's rotated hidden state
(numpy only: the algebra the kernels are told to run)."""

import numpy as np
import pytest
from adapter_fakes import CFG

from tensorfold.families.deepseek_v41.cuda import lora_format as lf
from tensorfold.families.deepseek_v41.cuda import lora_place as lp

DIMS = lf.Dims.of(CFG)


def make(kind, rank, seed=0):
    rng = np.random.default_rng(seed)
    width, out = DIMS.shape(kind)
    return lf.Factors((rng.standard_normal((rank, width)) * 0.1).astype(np.float32),
                      (rng.standard_normal((out, rank)) * 0.1).astype(np.float32), 0.75)


def test_the_hadamard_matrix_is_symmetric_orthonormal_and_in_natural_order():
    h = lp.hadamard()
    assert h.shape == (128, 128)
    np.testing.assert_allclose(h, h.T, atol=0)
    np.testing.assert_allclose(h @ h, np.eye(128), atol=1e-12)
    assert h[0, 0] > 0 and np.all(h[0] == h[0, 0]) and np.all(h[:, 0] == h[0, 0])         # the all-ones row
    np.testing.assert_allclose(h[1], h[0] * np.where(np.arange(128) & 1, -1, 1))          # Sylvester: bit 0 alternates


def test_a_rank_holds_its_own_whole_blocks_of_the_input():
    assert lp.span(512, 0, 2) == (0, 256) and lp.span(512, 1, 2) == (256, 512) and lp.span(512, 0, 1) == (0, 512)
    for total, rank, world in ((512, 2, 2), (512, 0, 3), (192, 0, 2), (512, -1, 2), (512, 0, 0)):
        with pytest.raises(ValueError):
            lp.span(total, rank, world)


@pytest.mark.parametrize("kind", [lf.WO_B, lf.SHARED_W2])
@pytest.mark.parametrize("world", [1, 2])
def test_the_ranks_partial_deltas_add_up_to_the_whole_matrices_delta(kind, world):
    f = make(kind, 3)
    width, out = DIMS.shape(kind)
    x = np.random.default_rng(1).standard_normal((5, width)).astype(np.float32)
    whole = 0.75 * (x @ f.a.T) @ f.b.T
    parts = 0
    for rank in range(world):
        p = lp.place(kind, f, DIMS, rank, world)
        lo, hi = lp.span(width, rank, world)
        assert p.a.shape == (3, width // world) and p.b.shape == (out, 3) and not p.folded and p.scale == 0.75
        parts = parts + p.scale * (x[:, lo:hi] @ p.a.T) @ p.b.T
    np.testing.assert_allclose(parts, whole, rtol=1e-5, atol=1e-5)


def test_strength_multiplies_the_scale_and_nothing_else():
    f = make(lf.WO_B, 2)
    p, q = lp.place(lf.WO_B, f, DIMS, 0, 2), lp.place(lf.WO_B, f, DIMS, 0, 2, strength=-2.0)
    assert q.scale == -2.0 * p.scale
    np.testing.assert_array_equal(p.a, q.a)
    np.testing.assert_array_equal(p.b, q.b)


def rotated_hidden(act, suh):
    """What the expert kernels leave for the down projection: blockwise Hadamard of (act * suh), in fp16."""

    h = lp.hadamard()
    z = (act.astype(np.float64) * suh).reshape(act.shape[0], -1, 128) @ h
    return z.reshape(act.shape).astype(np.float16)


@pytest.mark.parametrize("world", [1, 2])
def test_the_folded_factors_read_the_rotated_hidden_state_as_the_plain_ones_read_the_hidden_state(world):
    f = make(lf.SHARED_W2, 4, seed=5)
    rng = np.random.default_rng(6)
    act = rng.standard_normal((6, DIMS.inter)).astype(np.float32)
    suh = np.where(rng.random(DIMS.inter) < 0.5, -1.0, 1.0) * rng.uniform(0.5, 2.0, DIMS.inter)
    suh = suh.astype(np.float16).astype(np.float64)
    want = 0.75 * (act @ f.a.T) @ f.b.T
    got = 0
    for rank in range(world):
        lo, hi = lp.span(DIMS.inter, rank, world)
        xd = rotated_hidden(act[:, lo:hi], suh[lo:hi])                    # fp16, as the kernels store it
        p = lp.place(lf.SHARED_W2, f, DIMS, rank, world, suh=suh[lo:hi])
        assert p.folded and p.a.dtype == np.float32
        got = got + p.scale * (xd.astype(np.float32) @ p.a.T) @ p.b.T
    scale = np.abs(want).max()
    assert np.abs(got - want).max() <= 3e-3 * scale                       # fp16 storage of xd is the only difference


def test_folding_is_exact_without_the_fp16_rounding():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 256)).astype(np.float32)
    suh = rng.uniform(0.5, 2.0, 256) * np.where(rng.random(256) < 0.5, -1, 1)
    act = rng.standard_normal((4, 256))
    xd = (act * suh).reshape(4, 2, 128) @ lp.hadamard()
    np.testing.assert_allclose(xd.reshape(4, 256) @ lp.fold_rotated(a, suh).T.astype(np.float64), act @ a.T,
                               rtol=2e-5, atol=2e-5)


def test_folding_refuses_a_scale_vector_it_cannot_divide_by():
    a = np.ones((2, 128), dtype=np.float32)
    for bad in (np.zeros(128), np.full(128, np.inf), np.full(128, np.nan), np.ones(64)):
        with pytest.raises(ValueError):
            lp.fold_rotated(a, bad)
    with pytest.raises(ValueError):
        lp.fold_rotated(np.ones((2, 100), dtype=np.float32), np.ones(100))


def test_only_the_shared_expert_reads_a_rotated_input_and_the_factors_must_fit():
    f = make(lf.WO_B, 2)
    with pytest.raises(ValueError, match="only the packed shared expert"):
        lp.place(lf.WO_B, f, DIMS, 0, 2, suh=np.ones(256))
    with pytest.raises(ValueError, match="do not fit"):
        lp.place(lf.SHARED_W2, f, DIMS, 0, 2)
    with pytest.raises(ValueError, match="no placement"):
        lp.place("wq_b", f, DIMS, 0, 2)
