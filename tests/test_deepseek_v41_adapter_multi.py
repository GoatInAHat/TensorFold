"""Kept and spilled prompts never meet another adapter's rows, and the lane's steps carry the adapter to every rank,
through the real MultiDecoder admission, finish and follow paths on CPU tensors (no GPU, kernels or checkpoint)."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from tests.test_deepseek_v41_spill import CHUNK, LAYERS, RAW, _fill, _pool, _prompt, _rows

KEY_A, KEY_B = "a" * 32, "b" * 32


@pytest.fixture
def mods(monkeypatch, tmp_path):
    """multi.py and spill.py with the CUDA model and engine modules stubbed (PREFILL_CHUNK, REPLAY, RAW)."""

    monkeypatch.setenv("TF_DS_KEEP", "1")
    model = ModuleType("tensorfold.families.deepseek_v41.cuda.model")
    model.RAW = RAW
    engine = ModuleType("tensorfold.families.deepseek_v41.cuda.engine")
    engine.PREFILL_CHUNK, engine.REPLAY = CHUNK, False
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    for name in ("tensorfold.families.deepseek_v41.cuda.multi", "tensorfold.families.deepseek_v41.cuda.spill"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import tensorfold.families.deepseek_v41.cuda.multi as multi
    import tensorfold.families.deepseek_v41.cuda.spill as spill

    return multi, spill, tmp_path / "spill"


def ref(multi, slot, key):
    from tensorfold.families.deepseek_v41.cuda.lora_registry import AdapterRef

    return AdapterRef(slot, key)


def decoder(multi, spill, root, *, nonce="run1", held=("a", "b")):
    """A MultiDecoder whose fake prefill makes rows that depend on the adapter slot, as a real adapter's do."""

    pool, calls = _pool(), []

    def pool_view(p, slot, base, size):
        return SimpleNamespace(base=base, size=size, slot=slot, length=0, host=multi.HostIds())

    def prefill_steps(sc, dc, prompt, image=None, start=0, snap=None, adapter=0):
        sc.length = start
        sc.host.truncate(start)
        first = min(len(prompt), (start // CHUNK + 1) * CHUNK)
        ends = [first] + list(range(first + CHUNK, len(prompt), CHUNK))
        if ends[-1] != len(prompt):
            ends.append(len(prompt))
        calls.append((start, adapter))
        shifted = [t + 100000 * adapter for t in prompt]
        for s, e in zip([start] + ends[:-1], ends):
            _fill(pool, sc.base, sc.slot, shifted, s, e)
            sc.host.set(s, prompt[s:e])
            if snap is not None and e % CHUNK == 0:
                snap(e)
            if e < len(prompt):
                yield e
        return "logits"

    holds = {(1, KEY_A) if "a" in held else None, (2, KEY_B) if "b" in held else None}
    e = SimpleNamespace(rank=0, world=1, drafter=None, eos=(1,), replay_mode=False, prefill_steps=prefill_steps,
                        _sample=lambda last, pos, sampling: [7],
                        adapter_ok=lambda r: r is None or (r.slot, r.key) in holds)
    md = object.__new__(multi.MultiDecoder)
    md.e, md.pool, md.dpool, md.cap = e, pool, None, pool.cap
    md.m = SimpleNamespace(cfg=SimpleNamespace(window=128, compress_ratios=[4, 128, 1, 1], n_layers=LAYERS),
                           pool_view=pool_view, comm=None)
    md.slots = [multi.Slot(i, None) for i in range(pool.slots)]
    md.extents, md.free, md.max_rows, md.eos = multi.Extents(pool.cap), list(range(pool.slots)), 6, (1,)
    md.streams, md.filling, md.chunk_s, md.since_fill, md.next_id = {}, [], 0.0, 0.0, 0
    md.link = md.watch = md.round_end = None
    md.kept, md.next_kept, md.ticks, md.keep_stats = {}, 0, 0, {}
    md.spill = spill.Spill(root, engine="testengine", rank=0, pool=pool, ratios={0: 4, 1: 128}, budget=1 << 30,
                           max_age=3600.0, min_top=4096, device="cpu")
    md.spill_nonce = nonce
    return md, calls


def serve(multi, md, prompt, adapter=None):
    s = multi.Stream(list(prompt), 1, None, draft=False, stop_eos=False, adapter=adapter)
    s.emit = lambda new: None
    md.admit(s)
    while s in md.filling:
        md._fill(s)
    md.finish([s])
    return s


def test_a_prompt_continues_only_a_kept_prompt_of_the_same_adapter(mods):
    multi, spill, root = mods
    md, calls = decoder(multi, spill, root)
    a, b = ref(multi, 1, KEY_A), ref(multi, 2, KEY_B)
    p = _prompt(9000, 1)
    serve(multi, md, p, a)
    (k,) = md.kept.values()
    assert k.adapter == KEY_A
    longer = p + _prompt(2000, 2)
    for other in (None, b):                                      # the stock model and another adapter: a fresh fill
        calls.clear()
        s = serve(multi, md, longer, other)
        assert s.cached == 0 and calls == [(0, 0 if other is None else 2)]
    calls.clear()
    s = serve(multi, md, longer, a)                              # its own adapter: continues
    assert s.cached == 8192 and calls == [(8192, 1)]


def test_a_stock_prompt_is_never_continued_by_an_adapter_and_the_reverse(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root)
    p = _prompt(9000, 3)
    serve(multi, md, p, None)
    assert [k.adapter for k in md.kept.values()] == [""]
    assert serve(multi, md, p + [5, 6], ref(multi, 1, KEY_A)).cached == 0
    assert serve(multi, md, p + [5, 6, 7], None).cached == 8192


def test_a_kept_prompt_is_not_covered_by_a_finishing_prompt_of_another_adapter(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root)
    a, b = ref(multi, 1, KEY_A), ref(multi, 2, KEY_B)
    p = _prompt(9000, 4)
    serve(multi, md, p, a)
    serve(multi, md, p, b)                                       # the same ids, another adapter: both are kept
    assert sorted(k.adapter for k in md.kept.values()) == [KEY_A, KEY_B]
    serve(multi, md, p + [9] * 3000, a)                          # the same adapter's longer prompt covers its own
    assert sorted(k.adapter for k in md.kept.values()) == [KEY_A, KEY_B]
    assert len(md.kept) == 2


def test_spilled_prompts_are_read_back_by_their_own_adapter_only(mods):
    multi, spill, root = mods
    a, b = ref(multi, 1, KEY_A), ref(multi, 2, KEY_B)
    p = _prompt(20000, 6)
    md1, _ = decoder(multi, spill, root)
    serve(multi, md1, p, a)
    (k,) = md1.kept.values()
    assert k.disk is not None and k.disk["adapter"] == KEY_A
    want = _rows(md1.pool, k.base, k.top)
    for other in (None, b):                                      # a new run: an empty window, the same disk
        md, calls = decoder(multi, spill, root, nonce="run" + ("0" if other is None else "b"))
        s = serve(multi, md, p + [1, 2, 3], other)
        assert s.cached == 0 and md.keep_stats == {} and calls[0][0] == 0
    md, calls = decoder(multi, spill, root, nonce="runa")
    s = multi.Stream(list(p) + [1, 2, 3], 1, None, draft=False, stop_eos=False, adapter=a)
    s.emit = lambda new: None
    md.admit(s)
    assert s.cached == k.top and md.keep_stats == {"disk": 1}
    assert all(torch.equal(x, y) for x, y in zip(want, _rows(md.pool, s.base, k.top)))


def test_a_stock_spilled_prompt_has_no_adapter_in_its_manifest_and_the_hashes_it_always_had(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root)
    p = _prompt(20000, 7)
    serve(multi, md, p, None)
    (k,) = md.kept.values()
    assert "adapter" not in k.disk
    import hashlib

    keys = np.asarray(p, dtype=np.int64)
    plain = hashlib.sha256(keys[:k.top].tobytes()).hexdigest()
    assert k.disk["hashes"][str(k.top)] == plain
    assert spill.Spill._hashes(keys, [k.top], KEY_A)[str(k.top)] != plain


def test_a_manifest_naming_an_adapter_in_a_form_that_is_no_key_is_dropped_at_start(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root)
    serve(multi, md, _prompt(20000, 8), ref(multi, 1, KEY_A))
    (entry,) = md.spill.dir.glob("entry-*.json")
    manifest = json.loads(entry.read_text())
    manifest["adapter"] = "../../etc"
    entry.write_text(json.dumps(manifest))
    md2, _ = decoder(multi, spill, root, nonce="run2")
    assert md2.spill.manifests == {}


def test_a_stock_admission_is_the_op_it_always_was_and_an_adapters_rides_after_the_grammar(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root)
    sent = []
    md.link = SimpleNamespace(send=sent.append)
    serve(multi, md, _prompt(100, 9), None)
    serve(multi, md, _prompt(100, 9), ref(multi, 2, KEY_B))
    stock, held = [op for op in sent if op[0] == "admit"]
    assert len(stock) == 10 and len(held) == 11 and held[10] == [2, KEY_B] and held[:10] == stock[:10]


def test_a_follower_builds_the_stream_with_the_adapter_the_op_names_and_fills_under_it(mods):
    multi, spill, root = mods
    md, calls = decoder(multi, spill, root)
    p = _prompt(100, 10)
    ops = [["admit", p, 1, None, False, False, 0, [], None, [], [2, KEY_B]], None]
    link = SimpleNamespace(receive=lambda: ops.pop(0))
    with pytest.raises(RuntimeError, match="step link closed"):
        md.follow(link)
    (s,) = md.filling
    assert (s.adapter.slot, s.adapter.key) == (2, KEY_B)
    while s in md.filling:
        md._fill(s)
    assert calls == [(0, 2)]


def test_a_stock_follower_admission_still_parses_and_fills_stock(mods):
    multi, spill, root = mods
    md, calls = decoder(multi, spill, root)
    ops = [["admit", _prompt(100, 11), 1, None, False, False, 0, [], None, []], None]
    link = SimpleNamespace(receive=lambda: ops.pop(0))
    with pytest.raises(RuntimeError, match="step link closed"):
        md.follow(link)
    (s,) = md.filling
    assert s.adapter is None and md.slots[0].sc is not None
    while s in md.filling:
        md._fill(s)
    assert calls == [(0, 0)]


def test_the_plan_every_rank_agrees_on_names_the_adapter_and_whether_this_rank_holds_it(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root, held=("a",))
    stock = multi.Stream([1], 1, None)
    held, lost = (multi.Stream([1], 1, None, adapter=ref(multi, s, k)) for s, k in ((1, KEY_A), (2, KEY_B)))
    assert md._adapter_plan(stock) == [] and md._adapter_plan(stock, check=True) == []
    assert md._adapter_plan(held) == [[1, KEY_A]]
    assert md._adapter_plan(held, check=True) == [[1, KEY_A, True]]
    assert md._adapter_plan(lost, check=True) == [[2, KEY_B, False]]            # a rank without it plans differently


def test_a_request_for_an_adapter_the_engine_does_not_hold_fails_before_anything_is_sent(mods):
    multi, spill, root = mods
    md, _ = decoder(multi, spill, root, held=("a",))
    sent = []
    md.link = SimpleNamespace(send=sent.append)
    s = multi.Stream(_prompt(100, 12), 1, None, adapter=ref(multi, 2, KEY_B))
    s.emit = lambda new: None
    with pytest.raises(ValueError, match="does not hold"):
        md.admit(s)
    assert sent == [] and md.filling == [] and md.free == list(range(len(md.slots)))


def test_the_shape_each_round_checks_names_adapter_streams_and_is_unchanged_for_stock(mods, monkeypatch):
    multi, spill, root = mods
    monkeypatch.setattr(multi, "ADMIT_GIB", 0.0)                 # (no memory gate: two streams are live)
    md, _ = decoder(multi, spill, root)
    stock = multi.Stream(_prompt(100, 13), 1, None)
    stock.emit = lambda new: None
    md.admit(stock)
    entry = md._shape()[4][0]
    assert entry == [stock.sid, 0, 0]                            # a stock stream's entry has no slot in it
    held = multi.Stream(_prompt(100, 13), 1, None, adapter=ref(multi, 1, KEY_A))
    held.emit = lambda new: None
    md.admit(held)
    assert md._shape()[4] == [entry, [held.sid, 1, 0, 1]]
