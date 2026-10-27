"""A concurrent round's inputs carry each row's adapter slot: in window order, padded with copies of the last row, and
absent from a stock round (the round runner on CPU with fake graphs, no GPU)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import graph_keys, rounds

SIZES = graph_keys.row_sizes(16, graph_keys.ROUND_ROWS)
LONG = graph_keys.row_sizes(16, graph_keys.ROUND_ROWS_LONG)
CAP = 1 << 20


class FakeGraph:
    taps = None

    def __init__(self, rows):
        self.rows, self.runs, self.sets, self.captured = rows, [], [], False

    def set(self, *cols):
        self.sets.append(cols)

    def capture(self, pool):
        self.captured = True

    def run(self, *args):
        self.runs.append(args)
        return torch.zeros((self.rows, 4))


ADAPTERS = object()                     # (an engine that serves adapters: only whether lora is set matters here)


def runner(lora=ADAPTERS, graphs=None):
    r = rounds.RoundRunner.__new__(rounds.RoundRunner)
    r.m, r.pool = SimpleNamespace(engram=None, lora=lora), SimpleNamespace(cap=CAP)
    r.graphs, r.graph_pool, r.captures, r.warmed, r.out = graphs if graphs is not None else {}, object(), 0, True, None
    r.sizes, r.sizes_long = SIZES, LONG
    return r


def window(slot, p0, n, *adapter):
    return (slot, slot * 4096, 4096, p0, list(range(n)), None) + adapter


def test_the_columns_follow_the_windows_and_the_padding_copies_the_last_rows_adapter():
    g = FakeGraph(6)
    r = runner(graphs={(6, 1024): g})
    logits, taps = r.forward([window(0, 10, 3, 0), window(1, 20, 2, 2)])            # 5 rows run in the 6-row graph
    assert logits.shape == (5, 4) and taps is None
    ids, pos, slots, base, end, _, _, adapters = g.runs[0]
    assert adapters == [0, 0, 0, 2, 2, 2] and slots == [0, 0, 0, 1, 1, 1]
    assert pos == [10, 11, 12, 20, 21, 21] and ids == [0, 1, 2, 0, 1, 1]            # (the pad row is the last row again)
    assert base == [0, 0, 0, 4096, 4096, 4096] and end == [4096, 4096, 4096, 8192, 8192, 8192]


def test_a_window_without_an_adapter_is_slot_zero_and_a_stock_round_has_only_zeros():
    g = FakeGraph(4)
    r = runner(graphs={(4, 1024): g})
    r.forward([window(0, 5, 2), window(1, 9, 2)])
    assert g.runs[0][-1] == [0, 0, 0, 0]
    g2 = FakeGraph(4)
    runner(lora=None, graphs={(4, 1024): g2}).forward([window(0, 5, 2), window(1, 9, 2)])
    assert g2.runs[0][-1] == [0, 0, 0, 0]                                           # an engine without adapters


def test_a_row_of_an_adapter_in_an_engine_without_adapters_is_refused_not_run_as_stock():
    r = runner(lora=None, graphs={(2, 1024): FakeGraph(2)})
    with pytest.raises(ValueError, match="started without one"):
        r.forward([window(0, 5, 2, 1)])


def test_a_round_that_captures_a_graph_sets_it_up_with_the_same_columns(monkeypatch):
    made = []
    monkeypatch.setattr(rounds, "RoundDecoder", lambda m, pool, rows, bucket, taps, out=None: made.append(
        FakeGraph(rows)) or made[-1])
    r = runner()                                                                     # (warmed: a capture now is counted)
    r.forward([window(0, 5, 1, 1), window(1, 9, 3, 2)])                              # 4 rows: a 4-row graph
    (g,) = made
    assert g.captured and g.rows == 4 and g.sets[0][-1] == [1, 2, 2, 2] and g.runs[0][-1] == [1, 2, 2, 2]
    assert r.captures == 1 and (4, 1024) in r.graphs


def test_without_graphs_the_round_runs_unpadded():
    g = FakeGraph(5)
    r = runner()
    r.graphs = None
    r.sizes = SIZES
    made = []
    r.forward = rounds.RoundRunner.forward.__get__(r)
    rounds.RoundDecoder, saved = (lambda m, pool, rows, bucket, taps, out=None: made.append(g) or g), rounds.RoundDecoder
    try:
        r.forward([window(0, 5, 2, 1), window(1, 9, 3, 2)])
    finally:
        rounds.RoundDecoder = saved
    assert g.runs[0][-1] == [1, 1, 2, 2, 2]
