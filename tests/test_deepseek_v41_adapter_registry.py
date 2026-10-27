"""The adapter registry: identities, slots, what a request may select, and what two ranks must agree on (no torch)."""

import shutil
from types import SimpleNamespace

import pytest
from adapter_fakes import CFG, model_dir, standard

from tensorfold.families.deepseek_v41.cuda import lora_format as lf
from tensorfold.families.deepseek_v41.cuda import lora_registry as lr


@pytest.fixture
def base(tmp_path):
    return model_dir(tmp_path / "model")


def load(base, specs, reserved=("base",)):
    return lr.AdapterRegistry.load(specs, CFG, base, reserved)


def test_slots_follow_the_order_given_and_zero_is_the_stock_model(tmp_path, base):
    reg = load(base, [("one", str(standard(tmp_path / "a", 1)), 1.0), ("two", str(standard(tmp_path / "b", 2)), 1.0)])
    assert reg.names() == ["one", "two"] and reg.slots == 3 and len(reg) == 2
    assert [e.slot for e in reg.entries] == [1, 2] and reg.entry(2).name == "two"
    assert reg.model_map() == {"one": "one", "two": "two"}
    assert reg.ref(None) is None and reg.ref("two") == lr.AdapterRef(2, reg.entry(2).key)


def test_an_unknown_adapter_is_an_error_not_the_stock_model(tmp_path, base):
    reg = load(base, [("one", str(standard(tmp_path / "a")), 1.0)])
    with pytest.raises(lr.UnknownAdapter, match="no adapter named 'nope'"):
        reg.ref("nope")


def test_the_key_names_the_factors_and_their_scales_not_the_label(tmp_path, base):
    a = str(standard(tmp_path / "a", 1))
    shutil.copytree(a, tmp_path / "same")
    keys = [load(base, [(n, f, s)]).entry(1).key for n, f, s in
            [("x", a, 1.0), ("y", str(tmp_path / "same"), 1.0), ("x", a, 2.0), ("x", str(standard(tmp_path / "c", 3)), 1.0)]]
    assert keys[0] == keys[1]                                # the same files at the same strength: the same rows
    assert len({keys[0], keys[2], keys[3]}) == 3             # another strength or other factors: other rows
    assert all(len(k) == 32 and set(k) <= set("0123456789abcdef") for k in keys)


def test_one_changed_byte_of_the_weights_changes_the_key(tmp_path, base):
    a = standard(tmp_path / "a", 1)
    before = load(base, [("x", str(a), 1.0)]).entry(1).key
    path = a / lf.WEIGHTS
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    assert load(base, [("x", str(a), 1.0)]).entry(1).key != before


def test_the_digest_two_ranks_compare_follows_every_slot_name_and_key(tmp_path, base):
    a, b = str(standard(tmp_path / "a", 1)), str(standard(tmp_path / "b", 2))
    ref = load(base, [("one", a, 1.0), ("two", b, 1.0)])
    assert load(base, [("one", a, 1.0), ("two", b, 1.0)]).digest() == ref.digest()
    assert load(base, [("two", b, 1.0), ("one", a, 1.0)]).digest() != ref.digest()          # slots swapped
    assert load(base, [("one", a, 1.0), ("two", b, 0.5)]).digest() != ref.digest()          # a strength
    assert load(base, [("one", a, 1.0)]).digest() != ref.digest()
    other = model_dir(tmp_path / "other")
    (other / "config.json").write_text('{"model_type": "deepseek_v41", "hidden_size": 256, "x": 1}')
    assert load(other, [("one", a, 1.0), ("two", b, 1.0)]).digest() != ref.digest()         # another base config
    assert len(ref.words()) == 4 and all(0 <= w < 2**31 for w in ref.words())
    assert ref.words() == load(base, [("one", a, 1.0), ("two", b, 1.0)]).words()


def test_a_reference_is_good_only_where_the_same_adapter_holds_its_slot(tmp_path, base):
    a, b = str(standard(tmp_path / "a", 1)), str(standard(tmp_path / "b", 2))
    reg = load(base, [("one", a, 1.0), ("two", b, 1.0)])
    swapped = load(base, [("two", b, 1.0), ("one", a, 1.0)])
    assert reg.check(None) and reg.check(reg.ref("one"))
    assert not reg.check(swapped.ref("one"))                 # the same name, another slot and key
    assert not reg.check(lr.AdapterRef(3, "0" * 32)) and not reg.check(lr.AdapterRef(0, ""))


def pick_registry(tmp_path, base):
    reg = load(base, [("one", str(standard(tmp_path / "a", 1)), 1.0), ("two", str(standard(tmp_path / "b", 2)), 1.0)])
    return reg, reg.model_map()


def test_a_request_picks_its_adapter_by_model_id_or_field(tmp_path, base):
    reg, models = pick_registry(tmp_path, base)
    assert reg.pick({"model": "base"}, models) is None and reg.pick({}, models) is None
    assert reg.pick({"model": "one"}, models) == "one"
    assert reg.pick({"model": "base", "adapter": "two"}, models) == "two"
    assert reg.pick({"model": "two", "adapter": "two"}, models) == "two"
    assert reg.pick({"model": "no such model"}, models) is None        # an unknown model id was always the default


@pytest.mark.parametrize("body", [{"adapter": "nope"}, {"adapter": 3}, {"adapter": ""}, {"model": "one", "adapter": "two"}])
def test_an_unknown_or_contradictory_choice_is_refused(tmp_path, base, body):
    reg, models = pick_registry(tmp_path, base)
    with pytest.raises(lr.UnknownAdapter):
        reg.pick(body, models)


def test_names_and_strengths_that_cannot_be_served_are_refused(tmp_path, base):
    a = str(standard(tmp_path / "a"))
    for bad in ("", "-x", "a b", "a=b", "x" * 97):
        with pytest.raises(lf.AdapterError, match="letters, digits"):
            load(base, [(bad, a, 1.0)])
    with pytest.raises(lf.AdapterError, match="already a model id"):
        load(base, [("base", a, 1.0)])
    with pytest.raises(lf.AdapterError, match="already a model id"):
        load(base, [("one", a, 1.0), ("one", a, 1.0)])
    for bad in (0.0, float("nan"), float("inf")):
        with pytest.raises(lf.AdapterError, match="strength"):
            load(base, [("one", a, bad)])
    assert load(base, [("neg", a, -1.0)]).entry(1).strength == -1.0       # an inverted edit is a valid strength


def test_a_stream_or_a_reference_gives_its_slot_and_key():
    ref = lr.AdapterRef(2, "ab" * 16)
    assert (lr.slot_of(ref), lr.key_of(ref)) == (2, "ab" * 16)
    assert (lr.slot_of(None), lr.key_of(None)) == (0, "")
    stream = SimpleNamespace(adapter=ref)
    assert (lr.slot_of(stream), lr.key_of(stream)) == (2, "ab" * 16)
    assert (lr.slot_of(SimpleNamespace(adapter=None)), lr.key_of(SimpleNamespace())) == (0, "")


def test_the_description_names_what_each_adapter_edits(tmp_path, base):
    reg = load(base, [("one", str(standard(tmp_path / "a", ranks=(2, 3))), 1.0)])
    d = reg.describe()[0]
    assert d["slot"] == 1 and d["modules"] == {"wo_b": 1, "shared_w2": 1} and d["max_rank"] == 3
    assert d["weights_sha256"] == reg.entry(1).spec.sha256
