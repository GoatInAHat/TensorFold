"""Adapter selection on the CUDA server: model ids, the adapter field, refusals before a stream opens, and a server
without adapters answering exactly as it did (fake engine, no GPU)."""

import json

import pytest

pytest.importorskip("jinja2")

from adapter_fakes import CFG, model_dir, standard

from tensorfold import cli, serve_options
from tensorfold.cuda import server
from tensorfold.families.deepseek_v41.cuda import lora_registry as lr
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, Engine, app_for, request

NOT_PASSED = "<not passed>"


class AdapterEngine(Engine):
    """The fake engine, taking an adapter name as the DeepSeek-V4.1 engine does."""

    def __init__(self, registry=None):
        super().__init__()
        self.adapters = registry
        self.seen = []                                           # the adapter argument of each call

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, adapter=NOT_PASSED):
        self.seen.append(adapter)
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


def served(tmp_path):
    base = model_dir(tmp_path / "model")
    reg = lr.AdapterRegistry.load([("alpha", str(standard(tmp_path / "a", 1)), 1.0),
                                   ("beta", str(standard(tmp_path / "b", 2)), 1.0)], CFG, base, ("fake-cuda",))
    (tmp_path / "app").mkdir()
    app = app_for(tmp_path / "app")
    app.engine = AdapterEngine(reg)
    app.adapter_models = reg.model_map()
    return app


def models(port):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", "/v1/models")
        return json.loads(connection.getresponse().read())["data"]
    finally:
        connection.close()


def test_models_lists_each_adapter_beside_the_base_model_and_names_its_parent(tmp_path):
    app = served(tmp_path)
    with http_server(app) as port:
        data = models(port)
    assert [m["id"] for m in data] == ["fake-cuda", "alpha", "beta"]
    assert "parent" not in data[0] and [(m["parent"], m["root"]) for m in data[1:]] == [("fake-cuda", "fake-cuda")] * 2


def test_a_request_naming_an_adapters_model_id_runs_that_adapter_and_the_reply_names_it(tmp_path):
    app = served(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 8, "model": "beta"})
    assert status == 200 and json.loads(text)["model"] == "beta"
    assert app.engine.seen == ["beta"]


def test_the_adapter_field_selects_an_adapter_beside_the_base_model_id(tmp_path):
    app = served(tmp_path)
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 8, "model": "fake-cuda", "adapter": "alpha"})
        assert status == 200 and json.loads(text)["model"] == "fake-cuda"
        status, _, _ = request(port, {"prompt": "Hi", "max_tokens": 8, "adapter": "beta"}, chat=False)
        assert status == 200
        status, _, _ = request(port, {"messages": HI, "max_tokens": 8, "model": "alpha", "adapter": "alpha"})
        assert status == 200
    assert app.engine.seen == ["alpha", "beta", "alpha"]


def test_the_base_model_runs_without_an_adapter_argument_at_all(tmp_path):
    app = served(tmp_path)
    with http_server(app) as port:
        for body in ({"messages": HI, "max_tokens": 8}, {"messages": HI, "max_tokens": 8, "model": "fake-cuda"},
                     {"messages": HI, "max_tokens": 8, "model": "unlisted"}):
            assert request(port, body)[0] == 200
    assert app.engine.seen == [NOT_PASSED] * 3                          # the keyword is not sent for the base model


@pytest.mark.parametrize("extra", [{"adapter": "nope"}, {"adapter": ""}, {"adapter": 7}, {"adapter": ["alpha"]},
                                   {"model": "alpha", "adapter": "beta"}])
def test_an_unknown_or_contradictory_adapter_is_refused_before_any_stream_opens(tmp_path, extra):
    app = served(tmp_path)
    with http_server(app) as port:
        for stream in (False, True):
            status, kind, text = request(port, {"messages": HI, "max_tokens": 8, "stream": stream, **extra})
            assert status == 400 and kind == "application/json", text
            assert "adapter" in json.loads(text)["error"]["message"]
    assert app.engine.calls == []


def test_a_server_without_adapters_ignores_the_field_as_it_ignores_any_unknown_one(tmp_path):
    app = app_for(tmp_path)                                              # the plain fake engine: no adapters attribute
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 8, "adapter": "anything"})
    assert status == 200 and json.loads(text)["model"] == "fake-cuda"
    app = served(tmp_path / "again")
    app.engine.adapters = None                                           # an engine started without --adapter
    with http_server(app) as port:
        assert request(port, {"messages": HI, "max_tokens": 8, "adapter": "alpha"})[0] == 200
    assert app.engine.seen == [NOT_PASSED]


def test_an_engine_that_cannot_take_an_adapter_is_not_asked_to(tmp_path):
    app = served(tmp_path)
    app.engine = Engine()                                                # no adapter parameter on generate
    app.engine.adapters = served(tmp_path / "x").engine.adapters
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "max_tokens": 8, "adapter": "alpha"})
    assert status in (400, 500) and "does not serve adapters" in text and app.engine.calls == []


def test_the_prepared_request_carries_the_choice(tmp_path):
    app = served(tmp_path)
    assert app.prepare({"messages": HI, "model": "alpha"}, True).adapter == "alpha"
    assert app.prepare({"messages": HI}, True).adapter is None
    assert app.model_ids == ["fake-cuda", "alpha", "beta"]


# -- the command line ---------------------------------------------------------------------------------------------
def args(*argv):
    return cli.build_parser().parse_args(["serve", "/m", "--backend", "cuda", *argv])


def test_the_flags_parse_to_names_folders_and_strengths_in_order():
    specs = serve_options.adapter_specs(args("--adapter", "b=/x/b", "--adapter", "a=/x/a", "--adapter-strength", "a=0.5"))
    assert specs == [("b", "/x/b", 1.0), ("a", "/x/a", 0.5)]
    assert serve_options.adapter_specs(args()) == []


@pytest.mark.parametrize("argv,match", [
    (["--adapter", "nofolder"], "NAME=FOLDER"), (["--adapter", "=/x"], "NAME=FOLDER"), (["--adapter", "a="], "NAME=FOLDER"),
    (["--adapter", "a b=/x"], "letters, digits"), (["--adapter", "a=/x", "--adapter", "a=/y"], "twice"),
    (["--adapter", "a=/x", "--adapter-strength", "a"], "NAME=NUMBER"),
    (["--adapter", "a=/x", "--adapter-strength", "a=zero"], "NAME=NUMBER"),
    (["--adapter", "a=/x", "--adapter-strength", "a=0"], "NAME=NUMBER"),
    (["--adapter", "a=/x", "--adapter-strength", "a=nan"], "NAME=NUMBER"),
    (["--adapter", "a=/x", "--adapter-strength", "b=2"], "no --adapter defines"),
    (["--adapter-strength", "a=2"], "no --adapter defines"),
])
def test_malformed_adapter_flags_are_refused(argv, match):
    with pytest.raises(ValueError, match=match):
        serve_options.adapter_specs(args(*argv))


def family(adapters=True):
    from types import SimpleNamespace

    return SimpleNamespace(title="Fam", model_type="deepseek_v41", package=SimpleNamespace(CUDA_ADAPTERS=adapters))


def test_adapters_need_a_family_that_serves_them_the_cuda_backend_and_parallel_streams():
    ok = args("--adapter", "a=/x", "--parallel", "4", "--name", "base")
    serve_options.check(ok, family(), "cuda")
    with pytest.raises(ValueError, match="has none"):
        serve_options.check(ok, family(False), "cuda")
    with pytest.raises(ValueError, match="MLX has none"):
        serve_options.check(ok, family(), "mlx")
    with pytest.raises(ValueError, match="--parallel 2 or more"):
        serve_options.check(args("--adapter", "a=/x", "--name", "base"), family(), "cuda")
    with pytest.raises(ValueError, match="--parallel 2 or more"):
        serve_options.check(args("--adapter", "a=/x", "--parallel", "1", "--name", "base"), family(), "cuda")


def test_an_adapter_may_not_take_the_base_models_name_or_an_alias():
    for argv in (["--adapter", "base=/x", "--name", "base"], ["--adapter", "other=/x", "--alias", "other"]):
        with pytest.raises(ValueError, match="already a model id"):
            serve_options.check(args(*argv, "--parallel", "2", "--name", "base") if "--name" not in argv
                                else args(*argv, "--parallel", "2"), family(), "cuda")


def test_serve_options_without_adapters_check_exactly_as_before():
    serve_options.check(args("--name", "base"), family(False), "cuda")


def test_serve_hands_the_adapters_and_the_taken_model_ids_to_the_engine(tmp_path, monkeypatch):
    from types import SimpleNamespace

    seen = {}
    package = SimpleNamespace(CUDA_ADAPTERS=True, cuda_engine=lambda *a, **k: seen.update(k) or SimpleNamespace(max_len=8192))
    family = SimpleNamespace(title="Test family", model_type="test", package=package)
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8192))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    base = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--name", "base", "--alias", "chat"]
    parsed = cli.build_parser().parse_args(base + ["--parallel", "2", "--adapter", "x=/x", "--adapter-strength", "x=0.5"])
    assert cli._serve_cuda(parsed, family, tmp_path, 8192) == 0
    assert seen["adapters"] == [("x", "/x", 0.5)] and seen["adapter_reserved"] == ["base", "chat"]
    seen.clear()
    assert cli._serve_cuda(cli.build_parser().parse_args(base), family, tmp_path, 8192) == 0
    assert "adapters" not in seen and "adapter_reserved" not in seen        # no flag: the engine is called as before


def test_the_responses_route_forwards_the_adapter_field_and_the_model_id_to_the_chat_it_runs():
    from tensorfold.server.responses_translate import translate

    chat = translate({"model": "alpha", "input": "hi", "adapter": "beta"}, None).chat
    assert chat["model"] == "alpha" and chat["adapter"] == "beta"
    assert "adapter" not in translate({"model": "alpha", "input": "hi"}, None).chat
