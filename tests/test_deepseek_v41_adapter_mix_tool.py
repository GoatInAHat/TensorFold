"""tools/dsv41/adapter_mix.py against a stand-in server: equal hashes pass, a reply that depends on its neighbours (or on
drafting) is reported and fails the run, and the model ids rotate over the streams."""

import hashlib
import importlib.util
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "dsv41" / "adapter_mix.py"


def tool():
    spec = importlib.util.spec_from_file_location("adapter_mix_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serve(leaky: bool = False, no_hash: bool = False):
    """A server whose reply hash is a function of (model, prompt, seed): and, when ``leaky``, of how many requests
    were in flight."""

    live, lock, seen = [0], threading.Lock(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body["model"], body.get("draft", True)))
            with lock:
                live[0] += 1
                crowd = live[0]
            try:
                import time

                time.sleep(0.05)
                with lock:
                    crowd = max(crowd, live[0])
                key = f"{body['model']}|{body['prompt']}|{body['seed']}" + (f"|{crowd}" if leaky else "")
                sha = hashlib.sha256(key.encode()).hexdigest()[:16]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
                last = {"choices": [], "usage": {"completion_tokens": 3}}
                if not no_hash:
                    last["tensorfold"] = {"token_sha": sha}
                self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")
            finally:
                with lock:
                    live[0] -= 1

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


def run(module, port, monkeypatch, capsys, *argv):
    monkeypatch.setattr("sys.argv", ["adapter_mix.py", f"http://127.0.0.1:{port}", *argv])
    code = module.main()
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_equal_hashes_pass_and_every_id_takes_every_place_in_turn(monkeypatch, capsys):
    httpd, seen = serve()
    try:
        code, lines = run(tool(), httpd.server_port, monkeypatch, capsys, "--models", "base,a,b", "--streams", "4",
                          "--tokens", "8")
    finally:
        httpd.shutdown()
    assert code == 0 and len(lines) == 3
    assert [line["models"] for line in lines] == [["base", "a", "b", "base"], ["a", "b", "base", "a"],
                                                  ["b", "base", "a", "b"]]
    assert all(all(line["concurrent_equals_solo"]) and all(line["solo_equals_serial"]) for line in lines)
    assert {m for m, _ in seen} == {"base", "a", "b"} and (("a", False) in seen)        # serial runs ask draft: false


def test_a_reply_that_depends_on_the_streams_beside_it_fails_the_run(monkeypatch, capsys):
    httpd, _ = serve(leaky=True)
    try:
        code, lines = run(tool(), httpd.server_port, monkeypatch, capsys, "--models", "base,a", "--streams", "3",
                          "--tokens", "8")
    finally:
        httpd.shutdown()
    assert code == 1 and any(not all(line["concurrent_equals_solo"]) for line in lines)


def test_a_reply_without_a_hash_is_an_error_not_a_pass(monkeypatch, capsys):
    httpd, _ = serve(no_hash=True)
    try:
        code, lines = run(tool(), httpd.server_port, monkeypatch, capsys, "--models", "base", "--streams", "2",
                          "--tokens", "8")
    finally:
        httpd.shutdown()
    assert code == 1 and "no token_sha" in lines[0]["errors"][0] and not any(lines[0]["concurrent_equals_solo"])
