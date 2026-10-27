"""Exactness of mixed adapter rounds on a running server: each stream of a concurrent burst, whatever model id it asks
for (the base model or an adapter) and wherever the ids sit among the streams, must give the token-id hash (token_sha)
of its own solo run, and each solo run that of its "draft": false run. The ids rotate over the streams, so every
stream takes every id in turn. One JSON line a rotation; exit 1 on a mismatch or a failed request. Standard library only.

  python3 adapter_mix.py http://127.0.0.1:8080 --models BASE,adapter-a,adapter-b --streams 4 --tokens 128
"""

import argparse
import json
import sys
import threading
import urllib.request

PROMPTS = ["Write a short Python function that computes the Fibonacci sequence and explain it.",
           "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
           "List the planets of the solar system in order from the sun, one line each, with one fact about each.",
           "Summarize the plot of a heist story set on a space station in about five sentences."]


def ask(base: str, model: str, prompt: str, seed: int, tokens: int, temperature: float, draft: bool = True,
        gate: threading.Barrier | None = None) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "seed": seed}
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        if gate is not None:
            gate.wait()
        sha = None
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    sha = (json.loads(line[5:]).get("tensorfold") or {}).get("token_sha") or sha
        return {"sha": sha, "error": None if sha else "the reply carried no token_sha"}
    except Exception as exc:  # noqa: BLE001 - a failed request is reported
        return {"sha": None, "error": f"{type(exc).__name__}: {exc}"}


def burst(base: str, models: list[str], seeds: list[int], tokens: int, temperature: float) -> list[dict]:
    gate = threading.Barrier(len(models))
    out: list[dict] = [{} for _ in models]

    def run(i: int) -> None:
        out[i] = ask(base, models[i], PROMPTS[i % len(PROMPTS)], seeds[i], tokens, temperature, gate=gate)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(models))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("base")
    p.add_argument("--models", required=True, help="model ids, a comma list (the base model's and the adapters')")
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    ids = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [args.seed + i for i in range(args.streams)]
    bad = 0
    for turn in range(len(ids)):
        models = [ids[(i + turn) % len(ids)] for i in range(args.streams)]
        solo = [ask(args.base, models[i], PROMPTS[i % len(PROMPTS)], seeds[i], args.tokens, args.temperature)
                for i in range(args.streams)]
        serial = [ask(args.base, models[i], PROMPTS[i % len(PROMPTS)], seeds[i], args.tokens, args.temperature,
                      draft=False) for i in range(args.streams)]
        together = burst(args.base, models, seeds, args.tokens, args.temperature)
        errors = sorted({r["error"] for r in solo + serial + together if r.get("error")})
        concurrent = [t["sha"] == s["sha"] and t["sha"] is not None for t, s in zip(together, solo)]
        drafted = [s["sha"] == d["sha"] and s["sha"] is not None for s, d in zip(solo, serial)]
        line = {"models": models, "concurrent_equals_solo": concurrent, "solo_equals_serial": drafted,
                "errors": errors[:3]}
        print(json.dumps(line), flush=True)
        bad += (not all(concurrent)) + (not all(drafted)) + bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
