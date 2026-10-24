"""The prompt path's TFE taps against themselves at other chunk cuts (stdlib only). The engine runs with
TF_DS_EAGLE_TAPDUMP=FILE (rank 0), so every absorb writes its tap rows (eagle.zig tapDump).

  python3 -I tfe_tapcheck.py send [URL]        three prompts that share a prefix (lengths about N, N + 40, N - 40)
  python3 -I tfe_tapcheck.py check DUMP [--gate 0,1]   compare the "prompt" records at the positions they share

A prompt chunk that reaches the head drops its rows below the replay floor (the prompt's length - the model window) at
the middle layer; a tap of a layer before that keeps the chunk's rows from its first, so the absorb reads it at the
cut. The three prompts cut at three places, so a wrong offset gives one position three different rows (a difference
near 1). Pass: each gated tap's max abs difference over max abs at the shared positions is at most 3e-2 (bf16
rounding with other prefill chunk shapes), and the records have at least two different nonzero cuts.

--gate: the tap indices (engine order) that must pass; the default is every tap. Gate only the taps of layers below
layers / 2: a tap at or past the middle layer depends on the replay floor (rows below it run no upper layer, so the
first replayed rows see less history), so it differs between prompts at the same position. 2026-10-24, taps
[2, 26, 37, final]: [0.0195, 0.84, 0.54, 0.26].
"""

import json
import struct
import sys
import urllib.request

MODEL = "DeepSeek-v4.1-Flash-1017-TF"
WORDS = ("harbor lantern copper meadow signal river orchard basalt willow ember canyon ledger quartz monsoon "
         "glacier sparrow tundra cobalt prairie lattice").split()


def text(n_words: int) -> str:
    """Plain prose-like text of n_words words (a fixed sequence, so prompts share a prefix)."""

    out = []
    for i in range(n_words):
        w = WORDS[(i * 7 + i // 13) % len(WORDS)]
        out.append(w + ("." if i % 11 == 10 else ","if i % 5 == 4 else ""))
    return " ".join(out)


def send(url: str) -> None:
    base = text(2600)  # about 3,300 tokens: two prompt chunks of 2,048
    prompts = [base, base + " " + text(30), base[: int(len(base) * 0.985)]]
    for i, p in enumerate(prompts):
        body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 2, "temperature": 0}
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=300))
        print(json.dumps({"prompt": i, "prompt_tokens": r["usage"]["prompt_tokens"]}), flush=True)


def records(path: str):
    b = open(path, "rb").read()
    at = 0
    while at < len(b):
        n = struct.unpack_from("<I", b, at)[0]
        h = json.loads(b[at + 4:at + 4 + n])
        size = h["n"] * h["taps"] * h["dim"] * 2
        h["rows"] = memoryview(b)[at + 4 + n:at + 4 + n + size]
        at += 4 + n + size
        yield h


def bf16_row(mv, r: int, j: int, taps: int, dim: int):
    o = (r * taps + j) * dim * 2
    raw = mv[o:o + dim * 2].cast("H")
    return [struct.unpack("<f", struct.pack("<I", x << 16))[0] for x in raw]


def check(path: str, gate: list[int] | None = None) -> None:
    pr = [h for h in records(path) if h["kind"] == "prompt"]
    print(json.dumps({"prompt_records": [{k: h[k] for k in ("slot", "start", "n", "cut")} for h in pr],
                      "round_records": sum(1 for h in records(path) if h["kind"] == "round")}))
    if len(pr) < 2:
        raise SystemExit("FAIL: fewer than 2 prompt records")
    taps, dim = pr[0]["taps"], pr[0]["dim"]
    worst = [0.0] * taps
    pairs = 0
    for a in range(len(pr)):
        for b in range(a + 1, len(pr)):
            A, B = pr[a], pr[b]
            lo = max(A["start"], B["start"])
            hi = min(A["start"] + A["n"], B["start"] + B["n"]) - 16  # (the chat template's last tokens differ)
            for p in range(lo, hi, 4):
                pairs += 1
                for j in range(taps):
                    x = bf16_row(A["rows"], p - A["start"], j, taps, dim)
                    y = bf16_row(B["rows"], p - B["start"], j, taps, dim)
                    m = max(max(abs(v) for v in x), 1e-12)
                    worst[j] = max(worst[j], max(abs(u - v) for u, v in zip(x, y)) / m)
    cuts = sorted({h["cut"] for h in pr if h["cut"] > 0})
    gate = list(range(taps)) if gate is None else gate
    ok = pairs > 0 and len(cuts) >= 2 and max(worst[j] for j in gate) <= 3e-2
    print(json.dumps({"tapcheck": "pass" if ok else "fail", "positions_compared": pairs, "cuts": cuts, "gate": gate,
                      "worst_rel_by_tap": [round(w, 6) for w in worst]}))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "send":
        send(sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8899/v1/chat/completions")
    else:
        g = sys.argv[sys.argv.index("--gate") + 1] if "--gate" in sys.argv else None
        check(sys.argv[2], [int(x) for x in g.split(",")] if g else None)
