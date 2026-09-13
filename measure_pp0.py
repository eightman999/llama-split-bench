#!/usr/bin/env python3
"""Depth-0 prefill measurement with proper prompt lengths (standard: 512/2048/8192 tok).

Sends fresh (cache_prompt=false) prompts to the running test server and records
the server-reported prefill rate at effectively empty context. This replaces the
tiny-prompt artifact of the ladder harness's stage 0 in the figures; the plotter
uses the 2048-token value by default.

Usage: measure_pp0.py --tag <mode> --sizes 512,2048,8192 [--url ...] [--out FILE]
"""
import argparse
import json
import time
import urllib.request

TAIL = ("\n\nContinue the sequence with the next 100 numbered lines, in exactly "
        "the same style. Do not stop and do not summarize.")


def fill_text(n_chars):
    unit = "Sequence {i}: the quick brown fox jumps over the lazy dog while we measure context tokens. "
    parts, total, i = [], 0, 1
    while total < n_chars:
        p = unit.format(i=i)
        parts.append(p)
        total += len(p)
        i += 1
    return "".join(parts)[:n_chars]


def post(url, path, payload, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return time.time() - t0, json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sizes", default="512,2048,8192")
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"tag": args.tag}

    post(args.url, "/completion",
         {"prompt": "warmup", "n_predict": 8, "cache_prompt": False, "temperature": 0.0})

    for size in [int(x) for x in args.sizes.split(",")]:
        prompt = fill_text(int(size * 4.3)) + TAIL
        wall, res = post(args.url, "/completion",
                         {"prompt": prompt, "n_predict": 64, "cache_prompt": False,
                          "temperature": 0.0, "ignore_eos": True})
        t = res.get("timings", {})
        out[f"pp{size}"] = {
            "prompt_n": t.get("prompt_n"),
            "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "predicted_per_second": t.get("predicted_per_second"),
            "wall_s": round(wall, 2),
        }
        print(f"{args.tag} pp{size}: prompt_n={t.get('prompt_n')} cache_n={t.get('cache_n')} "
              f"prefill={t.get('prompt_per_second'):.1f} t/s", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
