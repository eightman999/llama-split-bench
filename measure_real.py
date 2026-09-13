#!/usr/bin/env python3
"""Real-prompt decode sanity runs on the standardized test server (port from config).

Default prompts = the built-in workload proxies (3 real-world-style tasks). Override with
--prompts-json '{"name": ..., "prompt": ...}' to measure with YOUR workload instead.
temperature 0.7 / top_p 0.9, n_predict 1200, fresh context per prompt. The plotter
derives the real-operation correction factor from results-real.json:
  factor = mean(real decode / synthetic decode at depth 0 of the reference series).

Usage: measure_real.py [--url http://127.0.0.1:18081] [--out results-real.json]
"""
import argparse
import hashlib
import json
import time
import urllib.request

PROMPTS = [
    ("design", "次の設計判断を整理して: 単一のV100 32GBで262kコンテキストの推論を回すとき、"
               "KVキャッシュの型・重みの量子化・投機デコードの3つは互いにどう影響し合うか。"
               "トレードオフを表にまとめ、優先順位と根拠を示して。"),
    ("review", "長文の技術文書をレビューする立場で、次の観点を順に検討して: 主張の根拠が測定に基づいているか、"
               "見落とされている交絡因子はないか、結論を出す前に必要な追加検証は何か。"
               "それぞれ具体的な反例を挙げて説明して。"),
    ("qa",     "以下を自分の言葉で説明して: なぜGPUのGEMVは帯域律速になるのか、"
               "なぜ量子化ブロックのメモリ配置が実効帯域を変えるのか、"
               "そして帯域を測る際に論理バイト数と実DRAM転送量がずれるのはどんな時か。"),
]


def post(url, path, payload, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return time.time() - t0, json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--out", default="results-real.json")
    ap.add_argument("--prompts-json", default="",
                    help='JSON file: [{"name": "...", "prompt": "..."}, ...]; default = 3 built-in workload proxies')
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_json:
        prompts = [(p["name"], p["prompt"]) for p in json.load(open(args.prompts_json))]
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)

    post(args.url, "/completion", {"prompt": "warmup", "n_predict": 8,
                                   "cache_prompt": False, "temperature": 0.0})

    records = []
    for name, prompt in prompts:
        wall, res = post(args.url, "/completion",
                         {"prompt": prompt, "n_predict": 1200, "cache_prompt": False,
                          "temperature": 0.7, "top_p": 0.9})
        t = res.get("timings", {})
        acc = t.get("draft_n_accepted") or 0
        dn = t.get("draft_n") or 0
        rec = {
            "prompt": name,
            "prompt_chars": len(prompt),
            "n_predict": 1200, "temperature": 0.7, "top_p": 0.9,
            "ttft_ms": round(t.get("prompt_ms") or 0, 1),
            "prompt_n": t.get("prompt_n"),
            "prefill_tok_s": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "decode_tok_s": t.get("predicted_per_second"),
            "draft_n": dn, "accepted": acc,
            "accept_rate": round(acc / dn, 3) if dn else None,
            "tokens_per_cycle": round(t["predicted_n"] / dn, 3) if dn and t.get("predicted_n") else None,
            "wall_s": round(wall, 1),
            "text_sha256": hashlib.sha256((res.get("content") or "").encode()).hexdigest(),
        }
        with open(os.path.join(outdir, f"real-response-{name}.txt"), "w") as f:
            f.write(res.get("content") or "")
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
