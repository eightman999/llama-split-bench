#!/usr/bin/env python3
"""Depth-staged prefill/decode measurement on the standardized test server (the
ladder). Context is reused across stages (cache_prompt=True, single slot): each
stage sends the previous prompt extended by a new filler chunk, so the server only
evaluates the new chunk (prompt_n) and prompt_per_second is the incremental prefill
rate at that depth. Right after each fill the same request generates --n-predict
tokens (default 1000, ignore_eos) and predicted_per_second is the decode rate
(MTP n_max=2 included; synthetic filler text -> optimistic acceptance).

Protocol matches the published 2x V100 measurements (see README); everything is parameterized.

Usage: measure_ladder.py --tag <mode> [--stages 0,32000,...] [--n-predict 1000]
                         [--url http://127.0.0.1:18081] [--out DIR] [--guard-devices 0,1]
                         [--ratio-init 4.3]
"""
import argparse
import gzip
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

TAIL = ("\n\nContinue the sequence with the next 100 numbered lines, in exactly "
        "the same style. Do not stop and do not summarize.")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_index_map():
    """nvidia-smi uuid -> GPU index (string)."""
    m = {}
    for line in sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader").splitlines():
        try:
            idx, uuid = [x.strip() for x in line.split(",")]
        except ValueError:
            continue
        m[uuid] = idx
    return m


def gpu_pids(guard_idx):
    """Compute-process PIDs on the guarded GPU indices. Empty guard set = no guard."""
    if not guard_idx:
        return set()
    idx_map = gpu_index_map()
    pids = set()
    for line in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").splitlines():
        try:
            uuid, pid = [x.strip() for x in line.split(",")]
        except ValueError:
            continue
        if idx_map.get(uuid) in guard_idx:
            pids.add(int(pid))
    return pids


def fill_text(n_chars):
    unit = "Sequence {i}: the quick brown fox jumps over the lazy dog while we measure context tokens. "
    parts, total, i = [], 0, 1
    while total < n_chars:
        p = unit.format(i=i)
        parts.append(p)
        total += len(p)
        i += 1
    return "".join(parts)[:n_chars]


def post(url, path, payload, timeout=7200):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return time.time() - t0, body


def gpu_state():
    return sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,"
              "clocks.sm,power.draw --format=csv,noheader")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stages", default="0,32000,64000,128000,196000,258000")
    ap.add_argument("--n-predict", type=int, default=1000)
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--out", required=True)
    ap.add_argument("--guard-devices", default="",
                    help="nvidia-smi GPU indices to guard against foreign compute, e.g. '0,1' (empty = no guard)")
    ap.add_argument("--ratio-init", type=float, default=4.3,
                    help="initial chars-per-token guess for filler sizing (adapts after stage 1)")
    args = ap.parse_args()

    guard_idx = set(args.guard_devices.split(",")) if args.guard_devices else set()
    stages = [int(x) for x in args.stages.split(",")]
    os.makedirs(args.out, exist_ok=True)
    resp_dir = os.path.join(args.out, f"responses-{args.tag}")
    os.makedirs(resp_dir, exist_ok=True)
    results_path = os.path.join(args.out, f"results-{args.tag}.json")
    status_path = os.path.join(args.out, f"status-{args.tag}.txt")

    records = []
    baseline = gpu_pids(guard_idx)
    if guard_idx:
        print(f"guarding GPU indices {sorted(guard_idx)}; server pids: {sorted(baseline)}", flush=True)
    else:
        print("no guard devices given - foreign-process guard disabled", flush=True)
    ratio = args.ratio_init

    for k, target in enumerate(stages):
        foreign = gpu_pids(guard_idx) - baseline
        if foreign:
            msg = f"ABORT: foreign GPU compute pids {sorted(foreign)}"
            print(msg, flush=True)
            records.append({"aborted": msg})
            with open(results_path, "w") as f:
                json.dump(records, f, indent=2)
            break

        if target == 0:
            prompt = "The capital of France is and the capital of Germany is"
        else:
            prompt = fill_text(int(target * ratio)) + TAIL

        before = gpu_state()
        t0 = time.time()
        shrunk = False
        payload = {"prompt": prompt, "n_predict": args.n_predict,
                   "cache_prompt": True, "temperature": 0.0, "ignore_eos": True}
        try:
            wall, res = post(args.url, "/completion", payload)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            print(f"stage {target}: HTTP {e.code} ({body}) - retrying at 96% fill", flush=True)
            prompt = fill_text(int(target * ratio * 0.96)) + TAIL
            payload["prompt"] = prompt
            shrunk = True
            try:
                wall, res = post(args.url, "/completion", payload)
            except Exception as e2:
                print(f"stage {target}: retry failed: {e2}", flush=True)
                records.append({"aborted": f"stage {target}: {e2}"})
                with open(results_path, "w") as f:
                    json.dump(records, f, indent=2)
                break
        except Exception as e:
            print(f"stage {target}: request failed: {e}", flush=True)
            records.append({"aborted": f"stage {target}: {e}"})
            with open(results_path, "w") as f:
                json.dump(records, f, indent=2)
            break
        after = gpu_state()

        t = res.get("timings", {})
        ev = res.get("tokens_evaluated", 0)
        if target and ev > 0:
            ratio = (len(prompt) - len(TAIL)) / float(ev)
        draft_n = t.get("draft_n") or 0
        acc = t.get("draft_n_accepted") or 0
        rec = {
            "stage": k,
            "target_tokens": target,
            "shrunk": shrunk,
            "prompt_chars": len(prompt),
            "tokens_evaluated": ev,
            "cache_n": t.get("cache_n"),
            "prompt_n": t.get("prompt_n"),
            "effective_depth": (t.get("cache_n") or 0) + (t.get("prompt_n") or 0),
            "prompt_ms": t.get("prompt_ms"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "predicted_ms": t.get("predicted_ms"),
            "predicted_per_second": t.get("predicted_per_second"),
            "draft_n": draft_n,
            "draft_n_accepted": acc,
            "draft_accept_rate": round(acc / draft_n, 3) if draft_n else None,
            "wall_s": round(wall, 1),
            "gpu_before": before,
            "gpu_after": after,
        }
        records.append(rec)

        with open(os.path.join(resp_dir, f"stage{k}-{target}.txt"), "w") as f:
            f.write(res.get("content") or "")
        with gzip.open(os.path.join(resp_dir, f"prompt{k}-{target}.txt.gz"), "wt") as f:
            f.write(prompt)

        with open(results_path, "w") as f:
            json.dump(records, f, indent=2)
        with open(status_path, "w") as f:
            f.write(json.dumps({kk: rec[kk] for kk in
                                ("stage", "target_tokens", "effective_depth", "cache_n",
                                 "prompt_n", "prompt_per_second", "predicted_n",
                                 "predicted_per_second", "draft_accept_rate", "wall_s")}) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
