#!/usr/bin/env python3
"""Standardized split-mode benchmark figures (ja/en) from a run-bench.sh run directory.

FROZEN CONVENTIONS (do not hand-tune figures outside this file):
  - 2x2 layout: (1) prefill  (2) decode  (3) ratio panel  (4) speedup-vs-baseline.
    Panels auto-hide: (3) needs >=2 series; (4) needs the baseline series and --vs on.
  - Depth-0 prefill point = fresh-prompt pp0 measurement (pp2048), not the ladder artifact.
  - Decode panel: thin dashed lines = real-operation estimate (measured x real-prompt
    factor from results-real.json); measured = solid, bold endpoint numbers.
  - Endpoint numbers: stacked at the right edge inside the axes (x=0.945, ha=left),
    bold for measured / light for estimates, near-equal duplicates dropped.
  - Titles left-aligned; suptitle left-aligned (x=0.005); grid on bar panels.

Reads: run-info.json, results-<series>.json, results-<series>-pp0.json (opt), results-real.json (opt).
Usage: plot_bench.py --dir RUN_DIR --series layer,tensor,single [--lang ja|en]
                     [--baseline single] [--ratio A,B] [--estimate on|off] [--vs on|off] [--out split-bench]
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {
    "layer":  {"ja": "layer (層分割・逐次)", "en": "layer (pipeline)"},
    "tensor": {"ja": "tensor (TP・同時計算)", "en": "tensor (TP, parallel)"},
    "tp2p":   {"ja": "tensor + GGML_CUDA_P2P=1", "en": "tensor + GGML_CUDA_P2P=1"},
}
COL = {"layer": "#1f77b4", "tensor": "#d62728", "single": "#7f7f7f", "tp2p": "#2ca02c"}
_FALLBACK_PALETTE = ["#8c564b", "#9467bd", "#e377c2", "#17becf", "#bcbd22"]
_MODE_DEV = {}   # mode name -> device, filled from run-info.json


def col(tag):
    """Series color; unknown (custom) mode names get a stable fallback color."""
    if tag not in COL:
        COL[tag] = _FALLBACK_PALETTE[len(COL) % len(_FALLBACK_PALETTE)]
    return COL[tag]


def disp(tag, lang):
    if tag in DISPLAY:
        return DISPLAY[tag][lang]
    dev = _MODE_DEV.get(tag, "")
    if "single" in tag or "gpu" in tag.lower():
        if not dev:
            return {"ja": "単一GPU", "en": "single GPU"}[lang]
        return {"ja": f"単一GPU ({dev}のみ)", "en": f"single GPU ({dev} only)"}[lang]
    return tag


def lighten(hexcol, amount=0.55):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def endpoints_dedupe(rows):
    out = []
    for v, c, style in rows:
        if any(abs(v - w) <= max(0.05, 0.004 * abs(v)) for w, _, _ in out):
            continue
        out.append((v, c, style))
    return out


def draw_endpoints(ax, rows):
    k = len(rows)
    for i, (v, c, style) in enumerate(rows):
        y = 0.045 + 0.085 * (k - 1 - i)
        txt = f"{v:.1f}" if v < 100 else f"{v:.0f}"
        ax.text(0.945, y, txt, color=c, fontsize=10.5 if style == "measured" else 9.5,
                fontweight="bold" if style == "measured" else "normal",
                alpha=1.0 if style == "measured" else 0.8,
                ha="left", transform=ax.transAxes)


def style_of(tag):
    ls = (0, (7, 3)) if tag in ("single", "tp2p") else "-"
    lw = 2.6 if tag == "single" else 2.0
    return ls, lw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--lang", default="ja", choices=["ja", "en"])
    ap.add_argument("--baseline", default="single")
    ap.add_argument("--ratio", default="")
    ap.add_argument("--estimate", default="on", choices=["on", "off"])
    ap.add_argument("--vs", default="on", choices=["on", "off"])
    ap.add_argument("--out", default="split-bench")
    args = ap.parse_args()
    ja = args.lang == "ja"
    plt.rcParams["font.family"] = ["Noto Sans CJK JP"] if ja else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    info = json.load(open(os.path.join(args.dir, "run-info.json")))
    series = [s.strip() for s in args.series.split(",") if s.strip()]
    for m in info.get("modes", []):
        if isinstance(m, dict) and m.get("name"):
            _MODE_DEV[m["name"]] = m.get("device", "")
    data = {}
    for s in series:
        recs = [r for r in json.load(open(os.path.join(args.dir, f"results-{s}.json")))
                if "effective_depth" in r]
        if len(recs) < 2:
            raise SystemExit(f"not enough records for series {s}")
        for r in recs:
            for key in ("prompt_per_second", "predicted_per_second"):
                if not isinstance(r.get(key), (int, float)):
                    raise SystemExit(f"series {s} stage {r.get('stage')}: missing {key} - incomplete run dir?")
        data[s] = recs
    lens = {s: len(data[s]) for s in series}
    if len(set(lens.values())) != 1:
        raise SystemExit(f"series have different stage counts: {lens} - re-plot a consistent set")

    stages = [int(x) for x in str(info.get("stages", "")).split(",")]
    if len(stages) != len(data[series[0]]):
        stages = [r["effective_depth"] for r in data[series[0]]]
    XL = ["0" if v == 0 else f"{round(v / 1000)}k" for v in stages]

    pp0 = {}
    for s in series:
        p = os.path.join(args.dir, f"results-{s}-pp0.json")
        if os.path.exists(p):
            d = json.load(open(p))
            sizes = sorted(int(k[2:]) for k in d if k.startswith("pp") and k[2:].isdigit())
            key = f"pp{min(sizes, key=lambda s: (abs(s - 2048), s))}" if sizes else None
            val = d[key].get("prompt_per_second") if key else None
            if isinstance(val, (int, float)):
                pp0[s] = val
            else:
                print(f"note: no usable pp0 value for {s}; depth-0 prefill falls back to the ladder value")

    factor = None
    if args.estimate == "on" and os.path.exists(os.path.join(args.dir, "results-real.json")):
        real = json.load(open(os.path.join(args.dir, "results-real.json")))
        ref = "tensor" if "tensor" in series else series[0]
        synth0 = data[ref][0]["predicted_per_second"]
        factor = sum(r["decode_tok_s"] / synth0 for r in real) / len(real)
        if not (0.4 < factor < 1.05):
            print(f"WARNING: real-prompt factor {factor:.3f} outside expected range")
    est = args.estimate == "on" and factor is not None

    rp = [x.strip() for x in args.ratio.split(",") if x.strip()]
    if len(rp) != 2:
        rp = series[:2] if len(series) >= 2 else []
        rp = [rp[0], rp[1]] if len(rp) == 2 else []
    show_ratio = len(rp) == 2 and rp[0] in data and rp[1] in data
    bl = args.baseline
    partners = [s for s in series if s != bl]
    show_vs = args.vs == "on" and bl in data and len(partners) >= 1
    if args.vs == "on" and not show_vs:
        print(f"note: vs-panel hidden (baseline '{bl}' not in series)")

    n_panels = 2 + int(show_ratio) + int(show_vs)
    if n_panels <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
        axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = list(axes.flatten())
    for ax in axes[n_panels:]:
        ax.axis("off")
    ax1, ax2 = axes[0], axes[1]
    nxt = 2

    # ---- (1) prefill ----
    for s in series:
        ys = [r["prompt_per_second"] for r in data[s]]
        xs = list(range(len(ys)))
        if s in pp0:
            ys = [pp0[s]] + ys[1:]
        ls, lw = style_of(s)
        ax1.plot(xs, ys, color=col(s), ls=ls, lw=lw, marker="o", ms=4, label=disp(s, args.lang))
    ax1.set_title(("① prefill (増分プロンプト評価速度) — 横軸: 深度ラダー(等間隔)" if ja else
                   "(1) prefill (incremental prompt-eval rate) - x: depth ladder (evenly spaced)"),
                  fontsize=11, loc="left")
    ax1.set_ylabel("tokens/s")
    draw_endpoints(ax1, endpoints_dedupe([(data[s][-1]["prompt_per_second"], col(s), "measured") for s in series]))

    # ---- (2) decode ----
    for s in series:
        ys = [r["predicted_per_second"] for r in data[s]]
        ls, lw = style_of(s)
        ax2.plot(range(len(ys)), ys, color=col(s), ls=ls, lw=lw, marker="o", ms=4, label=disp(s, args.lang))
    rows = []
    if est:
        for s in series:
            ys = [r["predicted_per_second"] * factor for r in data[s]]
            ax2.plot(range(len(ys)), ys, color=col(s),
                     ls=(0, (6, 3)) if s == "single" else (0, (3, 2.2)),
                     lw=1.3 if s == "single" else 0.9)
        ax2.text(0.015, 0.02,
                 (f"薄い線 = 実運用推定 (実プロンプト補正 ×{factor:.3f})" if ja else
                  f"thin dashed = real-operation estimate (real-prompt correction x{factor:.3f})"),
                 transform=ax2.transAxes, fontsize=8, color="#555555")
    for s in series:
        v = data[s][-1]["predicted_per_second"]
        rows.append((v, col(s), "measured"))
        if est:
            rows.append((v * factor, col(s), "estimate"))
    ax2.set_title(("② decode (平均生成速度)" if ja else
                   f"(2) decode (average generation rate, {info.get('n_predict', 1000)} tokens)"),
                  fontsize=11, loc="left")
    ax2.set_ylabel("tokens/s")
    draw_endpoints(ax2, endpoints_dedupe(rows))

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(XL)))
        ax.set_xticklabels(XL)
        ax.set_xlim(-0.35, len(XL) - 0.65)
        ax.grid(axis="y", color="#eeeeee")
        ax.legend(fontsize=8.5, loc="best")

    # ---- (3) ratio ----
    if show_ratio:
        ax3 = axes[nxt]; nxt += 1
        A, B = rp
        da, db = data[A], data[B]
        rel_pf = [(db[i]["prompt_per_second"] / da[i]["prompt_per_second"] - 1) * 100 for i in range(len(da))]
        rel_de = [(db[i]["predicted_per_second"] / da[i]["predicted_per_second"] - 1) * 100 for i in range(len(da))]
        if A in pp0 and B in pp0:
            rel_pf[0] = (pp0[B] / pp0[A] - 1) * 100
        xpos = range(len(da)); w = 0.38
        ax3.bar([x - w / 2 for x in xpos], rel_pf, w, color="#1f77b4",
                label=("prefill 差" if ja else "prefill delta"))
        ax3.bar([x + w / 2 for x in xpos], rel_de, w, color="#d62728",
                label=("decode 差" if ja else "decode delta"))
        ax3.axhline(0, color="#666666", lw=0.8)
        ax3.set_xticks(list(xpos)); ax3.set_xticklabels(XL, fontsize=9)
        ax3.set_ylabel(f"{disp(B, args.lang)} / {disp(A, args.lang)} − 1  (%)")
        vmax = max(rel_pf + rel_de + [1]); vmin = min(rel_pf + rel_de + [0])
        ax3.set_ylim(vmin - 10, vmax + 12)
        ax3.set_title((f"③ 相対差 ({disp(B, 'ja')} vs {disp(A, 'ja')})" if ja else
                       f"(3) relative difference ({disp(B, 'en')} vs {disp(A, 'en')})"),
                      fontsize=11, loc="left")
        ax3.set_axisbelow(True)
        ax3.grid(axis="y", color="#cccccc", lw=0.7)
        ax3.legend(fontsize=9)
        for x, v in zip(xpos, rel_de):
            ax3.text(x + w / 2, v + (1.5 if v >= 0 else -7), f"{v:+.0f}%", ha="center", fontsize=8.5)
        for x, v in zip(xpos, rel_pf):
            ax3.text(x - w / 2, v + (1.5 if v >= 0 else -7), f"{v:+.0f}%", ha="center", fontsize=8.5)

    # ---- (4) vs baseline ----
    if show_vs:
        ax4 = axes[nxt]; nxt += 1
        dep = data[bl]
        vals = []          # (partner, metric, values[])
        for s in partners:
            pf = [(data[s][i]["prompt_per_second"] / dep[i]["prompt_per_second"] - 1) * 100 for i in range(len(dep))]
            de = [(data[s][i]["predicted_per_second"] / dep[i]["predicted_per_second"] - 1) * 100 for i in range(len(dep))]
            if s in pp0 and bl in pp0:
                pf[0] = (pp0[s] / pp0[bl] - 1) * 100
            vals.append((s, "pf", pf))
            vals.append((s, "de", de))
        k = len(vals); bw = 0.2
        xpos = range(len(dep))
        for idx, (s, metric, vv) in enumerate(vals):
            off = (idx - (k - 1) / 2) * bw
            color = col(s) if metric == "pf" else lighten(col(s))
            label = (f"{disp(s, args.lang)} {'prefill' if metric == 'pf' else 'decode'}")
            ax4.bar([x + off for x in xpos], vv, bw, color=color, label=label)
        ax4.axhline(0, color="#666666", lw=0.8)
        ax4.set_xticks(list(xpos)); ax4.set_xticklabels(XL, fontsize=9)
        ax4.set_ylabel((f"{disp(bl, 'ja')}比 − 1  (%)" if ja else f"vs {disp(bl, 'en')} − 1  (%)"))
        allv = [v for _, _, vv in vals for v in vv]
        ax4.set_ylim(min(-4, min(allv) - 2), max(allv) * 1.15 + 4)
        ax4.set_title((f"④ {disp(bl, 'ja')} 比の伸び率 (同一条件)" if ja else
                       f"(4) speedup vs {disp(bl, 'en')} (same host / binary / argv)"),
                      fontsize=11, loc="left")
        ax4.set_axisbelow(True)
        ax4.grid(axis="y", color="#cccccc", lw=0.7)
        ax4.legend(fontsize=8.5, ncol=2)
        for idx, (s, metric, vv) in enumerate(vals):
            off = (idx - (k - 1) / 2) * bw
            for x, v in zip(xpos, vv):
                ax4.text(x + off, v + (1.2 if v >= 0 else -5.5), f"{v:.0f}", ha="center", fontsize=6.5)

    # ---- suptitle ----
    deep = max(r["effective_depth"] for r in data[series[0]]) / 1000
    date = str(info.get("date", ""))[:10]
    m = re.search(r"commit ([0-9a-f]+)", info.get("bin_version", ""))
    sha8 = info.get("bin_sha256", "")[:8]
    ver = f"llama.cpp master {m.group(1)} (sha {sha8})" if m else f"llama.cpp (sha {sha8})"
    machine = info.get("machine", "")
    npred = info.get("n_predict", 1000)
    ctxk = info.get("ctx", 0) // 1024
    kvk = info.get("cache_k", "")
    kvv = info.get("cache_v", "")
    kvlabel = f"KV {kvk}/{kvv}" if (kvk or kvv) else "KV default"
    line1 = (f"{machine} — {' / '.join(disp(s, 'ja') for s in series)}、{ctxk}kコンテキスト深度ラダー (実測最深 {deep:.1f}k, {date})" if ja else
             f"{machine} - {' / '.join(disp(s, 'en') for s in series)}, depth ladder to {ctxk}k context (deepest measured {deep:.1f}k, {date})")
    line2 = (f"{ver}・{kvlabel}・生成{npred}tok・合成テキスト・コンテキスト再利用方式" if ja else
             f"{ver} - {kvlabel} - {npred} tokens generated - synthetic text - context-reuse protocol")
    fig.suptitle(line1 + "\n" + line2, fontsize=11.5, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94 if n_panels > 2 else 0.90))
    out = os.path.join(args.dir, f"{args.out}-{args.lang}.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  (panels={n_panels}, estimate={'on' if est else 'off'}, factor={factor if factor else '-'})")


if __name__ == "__main__":
    main()
