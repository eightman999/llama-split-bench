#!/usr/bin/env bash
# llama-split-bench: measure layer vs tensor (vs single-GPU) for llama.cpp on any multi-GPU box.
# Usage: run-bench.sh <tag> [--modes a,b] [--devices CUDA0,CUDA1] [--ctx N] [--stages S]
#                      [--n-predict N] [--pp0-sizes S] [--bin PATH] [--model PATH]
#                      [--port N] [--no-real]
# Launch detached:  setsid nohup bash run-bench.sh <tag> > runs/<tag>.log 2>&1 &
# Everything tunable lives in bench.conf / bench.local.conf - no other edits are needed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/bench.conf"
[ -f "$HERE/bench.local.conf" ] && source "$HERE/bench.local.conf"
RUNS_DIR="${RUNS_DIR:-$HERE/runs}"

TAG=""; MODES_R=""; CTX_R="$CTX"; STAGES_R="$STAGES"; NP_R="$N_PREDICT"; PP0_R="$PP0_SIZES"
DEVICES_R=""; BIN_R="$BIN"; MODEL_R="$MODEL"; PORT_R="$PORT"; DO_REAL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --modes) MODES_R="$2"; shift 2;;
    --devices) DEVICES_R="$2"; shift 2;;
    --ctx) CTX_R="$2"; shift 2;;
    --stages) STAGES_R="$2"; shift 2;;
    --n-predict) NP_R="$2"; shift 2;;
    --pp0-sizes) PP0_R="$2"; shift 2;;
    --bin) BIN_R="$2"; shift 2;;
    --model) MODEL_R="$2"; shift 2;;
    --port) PORT_R="$2"; shift 2;;
    --no-real) DO_REAL=0; shift;;
    -*) echo "unknown option: $1"; exit 1;;
    *) TAG="$1"; shift;;
  esac
done
[ -z "$TAG" ] && { echo "usage: run-bench.sh <tag> [--modes a,b] [--devices ...] [--ctx N] [--stages S] [--n-predict N] [--no-real]"; exit 1; }
[ -z "$MODEL_R" ] && { echo "ERROR: MODEL is not set - put it in bench.conf or bench.local.conf (see README)"; exit 1; }
command -v "$BIN_R" >/dev/null 2>&1 || [ -x "$BIN_R" ] || { echo "ERROR: binary '$BIN_R' not found"; exit 1; }

# --devices rebuilds the standard mode set (layer/tensor on all, single on the first)
if [ -n "$DEVICES_R" ]; then
  DEVICES="$DEVICES_R"
fi
if ! declare -p MODES >/dev/null 2>&1; then
  SD="$SINGLE_DEV"; [ "$SD" = auto ] && SD="${DEVICES%%,*}"
  MODES=("layer|$DEVICES|layer" "tensor|$DEVICES|tensor" "single|$SD|")
fi

SEL=()
for m in "${MODES[@]}"; do
  name="${m%%|*}"
  if [ -z "$MODES_R" ] || [[ ",$MODES_R," == *",$name,"* ]]; then SEL+=("$m"); fi
done
[ ${#SEL[@]} -eq 0 ] && { echo "no modes selected - check the MODES array in bench.conf / --modes"; exit 1; }

TAGDIR="$RUNS_DIR/$TAG"; mkdir -p "$TAGDIR"
URL="http://127.0.0.1:$PORT_R"

# which mode runs the real-prompt correction prompts
REAL_TARGET="$REAL_MODE"
if [ "$REAL_TARGET" = auto ] || [ -z "$REAL_TARGET" ]; then
  REAL_TARGET=""
  for m in "${SEL[@]}"; do [ "${m%%|*}" = tensor ] && { REAL_TARGET=tensor; break; }; done
  [ -z "$REAL_TARGET" ] && REAL_TARGET="${SEL[0]%%|*}"
fi

# optional fan logging from a hwmon directory
FANF=""
if [ "$FAN_HWMON" = auto ]; then
  FAN_HWMON=$(grep -lE "nct6[0-9]+" /sys/class/hwmon/hwmon*/name 2>/dev/null | xargs -r dirname | head -1)
fi
if [ -n "$FAN_HWMON" ] && [ -d "$FAN_HWMON" ]; then
  for ch in ${FAN_CHANNELS//,/ }; do
    [ -r "$FAN_HWMON/pwm$ch" ] && FANF="$FANF pwm$ch=$(cat "$FAN_HWMON/pwm$ch")"
  done
fi

guard_of() { echo "$1" | tr ',' '\n' | sed -n 's/^CUDA\([0-9][0-9]*\)$/\1/p' | paste -sd, -; }

REAL_DONE=0
echo "BENCH-START $TAG $(date +%H:%M:%S)  ctx=$CTX_R stages=$STAGES_R np=$NP_R"
for m in "${SEL[@]}"; do
  IFS='|' read -r name dev split <<< "$m"
  echo "=== mode $name (device=$dev split=${split:-default}) $(date +%H:%M:%S) ==="

  STOP="$TAGDIR/sampler-$name.stop"; rm -f "$STOP"
  ( while [ ! -f "$STOP" ]; do
      echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,power.draw,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' '|')$FANF"
      sleep 2
    done > "$TAGDIR/sampler-$name.log" 2>&1 ) &
  SAMPLER=$!

  ARGS=("$BIN_R" -m "$MODEL_R" --host 127.0.0.1 --port "$PORT_R" --device "$dev")
  [ -n "$split" ] && ARGS+=(--split-mode "$split")
  ARGS+=(-ngl "$NGL" -fa on --jinja -c "$CTX_R" --parallel 1 -t "$THREADS")
  [ -n "$KV_K" ] && ARGS+=(--cache-type-k "$KV_K")
  [ -n "$KV_V" ] && ARGS+=(--cache-type-v "$KV_V")
  if [ -n "$SPEC_ARGS" ]; then
    ARGS+=($SPEC_ARGS)
    case "$SPEC_ARGS" in *draft*)
      SDV="$SPEC_DEVICE"; [ "$SDV" = auto ] && SDV="$dev"
      ARGS+=(--spec-draft-device "$SDV");;
    esac
  fi
  [ -n "$LOAD_MODE" ] && ARGS+=(--load-mode "$LOAD_MODE")
  [ -n "$CACHE_ARGS" ] && ARGS+=($CACHE_ARGS)
  [ -n "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")
  [ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)
  ARGS+=(--alias "$(basename "$MODEL_R" .gguf)")

  "${ARGS[@]}" > "$TAGDIR/server-$name.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 120); do
    if curl -s -m 2 "$URL/health" 2>/dev/null | grep -q '"status":"ok"'; then ok=1; break; fi
    sleep 5
  done
  echo "health: $(curl -s -m 3 "$URL/health")"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; echo
  if [ "$ok" != 1 ]; then
    echo "BENCH-ABORT: $name server not ready (see $TAGDIR/server-$name.log)"
    kill "$SRV" 2>/dev/null; pkill -P "$SRV" 2>/dev/null; touch "$STOP"; wait $SAMPLER 2>/dev/null
    exit 1
  fi

  python3 "$HERE/measure_ladder.py" --tag "$name" --stages "$STAGES_R" --n-predict "$NP_R" \
    --url "$URL" --out "$TAGDIR" --guard-devices "$(guard_of "$dev")"
  if grep -q '"aborted"' "$TAGDIR/results-$name.json" 2>/dev/null; then
    echo "BENCH-ABORT: $name ladder aborted"
    kill "$SRV" 2>/dev/null; pkill -P "$SRV" 2>/dev/null; touch "$STOP"; wait $SAMPLER 2>/dev/null
    exit 1
  fi
  python3 "$HERE/measure_pp0.py" --tag "$name" --sizes "$PP0_R" --url "$URL" --out "$TAGDIR/results-$name-pp0.json"
  if [ "$DO_REAL" = 1 ] && [ "$REAL_DONE" = 0 ] && [ "$name" = "$REAL_TARGET" ]; then
    python3 "$HERE/measure_real.py" --url "$URL" --out "$TAGDIR/results-real.json"
    REAL_DONE=1
  fi

  kill "$SRV" 2>/dev/null; sleep 3; kill -9 "$SRV" 2>/dev/null
  ORPHAN=$(pgrep -f "llama-serve[r].*--port $PORT_R" | head -1)
  [ -n "$ORPHAN" ] && { kill "$ORPHAN" 2>/dev/null; sleep 2; }
  touch "$STOP"; wait $SAMPLER 2>/dev/null
done

# config snapshot for the plotter
MODESPEC=$(printf '%s;' "${SEL[@]}")
python3 - "$TAGDIR" "$TAG" "$CTX_R" "$STAGES_R" "$NP_R" "$MODESPEC" "$MACHINE" "$BIN_R" "$KV_K" "$KV_V" <<'PYEOF'
import json, os, socket, subprocess, sys, datetime
from collections import Counter

d, tag, ctx, stages, npred, modespec, machine, binp, kvk, kvv = sys.argv[1:11]

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

modes = []
for m in modespec.split(";"):
    if not m:
        continue
    parts = (m.split("|") + ["", ""])[:3]
    modes.append({"name": parts[0], "device": parts[1], "split": parts[2]})

path = binp if os.path.exists(binp) else sh(f"command -v {binp}")
sha = sh(f"sha256sum '{path}'").split()[0] if path else ""
_v = subprocess.run([binp, "--version"], capture_output=True, text=True)
ver = (_v.stdout + " " + _v.stderr).replace("\n", " ").strip()   # some builds print --version to stderr

if machine == "auto":
    gn = [x.strip() for x in sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines() if x.strip()]
    c = Counter(gn)
    gs = " + ".join(f"{n}x{v}" if n > 1 else v for v, n in c.items())
    host = socket.gethostname().split(".")[0]
    machine = f"{host} \u2014 {gs}" if gs else host

info = {"tag": tag, "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ctx": int(ctx), "stages": stages, "n_predict": int(npred),
        "modes": modes, "machine": machine,
        "bin": binp, "bin_version": ver, "bin_sha256": sha,
        "cache_k": kvk, "cache_v": kvv}
with open(f"{d}/run-info.json", "w") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print("run-info.json written")
PYEOF

# figures (ja + en)
SERIES_NAMES=$(printf '%s\n' "${SEL[@]}" | cut -d'|' -f1 | paste -sd,)
for LANG in ja en; do
  "$VENV_PY" "$HERE/plot_bench.py" --dir "$TAGDIR" --series "$SERIES_NAMES" --lang "$LANG" --out split-bench \
    || echo "plot($LANG) failed - check VENV_PY/matplotlib (see README)"
done
echo "BENCH-DONE $TAG $(date +%H:%M:%S)"
ls -la "$TAGDIR"/*.png 2>/dev/null
