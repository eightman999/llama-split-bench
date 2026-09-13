#!/usr/bin/env bash
# List the llama.cpp devices and the NVIDIA GPU inventory, e.g. for filling bench.local.conf.
# Usage: ./list-devices.sh [path-to-llama-server]   (default: llama-server from PATH)
BIN=${1:-llama-server}
echo "== llama.cpp devices =="
OUT=$("$BIN" --list-devices 2>&1); rc=$?
printf '%s\n' "$OUT" | sed -n '1,40p'
if [ "$rc" -ne 0 ]; then
  echo "(failed to run '$BIN --list-devices' - if it is not on PATH, pass the path: ./list-devices.sh /path/to/llama-server)"
fi
echo
echo "== NVIDIA GPUs =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,compute_cap --format=csv || echo "(nvidia-smi not found)"
