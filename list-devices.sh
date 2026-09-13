#!/usr/bin/env bash
# List the llama.cpp devices and the NVIDIA GPU inventory, e.g. for filling bench.local.conf.
# Usage: ./list-devices.sh [path-to-llama-server]
BIN=${1:-llama-server}
echo "== llama.cpp devices =="
"$BIN" --list-devices 2>&1 | sed -n '1,40p' || echo "(failed to run '$BIN --list-devices')"
echo
echo "== NVIDIA GPUs =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,compute_cap --format=csv || echo "(nvidia-smi not found)"
