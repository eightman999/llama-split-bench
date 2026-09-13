#!/usr/bin/env bash
# Static checks for llama-split-bench (no GPU needed). Run before committing.
set -u
cd "$(dirname "$0")"
rc=0
for f in run-bench.sh list-devices.sh check.sh; do
  if bash -n "$f"; then echo "ok: bash -n $f"; else rc=1; fi
done
for f in measure_ladder.py measure_pp0.py measure_real.py plot_bench.py; do
  if python3 -m py_compile "$f"; then echo "ok: py_compile $f"; else rc=1; fi
done
if python3 -c "import pyflakes" 2>/dev/null; then
  if python3 -m pyflakes ./*.py; then echo "ok: pyflakes (undefined names etc.)"; else rc=1; fi
else
  echo "note: pyflakes not installed - 'pip install pyflakes' enables undefined-name checks"
fi
exit $rc
