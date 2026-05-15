#!/usr/bin/env bash
# Run rfd3, rf3, or mpnn on CPU, bypassing the broken MPS auto-select.
# Usage: run_cpu.sh rfd3 design out_dir=... inputs=...
#        run_cpu.sh rf3 fold inputs=... out_dir=...
#        run_cpu.sh mpnn --model_type protein_mpnn --checkpoint_path ... --structure_path ... --out_directory ...

set -e
: "${RFD3_DIR:?Set RFD3_DIR to the foundry install root}"
source "$RFD3_DIR/.venv/bin/activate"

[[ $# -ge 1 ]] || { echo "Usage: $0 {rfd3|rf3|mpnn} <args...>" >&2; exit 1; }

cli="$1"; shift
case "$cli" in
  rfd3) module="rfd3.cli";       attr="app"  ;;
  rf3)  module="rf3.cli";        attr="app"  ;;
  mpnn) module="mpnn.inference"; attr="main" ;;
  *) echo "Unknown CLI: $cli (expected rfd3, rf3, or mpnn)" >&2; exit 1 ;;
esac

exec python -c "
import torch, sys, importlib
torch.backends.mps.is_available = lambda: False
sys.argv = ['$cli'] + sys.argv[1:]
getattr(importlib.import_module('$module'), '$attr')()
" "$@"
