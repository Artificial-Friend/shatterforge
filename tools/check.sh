#!/usr/bin/env bash
# The checks, then docs/STATUS.md from their results.
#   tools/check.sh            engine tests, wasm + page build, GPU suite            (~15 min on a software GPU)
#   tools/check.sh lab        the same, plus the lab's quick suite (collapse, ball)  (+ ~60 min)
#   tools/check.sh lab-full   the same, plus all five lab scenarios                  (+ ~3 h)
#   tools/check.sh status     only rewrite docs/STATUS.md from the last results
# Results go to lab-results/ (not committed); STATUS.md is committed.
set -u
cd "$(dirname "$0")/.."
mkdir -p lab-results
level="${1:-base}"
if [ "$level" != status ]; then
  echo "== engine tests"
  (cd engine && cargo test --release 2>&1) > lab-results/engine.txt; grep -E "test result|FAILED|panicked" lab-results/engine.txt | head -5
  echo "== wasm and page build"
  # (a failed wasm build stops here: the page was built on with the last good wasm, and the checks ran on it)
  if ! (cd engine && RUSTC_BOOTSTRAP=1 cargo build --release --target wasm32-unknown-unknown -Zbuild-std=core,alloc,panic_abort > ../lab-results/wasm.txt 2>&1); then
    grep -E "^error" -A4 lab-results/wasm.txt | head -20; echo "FAIL wasm build"; exit 1
  fi
  python3 game/build_tower.py
  echo "== GPU suite"
  node gpu/test_hulls.cjs
  node gpu/test_gpu.cjs 2>&1 | grep -E "^(PASS|FAIL)" | cut -c1-120
  case "$level" in
    # (and determinism: the collapse twice, 150 frames, the same frame for frame)
    lab) for c in rest debris; do node game/page_check.cjs $c 2>&1 | grep -E "^(PASS|FAIL)"; done
         node game/lab.cjs quick 2>&1 | grep -v "GL Driver" | grep -v " of "
         node game/lab.cjs collapse --frames 150 --twice --out lab-results/twice 2>&1 | grep -E "deterministic|    frame" ;;
    lab-full) node game/lab.cjs full 2>&1 | grep -v "GL Driver" | grep -v " of " ;;
  esac
fi
python3 tools/status.py
