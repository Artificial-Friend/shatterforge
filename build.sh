#!/usr/bin/env bash
# Build the engine to WebAssembly and bundle the single-file game page into dist/shatterforge.html.
#   ./build.sh          release build
#   ./build.sh debug    release build with the test/diagnostic exports (tools/debug_exports.rs) appended
set -euo pipefail
cd "$(dirname "$0")"
TARGET=wasm32-unknown-unknown
if [[ "${1:-}" == "debug" ]]; then
  cp src/lib.rs /tmp/shatterforge_lib.rs.orig
  cat tools/debug_exports.rs >> src/lib.rs
  trap 'cp /tmp/shatterforge_lib.rs.orig src/lib.rs' EXIT
fi
if rustup target list --installed 2>/dev/null | grep -q "^$TARGET$"; then
  cargo build --release --target "$TARGET"
else
  # no prebuilt wasm std available: build core from source (needs the rust-src component)
  RUSTC_BOOTSTRAP=1 cargo build --release --target "$TARGET" -Zbuild-std=core,panic_abort
fi
python3 - <<'PY'
import base64
w = base64.b64encode(open('target/wasm32-unknown-unknown/release/shatterforge.wasm', 'rb').read()).decode()
page = open('web/page.template.html').read().replace('__WASM__', w)
open('dist/shatterforge.html', 'w').write(page)
print('dist/shatterforge.html', len(page) // 1024, 'KB')
PY
