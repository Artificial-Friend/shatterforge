#!/usr/bin/env python3
"""Embed the engine wasm into the bench page: writes dist/bench.html."""
import base64, pathlib
root = pathlib.Path(__file__).resolve().parent.parent.parent
wasm = root / "target/wasm32-unknown-unknown/release/sf_demo.wasm"
page = (root / "engine/bench/page.template.html").read_text().replace("__WASM_B64__", base64.b64encode(wasm.read_bytes()).decode()).replace("__GPU_WORLD_JS__", (root / "engine/gpu/gpu_world.js").read_text())
out = root / "dist/bench.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(f"{out} ({len(page) // 1024} KB)")
