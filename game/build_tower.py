#!/usr/bin/env python3
"""Build dist/tower.html: the phase-1 playground with the engine wasm and the GPU physics inlined."""
import base64, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
wasm = root / "engine/target/wasm32-unknown-unknown/release/sf_engine.wasm"
page = (root / "game/tower.template.html").read_text()
page = page.replace("__WASM_B64__", base64.b64encode(wasm.read_bytes()).decode()).replace("__GPU_WORLD_JS__", (root / "gpu/gpu_world.js").read_text())
out = root / "dist/tower.html"
out.write_text(page)
print(f"{out} ({len(page) // 1024} KB)")
