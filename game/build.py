#!/usr/bin/env python3
"""Build dist/tower.html: the demo game as one file, with the wasm (sf-demo), the GPU world and the page's scripts inlined.

The page's script is these files joined in this order (classic script, one scope). engine/web is the engine's page
runtime; game/web/js is the demo; game/lab is the lab. See engine/README.md for what the runtime expects of its host.
"""
import base64, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = [
    "game/web/js/base.js",       # $, TICK
    "game/lab/scenarios.js",     # the lab's scenarios (they choose the level when the lab runs)
    "game/web/js/level.js",      # which level: kind, size, lots
    "game/web/js/scene.js",      # renderer, camera, sun, ground
    "engine/web/draw.js",        # the structure's meshes and the debris, drawn
    "engine/web/bodies.js",      # the static shapes, falling pieces, the debris budget, clean-up
    "engine/web/worker.js",      # the engine in a worker: jobs, structure steps, rest loads
    "engine/web/view.js",        # the engineer's view
    "engine/web/breaking.js",    # breaking on the GPU, blasts, impacts
    "game/web/js/input.js",      # mouse and keys
    "game/web/js/settings.js",   # the menu, graphics and physics settings
    "game/web/js/ball.js",       # the steel ball
    "game/web/js/bench.js",      # the benchmark
    "game/web/js/loop.js",       # the frame loop and the HUD
    "game/lab/lab.js",           # the lab
    "game/web/js/start.js",      # start
]
wasm = root / "target/wasm32-unknown-unknown/release/sf_demo.wasm"
page = (root / "game/web/page.template.html").read_text()
js = "".join((root / f).read_text() for f in SCRIPTS)
page = page.replace("__WASM_B64__", base64.b64encode(wasm.read_bytes()).decode()).replace("__GPU_WORLD_JS__", (root / "engine/gpu/gpu_world.js").read_text()).replace("__JS__\n", js)
out = root / "dist/tower.html"
out.write_text(page)
print(f"{out} ({len(page) // 1024} KB)")
