# Shatterforge

A structural destruction engine for the browser (Rust → WebAssembly, WebGPU), and a demo game that shows it.
Buildings are made of real-size bricks of real materials; what stands, bends, breaks and falls is computed, not
scripted.

| | |
| --- | --- |
| `engine/` | **The engine: the future SDK.** No levels, no player, no page of its own. |
| `engine/src/` | Rust crate `sf-engine` (`no_std` on wasm): the brick store (`brick`), the structure at rest and dynamic with plastic joints (`structure`), break plans and falling pieces' records, meshes (`mesh`), hulls (`hull`), and the simulation that ties them (`sim::Sim`). |
| `engine/gpu/` | The GPU world (`gpu_world.js`, WebGPU): rigid bodies and particles; its test scenes. |
| `engine/web/` | The page runtime: the engine in a worker, bodies and pieces between engine and GPU, breaking, impacts, the debris budget, drawing with three.js, the engineer's view. |
| `game/` | **The demo game.** |
| `game/src/` | Rust crate `sf-demo`: the levels (five kinds of tower, the complex building, the city), the player, and the wasm the page loads (`wasm.rs`). |
| `game/web/` | The page: template, menu, settings, tools, benchmark, loop. |
| `game/lab/` | The lab: scripted scenarios with physical checks, run headless on a virtual clock. |
| `docs/` | How it works and where it stands. `docs/STATUS.md` is the state, generated from the last runs. |
| `tools/` | `check.sh` (the checks, then STATUS.md), `compare.cjs` (dynamic against at rest), `status.py`. |
| `bench/` | Results from real machines (machine and date in the file name). |
| `legacy/` | The retired first engine and its game (its own README and build). |

**Play:** open `dist/tower.html` in a desktop browser with WebGPU. One self-contained file; three.js loads from cdnjs.

## Build and check

```bash
cargo test --release                                   # both crates (the engine's tests run on the demo's buildings)
RUSTC_BOOTSTRAP=1 cargo build -p sf-demo --release --target wasm32-unknown-unknown -Zbuild-std=core,alloc,panic_abort
python3 game/build.py                                  # dist/tower.html
tools/check.sh            # tests, build, GPU suite, STATUS.md
tools/check.sh lab        # the same, plus page checks and the lab's quick suite
node game/lab/lab.cjs =collapse                        # one lab scenario by name (needs SF_NODE_MODULES, Playwright)
```

The rules for changing anything are in `CLAUDE.md`.

## Where the line between engine and game is, and is not yet

- **Rust:** the line is a crate boundary. `sf-engine` knows nothing of the game. `sf-demo` builds bricks and hands
  them to `Sim::from_bricks`. One exception, on purpose: the engine's tests need buildings to stand on, and take
  the demo's level files in as test fixtures (`engine/src/lib.rs`).
- **The wasm** is the demo's: the exported functions (`game/src/wasm.rs`) are what this page calls, not yet an
  engine API. The GPU test scenes use that wasm too.
- **The page:** the line is by file. `engine/web/*.js` and `game/web/js/*.js` are joined into one script
  (`game/build.py` has the order); the runtime still reads the page's globals (the scene, the level, the
  settings). Making that a module with a stated interface is the next step toward an SDK.
