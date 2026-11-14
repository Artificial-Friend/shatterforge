# The engine (`sf-engine`, the GPU world, the page runtime)

- `src/`: the Rust crate. Start at `sim.rs` (`Sim::from_bricks`, `structure_step`, `blast`, `impacts`) and
  `structure.rs` (`analyse`, `analyse_dyn`). docs/STRUCTURE.md, docs/DYNAMIC.md, docs/DEFORM.md.
- `gpu/`: `gpu_world.js` and its tests (`node engine/gpu/test_gpu.cjs`, `test_hulls.cjs`). docs/GPU.md.
- `web/`: the page runtime, plain scripts joined by the host's build:
  `draw.js` (meshes, debris), `bodies.js` (static shapes, falling pieces, the debris budget, clean-up),
  `worker.js` (the engine in a worker, jobs), `view.js` (the engineer's view), `breaking.js` (breaking on the GPU,
  blasts, impacts).
- `bench/`: the phase-0 benchmark page of the rigid-body prototypes (`src/phys.rs`).
- `examples/`: native probes of the rigid-body prototypes.

Tests: `cargo test --release -p sf-engine`. They run on the demo's buildings (`../game/src/{tower,complex,city,player}.rs`,
taken in as fixtures for tests only).

What the runtime expects of its host today (not an interface yet): three.js and a `scene`; the wasm's exports as
`E` (page copy) and in the worker; `gpu` (a `GpuWorld`); the level (`KIND`, `STOREYS`, `lots()`); `setting` and
`LAB`. See the root README for where this is going.
