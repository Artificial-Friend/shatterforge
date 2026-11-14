# The demo game (`sf-demo`)

- `src/`: the levels (`tower.rs`: concrete frame, masonry, mass timber, mixed; `complex.rs`; `city.rs`), the player
  (`player.rs`), and the wasm's exports (`wasm.rs`). `tower::new_kind(storeys, kind)` makes a level's `Sim`.
- `web/`: `page.template.html` and `js/` (level, scene, input, settings, ball, benchmark, loop, start).
- `lab/`: `scenarios.js`, `lab.js` (in the page), `lab.cjs` (headless runner), `page_check.cjs`, `shot.cjs`. docs/LAB.md.
- `build.py`: writes `dist/tower.html` from the wasm, the engine's GPU world and runtime, and this page.
- `examples/`: native probes on the buildings.
