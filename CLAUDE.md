# Rules for working on Shatterforge

These hold for every change, by anyone (people or Claude).

## Tests follow the code
1. **Run the checks that cover what you changed, before you commit.**
   - `engine/` (Rust): `cd engine && cargo test --release`, then rebuild the wasm and the page.
   - `gpu/gpu_world.js`: `node gpu/test_gpu.cjs` (all scenes; `tools/check.sh` does both).
   - `game/` (the page, the lab): `python3 game/build_tower.py`, then at least `node game/lab.cjs collapse --frames 150`.
   - Anything that changes physics, breaking, read-backs or drawing: `tools/check.sh lab` (the lab's quick suite) before publishing a build.
2. **Every bug fixed gets a check that would have caught it**: a GPU test scene, an engine test, or a lab check or scenario. Say in the test what it reproduces and the numbers before and after the fix.
3. **Never loosen a limit to make a check pass.** If a limit was wrong, change it in its own commit and give the physical reason in the code next to it and in docs/LAB.md or docs/GPU.md.
4. **New features get scenarios.** A new material, building kind, tool or physics feature gets a lab scenario or GPU test scene, and its limits are written down when it is added.
5. **A check that fails and stays failing is a known problem, written down**: in docs/GPU.md ("Still to do") or docs/LAB.md, with what was tried. It may leave the default run only with that note (as `compound` has).

## The state is always current
6. **docs/STATUS.md is the state.** It is generated from the last results: run `tools/check.sh` (or `tools/check.sh status` after running checks by hand) and commit STATUS.md with the change. A commit that changes code without an updated STATUS.md says why in its message (for example, docs only).
7. **Numbers in docs come from runs, with the run named** (test, scenario, frames, software or real GPU). Software-GPU timings compare only with other software-GPU timings.
8. **Results from real machines** (the lab's Copy JSON, benchmark reports) go under `bench/` with the machine and date in the file name, and STATUS.md or the docs say which numbers came from them.

## Practical
- The lab runs on a virtual clock: a slow GPU changes only the wall time. On the software GPU a 4-storey collapse step takes ~4 s, so use `--frames` for quick runs and the full length before publishing.
- `SF_NODE_MODULES` must point to a `node_modules` folder with `three` for `game/lab.cjs`; Playwright must be installed.
