# Shatterforge

A destructible-environment game and structural-physics engine written in Rust (`no_std`) and compiled to WebAssembly. Inspired by Volition's Geo-Mod. Everything in the world is built from 0.5 m bricks of real materials, and buildings stand or fall by their actual load paths.

> **Engine rewrite in progress.** A new engine with real-size adaptive bricks is being built in `engine/` (see `docs/SPEC.md` and `docs/PHASE0.md`). The engine described below (`src/`) is retired and kept only as a reference until the new one runs the tower.

**Play:** open `dist/shatterforge.html` in a desktop browser. It is a single self-contained file: the engine is embedded as base64, and three.js loads from cdnjs.

## The game

The game takes place on Kessel Basin, a 320 m open site with 37 structures and about 378,000 bricks. These include:

- **Downtown:** a 16-storey admin tower, 8–10-storey office towers and a hotel, all with stairwells to the roof, offices and corridors.
- **Housing:** apartment blocks with balconies, and three-storey row houses with working staircases.
- **Landmarks:** a church with a bell tower, a castle with a keep, and a parking garage.
- **Industry and transport:** a refinery with a 56 m smokestack, silos, a rail viaduct, a container yard under a gantry crane, a comms mast and a water tower.

Eight structures are targets.

**Tools and characters**

- **Weapons:** a hammer, rockets and remote charges.
- **Vehicles:** you can take over any vehicle, including AI patrol trucks that drive a road network and ram the player when provoked.
- **Site crew:** they flee from explosions and from structures that creak.

**Controls:** WASD to move, mouse to look, 1/2/3 to pick a tool, F to detonate, E to enter or exit a vehicle, M for the map, L for the load view, and Esc to pause. While paused, the HUD buttons stay clickable. Touch controls are included.

## Build

```bash
./build.sh          # release build, writes dist/shatterforge.html
./build.sh debug    # same, plus the diagnostic exports in tools/debug_exports.rs (needed by the tests)
```

You need Rust and Python 3. With the wasm target installed (`rustup target add wasm32-unknown-unknown`), the script does a normal `cargo build`. Without it, the script falls back to `-Zbuild-std` (which needs the `rust-src` component). The engine imports a single host function, `env.now`.

## Tests

The tests are headless simulations run with Node 18+, against the debug build:

```bash
./build.sh debug
node tests/stab.mjs                          # every structure awake for 20 s: nothing may collapse on its own
node tests/phys3c.mjs                        # admin tower: 1, 2, 6 columns, 4 corners, 8 charges
node tests/test3.mjs                         # full playthrough: walking, driving, demolishing all targets, perf per phase
node tests/frac.mjs tests/office_collapse.json   # fragment statistics per material after a collapse
node tests/frac.mjs tests/masonry_collapse.json
node tests/tower.mjs '[[-76.5,0.6,-14.5,2.2],[-72.5,0.6,-14.5,2.2],[-76.5,0.6,-10.5,2.2],[-72.5,0.6,-10.5,2.2]]' 72 17
                                             # bell tower base blown: position, lean, speed of the biggest piece over time
node tests/walk.mjs '[0,10,[[-0.5,3],[-0.5,0.6]]]'   # scripted walk (stairs reachability)
```

## Engine overview (`src/lib.rs`)

**World**

- A sparse chunked brick grid, 640 × 136 × 640 cells of 0.5 m.
- Up to 500,000 bricks.
- Materials: concrete, masonry, steel, glass and timber, with paint palettes.
- The meshing jitters lattice corners and uses procedural material shaders, so walls do not look like cubes.
- Static geometry is meshed per structure and per 12 m height band.

**Load paths of standing structures**

- Each structure solves a scalar potential flow from every brick's weight down to the ground (SOR, adaptive sweeps, budgeted per frame).
- Joints are checked in compression, tension (masonry is weak in tension) and shear.
- Glazing and stud partitions are non-bearing infill.
- A joint is judged only once the load field has settled. Overloaded joints creak, then fail.
- Losing connection to the ground detaches a section as a rigid body.

**Falling pieces**

- Each piece is a compound rigid body: one sphere per brick, a sequential-impulse contact solver and friction.
- Every brick is loaded by the effective gravity it feels (g − a). That is nothing in free fall, its weight at rest, and 50–150× its weight in the instant of an impact.
- Pieces therefore break where the dynamic load exceeds their strength: on landing, or in mid-air while rotating.
- Brittle materials snap at once. Concrete and timber yield briefly first.
- Strength depends on each joint's present orientation, so a toppled tower is weak where it actually bends.

**Collisions**

- Damage happens only where things collide.
- Each contact delivers energy ½·J·v_approach. That is zero for resting or sliding contacts.
- The energy is split between the two sides by how easily each material crushes. Each side crushes the bricks nearest the contact, growing outward, and cracks the next layer.
- Crushed material partly turns to dust and fragments, and partly breaks loose as rubble.

**Compression vs tension**

- A bearing joint failing in compression crushes material, so the load above sinks.
- Tension and shear failures open cracks that can separate.

**Safety limits**

- A small piece pinched between heavy ones is crushed instead of being ejected.
- Spin picked up from collisions is limited.
- Penetration is recovered gently.

**Debris housekeeping**

- Single loose bricks turn to dust 20 s after breaking off, unless something rests on them.
- In very large collapses, lone crumbs become particle fragments.
- The body pool frees resting crumbs before it would ever drop real debris.

## Known limits and next steps

- **Bending is not modelled yet.** The load model handles compression, tension and shear but not moments. Floors cannot sag, and when a column is lost the floors around it fail at their joints instead of folding as a whole bay. Adding bending to the same structure is the next major step.
- **Big frame buildings do not pancake quickly.** A frame building whose whole ground floor is removed settles onto its own rigid rubble instead of pancaking in a few seconds. Rubble that can compact is needed.
- **Strengths are gameplay-tuned,** at about a tenth of real concrete.
- **Large collapses are expensive.** The worst frame is about 100–170 ms when every target is demolished at once. The load solver and crushing are candidates for multithreading or the GPU.
