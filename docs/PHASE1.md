# Phase 1: adaptive brick store and the tower

**Gate: an office tower at real scale, under 100 bytes a brick. Passed.**

The first of the 12 tower storeys is the lobby; the roof carries a stair penthouse. There are 2,244 bricks and 6,809 joints. They take 84.7 B per brick as records and joints, and 91.3 B per brick including the spatial index. The tower builds in about 55 ms natively.

## The tower (`engine/src/tower.rs`)

The tower follows the SPEC scale standard.

| Element | Size |
| --- | --- |
| Footprint | 3 × 3 bays of 7.5 m (22.5 m square) |
| Storey | 3.75 m: 0.25 m slab + 3.5 m clear |
| Columns | 0.625 m square on the column grid |
| Facade | 0.25 m concrete spandrel and header, glass band with mullions every 1.875 m |
| Core | central bay, concrete walls |
| Stair | switch-back, 1.25 m flights, 0.25 m rise and going, half-landing, opening in each slab above |
| Corridor | 1.75 m ring around the core |
| Offices | 5.5 m deep, 0.125 m stud partitions, a 0.875 × 2.125 m door to the corridor for each office |
| Lobby | open plan, 1.75 m entrance |
| Roof | 1 m parapet, penthouse with a roof door |

Each storey is painted into a tick grid of element ids, then merged greedily into the largest single-element boxes. Most bricks are large as a result: a slab with its stair opening is a few bricks, and a wall between two openings is one brick.

## Store changes (`engine/src/brick.rs`)

- **Spatial index.** Each brick is in exactly one cell, the one holding its min corner, at the finest of four levels (1, 4, 16 or 64 m) that fits it. The cells are intrusive linked lists in a hash map, costing 4 B per brick plus about 15 B per occupied cell. The first two-level grid cost 25–30 B per brick for large bricks. Splits also got faster: median 0.07 ms.
- **Removal.** A brick can be removed with its joints. Its record stays, so indices are stable.
- **Change log.** Bricks that were added, resized or removed are logged, so meshes and physics shapes update incrementally.
- **Connectivity.** Pieces no longer joined to the ground are found by a flood fill over joints from the bricks resting on y = 0.
- **Ray cast.** Rays can be cast against the store.

## Mesh (`engine/src/mesh.rs`)

The mesh is each brick face minus the parts covered by touching neighbours. This is exact rectangle subtraction driven by the joint graph. Faces resting on the ground are dropped.

The tower is 10,732 quads, meshed per storey. A blast remeshes only the storeys it touched: a 0.6 m blast in a slab took 0.4 ms natively, including the refine.

## Walker (`engine/src/player.rs`)

The walker is a 0.6 × 1.8 m body with gravity and a 0.35 m step-up.

The test walks in through the core door, up flight A, across the landing and up flight B, and arrives at the next floor at 4.00 m (storey 3.75 m + slab 0.25 m).

## Blasts (`engine/src/game.rs`)

A blast works in four steps:

1. The bricks around the hit are refined to 0.25 m.
2. What lies inside the blast radius crumbles into loose pieces.
3. Anything no longer connected to the ground detaches as rigid compound pieces.
4. The result goes out as a log:
   - static shapes to set or remove in the GPU world
   - new bodies, each with its boxes

## Playground (`dist/tower.html`, built by `game/build_tower.py`)

- **World:** the tower at 1:1 scale, with shadows, a walk mode and a fly mode.
- **Blasting:** click to blast at the crosshair.
- **Debris:** handled by the GPU engine, which needs WebGPU:
  - The standing tower is one fixed body made of 2,000+ static shapes.
  - Crumbs leave along the shot at 2–4 m/s.
  - Detached pieces fall as single compound bodies.
- **When physics runs:** only while something is awake, so an idle tower costs no GPU time.

GPU engine changes:
- Static and sleeping shapes no longer search for pairs: only shapes of awake bodies do, across all grid levels.
- Bodies and static shapes can be added, and static shapes changed or removed, while the world runs.
