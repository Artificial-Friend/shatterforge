# Shatterforge engine spec (phase 0)

This spec fixes the units, real-world dimensions, brick format and splitting rules for the new engine, plus the performance gates the prototypes must meet. The engine is a new codebase in `engine/`. The old 0.5 m cube engine (`src/`) is retired: it stays only as a reference until the new engine runs the tower.

## Units and axes

- Metres, kilograms, seconds. Y is up.
- **Tick:** the engine grid is 0.125 m. Every brick position and size is a whole number of ticks, so brick faces meet exactly and joints have exact areas.
- **Coordinate range:** 16-bit signed tick coordinates cover ±4,096 m. That is enough for a 2 km city with margin.

## Scale standard

Every generated building and the player must meet these dimensions.

| Element | Dimension | Ticks |
| --- | --- | --- |
| Office floor to floor | 3.75 m | 30 |
| Residential floor to floor | 3.0 m | 24 |
| Floor slab | 0.25 m | 2 |
| Exterior wall | 0.25–0.375 m | 2–3 |
| Interior partition | 0.125 m | 1 |
| Door | 0.875 × 2.125 m (1.75 m double in lobbies) | 7 × 17 |
| Corridor | 1.75 m | 14 |
| Stair | 1.25 m wide, 0.25 m rise, 0.25 m going (with a ramp collider) | 10, 2, 2 |
| Column grid | 7.5 m, columns 0.5–0.75 m square | 60, 4–6 |
| Player | 1.8 m tall, 0.3 m radius, 0.35 m step-up | — |
| Street | 14 m with sidewalks | 112 |
| City block | 100 m | 800 |

Notes:

- **Stairs:** the tick grid quantises steps to 0.125 m, so stairs use 0.25 m rises on a 0.25 m going: 15 steps per 3.75 m storey, a 45° stair. A ramp collider under the steps gives smooth walking. Real 0.17 m rises would need a 1/16 m grid. That is possible later without changing the format, by halving the tick.
- **Doors** are 0.875 m because 0.9 m is not a tick multiple.

## Brick format

A brick is an axis-aligned box in its structure's frame, made of one material. Everything is stored as arrays, one entry per brick, so solvers stream them.

| Field | Type | Bytes | Meaning |
| --- | --- | --- | --- |
| x, y, z | i16 × 3 | 6 | min corner, ticks |
| sx, sy, sz | u8 × 3 | 3 | size, ticks (up to 31.875 m) |
| material | u8 | 1 | concrete, masonry, steel, glass, timber |
| paint | u8 | 1 | palette index |
| flags | u8 | 1 | alive, infill, dynamic |
| structure | u16 | 2 | owning structure |
| body | u32 | 4 | rigid body when it has broken off, else none |
| joint list head | u32 | 4 | first joint touching this brick |
| damage | u8 | 1 | accumulated damage, 0–255 |
| **Brick total** | | **23 (24 aligned)** | |

A joint is a shared face between two bricks, stored once:

| Field | Type | Bytes |
| --- | --- | --- |
| brick a, brick b | u32 × 2 | 8 |
| next joint of a, next joint of b | u32 × 2 | 8 |
| contact area | u16 | 2 (tick², up to 31.875 × 31.875 m) |
| axis | u8 | 1 (0, 1, 2: the face normal axis, from a to b) |
| damage | u8 | 1 |
| **Joint total** | | **20** |

A regular brick wall averages 3 joints per brick, so a brick plus its share of joints is about 84 bytes. That is under the 96-byte target and a seventh of the old engine's 600.

The rest is derived on the fly or held only while needed:

- Mass, centre and inertia come from size and material.
- Solver state (load, rotation) lives in the solver's own arrays while a structure is active.
- A broken-off brick's offset in its body is kept in a side table.

## Splitting rules

- A brick splits into two along one axis, at a tick boundary strictly inside it. The lower part keeps the brick's index; the upper part is a new brick.
- **Joints are redistributed exactly.**
  - Each joint of the parent goes to whichever child still touches that neighbour.
  - If both children touch the neighbour, the joint is kept and a second one is added, with each area recomputed from the face overlap.
  - The two children are joined by a new joint across the cut, at full strength.
  - Damage carries over.
- **Refinement around a point.** Every brick inside a sphere is split repeatedly, always along its longest axis at the midpoint (snapped to ticks), until no brick inside the sphere is larger than the target size. The minimum size is 1 tick (0.125 m).
- **Material grids.** Masonry and concrete split along their course grid (0.25 m) before going finer, so cracks follow brick courses and pours.

## Joint strength (to be tuned in phase 3)

Capacity of a joint = area × material strength for that axis:

- **Vertical joints** (bearing): compressive and tensile strength.
- **Horizontal joints:** shear strength.

The load solver (phase 3) adds bending through rotations at each brick.

## Phase 0 gates

| Gate | Target | Measured where |
| --- | --- | --- |
| Split | A 7.5 × 3.75 × 0.25 m wall panel refined to 0.125 m within 0.5 m of an impact point in under 1 ms | Native and WebAssembly, single thread |
| Brick memory | 96 bytes or less per brick including joints | Counted in the store |
| Stacking | 20 stacked 0.5 m cubes rest still: under 1 cm drift and under 1 cm/s speed after 10 s | Headless test |
| Throughput | 20,000 awake box bricks stepped in under 5 ms at 60 Hz | Target gaming PC. The CPU baseline is measured here; the WebGPU path is built in phase 2 |
