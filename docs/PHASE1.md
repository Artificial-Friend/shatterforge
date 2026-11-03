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

## Collapse under load and breaking on impact

### Standing structure (`structure::analyse`, `Game::structure_step`)

After a blast, the page runs one structural round per frame until nothing more fails.

- **Cells.** Every standing brick is cut into horizontal cells of at most 2 m.
- **Load path.** The weight is solved as a flow to the ground by conjugate gradients, with stiffness-based conductances: EA/L vertically and w·d³/L³ for horizontal (bending) links.
- **Moments.** Moments are carried down the flow and end at bearings.
- **Checks.** Joints are checked in compression, tension and shear. Bending is checked across side joints and inside bricks.
- **Failures.** A failed joint is removed. A failure inside a brick splits the brick at the critical section and leaves the new joint out (a crack).
- **Pieces.** Whatever is no longer connected to the ground becomes new falling pieces.
- **Exceptions.** Infill (glass, partitions, cladding) carries no load. Stair flights are monolithic: they are not checked in bending and have a low conductance.

| Case | Result |
| --- | --- |
| Intact tower | stands; worst joint at 82 % of strength |
| Outer columns cut | 177 joints fail over 6 rounds |
| 7.5 m cantilever | cracks |
| 2 m cantilever | holds |
| One side of the ground storey cut (page test) | holds at 89 %; 2 joints fail |

One round over the whole tower takes about 60 ms native and 90–190 ms in wasm.

### Impacts (`Game::impacts`)

Every frame, the page sends the engine the GPU's hard contacts, from both sides of each contact: up to 32 per falling piece and 96 for the standing structure. Each contact is passed in the tower frame with its point, the direction it pushes the body, its impulse J, its approach speed v and the material on the other side.

- **Crushing on both sides.** The impact energy ½·J·v crushes the same volume on each side: V = E / (c_self + c_other), where c is the crush energy per m³ of each material. Concrete on concrete crushes half each; glass on concrete crushes mostly the glass. The radius is capped at 1.5 m.
- **The standing structure takes the hit.**
  - The downward part of the contact force, J/dt, is added as a load on the bricks around the contact.
  - A structural check runs on the whole structure, and up to 256 joints fail.
  - Whatever comes loose detaches as new pieces.
  - GPU bodies wake when static bricks are removed under them.
- **A falling piece cracks.** Its bricks decelerate with it, and their inertia flows to all of its contacts. Only failures within one storey of the contacts count: a rigid stop in one frame would otherwise load every storey with the full deceleration.
- **Crush-down.**
  - **Required force.** The force needed to stop the piece is (ΣJ + |ΣJ·d|) / 2 / dt. For a piece hit from one side this is the net impulse; for a piece squeezed between two sides it is the larger side.
  - **Weakest section.** The engine finds the weakest section within a storey of the contacts: the compressive strength times the area of the structural bricks and joints it cuts.
  - **Cut.** If the required force exceeds that capacity, the piece is cut at that section.
  - **Rubble.** The contact side of the cut is the crushed material. It falls apart into rubble, as single bricks within the debris budget; the rest becomes dust. It does not stay a solid block for the floors above to land on.
  - **Slowing.** The part beyond the cut keeps its velocity from before the hit. Its speed into the contacts drops to √(v² − 2E/m), where E is the crush energy of the section's area over 0.5 m of crushing.
  - **Result.** The floors above carry on, meet what is below again and crush the next section, storey by storey. A failed section stops resisting once it is crushed: it is not charged its peak force on every frame.
- **Contact data (page).**
  - **Velocity before the hit.** This is the fastest of the last four read-backs, because events arrive a read-back or two late. A body cut free in the last few steps also counts its starting velocity.
  - **Scaling.** The page sends only a piece's 32 strongest contacts, so their impulses are scaled to its actual change of momentum, M·|v_before − v_now|.
  - **Big pieces and light rubble.** A piece of more than 100 bricks whose velocity changes by under 0.5 m/s is not broken: rubble pelting it does no damage.
- **Infill holds nothing together.** Glass, cladding and partitions carry no load. Where the structure behind them has come apart, their joints are cut, and each infill brick stays with the structural part it rests on.
- **Budgets per call.**
  - Fragments under 48 bricks count as rubble, and at most 64 stay as bodies.
  - At most 256 crumbs and rubble bricks stay as bodies, half of them for the standing structure.
  - The rest of the crushed material is removed as dust and counted.

The engine test drops the tower 1.4 m at 5.2 m/s onto 285 stumps:
- **The stumps.** 460 crumbs, and 256 joints fail.
- **The tower.** It breaks just above its bottom slab: that section carries 354 MN, and stopping the tower in one frame takes 1,395 MN.
- **The crush-down.** The upper tower is cut free and goes on crushing in later rounds.

### Connections, building kinds, reinforced concrete

- **Connections.** A joint is a connection between two elements, and its strength is the weaker brick's strength times a factor per mode (compression, tension, shear):
  - a seam inside one element: 1 / 1 / 1
  - concrete slab to column or core: 1 / 0.5 / 0.45 (the shear is the slab's punching strength, about 0.33 √f_c)
  - timber elements joined to each other (nailed or bolted): 1 / 0.1 / 0.25
  - any other connection between different elements: 1 / 0.7 / 0.7
- **Building kinds.** The same layout can be built three ways (settings):

  | Building | Storeys | Built of | Worst joint, intact |
  | --- | --- | --- | --- |
  | Concrete frame | 12 | reinforced concrete | 82 % of strength |
  | Masonry | 5 | masonry piers and facade, with a concrete core and floors | 94 % of strength |
  | Mass timber | 6 | timber floors, posts, core and stairs | 53 % of strength |

  Each fails through its own connections.
- **Reinforced concrete.** Cast concrete elements (slabs, columns, core, stairs) are flagged as reinforced and fail in two stages. The first failure cracks the joint, or splits a brick but keeps the joint: the steel holds, and half the strength is left. A second failure separates it. Precast cladding is plain and fails at once. With this, cutting the perimeter takes 8 rounds and 349 joint failures (it was 6 rounds and 177).

### Blasts on debris

- **The ray.** The click ray tests the debris as drawn (oriented boxes) as well as the building.
- **A falling piece** that is hit crumbles inside the blast radius (`blast_piece`), and the rest splits into its connected parts.
- **The blast wave.** Every blast throws the debris within 2.5 radii outward.
- **Concentrated hits.** A big piece is still broken by a hit that puts 100 kJ or more into one contact, such as the steel ball, even when the hit barely changes the piece's motion.

### Steel ball and settings

- **Steel ball.** Key 4 switches the click from blasting to a steel ball fired from the camera: 0.5 m and 60 m/s by default. Its hits go through the same impact path as everything else.
- **Energy split.** Impact energy divides by the other side's crush energy, V_self = E·c_other / (c_self·(c_self + c_other)). The weaker material takes most of it: steel on concrete crushes the concrete, and concrete on concrete crushes half each.
- **Settings.** Key O opens the settings panel. It sets the crush depth (how much a crushed section absorbs), the crumbs and rubble kept per impact, and the ball's speed and size. The engine parameters go to the worker and to the page's copy (`set_param`), and the settings are remembered in the browser.
- **Timing.** The status panel shows main-thread time per stage and GPU time per stage.

### Engine in a worker

The authoritative engine runs in a Web Worker: blasts, the structural rounds and impacts. One job runs at a time, in this order: blasts, then impacts, then structural rounds.

- **The page's copy.** The page keeps a copy of the brick store for walking, aiming and meshes. After each job that changed the store, the copy is replaced with the worker's (`Bricks::save`/`load`, about 88 B per brick).
- **Frame rate.** Analysis therefore no longer stops the frame. A blast reaches the screen when its job is done.
- **Fallback.** Without workers, the same jobs run inline.

### Debris display

- **Colours.** Debris boxes carry material + 16 × paint and use the tower mesh's colours.
- **No GPU backlog.** GPU steps are only queued while fewer than two are unfinished, so read-backs never fall behind.
- **Smooth motion.** Debris is drawn blended between the last two read-backs, matched shape by shape.

### Limits

- **Vertical loads only.** The standing structure takes only the vertical part of an impact. A sideways hit crushes material but does not load the frame sideways.
- **Mass is not conserved.** Crushed material beyond the rubble and crumb budgets is removed as dust.
- **Frame rate depends on GPU throughput.** The worker keeps analysis off the drawing thread, but when the GPU cannot keep up, physics runs in slow motion rather than dropping frames.
- **Snapshot cost.** The page's copy of the store is replaced as a whole after each changing job. That is about 0.4 MB for the intact tower and more after a collapse; sending only the changes would be cheaper.
