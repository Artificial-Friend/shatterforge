# WebGPU rigid-body engine

`gpu/gpu_world.js` is the engine's rigid-body physics. It runs entirely on the GPU, with no read-back during a step.

The CPU engine in `engine/src/phys.rs` is the reference for the shared core:
- box contacts
- speculative contacts
- the soft step: 12 sub-steps, with contact stiffness at one sixth of the sub-step rate
- anchored patch friction with twist

The GPU engine adds four things on top of that core:
- compound bodies
- materials
- restitution
- a multi-level grid

## Bodies and shapes

A body is rigid and made of one or more box shapes. A wall chunk that breaks off is one body of many bricks.

`buildBodies` (JavaScript) takes shapes in world space and computes the body's physical properties:
- mass and centre of mass
- the full inertia tensor, with box inertias rotated and shifted by the parallel axis theorem
- the tensor's principal axes (Jacobi eigen-decomposition)

The body is stored in its principal frame, so the GPU only needs a diagonal inertia.

Collision works on shapes; the solver works on bodies.

## Materials

| Material | Density kg/m³ | Friction | Restitution |
| --- | --- | --- | --- |
| concrete | 2400 | 0.6 | 0.1 |
| masonry | 1900 | 0.6 | 0.05 |
| steel | 7850 | 0.45 | 0.25 |
| glass | 2500 | 0.5 | 0.2 |
| timber | 500 | 0.5 | 0.3 |
| ground | — | 0.6 | 0 |

A pair uses the geometric mean of the two frictions and the larger restitution. Restitution is applied after the sub-steps, Box2D v3 style: points that approached faster than 1 m/s have their approach speed reflected by the restitution. The values are first estimates, to be tuned against reference footage.

## Step

| Stage | Kernels | Work |
| --- | --- | --- |
| Broad phase | `prep_bodies`, `prep_shapes`, `scan`, `scatter`, `pairs` | Shape poses and AABBs, then a multi-level hashed grid: level L has 4^L × the base cell, and each shape goes to the level that fits its AABB. A shape queries its own level (j > i) and each non-empty coarser level, so every pair is found once, from its smaller shape. |
| Narrow phase | `narrow`, `hull_args`, `narrow_hull` | `narrow` takes the box pairs: box-box SAT, clipping and 4-point reduction. Pairs with a convex chunk are listed and tested by `narrow_hull`, one indirect dispatch sized by that list (stage `chunks` in the timings). Then for both: warm-start match through a GPU hash table keyed by shape pair, constraint masses and approach speeds. |
| Colouring | `color`, `color_scan`, `color_scatter`, `adj_scan` | Greedy colouring with atomic per-body masks, plus per-body contact lists |
| Sub-steps | `integrate_warm`, `solve`, `integrate_pos`, `relax` | Warm start per body (no colouring needed). Gauss-Seidel solve per colour, with one indirect dispatch per colour. |
| After the sub-steps | `restitute`, `finish` | Restitution per colour, positions, sleep timers |
| Sleep | `uf_union`, `uf_min`, `uf_sleep` | Islands by GPU union-find |

Records:

| Record | Size |
| --- | --- |
| Contact (hot) | 272 B. The moment arms and inverse-inertia products are recomputed in the shader, not stored. |
| Contact (cold) | 96 B: warm-start matching data |
| Solver, per body | 96 B |
| Static, per body | 80 B |
| Shape | 128 B |

Each contact also records two values for the damage model (`readContacts`), which uses impact energy ≈ ½ J v:
- the total normal impulse J it applied during the step
- the approach speed v at the start of the step

## Bodies with many contacts

A body can only take one contact per colour, and there are at most 31 colours. A big falling piece touching the ground, the remaining stumps and hundreds of crumbs has far more contacts than that. Those extra contacts used to be solved one at a time by a single GPU thread. That made the tower page freeze when a whole tower detached (3,306 bricks, about 2,200 extra contacts).

Extra contacts are now solved in parallel by Jacobi iteration with mass splitting:
- A body with k extra contacts acts as if it had 1/k of its mass in each of them.
- The velocity changes are summed with fixed-point atomics and averaged per body.
- Two rounds run per solve and per relax pass.

The test (`plate`) drops a slab piece of 144 bricks, as one body, onto 144 blocks. 119 of its contacts are extra contacts, and it settles flat at 0.6249 m (expected 0.625), at rest.

## Broad phase

Every shape, whether moving, resting or static, searches only its own grid level and the coarser ones: a 3 × 3 × 3 neighbourhood at each.

- **Why this is enough.** A shape's level fits its size, so anything it overlaps on those levels sits in a neighbouring cell. A big shape never scans finer levels, because the small shapes there find it.
- **Who is checked.** A moving shape checks everything in its cells. A resting or static shape looks only into cells flagged as holding a moving shape, and only at the moving shapes there.
- **Uniqueness.** Each pair is found once: on one level by the lower index, across levels by the finer shape.

This replaces the previous rule, where only moving shapes searched and a large moving piece scanned finer levels over its whole extent. That cost about 10 ms per step in a collapse. On the first frame of the 2,000-box pile, the old and new searches find the same 5,871 candidate pairs (test `paircount`).

## Dispatch

All dispatches are direct, so the browser does not validate them one by one; Chromium validates indirect dispatches individually, and about 800 per step stalled whole frames.

- **Colour passes loop.** Each colour pass is sized from the recent per-colour count (×1.5 + 256) and its threads take every stride-th contact, so a colour that grew since the stats were read is still solved in full, only by fewer threads. (Sizing from stats without the loop left contacts unsolved in fast-changing scenes.)
- **Narrow phase loops too.** `narrow` is sized from the recent candidate-pair count (×2, at least 8,192) and loops the same way. Sized by the capacity (24 pairs per shape) it spent most of its time in threads with nothing to do.
- **Chunk pairs apart.** The chunk test needs both hulls' corners and directions and 16 clipped points per thread; in the same kernel as the boxes, every box pair paid for that too. GPU test `pile` (2,000 boxes, software GPU) checks the narrow phase against the broad phase: 260 against 49 ms with chunks and capacity sizing, 63 against 53 ms after both changes (57 against 49 before chunks).
- **Everything else by capacity.** The kernels skip work past their real counts.
- **Colour cap.** At most 16 sequential colours are solved per sub-step by default (`maxColors`); contacts beyond them go to the parallel pass.
- **Read-backs in flight.** Up to three stats read-backs and three debris read-backs are in flight at once. A late, older result is dropped, but impact events are never dropped.

## Slots

The slots of retired bodies and shapes are reused (`addBodies` returns the bodies' slots). `roomBodies()` and `roomShapes()` give the free capacity. `setVelocity` pushes a body and wakes it.

## Sleeping

A body counts as at rest when, over the last 0.5 s, both hold:
- it has stayed within 1 cm and 0.02 rad of where it was when its timer restarted
- it has stayed below 0.5 m/s

An island sleeps when all of its bodies are at rest. Testing the pose, not the instantaneous speed, matters. Settled piles carry solver chatter of 2–4 cm/s that goes nowhere, and a speed test kept a 2,000-brick pile awake indefinitely.

A body also sleeps on its own once it has been at rest for 1 s, even when its island is still awake. A pile of rubble with a few creeping pieces would otherwise stay awake as a whole: in the rubble test, 99% of the "awake" bodies were already still. The still ones become static support for the rest.

A sleeping body is static to the solver. It wakes when a body moving faster than 0.25 m/s at the contact touches it (`wakeSpeed`). A body just woken, or creeping, does not wake its neighbours, so a pile wakes only as far as the disturbance really goes.

**A world at rest costs nothing.** When the last stats show nothing awake (bodies and particles) and nothing has changed from JS since (no bodies added or removed, no velocities set, no blast), the step is skipped (`skipped` counts them).

## Particles (granular debris)

Crumbs are particles: spheres of the brick's volume, drawn as the bricks they are (`maxParticles`, `addParticles`, `readParticles`). Each step, after the bodies moved:
1. **Prepare.** Sleeping particles wake when a moving rigid shape is in a grid cell around them (any level), when a blast reaches them (`blastParticles`) or on `wakeAll`. All live particles go into their own hash grid.
2. **Four sub-steps of position-based dynamics.** Predict with gravity; push out of other particles (averaged, over-relaxed; a sleeping particle does not give way), then out of the ground and every rigid shape in sequence, each in full, so a particle never ends inside a body. Friction is Coulomb: the slide of the sub-step is taken back up to μ × the push. Velocity is the move over the sub-step.
3. **Push limits.** A sub-step's own approach is undone in full. Overlap beyond it (from a spawn, or a body driven into a heap) separates at 2 m/s, so nothing is launched from a deep overlap.
4. **One way.** Particles do not push bodies: a box dropped on a heap sinks to the floor and pushes the particles aside. A particle pressed from opposite sides by two shapes (that box on the floor) is crushed to dust (state 3; the page frees it).
5. **Rest.** A particle sleeps after 0.5 s touching something and staying within 1 cm of where its timer restarted. A particle wedged among sleeping ones may jitter in place without going anywhere, and a speed test kept such particles awake indefinitely.

**Candidates once per step.** Each particle gathers the particles (up to 24) and shapes (up to 12) it can reach this step, from its speed, and the sub-steps test only those. In the tower benchmark (12,600 crumbs in the air) this cut the particle stage from 18 s to 0.8 s per step on the software GPU, below the rigid sub-steps; the poured heap comes out the same.

Particles cut from a moving body wait (state 4) and are placed at the next step from the parent's pose then, like its pieces (`breakBody(parent, kids, pr, qr, dvs, parts, pdvs)`).

## Validation (`node gpu/test_gpu.cjs`, headless Chrome with SwiftShader WebGPU)

| Check | Expected | GPU result |
| --- | --- | --- |
| Box dropped tilted | rests flat at 0.125 m | 0.125 m, at rest |
| Stack of 20 × 0.5 m cubes, 10 s | CPU reference: 6.93 mm drift, 1.7 mm/s | 6.93 mm, 1.0 mm/s |
| Offset column of 10 | topples, no cube left standing on another | topples |
| Pile of 2,000 after 2 s, 4 sub-steps | CPU: y 0.117–4.828 m | y 0.118–4.823 m, 0 NaN |
| Drop from 2 m, restitution 0.5 / 0.3 / 0.05 | rebound e² h: 0.50 / 0.18 / 0.005 m | 0.490 / 0.176 / 0.005 m |
| Block on a slope, friction 0.6 | holds at 25°; at 40° slides g (sin θ − μ cos θ) t²: 1.797 m from 0.5 s to 1.5 s | 0.000 m; 1.798 m |
| 2 × 2.5 m wall section, one body of 45 bricks, tipped 12° | falls flat as one piece | all brick centres at 0.125 m |
| Steel brick at 80 m/s at a 0.125 m concrete wall | does not pass through | stops at the face and rebounds at 17.7 m/s (e = 0.22) |
| 8 m steel beam dropped across three blocks | rests on them at 0.65 m (coarse grid level) | 0.650 m |
| Pile of 300 with sleeping | settles and sleeps | all asleep by 8 s |
| Heap of 1,500 mixed rubble bodies | settles and sleeps | all asleep by 5 s (island sleep alone: 8 s) |
| 2,000 particles poured on a 2 × 2 m table | none under the ground or inside the table; heap on it; all asleep, then steps skipped; a box dropped on it wakes it | 0 / 0; 0.6 m heap on the table, the rest at the angle of repose around; asleep by 5 s, all steps skipped; woken; 34 crushed under the box |
| Particle cut from a tumbling body, placed from a pose 10 steps old | where the body is now | within 2 mm |
| Fewer sub-steps (20-box stack drift) | | 12: 7 mm · 8: 16 mm · 6: falls (so adaptive sub-steps stop at 8) |

SwiftShader runs WebGPU on the CPU, so its timings mean nothing. Throughput is measured by the bench page on real GPUs.

## Contact reduction

Two bodies touching through many shape pairs keep only representative contacts: per body pair with more than 32, the ones furthest out in 26 directions (the corners, edges and faces of a cube) and the deepest. The others are marked (+8 on the point count) so colouring and the solver skip them, but they keep their points and impulses for the warm start of the next step, where they may be picked again. This is what keeps the upper part of a building (thousands of bricks on a floor) out of the parallel overflow pass.

GPU test `compound` (800 bricks as one body landing at 5.9 m/s on a brick floor): rebound 1.1 → 0.44 m/s (concrete restitution allows 0.59); overflow ~1,300 contacts → 0.

## Crushing on contact

A light piece pressed by a body at least 20 times heavier is crushed on the GPU (kernel `crush`), when the impulse it took in a step is more than 1.5 times what its own approach momentum needs (it is being squeezed against something) and it approached at 1 m/s or more. From then on it collides only with the static world, so the heavy body goes through it. The page (`takeCrushed`) turns it into rubble particles, its bricks split into cubes of at most 0.4 m, placed from its current pose.

GPU test `crush`: a 7,700 t block landing on 24 pieces of 11 t crushes all 24 and comes down to the ground; a piece rammed from the side by a 600 t block is not crushed. In the lab's collapse (150 frames), launches went from 14 to 1 and bounces from 11 to 7.

## Convex chunks

A shape can be a convex hull instead of a box (`hull` on the shape, stored in its record's fourth word). The library (`GpuWorld.hulls`) holds 16 fracture patterns of the cube (Voronoi cells of 2–6 seeds, 63 cells); `GpuWorld.hullShapes(p, q, h, pattern)` gives the cells of a brick, which tile it exactly. Each cell is kept in its own box mapped to [-1, 1]^3, so a shape's `h` is its cell's half size and the cells need no per-shape storage. Mass and inertia come from the cell's real volume, centroid and covariance (a wedge's box centre can lie outside it).

Collisions: box against box keeps its own routine. Anything with a chunk uses separating axes over both shapes' faces and the cross products of their edge directions. The contact comes from a reference face clipped against the incident face, or from the closest points of two edges, whose axis takes the sign where the shapes are most clear of each other. A chunk on the ground gives up to 8 corners, as two contacts of 4; one patch of 4 let the others swing into the ground.

| GPU test | Expected | Result |
| --- | --- | --- |
| `hulls` (a): every cell dropped tumbling | rests on a face | rest speed 0.0001 m/s, lowest corner -0.1 mm |
| `hulls` (b): the 6 chunks of a brick placed together | stay put | 0.3 mm |
| `hulls` (c): heap of 300 chunks | settles, nothing under the ground at any time, no lost pairs | asleep by 5 s, lowest corner -0.6 mm |
| `hullpairs`: 200 random chunks dropped on fixed (or resting) chunks | no corner > 3 cm inside | 0 mm |
| `hullthrow`: 200 chunks thrown at 4–14 m/s, spinning up to 26 rad/s | no corner > 5 cm into the ground | 4 mm (11 cm with one 4-point ground contact) |

Particles still collide with a chunk's box.

## Capacity

Pairs beyond `maxPairs` are lost, and a lost ground pair lets a piece fall through the ground. The default is now 24 per shape (it was 8, and a heap of chunks overflowed it: chunks fell to -7.8 m). `stats.pairsLost` and `lost` (steps that lost contacts or pairs) report it, and the lab checks it.

## Still to do

- **Mass ratio limit.** Across one contact the heavier body responds as if at most 20 times heavier (GPU test `sandwich`: a 7,700 t block landing on 11 t rubble went from a 2.52 m/s rebound to 0.37 m/s).
- **Depenetration** is at most 1 m/s (it was 3; pieces made overlapping the slab they were crushing into popped out).

- **Contact sag.** Stiffer contacts are still needed. A 20-high stack sags 7 mm (0.35 mm per contact), within budget but not the micron level of real masonry.
- **CPU reference.** The CPU engine does not have compound bodies, materials or restitution. It stays the reference for the box core only.
