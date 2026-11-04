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
| Narrow phase | `narrow` | One thread per shape pair: box-box SAT, clipping and 4-point reduction. Warm-start match through a GPU hash table keyed by shape pair, then constraint masses and approach speeds. |
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

## Still to do

- **Contact sag.** Stiffer contacts are still needed. A 20-high stack sags 7 mm (0.35 mm per contact), within budget but not the micron level of real masonry.
- **CPU reference.** The CPU engine does not have compound bodies, materials or restitution. It stays the reference for the box core only.
