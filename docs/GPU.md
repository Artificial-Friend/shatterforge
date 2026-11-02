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

## Sleeping

A body counts as at rest when, over the last 0.5 s, both hold:
- it has stayed within 1 cm and 0.02 rad of where it was when its timer restarted
- it has stayed below 0.5 m/s

An island sleeps when all of its bodies are at rest. Testing the pose, not the instantaneous speed, matters. Settled piles carry solver chatter of 2–4 cm/s that goes nowhere, and a speed test kept a 2,000-brick pile awake indefinitely.

A sleeping body is static to the solver and wakes when a moving body touches it.

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
| Pile of 300 with sleeping | settles and sleeps | all asleep by 9 s |

SwiftShader runs WebGPU on the CPU, so its timings mean nothing. Throughput is measured by the bench page on real GPUs.

## Still to do

- **Contact sag.** Stiffer contacts are still needed. A 20-high stack sags 7 mm (0.35 mm per contact), within budget but not the micron level of real masonry.
- **CPU reference.** The CPU engine does not have compound bodies, materials or restitution. It stays the reference for the box core only.
