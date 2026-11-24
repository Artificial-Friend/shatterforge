# WebGPU rigid-body engine

`engine/gpu/gpu_world.js` is the engine's rigid-body physics. It runs entirely on the GPU, with no read-back during a step.

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
| steel | 7850 (a member: 1,256, see "Mass") | 0.45 | 0.25 |
| glass | 2500 | 0.5 | 0.2 |
| timber | 500 | 0.5 | 0.3 |
| ground | — | 0.6 | 0 |

A pair uses the geometric mean of the two frictions and the larger restitution. Restitution is applied after the sub-steps, Box2D v3 style: points that approached faster than 1 m/s have their approach speed reflected by the restitution. The values are first estimates, to be tuned against reference footage.

### What comes in too fast does not bounce (18 November)

What crushes does not spring back. A contact's restitution is multiplied by 1 − (v/vc)², v its approach speed as the step begins, and is nothing above vc. vc is the lower of two limits:

- **The faces.** Two faces that meet at v are stressed to v × Za·Zb / (Za + Zb): the elastic wave each sends into the other, Z = √(E × density) the wave impedance. Over the weaker one's crushing strength s the face crushes: v1 = s × (1/Za + 1/Zb).
- **The energy.** The two shapes that touch can each hold s²V / 2E before they crush (V the shape's volume), and the blow brings ½ m v², m the pair's reduced mass: v2² = s² (Va/Ea + Vb/Eb) / m. For one box on the rigid ground v2 equals v1. For a body of many shapes it is far lower: the 307 t upper part of the complex building on the stump of one steel column, 0.7 m/s.

| Material | E (GPa) | Z (MPa per m/s) | Strength s (MPa) | v1 against itself (m/s) | v1 against concrete (m/s) |
| --- | --- | --- | --- | --- | --- |
| concrete | 30 | 8.49 | 30 | 7.1 | 7.1 |
| masonry | 5 | 3.08 | 10 | 6.5 | 4.4 |
| steel | 200 | 39.6 | 250 (yield) | 12.6 | 4.3 |
| glass | 70 | 13.2 | 50 | 7.6 | 5.8 |
| timber (across the grain) | 0.5 | 0.5 | 8 | 32 | 16.9 |

The ground and the test materials are rigid and never crush (against the ground, v1 = s / Z of the body: concrete 3.5 m/s). The strengths are the ones the crushing passes use (`CRUSH_MPA`); the moduli are textbook values, not measured ones.

What the rule is and is not. Both limits are the most there can be: two whole faces meeting flat, and the whole of both shapes strained to the limit. A corner crushes sooner and the rule does not know it. Each contact patch is judged as if it took the whole blow, so a part landing on twelve stumps at once is allowed a twelfth of the energy its stumps can hold together (it errs towards no bounce for bodies of many shapes on many supports, which is what such bodies do). The factor 1 − (v/vc)² is a choice: it is the share of the impact's energy that could still be elastic, applied to the restitution, which is a ratio of speeds (the square root of that share would be the consistent form, and bounces more just under vc).

Why: lab `complex`, 18 November. With the give-back below, the 307 t steel-framed upper part came down on the stumps of its steel columns at the speed it had really reached, 6.14 m/s, and went back up at 1.52 m/s, steel's 0.25 of it, throwing every chair in it at the ceilings (10 rebounds over half the impact speed in 70 frames). With the faces' limit alone (steel on steel, 12.6 m/s) it still went up at 1.18 m/s; with the energy limit, 0.11 m/s in the step it lands and 0.00 in the next.

- GPU test `hardland` (the faces): a 63 t steel block flat onto fixed concrete at 6 m/s goes up at **1.54 m/s** without the rule and **0.07** with it; at 2 m/s, 0.54 and 0.48.
- GPU test `stumpland` (the energy): a 300 t steel part of 64 boxes onto four steel stumps at 6.8 m/s goes up at **1.12 m/s** with the faces' limit alone and **0.00** with the energy limit.
- GPU test `bounce` now expects what the rule predicts for a box dropped 2 m onto the ground (6.26 m/s): timber 0.129 m (0.18 m by its restitution alone; measured 0.127), masonry none (vc 3.2 m/s), the test material 0.5 m as before.

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

**Resting shapes far from anything awake skip the search.** Each awake shape marks the 4 m cells under its box in a small hashed map (8,192 slots; a box over more than 64 cells marks everything). A resting or static shape whose box touches no marked cell cannot pair with anything awake and returns at once. GPU test `quietcity` (a standing building of 20,000 boxes, 8 boxes falling beside it, software GPU): broad phase 243 → 122 ms per step.

This replaces the previous rule, where only moving shapes searched and a large moving piece scanned finer levels over its whole extent. That cost about 10 ms per step in a collapse. On the first frame of the 2,000-box pile, the old and new searches find the same 5,871 candidate pairs (test `paircount`).

## Dispatch

All dispatches are direct, so the browser does not validate them one by one; Chromium validates indirect dispatches individually, and about 800 per step stalled whole frames.

- **Colour passes loop.** Each colour pass is sized from the recent per-colour count (×1.5 + 256) and its threads take every stride-th contact, so a colour that grew since the stats were read is still solved in full, only by fewer threads. (Sizing from stats without the loop left contacts unsolved in fast-changing scenes.)
- **Narrow phase loops too.** `narrow` is sized from the recent candidate-pair count (×2, at least 8,192) and loops the same way. Sized by the capacity (24 pairs per shape) it spent most of its time in threads with nothing to do.
- **Chunk pairs apart.** The chunk test needs both hulls' corners and directions and 16 clipped points per thread; in the same kernel as the boxes, every box pair paid for that too. GPU test `pile` (2,000 boxes, software GPU) checks the narrow phase against the broad phase: 260 against 49 ms with chunks and capacity sizing, 63 against 53 ms after both changes (57 against 49 before chunks). Lab `collapse`, software GPU, at the slowest step: narrow 5,483 ms (600-frame run before the change) against 23 ms, plus 0.04 ms for chunks (150-frame run after it); median GPU step 6,693 ms against 3,525 ms.
- **The once-a-step contact kernels: measured.** About 40 kernels a step run over the contacts (colouring, events, crushing, sleep islands, loads). Sized by capacity (160,000 contacts: 2,500 workgroups each, all but a few returning at once) or by their real count through an indirect dispatch (which the browser validates): which costs less depends on the GPU and the browser, so `calibrateDispatch` times 40 of each when the world is made and keeps the faster (`dispatchCal`). Only the timing differs. GPU test `quietcity`, software GPU, ms per step: narrow 77 → 0.02, colour 633 → 1.4, sub-steps 79 → 16, sleep 15 → 0.04 (it chose indirect there).
- **Everything else by capacity.** The kernels skip work past their real counts.
- **Colour cap.** At most 16 sequential colours are solved per sub-step by default (`maxColors`); contacts beyond them go to the parallel pass.
- **Read-backs in flight.** Up to three stats read-backs and three debris read-backs are in flight at once. A late, older result is dropped, but impact events are never dropped.

## Slots

The slots of retired bodies and shapes are reused (`addBodies` returns the bodies' slots). `roomBodies()` and `roomShapes()` give the free capacity. `setVelocity` pushes a body and wakes it.

Each time a body slot is given out (a new body or a spare) it gets a new generation (`genOf`). An answer that arrives later is applied only if the slot still holds the body it is about:
- The engine's answer to a break on the GPU (`onSnap`) checks the generations of both parts. In the lab collapse, an answer re-registered a spare after it had been broken, and a later break of that piece broke the spare made since in the same slot. The slot was then freed twice, and a new body and another spare shared it. The new body's pieces were drawn 1,000 m below the ground (8 jumps in 150 frames).
- A slot is never freed twice (`freeBody`; the lab counts attempts as "body slots freed twice", limit 0).
- Crush records carry the step they come from, and one from before the slot was given out again is dropped.

## Sleeping

A body counts as at rest when, over the last 0.5 s, both hold:
- it has stayed within 1 cm and 0.02 rad of where it was when its timer restarted
- it has stayed below 0.5 m/s

An island sleeps when all of its bodies are at rest. Testing the pose, not the instantaneous speed, matters. Settled piles carry solver chatter of 2–4 cm/s that goes nowhere, and a speed test kept a 2,000-brick pile awake indefinitely.

A body also sleeps on its own once it has been at rest for 1 s, even when its island is still awake. A pile of rubble with a few creeping pieces would otherwise stay awake as a whole: in the rubble test, 99% of the "awake" bodies were already still. The still ones become static support for the rest.

A sleeping body is static to the solver. It wakes when a body moving faster than 0.25 m/s at the contact touches it (`wakeSpeed`). (It joins the solver in the next step; in the way of a body faster than 3 m/s it is woken before the contacts are made: "A sleeping body was a wall", below.) A body just woken, or creeping, does not wake its neighbours, so a pile wakes only as far as the disturbance really goes.

**When a particle rests.** On its pose alone, as for bodies: within 1 cm of where its timer restarted for 0.5 s; or within its own radius, counted a quarter as fast (2 s). A speed limit (2 m/s) and touching something were required too, and the last crumbs of a collapse never slept: lodged against the foot of the standing walls or the ground, they flip between two places 6–7 cm apart every step at 2–6 m/s, some 1 mm off the ground touching nothing (their GPU records, read by the lab at the end of a run). They kept every step of a settled scene running.

**What wakes particles.** A resting particle wakes near a body that is going somewhere: faster than `wakeSpeed`, or moved over 1 cm in the last 0.25 s (its rest timer). Any awake body woke them before, one chattering in place too, and a settled heap never slept: lab `mixed` at 10 s, 36,879 of 36,882 particles awake, most under 0.1 m/s (GPU test `pilesleep`: 312 awake to the end → 0 from 4 s). A particle holding up a moving body does not fall asleep while it does: asleep, it would no longer bear it, and the body would sink through the heap (`heapcreep`, slab awake).

**Parked spares stay asleep.** Spare bodies wait 1 km under the ground for a break on the GPU; `wake` skips them. A blast's wake-all woke them, they fell for the rest of the run, and the world was never at rest (lab `collapse`, 200 frames: 38 of 234 awake bodies; lab check "spare bodies awake while parked", limit 0).

**A world at rest costs nothing.** When the last stats show nothing awake (bodies and particles) and nothing has changed from JS since (no bodies added or removed, no velocities set, no blast), the step is skipped (`skipped` counts them). The page then issues no read-back and rebuilds no instance matrices: the last read-back stays drawn (settled rubble is scenery until something moves).

## Particles (granular debris)

Crumbs are particles: spheres of the brick's volume, drawn as the bricks they are (`maxParticles`, `addParticles`, `readParticles`). Each step, after the bodies moved:
1. **Prepare.** Sleeping particles wake when a moving rigid shape is in a grid cell around them (any level), when a blast reaches them (`blastParticles`) or on `wakeAll`. All live particles go into their own hash grid.
2. **Four sub-steps of position-based dynamics.** Predict with gravity; push out of other particles (averaged, over-relaxed; a sleeping particle does not give way), then out of the ground and every rigid shape in sequence, each in full, so a particle never ends inside a body. Friction is Coulomb: the slide of the sub-step is taken back up to μ × the push. Velocity is the move over the sub-step.
3. **Push limits.** A sub-step's own approach is undone in full. Overlap beyond it (from a spawn, or a body driven into a heap) separates at 2 m/s, so nothing is launched from a deep overlap.
4. **Two ways: heaps hold bodies up.** A particle pressed between a moving body and something fixed (the ground, a static or sleeping shape, or particles resting on the ground) records itself as holding that shape up (`sup_record`: the deepest-pressed particle in each quarter of the shape, packed for `atomicMax`, so the same in any order, and its cross-section). At the next step, after the narrow phase, `p_support` makes those (up to four) particles one contact of the body against the static world, solved with the rest, with friction but no push, no restitution and no warm start: the body is stopped where it is (the contact comes a step late, by which time the body is into the particles). Particles whose centre is inside the shape are left out, and the points' normals must agree. A particle pressed from opposite sides by two moving bodies is crushed to dust (state 3; the page frees it); between two fixed things nothing presses.
   **Bearing limit.** The support contact pushes at most what loose rubble bears (`RUBBLE_BEAR`, 1 MPa, like gravel fill) over the supporting particles' cross-section, a per-point impulse limit in the contact (`M_RV`, unused by support contacts for restitution; restitution −1 marks them). A body heavier than that sinks in, decelerated by it; `sup_acc` adds up each shape's support impulse after the solve, and a shape that took its limit is overloaded (`p_sup_load`): the particles squeezed deep (over 0.3 r on both sides) between it and what holds them are crushed, until enough of them bear it. They crush once the overload has lasted 3 steps (50 ms), time for the rest to come to bear: at once, a 21.6 t block set on the peak of a sleeping heap crushed the 30 particles it first came down on (`heapcreep`, slab awake). A heap holds a slab; crumbs do not hold a building. GPU tests `heap` and `heapcreep`:
   - a 2.9 t slab dropped on a heap at 6.8 m/s sinks to 0.16 m up, none crushed, back up at 0.06 m/s (one-way particles: through to the ground, 981 of 1,500 crushed);
   - a 120 t block on 25 scattered particles (1.5 MPa) crushes them all and reaches the ground;
   - a 21.6 t block set down on crumbs on a sleeping slab is held 0.30 m above it, none crushed.
   **On a moving body.** A particle between two moving bodies holds the upper one up against the lower (a part resting on rubble on a slab still awake): the support contact is then between the two bodies (`sup_record` keeps one body under each shape, the static world first). Before, such a particle was crushed once both pressed it 0.3 r deep; `heapcreep` with the slab awake: the block went through to the slab, 228 crushed → held 0.16 m above it, none crushed. Lab `mixed`, 600 frames: particles crushed 48 → 26 m³, volume lost 3.1% → 1.7%.
   Tried first: the shape held outright and the particles crushed when overloaded, judged on the load smoothed over 5 steps. A landing's spike kept the overload on for 30 steps and crushed every particle pressed meanwhile (`heapcreep`: through to the slab, 239 crushed); capped at twice the weight instead, the mixed building's upper part (2,140 t) stopped 2.5 m down on the cut storey's crumbs (roof fall 2.54 m, limit 3).
5. **Rest.** A particle sleeps after 0.5 s touching something and staying within 1 cm of where its timer restarted. A particle wedged among sleeping ones may jitter in place without going anywhere, and a speed test kept such particles awake indefinitely.

**Candidates once per step.** Each particle gathers the particles (up to 24) and shapes (up to 12) it can reach this step, from its speed, and the sub-steps test only those. In the tower benchmark (12,600 crumbs in the air) this cut the particle stage from 18 s to 0.8 s per step on the software GPU, below the rigid sub-steps; the poured heap comes out the same.

Particles cut from a moving body wait (state 4) and are placed at the next step from the parent's pose then, like its pieces (`breakBody(parent, kids, pr, qr, dvs, parts, pdvs)`).

### Nothing leaves a squeeze faster than what squeezes it (18 November)

A particle is pushed out of each shape it is in, one after another in the same sub-step, and each push is
bounded on its own: as far as that body moved into the particle, plus the separation limit (2 m/s). A crumb that
a part of many bricks has come down over is pushed out of one brick into the next, and on by that one, up to
twelve times. The whole move of the sub-step is now bounded too: as far as the particle was going by itself, or
as far as the fastest particle or body that pushed it went plus the separation limit, once.

Found by a crumb that was not there to be asleep: lab `collapse`, 600 frames, did not settle in 10 s because one
crumb was still falling, from about 80 m, at 28 m/s in the last frame. The lab had no check for it; with one
(docs/LAB.md, "crumbs thrown up"), the first 240 frames show 32 crumbs going up at 10 to 25 m/s between frames 54
and 73, as the upper part lands. GPU test `crumbpop`, a 300 t part of 128 bricks landing at 6 m/s on 64 crumbs:
the fastest crumb goes up at **39.9 m/s** before (all that `p_apply` allows) and **6.8 m/s** with the limit.

### Crumbs do not throw one another (18 November)

The limit above took the fastest pusher's speed, whatever its direction, and crumbs were among the pushers. Each
push of one crumb on another is bounded by how fast the two close plus the separation limit, and that limit was
added again by every neighbour and in every sub-step: up a column of crumbs made on top of one another each left
2 m/s faster than the one under it, and two crumbs parting let each other go faster by it each sub-step.

The move the other crumbs give a crumb in a sub-step is now bounded by speed with nothing added: as far as it
was going by itself, or as far as the fastest crumb that pushed it went, or the separation limit, whichever is
most. No crumb leaves the others faster than the fastest of them, so their speeds cannot grow among themselves.
The shapes' pushes that follow keep their own limit (the fastest body plus the separation limit, once).

Tried first and not kept (commit f259dd1): the same bound taken along the way the crumb ends up going. A crumb
could then not get out sideways from under what came down on it faster than the separation limit, 2 m/s. Lab
`qglass` (600 frames, software GPU): the glass house, its four north columns cut, stopped on the crumbs of the cut
0.29 m down; 1.14 m with the bound by speed (limit at least 1). Applied to the shapes' pushes as well, a crumb a
box came down on ended 5 cm inside the table (GPU test `particles`, 3 inside).

What the change does to a heap: crumbs no longer stir one another, so a heap spreads less. GPU test `particles`
(2,000 crumbs poured on a table 2 m square): 471 stay on the table, where 385 did, the heap 8 cm higher, and the
last two crumbs come to rest between 6 and 7 s, where the last did between 5 and 6. That test counted the steps
the world skips once all is asleep in 30 steps taken in a row, in which no read-back can arrive: 30 or none on
whether the last crumb slept before the last read-back that had arrived by 7 s; none now, on a machine doing
nothing else. It steps frame by frame now, as a page does: 29 of 30 (limit over 20, unchanged; asleep at 7 s,
unchanged). GPU test `heap` now fails: see "Still to do".

Found by the lab's check for crumbs thrown up: `tallball` and `qballoffice` (build 228b18a, 480 frames, software
GPU) each had one crumb going up at 13.2 and 11.3 m/s, 85 and 45 frames after the ball had stopped, with no body
moving within reach. GPU test `crumbnest`, 24 crumbs made 0.1 m apart in a well 0.5 m square: the top of the
column leaves at **7.09 m/s** before and **1.8 m/s** after (2.07 the fastest in any direction).

## Validation (`node engine/gpu/test_gpu.cjs`, headless Chrome with SwiftShader WebGPU)

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
| 2,000 particles poured on a 2 × 2 m table | none under the ground or inside the table; heap on it; all asleep, then steps skipped; a box dropped on it wakes it | 0 / 0; 0.6 m heap on the table, the rest at the angle of repose around; asleep by 5 s, all steps skipped; woken; the box rests on the heap, none crushed (particles one way: 34 crushed under it) |
| Slab (2.9 t) dropped flat on 1,500 particles at 6.8 m/s; 120 t block on 25 scattered particles | slab held up by the heap, few crushed, bounce ≤ 0.1 × impact + 0.1 m/s; block crushes them and comes down | slab 0.16 m up, 0 crushed, back up at 0.06 m/s; block on the ground, 25 crushed |
| 21.6 t block set down on crumbs on a sleeping slab (`heapcreep`) | held above the slab, few crushed, no rise | 0.30 m above, 0 crushed, no rise |
| Particle cut from a tumbling body, placed from a pose 10 steps old | where the body is now | within 2 mm |
| Fewer sub-steps (20-box stack drift) | | 12: 7 mm · 8: 16 mm · 6: falls (so adaptive sub-steps stop at 8) |

SwiftShader runs WebGPU on the CPU, so its timings mean nothing. Throughput is measured by the bench page on real GPUs.

## Contact reduction

Two bodies touching through many shape pairs keep only representative contacts: per body pair with more than 32, the ones furthest out in 26 directions (the corners, edges and faces of a cube) and the deepest. The others are marked (+8 on the point count) so colouring and the solver skip them, but they keep their points and impulses for the warm start of the next step, where they may be picked again. This is what keeps the upper part of a building (thousands of bricks on a floor) out of the parallel overflow pass.

GPU test `compound` (800 bricks as one body landing at 5.9 m/s on a brick floor): rebound 1.1 → 0.44 m/s (concrete restitution allows 0.59); overflow ~1,300 contacts → 0.

## Shock propagation (relax pass)

A light piece on the ground under a heavy one is a sandwich the Gauss-Seidel solve does not settle. In each sub-step, the push out of the ground (the soft contact's bias) goes up through the light piece into the heavy one. The relax pass should take that push back, but at 20:1 each of its iterations returns only about 1/21 of it.

In GPU test `landing` (12 sub-steps, software GPU), a 960 kg slab landing at 7.7 m/s on a 47 kg piece came back up at 3.65 m/s, 47% of the impact speed against concrete's 0.1. More relax iterations helped only slowly: 2.84 m/s with 2, 2.61 with 4, 1.41 with 8. Restitution off, a 0.1 m/s push limit, 48 sub-steps and the mass ratio limit off all left it at 2.8–3.6 m/s.

**The fix.** Each body gets a level: the static world and sleeping bodies are 0, a body resting on them 1, and so on. Levels come from the contacts whose normal is within 45° of vertical, over 6 passes. Each pass reads one copy and writes the other, so the result does not depend on thread order. In the relax pass, where a vertical contact is separating (the pass is taking back a push), the lower body is held: the upper body alone gives the push back, as it would once the pass had converged. Approaching contacts, the biased solve and restitution are unchanged.

A first version that also held the lower body for approaching contacts was unstable: the impulse it built up, warm-started into the light piece at its real mass, threw both bodies at 180–2,800 m/s.

`landing` (rebound limit 0.1 × impact speed + 0.1 m/s), before → after:

| Case | Heavy body's rebound (m/s) |
| --- | --- |
| 960 kg on 47 kg at 7.7 m/s | 3.65 → 0.42 |
| 960 kg on 47 kg at 2 m/s | 0.96 → 0.10 |
| 7,680 kg on 47 kg at 5 m/s | 0.75 → 0.37 |
| 207 kg on 47 kg at 7.7 m/s | 2.45 → 0.20 |

All other GPU tests pass unchanged. `GpuWorld.shock = false` turns it off.

## Crushing on contact

**After a crush.** A crushed piece stops where it is (the page turns it into rubble at rest there), and the next step's contacts of the bodies it was pressing start afresh (kernel `crush_release`; `emit`): their warm start carried the load the crushed piece had passed on, and without it threw them. GPU test `chain`: a 223 kg piece under a block crushed by a 259 t part went up at 20.6 m/s (13,800 N s warm-started, 62 m/s of its own); now 2.97 m/s at most.

A light piece pressed by a body at least 20 times heavier is crushed on the GPU (kernel `crush`), when the impulse it took in a step is more than 1.5 times what its own approach momentum needs (it is being squeezed against something) and it approached at 1 m/s or more. From then on it collides only with the static world, so the heavy body goes through it. The page (`takeCrushed`) turns it into rubble particles, its bricks split into cubes of at most 0.4 m, placed from its current pose.

GPU test `crush`: a 7,700 t block landing on 24 pieces of 11 t crushes all 24 and comes down to the ground; a piece rammed from the side by a 600 t block is not crushed. In the lab's collapse (150 frames), launches went from 14 to 1 and bounces from 11 to 7.

**Pinching (on by default since 14 November, with the velocity bound; `GpuWorld.pinching = false` turns it off).** The crush test looks at one contact at a time and needs an approach of 1 m/s. A piece pressed between two bodies, or between a body and the ground, at almost no approach speed is never crushed by it. In the lab collapse a 49 kg piece took 76,000 N s in one step between the ground and a 773 kg body, spun at 30 rad/s, then flew up at 7 m/s.

Kernels `pinch_acc` and `pinch` sum each body's normal impulses as vectors and as magnitudes (fixed point, so in any order); the part that cancels is the squeeze. A piece is crushed when:
- it is at least 20 times lighter than what squeezes it (the static world counts as infinitely heavy);
- the squeeze is more than 3 m/s of its own momentum;
- the squeeze over the step, over all its contact patches (the largest one alone until 14 November), passes the crushing strength;
- something that moves and is at least twice its mass is among what presses it (14 November: the static world is the anvil, something has to be the hammer). Without that, the day the test went on by default, lab `timber` lost its whole upper building: one body of 7,500 boxes, wedged on the stumps it was cut from and pushed from all sides by them and by the cut's crumbs, read as squeezed and was crushed to 29,277 crumbs 0.3 s into its fall; the roof went from 15 m to 1.5. The scenario now has an upper limit on its roof's fall (6 m) beside the lower one, which alone took that for a pass.

GPU test `pinch` turns it on and passes, but it passes with it off too, because the crush test crushes those pieces once the block reaches them. In the lab collapse (150 frames, seeds 0 / 7 / 8, before shock propagation): rebounds 13 / 26 / 16 → 12 / 11 / 15, launches 7 / 6 / 8 → 4 / 2 / 2, but one energy gain in each run (0 before). With shock propagation: rebounds 10 / 6 / 11 → 9 / 17 / 16, launches 5 / 2 / 0 → 3 / 1 / 0. Three runs do not tell it apart from luck, and it adds energy gains on its own, so it stays off.

## The velocity bound (14 November)

**What it is.** After the solve, each body's velocity is held to what its contacts can physically have given it (kernels `vh_init`, `vh_acc`, `vh_clamp`; `GpuWorld.velocityBound = false` turns it off). Contacts here lose energy (restitution 0.05 to 0.5), so what a step can add to a body's speed is bounded by how fast its contacts were closing when the step began:

- speed: `|v'| ≤ |v| + (1 + e) × the fastest closing at any of its contact points + a step of gravity`;
- upward: `v'y ≤ vy + (1 + e) × the fastest closing along a contact normal × that normal's upward part`;
- spin: what it had, plus `(1 + e) ×` the closing over the arm from the contact to its centre, plus 1 rad/s;
- the distance gone in the step (what `finish` applies): no further than at the bounded speed, plus what the contacts push it out of an overlap by (`maxPush`).

`e` is the greatest restitution among the body's contacts plus 0.1 (`boundSlack`) for what the solve leaves over. A closing against a body that can itself give way counts for `m_other / (m + m_other)` of it; against the static world or a sleeping body, all of it. A body that stands on the static world through the contacts under it (the levels of shock propagation, up to 6 bodies deep) can always come to rest, with `e` of its fall returned.

**Why.** The solver does not settle a squeeze between very unequal masses in a few iterations, and what it leaves is a velocity nothing physical gave. Lab `masonry`, frames 59 to 61: a 5.2 t piece on a 0.56 t one, both falling at 3.6 m/s, under a 495 t part coming down at 10 m/s, left the step going **up** at 3.55 m/s; two pieces lying on it left at 5.5 and 7.7 m/s. Crushing and shock propagation each remove some cases (above); the bound holds in every case, crushed or not, converged or not. The velocities *inside* a step are the unbounded ones, which is why the distance is bounded too: a piece crushed in the step it was squeezed in ended 0.23 m from where a step at its speed puts it, and was drawn jumping (lab `masonry`, frame 52).

**Pinching is on with it** (`GpuWorld.pinching`, off since 9 November as "not told apart from luck"). With the bound, pieces that used to be thrown out of a squeeze stay in it; the lab's collapse then kept breaking its upper part against them (measured on the first form of the bound, before the engine's fixes of the same day: 915 bodies after 240 frames, 437 with the pinch test crushing them by the stress on them). The table below is with both on, against both off.

**What it does not bound:** anything in free flight. **What it takes away:** a hit passed down a chain of contacts within one step is passed on in the next step instead.

**Tried first, each dropped for a reason a test found:**
- A box of velocities per axis. A chunk landing on a sloping face could not turn its fall into a slide (sideways 0 before, so 0 after), slid at 0.45 m/s for 0.6, and GPU test `hullpairs` found it still falling off after 2 s.
- The fastest partner's velocity as the limit, with allowances for the body's own spin and a step of gravity upward. A body the solver kept kicking climbed by that much every step it stayed in contact: GPU test `bound`, 2.25 m/s where 1.8 was meant. Closing speeds have no such ratchet: a contact that is not closing gives nothing.
- A grounded *partner* imposing its whole contact-point velocity: a 238 kg piece spinning under a 700 kg one sent it up at 3.6 m/s.
- Bounding the velocity only: the distance, above.

**Numbers (software GPU, this build):**

GPU test `bound`: the `landing` cases with shock propagation and crushing off, the solver on its own. The heavy piece's rebound, m/s:

| Heavy on light, at | Without the bound | With | The light piece, with | Bound (0.2 × the fall) |
| --- | --- | --- | --- | --- |
| 960 kg on 47 kg, 7.7 m/s | 3.65 | 0.05 | 0.77 | 1.54 |
| 960 kg on 47 kg, 2 m/s | 0.96 | 0.02 | 0.22 | 0.40 |
| 7,680 kg on 47 kg, 5 m/s | 0.75 | 0.00 | 0.50 | 1.00 |
| 207 kg on 47 kg, 7.7 m/s | 2.45 | 0.05 | 0.77 | 1.54 |

The lab, 240 frames, three seeds each (`SF_SEED` 0 / 7 / 8), with the bound and the pinch test against neither (`SF_NOBOUND=1`):

| | Without | With |
| --- | --- | --- |
| `collapse`: rebounds over half the impact speed | 9 / 2 / 1 | 0 / 0 / 0 |
| `collapse`: pieces thrown up faster than 5 m/s | 0 / 1 / 3 | 0 / 0 / 0 |
| `collapse`: crumbled share of a big part (worst) | 0.29 / 0.28 / 0.09 | 0.11 / 0.08 / 0.10 |
| `collapse`: bodies at frame 240 | 694 / 430 / 355 | 386 / 322 / 328 |
| `masonry`: rebounds | 5 / 1 / 0 | 1 / 1 / 1 |
| `masonry`: thrown up | 2 / 1 / 1 | 0 / 0 / 0 |
| both: pieces moving further than their speed allows | 0 / 1 / 2 and 2 / 2 / 2 | 2 / 0 / 2 and 3 / 3 / 2 |

Over the six pairs: rebounds 18 → 3, launches 8 → 0. One rebound is left in each masonry run (−3.2 → +1.8, −3.2 → +1.6, −2.3 → +1.8 m/s). In seed 0 it is a part broken off a tumbling piece on the GPU (`snap_split`), which leaves with the piece's motion at its own centre: the spin's, not a contact's. In the other two it is a piece of 10 and of 6 boxes, not traced. The jumps are crumbs, the same with and without: "Still to do".

All 41 GPU scenes pass with it (40 before, and `bound`). Cost: three more dispatches a step (one a manifold, two a body), and the pinch test's two.

## Fast pairs that only pass (15 November)

A pair's margin covers all the two can travel in the step, so that nothing goes through a thin wall between two
steps (test `bullet`). The other side of that: a body at 47 m/s has contacts with everything within 0.8 m of it,
and a contact with a shape it only passes holds it back. Its approach along that contact's normal (edge against
edge: a slanted normal) is more than the gap divided by the step allows, and the solver takes the excess out.

It was found on the steel ball (docs/LAB.md, 15 November). The engine had already made a hole for it through a
roof slab, 1 m square for a ball whose corners reach 0.49 m from its middle. In the step it went through
(lab `tallball`, frame 28, body trace `SF_TRACE_BODIES`), the ball had **104 contacts with the hole's rim**, their
normals slanted (0, -0.87, 0.50), (0, -0.62, 0.79), (0.59, -0.80, -0.11), their approach speeds 27 to 45 m/s, and
it left the step at 8.3 m/s where it entered at 44.7, turned sideways at 6.9 m/s.

The narrow phase now asks, for a pair that moves more than 0.3 m against each other in the step (`SWEEP_MIN`:
18 m/s at 60 steps a second), whether the two meet in the step at all. Each separating axis gives the part of the
step during which they overlap along it, the one moving by its relative travel against the other (and a
tolerance: the margin without the straight travel, which is the contact slop and what their spin can add). If no
moment of the step lies in all those parts, there is no contact (`sweep`, in `box_box` and `hull_hull`; a hull's
face axes are one-sided, so they only bound when the overlap begins). Slower pairs keep every contact within the
margin, as before: a body that is turned in the step's first sub-steps may still meet what it was not heading for.

GPU test `passby` (software GPU): the game's ball (0.8 m, four crossed boxes, 2,104 kg), turned so its edges are
skew, sent down at 47 m/s through the middle of a 1.2 m hole in a fixed slab with eight turned blocks round the
hole; and the same 0.35 m off centre, where its path does meet the rim.

| | Before (`SWEEP_OFF=1`) | After |
| --- | --- | --- |
| Through the middle: slowest it went down while above the slab's underside | came back up at 9.1 m/s | 47.2 m/s (free fall) |
| Through the middle: fastest sideways | 22.5 m/s | 0.00 m/s |
| Through the middle: most contacts in a step | 38 | 4 |
| Off centre (it must hit) | back up at 10.0 m/s, 17.6 m/s sideways | back up at 6.0 m/s, 16.6 m/s sideways |

`bullet` still holds (the brick at 80 m/s stops 0.13 m short of the wall's face, as before).

## Mass: a body's own, and a steel member's (15 November)

- **A body's mass can be given** (`addBodies`, `mass`): the shapes share it by their volumes. The game's ball is
  four boxes that overlap, and the world added their masses: a 0.5 m ball weighed 1,668 kg (13.6 r³ of steel)
  where a steel sphere weighs 514 kg (4.19 r³), 3.2 times too much; the lab's 0.8 m ball 6,833 kg for 2,104 kg.
  The lab checks it on every ball ("its mass, of a steel sphere of its size": 0.99 to 1.01).
- **A shape can have its own density** (`density`). A steel member is a rolled section in the box it is drawn as:
  the engine weighs and rates it at 16% of the box (`structure::STEEL_AREA`). The page gave the GPU the material
  alone, so the same member, once it fell, weighed the solid box, 6.25 times as much (a 6 m beam of 0.25 m:
  2,944 kg for 471 kg), and landed with that. Steel debris now carries the member's density
  (`engine/web/bodies.js`, `debrisDensity`); the lab checks every piece it makes against the engine
  (`piece_mass`): "a piece's mass on the GPU over the engine's (worst)", 0.98 to 1.02.

## What stopped the ball, and the falling building: four things found on 15 and 18 November

The game's steel ball "often bounces off structures instead of going on". Three causes were in the GPU world
(another, the look-ahead, is the page's and the engine's: docs/LAB.md, 15 November); the fourth here was found
when the third was fixed. Each has a GPU test that fails without the change (the switch is named) and the lab
scenario it was found in. Software GPU.

### A sleeping body was a wall for the step it was first hit in

A sleeping body is static to the solver, and is woken by the narrow phase when a moving body touches it, to join
the solver in the **next** step. For a piece that creeps onto it nothing is lost by that. For a fast one, that
step is the hit: the sleeper has no mass in it, and the fast body leaves it as it would leave a wall.

Lab `qballglass` (the 0.5 m, 514 kg ball through the glass house at 60 m/s): it reached the wardrobe (383 kg,
asleep on the floor since the level began) at 53 m/s and came back at 15.9 m/s; one contact had taken 26,335 N s.

Now a sleeping body in the way of one faster than 3 m/s (`WAKE_FAST`) is woken **before the contacts are made**:
the broad phase marks it when the two boxes overlap (they hold a step's travel), and `prep_woken` wakes it and
gives it its mass in the solver, between the broad and the narrow phase. The mark is a bit in a per-body table,
not the sleep flag itself, which the broad phase reads from many threads (the pairs it lists must not depend on
their order).

GPU test `sleeper`: the same wardrobe, left until it sleeps (40 frames), then the ball at 53 m/s into its long
side. Sharing the ball's momentum the two go on at 53 x 514 / 897 = 30 m/s.

| | Before (`WAKE_OFF=1`) | After |
| --- | --- | --- |
| The ball, after the hit | back at 15.9 m/s | on at 26.5 m/s |
| The wardrobe | at rest | on at 29.4 m/s |

### Covered faces: a floor's seam was an edge to catch on

The standing structure is many boxes, face to face: a floor is panels, a wall is bricks. A piece that lies a few
centimetres into a floor (pressed, tipping, landing) and slides to the next panel is, to that panel alone, a piece
beside it and below its top; the axis that separates the two is the panel's **side** face, and the contact made
there is across the piece's way.

GPU test `wedge`: a floor of 2 m panels; a 300 kg steel block on it, and the ball into the block at 60 m/s. The
block tips 4.7 cm into the floor, meets the next panel's edge (normal 0.92 along the way), takes 23,315 N s there
in one step and stops, the ball behind it. And two of the glass house's chairs (12 kg each) in the ball's way.

The world now knows which faces of the fixed, axis-aligned boxes are covered by other fixed boxes (`coverFlush`:
a face is covered when all of it is, 1 cm out from it, inside other fixed boxes (see "A face covered in part"
below; until 19 November: when its middle and four points 0.6 of the way to its edges were); kept up as shapes are set and killed, a shape and what touches it looked at again; the mask is in
the shape's record, and counts only while the shape's body is fixed). A contact whose normal pushes out through a
covered face by more than 0.3 of itself is no contact (`cover_fix`): what lies over the seam is held by the contact with the
panel it lies on. (First the normal was turned to the faces that are there; it kept the depth measured along the
way it had pointed, which is wrong for the way it then pushed. Lab `collapse` on that build did not settle: 92
of 364 bodies awake at 10 s; over 450 frames, 109 with the early waking off, 48 with covered faces off, 4 with
both and the swept test off. Those runs do not separate the causes: each collapse falls differently. And a contact that leans through a covered face by less than that
stays: without it a chair's leg, a hair off the vertical, lost its contact at a seam and rocked there for good.)

| | Before (`COVER_OFF=1`) | After | On a floor of one piece |
| --- | --- | --- | --- |
| The ball after the block | 4.2 m/s | 31.9 m/s | 28 m/s |
| The ball after the two chairs | 54.6 m/s | 56.6 m/s | |

Limits: boxes only (a free piece of the temple, a hull, has no mask), fixed bodies only (a fallen floor that
debris slides over is one moving body of several boxes: its seams are as before), and a face partly covered
counts as open.

**A face covered in part (19 November).** "Counts as open" was what the rule meant and not what the code did. It
looked at five points of a face: its middle and four 0.6 of the way to its edges. A step under a shorter step
has all five under the upper one and a tread in the open: the temple's lower step is 5.25 m wide and the upper
one stops 0.375 m short of its edge. Its top was marked covered, the contact with it was dropped, and what
landed on the tread fell through the step and lay inside it, between its underside and the ground. Found on lab
`qtemple` (a 64 kg piece that landed on the tread at 7.7 m/s, then lay in the step, pressed by 13 to 15 kN s a
step from both sides, turned 7 to 14 mrad every few frames by the two pushes and so never slept; LAB.md,
19 November). The same holds wherever fixed boxes are stacked with a set-back: plinths, cornices, a wall on a
wider footing.

Now the five points only rule faces out; a face that passes them is cut along the edges of the fixed boxes over
it, and it is covered when the middle of every part is inside one of them (to within 1 cm of the face's own
edges). A brick under two bricks in bond is still covered; the step is not.

GPU test `tread` (new, default run; software GPU): two fixed steps as the temple's, a 10 kg piece dropped on the
tread at 3 m/s. Before: at rest 0.021 m up, inside the step (lower step's mask 4: top covered). After: 0.425 m
up, on the tread (mask 0; the brick under two keeps 4).

### A body thrown is solved at its own mass

Across one contact the heavier body responds as if at most 20 times heavier than the lighter (`P_RATIO`; "Bodies
with many contacts"): for a part of thousands of tonnes lying on light rubble, which the solver cannot settle
otherwise. For the ball it meant that every piece lighter than a twentieth of it cost it as much as one of a
twentieth: a 12 kg chair as 25.7 kg to the 514 kg ball; a crumb as 105 kg to the lab's 2.1 t ball.

A body can now be marked as thrown (`setShot(body)`; the page marks the game's ball). While it moves faster than
10 m/s (`SHOT_MIN`) its contacts are solved at its real mass. At rest on rubble it is limited as everything is.

GPU test `wedge`, the two chairs (carried along whole they leave the ball 60 x 514 / 538 = 57.3 m/s):
54.7 m/s without (`SHOT_OFF=1`), **56.6 m/s** with.

`setShot(body, 2, pass)` also gives the body a part of its way (a point, a direction, from where to where
along it) in which it does not touch what of the fixed world is across it: a contact there whose normal is more
than 14 degrees out of the plane across the way is not made. The page sets it to the depth of each thing broken
for the ball, 0.3 m each side, for 0.3 s at most (docs/LAB.md). Two forms were tried first, and measured: no
contact with the fixed world at all for those steps (coming down at 5.6 m/s of 58, the ball was 0.4 m into the
glass house's floor after two partitions), and no contact within 60 degrees of its velocity, anywhere (lab
`ball`: at 115 m/s a rim met at 70 degrees took it to 42 m/s).

### What a crushed piece gives back (18 November)

With the seams gone, lab `collapse` showed what they had hidden. The 2,005 t upper part of the frame, falling at
2.26 m/s onto the rubble of its own cut storey, left one step at 0.75 m/s: 18,057 N s through one contact with
one piece of that rubble, which stopped 2,005 t as it would stop 20 times its own mass (`P_RATIO`). The piece was
crushed in that step (pinch), as it should be; the speed was gone. The building then came down on its columns'
stumps at under 2 m/s, too slowly to count as an impact, and stood on them whole, its roof 1.74 m lower. Until
15 November one contact with a floor panel's covered side had given the one event that broke it up.

A piece that fails does not stop what breaks it: it can take the momentum that carries its own mass along at the
speed they met at, and no more. Each body that pressed a piece crushed in this step gets back the velocity that
piece's contact took from it, less that share, and no more than brings it back to the speed it began the step
with (`crush_refund`, after the crush passes; sums in fixed point per body, the same in any order).

GPU test `crushgive`: a 200 t block down at 3.12 m/s on a 300 kg piece on the floor. After the step the piece is
crushed in: **1.65 m/s** without (`REFUND_OFF=1`), **3.12 m/s** with.

Only the heavier of the two gets anything back, the one that did the crushing: a light piece the crushed one was
pressed onto was pushed by it, not slowed (GPU test `chain`: given back, that push threw the piece under it up at
7.8 m/s; 2.96 m/s now, what it was before there was a give-back, against a limit of 5).

And only what slowed it. The push of the crushed piece's contact must have been against the way the body was
going as the step began (the velocity bound's record now keeps that velocity whole), and what is given back is
given along that way, up to the speed it had: never sideways, never back. A piece of furniture on a falling
part's own floor, crushed as the part stops under it, had pushed the part down; given back, that sent the upper
part of the complex building from 6.14 m/s down to 0.99 and then 1.83 m/s up in the two steps it landed in (lab
`complex`, 18 November: 22 rebounds, 3 energy gains over 5%). GPU test `crushride`: a 100 t slab falling at
2.49 m/s with a 300 kg piece on it, which a 200 t block crushes from above: **2.67 m/s up** after that step
without the rule, **2.67 m/s down** with it.

Lab `collapse`, 600 frames, this build: the roof falls 3.27 m (18.64 to 15.37), 50 breaks, 293 bodies, every
body asleep at 8.0 s (on 14 November: 3.21 m, 375 bodies, 10 of them awake at 10 s). Without the give-back and
with the seams gone it fell 1.74 m and did not break.

## Breaking in the step of the hit

The engine answers a hit about 4 frames late. So each new falling piece carries a break plan from it (`Sim::plan_piece`): where it breaks first, and how hard a hit must be to break it there.

How the engine makes the plan:
- The piece is loaded with its own weight, resting on its bricks within 0.5 m of its lowest.
- The plan's section is the horizontal plane (a bed joint, a storey line) whose worst bearing joint is the worst, among the planes that part the piece with at least a twentieth of its mass on the smaller side.
- The threshold is the impulse in one step that decelerates the piece by g / (that joint's ratio): M g dt / ratio.
- Cutting single joints worst first did not work: the 4-storey tower's worst joints are spread over the building, and it took 4,096 of its 5,140 joints to part it.

The page arms the body with `setSnap(body, impulse, shapes of the other part)`. That makes a spare body in advance, parked asleep, with the other part's mass and inertia. Each step on the GPU:
- `snap_acc` adds up the normal impulse of the body's contacts (N s, integer).
- `snap_split` runs when the total passes the threshold. The spare takes the other part's place and the piece's motion there; the body keeps its own part, with that part's mass.
- `snap_shapes` moves the other part's shapes to the spare, in its frame.

What follows:
- The read-back lists the break (`takeSnaps`).
- The page sends the engine a `snap` job, first in line. The engine cuts the plan's joints and answers with the two pieces, each with its own new plan.
- If the engine broke the piece first, the plan no longer holds, the answer is empty, and the spare is given up: the engine's parts already include its bricks.

GPU test `snap`: a column of two 0.5 m cubes (600 kg) falls 1 m, armed to break between them at 500 N s. A second column is armed at 10 MN s.
- The first breaks in the frame it lands (frame 27 of both). The top cube moves no further between frames than its speed allows. It ends resting on the lower cube at 0.75 m.
- The second never breaks.

Engine test `a_falling_piece_has_a_break_plan_and_snaps_by_it`: the 4-storey tower's ground storey is cut.
- Its upper part (2,049 bricks, 2,225 t) breaks across a plane of 77 joints, letting 17 bricks (at least a twentieth of its mass) come away.
- The threshold is 3,116 kN s in a step. Stopping it from 5 m/s takes 11,125 kN s.
- Snapping parts it there into two pieces with no joint between them.

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

| `freehull`: runtime hulls (free pieces): a wedge dropped 0.2 m onto a slab; three courses of wedge pairs; a block on a static slab turned 15 degrees | lands flat; the stack stands; rests on the slope | down 0.2000 m, turned 0.00 degrees; 6.6 mm; 5.2 mm and 0.00 degrees |
| `freehullp`: particles dropped over a static wedge whose box is half empty air | rest on the slope, never inside | 0.050 m over it (r 0.05); with the old box test 0.850 m, on the box's top |

**Runtime hulls (phase 2, docs/PHASE2.md step 5).** After the library come up to 4,096 hulls registered at runtime (`GpuWorld.pieceShape`, `registerHull`), each a free piece's corners and faces in its own box mapped to [-1,1]^3, registered by content (the same piece shape takes the same slot) and uploaded to every world at its next step (`uploadHulls`). Static shapes may be turned (`setStaticShape(..., hull, q)`).

**Particles against a free piece** (a runtime hull, id from `HULL_LIB` = 63 on) test its faces (`hull_sd`: the face plane the particle is farthest outside, its distance and normal), in `p_contact` and `p_support`; boxes and the library's cells keep the clamp to the box (the cells fill their boxes; with their faces, a crumb coming off one in the lab's `ball` left another way and jumped 0.31 m, over the 0.26 m limit). Particles took every hull for its box: near enough for the library's cells, which fill their boxes, but a free piece's box (an inclined brace's, built unturned) is mostly air, and crumbs by one were pushed out of nothing (lab `complex`: 0.45 m jumps in a frame).

## Determinism

The same inputs give the same world, bit for bit. Before this, two runs of a 600-box pile differed from the first sample on (1.4 m after 90 frames), and each lab run of the collapse came out differently.

What made runs differ, and what each is now:

| Source | Now |
| --- | --- |
| The grid's cells were filled by atomics, in whatever order threads arrived. Pairs and the particle candidate lists (capped at 24) walk them. | Each cell's list is sorted by shape index (`sort_cells`, `p_sort`). |
| Contacts get their slots from an atomic counter, so slot numbers mean nothing. | Every choice between contacts goes by a key mixed from the two shapes (`mkey`). |
| Colouring was first come, first served (`atomicOr` on the bodies' masks). | It is claimed in rounds (`color_claim`, `color_tie`, `color_resolve`; details below). |
| A body's warm-start sum went over its contacts in the order `color_scatter` left them. | Its list is sorted by key first (`blist_sort`). |
| Restitution of the overflow contacts goes through them one by one, in scatter order. | They are sorted by key first: a bitonic sort in one workgroup (`ovf_sort`). |
| Contact reduction broke ties of the extreme contacts by slot. | Ties are broken by key. |
| The reduction's table of body pairs hashed a pair as its key times a constant. Every pair with the static building (body 1) then started from the same slot, chains ran past the 64 probes, and contacts went uncounted depending on thread order. So the building's pair with a piece was reduced in one run and not in the next (lab collapse, frames 91–102, found by comparing every frame's contact colours of two runs). | The key is mixed (a murmur-style finaliser). |
| A body pair to reduce got its place among the reduced pairs (at most 4,096) from whichever of its contacts came first. Every contact that saw no place yet took a new one, so a pair with hundreds of contacts used hundreds of places, and past 4,096 the pairs that came last went unreduced. In the lab collapse the colours parted at frame 91 from it. | One thread per table slot hands each pair one place (`red_big`). |
| Events, crushes and breaks came back in the order threads wrote them. | JS sorts each batch (by shapes, by body). |
| With more events in a step than the list holds (512), the ones kept were whoever wrote first. | Events are counted in energy bands (eighths of a doubling over the threshold), and the lowest bands that do not fit are left out whole (`ev_count`, `ev_cut`). |
| Pieces placed from a parent that is itself placed in the same batch (a piece broken twice in one job) read it mid-placement. | One generation is placed per step; the next waits for the following step. |
| `readPose` read the velocities only once the poses had come back, some steps later. | It copies both at once. |
| `readPose` computed shapes' poses from the live JS records, which a GPU break changes when its stats arrive, sometimes before the read-back does. | It computes each shape's pose from the GPU's own shape record (body, local pose, size), copied with the bodies. A first fix copied the JS records at the moment of the read instead; pieces of a body that had just broken were then drawn about its old centre, up to 8.7 m off (lab collapse, 150 frames: 6,709 jumps). |

How the colouring rounds work:
- Each round, a contact not yet coloured picks one of the lowest four colours free at both its bodies (by its key and the round), and claims it at each body with a priority.
- It takes the colour where its claim is the strongest at both bodies.
- Claims carry the round in their top bits, so an old claim never blocks a new one.
- Supports (contacts with the static world or a resting body) get 2 rounds of their own first, then everything gets 4 more (`colRoundsS`, `colRounds`). With 12 rounds the colour stage of the lab collapse took 880 ms at its slowest step on the software GPU, against 38 ms before; a dispatch there costs milliseconds. With 2 + 4, the 2,000-box pile left 10 contacts to the overflow pass against 1 with 12 rounds; with 1 + 3 it left 67.
- What is still uncoloured, or finds no colour free, goes to the overflow pass.
- Two contacts at a body could have equal 26-bit priorities. In a 600-box pile over 90 frames that is about a one-in-ten chance, and both would take the colour and be solved on the same body at once. So tied contacts claim again with 27 bits of a second hash (`color_tie`).
- Supporting rounds of their own came from GPU test `stumps`. With supports only first within each claim, the other contacts at the slab took colours its supports then could not get, and the slab rebounded at 0.71 m/s.
- That test's limit was 0.7 m/s, taken from the 6.3 m/s the slab starts at. It hits at 7.6 m/s, so concrete's restitution of 0.1 allows 0.76 m/s; the limit is now 0.1 × the impact speed (its own commit).
- Each round is four dispatches (claim, tie, resolve, the round counter).

What was already order-free:
- The overflow pass sums in fixed point (integer atomics commute).
- The contacts of one colour share no moving body.
- Sleep islands link to the smaller root.
- Warm-start and reduction tables are looked up by key.

GPU test `determinism`: three scenes, each run twice, every pose (and particle) compared: no difference.
- A 600-box pile, 90 frames.
- A pile of 120 convex chunks, 90 frames.
- Everything at once: a 7,700 t block crushing 24 pieces (24 crushed), 1,500 particles poured, a column broken on the GPU (1 break), and a body broken into pieces as it falls, one of them broken again in the same step.

The lab checks the whole system (page, engine, GPU): `node game/lab/lab.cjs collapse --frames 150 --twice` runs the collapse twice and compares a digest of every frame. The digest covers poses, particles, bodies, breaks, dust, bricks, joints, contacts and the GPU's inputs, and it reports where the two runs part. `tools/check.sh lab` runs it.

Three things made lab runs differ besides the GPU's atomics, all fixed:
- the stats were read whenever the browser had mapped them (the lab now waits for them every frame, `flushStats`);
- the engine's maps used a seeded hasher (it is now fixed, `crate::Map`);
- the two `readPose` faults in the table above.

What is not covered:
- **Capacities.** Contacts or pairs beyond capacity are dropped by whichever threads come last. The lab counts such steps, and there are none.
- **Crush records.** When two heavy bodies crush the same light one in the same step, which one is named as the crusher can vary. The crush itself does not.
- **Hardware.** The same GPU and driver give the same result; different GPUs need not, because of float rounding in their math.

## Capacity

Pairs beyond `maxPairs` are lost, and a lost ground pair lets a piece fall through the ground. The default is now 24 per shape (it was 8, and a heap of chunks overflowed it: chunks fell to -7.8 m). `stats.pairsLost` and `lost` (steps that lost contacts or pairs) report it, and the lab checks it.

## Still to do

- **A part cut free can be undrawn for 2 to 5 frames** (24 November; `play_check hz144`: 52.85 m³ for up to 5 frames of 144 Hz). The page side: docs/LAB.md, 24 November, "Open". Not tried: drawing a new body at its birth pose.

- **Standing structure pushes what rests on it and is not pushed back within a step** (24 November; lab `qroofball`,
  frame 444: a 514 t ball at rest rose 3.8 mm and took 0.17 m/s as the floor under it moved; 17 kJ, the energy
  check's one failure there). Not proven to be the cause: docs/LAB.md, 24 November. Not tried: anything.
- **Firefox (naga) is checked for compiling only** (`engine/gpu/test_wgsl.cjs`; `find` had no final `return`, and
  Firefox made no GPU world at all). No run of the world on Firefox has been made.

- **A heavy part at rest on pieces thousands of times lighter is not settled** (19 November; lab `complex`: 290 t on
  three cabinets of 25 to 31 kg). The overlap is shared out while the light piece is awake and pushed out of the heavy
  part alone when it falls asleep: a 9 mm rise in one step. Tried: furniture crushing at a fifth of its material's
  strength (it is a shell); the part then creeps and hops on the next small pieces. Not kept. docs/LAB.md, 19 November.
- **A part broken off on the GPU and crushed within three steps is drawn 0.3 to 0.5 m off for one frame** (lab
  `qoffice`, frame 57). Not found.

- **A slab landing on a heap rocks (open, 18 November; GPU test `heap`, out of the default run with this note).** A slab of 2 x 0.3 x 2 m, 2.9 t, dropped flat at 6.6 m/s on 1,500 crumbs poured on the ground. Until the crumbs stopped throwing one another the heap had spread to a layer 0.18 m thick under the slab, the slab came down nearly flat (turning at 1.6 rad/s after the hit) and its centre went back up at **0.59 m/s** (limit 0.1 x the impact speed + 0.1 = 0.76). Now the heap is 0.27 m thick and uneven; the slab meets it at one corner a step before the rest (turning at 3.1 and 3.2 rad/s about both axes, its centre still going down at 5.3 m/s), and as the other corners land its centre goes up at **1.04 m/s** and it hops 5 cm for 0.2 s. None crushed, at rest by frame 53. It is the turning that lifts the centre, not a push of the heap: two changes to the contact through particles were tried and changed nothing to two decimals (no push out of an overlap; no friction anchor), and with the particles holding nothing (`NOSUP=1`) the slab goes to the ground with no speed upward. What would be needed: the particles under a corner give way as rubble does (each quarter of the shape bearing on its own particles' cross-section, not on all of them), or the contact made in the step the slab arrives rather than the next. Not done. The limit is not changed. (The scene's reading of the heap before the drop returns no particles while the world has no body, so the slab always starts 2.2 m up; `particles: 0, heapTop_m: 0` in its output is that.)

- **Crumbs that hop (closed, 15 November: the lab's measure, not the physics).** The check compared the distance a piece is DRAWN to have gone in a frame with its speed in the newest read-back. The drawn place is blended between two read-backs, so the speed that allows it is the greater of the two: a crumb that fell at 4.2 m/s in the older and was stopped in the newer had moved as far as 4.2 m/s goes and was held to 0. GPU test `pland` reads the same landings every step, from the GPU: nothing moves further than its speed. The lab's bound is now the greater speed of the two read-backs (its own commit; docs/LAB.md, 15 November). What follows is the note as it stood. In a collapse of 20,000 crumbs, two or three are drawn moving 0.3 to 0.5 m in one frame, upward, with a velocity that says otherwise (falling at 4.2 m/s and 0.41 m higher than a frame before: lab `masonry`, frame 14). Each is beside or inside a large body falling at the frame it happens. The same with and without the velocity bound (9 and 12 over six runs). What was tried: the separation limit of a particle against shapes (2 m/s past its own approach) taken per particle and sub-step and not per shape; GPU test `particles` then left a particle inside a body, which the rule "a particle never ends inside a body" forbids, and it was taken back out. Not found: what moves them. The lab's check counts them ("pieces moving further than their speed allows").

- **A fast-spinning chunk sinks into the ground (open, 9 November).** The GPU test `hullthrow` throws the hull library's cells; with six runtime hulls registered before it (`freehull` ran first), it drew another random sample, and library cell 39, thrown as body 75 spinning at 49 rad/s (corners at about 15 m/s), had a corner 7.6 cm inside the ground at frame 8 (limit 5 cm; the scene's own sample: 1.2 cm). Reproduce: `HTHROW_ALL=1 node engine/gpu/test_gpu.cjs freehull hullthrow`. The scene now throws the library's cells only, as written (the runtime hulls have `freehull`); the deeper case is not the margin (`ground_margin` covers w r dt) but the contacts: found once a frame, they cannot follow a body that turns 0.8 rad (47 degrees) in that frame, and corners that were not touching at its start come round into the ground. Candidates: a cap on the turn per frame (0.25 rad: 15 rad/s at 60 Hz) for bodies off the ground, or contacts found again in the sub-steps for the few spinning that fast.

- **Rebounds in the collapse (mostly closed, 14 November).** With the velocity bound and the pinch test, 240 frames, seeds 0 / 7 / 8: `collapse` 0 / 0 / 0 rebounds and 0 / 0 / 0 launches (9 / 2 / 1 and 0 / 1 / 3 without), `masonry` 1 / 1 / 1 and 0 / 0 / 0 (5 / 1 / 0 and 2 / 1 / 1). Open: the one rebound in each masonry run, above. Before: With shock propagation, the lab collapse (150 frames, seeds 0 / 7 / 8) counts 10 / 6 / 11 rebounds and 5 / 2 / 0 launches, against 13 / 26 / 16 and 7 / 6 / 8 without it. What is left is mostly pieces landing at 2–3 m/s and leaving at 1–2 m/s. Some may be real: the check reads the centre's speed, and a spinning piece that lands on one end can send its centre up. The launches left are spares and slivers at 5–14 m/s.

- **Mass ratio limit.** Across one contact the heavier body responds as if at most 20 times heavier (GPU test `sandwich`: a 7,700 t block landing on 11 t rubble went from a 2.52 m/s rebound to 0.37 m/s). The limit only takes energy from the heavy body: of its response at the limited mass, the part that brings its speed down to its least is kept, and the rest is taken at its real mass (`sandwich`: 0.37 → 0.03 m/s; dropping the whole change once it overshot zero lost the slowing too, and the block sank and rebounded at 1.9 m/s). Debug switches (`dbgBits`): 1 limits before splitting, 2 turns this off, 4 colours supports with the rest. The limit is between the masses as split over a body's overflow contacts, so a split body is never made lighter than the light one. Applied both ways and before splitting, it made momentum.
- **Supports first.** Contacts with the static world or a resting body are coloured in their own pass before the others, so a body's supports are solved in sequence, not left to the parallel overflow pass.
- GPU test `stumps` covers all three: a slab with columns lands at 6.3 m/s on 5 stumps carrying 450 pieces. Before, the slab rose at 2.1–8.7 m/s and the pieces were thrown at up to 15 m/s, varying from run to run. After, the slab rose at 0.11–0.17 m/s (rebound 0.63 m/s, as concrete's e = 0.1 gives) and the pieces at 1.5 m/s, over three runs. In the lab collapse (frame 50), the upper part of the building had gone from −6.34 to +1.11 m/s on landing and then kept rising.
- **Overflow passes** are sized by the recent overflow count and loop, like the colours (they were sized by the contact capacity, every round of every sub-step).
- **Depenetration** is at most 1 m/s (it was 3; pieces made overlapping the slab they were crushing into popped out).

- **Rare bursts in mixed rubble (open).** In the lab collapse, about one full run in four throws a pile apart that was at rest: at frame 138 of one run, a 2-brick piece that had just woken went from rest to 2,800 m/s and 20,000 rad/s in one frame (575 MJ); in another, 24 MJ at frame 164. Runs are not repeatable (GPU atomics), so the lab's flight recorder (docs/LAB.md) keeps the read-backs before a burst. GPU test `stress` (boxes of 1–12 bricks, 60 chunks and a 288-brick slab piled between stumps, then pieces dropped on the sleeping pile) fails in some runs: a chunk thrown at about 15 m/s (seed 2: 22.5 m/s falling back at frame 340). Box-only and chunk-only piles pass. Over seeds 1–8 (software GPU, 900 frames each), runs over 15 m/s: 1 of 8 with the current solver (seed 4, 20.9 m/s), 0 of 8 with the three solver changes above switched off (`dbgBits` 7), 1 of 8 with only the energy-only limit off (seed 6, 20.9 m/s), 0 of 8 with only supports-first off; this morning's solver passed seeds 2 and 5. At one failure in eight these do not tell the changes apart. Also tried: restitution off, contact reduction off (no effect on the bursts), and dropping warm starts of contacts made while a body slept (GPU test `wakeup`: passes with and without it, not kept). `stress` is outside the default run until this is fixed.
- **A piece squeezed under a slab 163 times its mass, with crushing off (open; GPU test `landing163`, not in the default run, 18 November).** A 7,680 kg slab and the 47 kg piece under it land on the ground together at 5 m/s, the solver alone (`crushing = false`). The piece is pressed 4.4 cm into the ground by frame 5 and is back out by frame 13; meanwhile it spins, its corners closing on the ground and on the slab at 3 to 5 m/s while its centre moves at under 0.4, and its velocity read each frame swings: with the restitution rule of 18 November, +0.05, +0.36, **+0.91**, −0.13 m/s in frames 3 to 6 (the test allows 0.6 up); before the rule, +0.02, +0.05, −0.54, +0.06, and its largest reading was the 0.50 of its own bounce in the first frame, which the rule has taken away (concrete on the ground at 5 m/s does not bounce). It was as deep in the ground before as after: the rule changed which way the reading swings, not the squeeze. It is not thrown: its highest place is where it came to rest. Tried: the trace of every frame and every contact, with the rule and without it (the numbers above); the energy limit of the same day changes nothing here (one shape each). No change to the solver has been tried. In the game such a piece is crushed in that step (pinch), which is the engine's answer to it; the other three cases of `landing` (960 kg at 7.7 and 2 m/s, 207 kg at 7.7) pass and stay in the default run.
- **Contact sag.** Stiffer contacts are still needed. A 20-high stack sags 7 mm (0.35 mm per contact), within budget but not the micron level of real masonry.
- **CPU reference.** The CPU engine does not have compound bodies, materials or restitution. It stays the reference for the box core only.
