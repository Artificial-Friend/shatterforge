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
- **Chunk pairs apart.** The chunk test needs both hulls' corners and directions and 16 clipped points per thread; in the same kernel as the boxes, every box pair paid for that too. GPU test `pile` (2,000 boxes, software GPU) checks the narrow phase against the broad phase: 260 against 49 ms with chunks and capacity sizing, 63 against 53 ms after both changes (57 against 49 before chunks). Lab `collapse`, software GPU, at the slowest step: narrow 5,483 ms (600-frame run before the change) against 23 ms, plus 0.04 ms for chunks (150-frame run after it); median GPU step 6,693 ms against 3,525 ms.
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

A light piece pressed by a body at least 20 times heavier is crushed on the GPU (kernel `crush`), when the impulse it took in a step is more than 1.5 times what its own approach momentum needs (it is being squeezed against something) and it approached at 1 m/s or more. From then on it collides only with the static world, so the heavy body goes through it. The page (`takeCrushed`) turns it into rubble particles, its bricks split into cubes of at most 0.4 m, placed from its current pose.

GPU test `crush`: a 7,700 t block landing on 24 pieces of 11 t crushes all 24 and comes down to the ground; a piece rammed from the side by a 600 t block is not crushed. In the lab's collapse (150 frames), launches went from 14 to 1 and bounces from 11 to 7.

**Pinching (tried, off by default; `GpuWorld.pinching = true`).** The crush test looks at one contact at a time and needs an approach of 1 m/s. A piece pressed between two bodies, or between a body and the ground, at almost no approach speed is never crushed by it. In the lab collapse a 49 kg piece took 76,000 N s in one step between the ground and a 773 kg body, spun at 30 rad/s, then flew up at 7 m/s.

Kernels `pinch_acc` and `pinch` sum each body's normal impulses as vectors and as magnitudes (fixed point, so in any order); the part that cancels is the squeeze. A piece is crushed when:
- it is at least 20 times lighter than what squeezes it (the static world counts as infinitely heavy);
- the squeeze is more than 3 m/s of its own momentum;
- the squeeze over the step, over its largest contact patch, passes the crushing strength.

GPU test `pinch` turns it on and passes, but it passes with it off too, because the crush test crushes those pieces once the block reaches them. In the lab collapse (150 frames, seeds 0 / 7 / 8, before shock propagation): rebounds 13 / 26 / 16 → 12 / 11 / 15, launches 7 / 6 / 8 → 4 / 2 / 2, but one energy gain in each run (0 before). With shock propagation: rebounds 10 / 6 / 11 → 9 / 17 / 16, launches 5 / 2 / 0 → 3 / 1 / 0. Three runs do not tell it apart from luck, and it adds energy gains on its own, so it stays off.

## Breaking in the step of the hit

The engine answers a hit about 4 frames late. So each new falling piece carries a break plan from it (`Game::plan_piece`): where it breaks first, and how hard a hit must be to break it there.

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

Particles still collide with a chunk's box.

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

The lab checks the whole system (page, engine, GPU): `node game/lab.cjs collapse --frames 150 --twice` runs the collapse twice and compares a digest of every frame. The digest covers poses, particles, bodies, breaks, dust, bricks, joints, contacts and the GPU's inputs, and it reports where the two runs part. `tools/check.sh lab` runs it.

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

- **Rebounds in the collapse (open).** With shock propagation, the lab collapse (150 frames, seeds 0 / 7 / 8) counts 10 / 6 / 11 rebounds and 5 / 2 / 0 launches, against 13 / 26 / 16 and 7 / 6 / 8 without it. What is left is mostly pieces landing at 2–3 m/s and leaving at 1–2 m/s. Some may be real: the check reads the centre's speed, and a spinning piece that lands on one end can send its centre up. The launches left are spares and slivers at 5–14 m/s.

- **Mass ratio limit.** Across one contact the heavier body responds as if at most 20 times heavier (GPU test `sandwich`: a 7,700 t block landing on 11 t rubble went from a 2.52 m/s rebound to 0.37 m/s). The limit only takes energy from the heavy body: of its response at the limited mass, the part that brings its speed down to its least is kept, and the rest is taken at its real mass (`sandwich`: 0.37 → 0.03 m/s; dropping the whole change once it overshot zero lost the slowing too, and the block sank and rebounded at 1.9 m/s). Debug switches (`dbgBits`): 1 limits before splitting, 2 turns this off, 4 colours supports with the rest. The limit is between the masses as split over a body's overflow contacts, so a split body is never made lighter than the light one. Applied both ways and before splitting, it made momentum.
- **Supports first.** Contacts with the static world or a resting body are coloured in their own pass before the others, so a body's supports are solved in sequence, not left to the parallel overflow pass.
- GPU test `stumps` covers all three: a slab with columns lands at 6.3 m/s on 5 stumps carrying 450 pieces. Before, the slab rose at 2.1–8.7 m/s and the pieces were thrown at up to 15 m/s, varying from run to run. After, the slab rose at 0.11–0.17 m/s (rebound 0.63 m/s, as concrete's e = 0.1 gives) and the pieces at 1.5 m/s, over three runs. In the lab collapse (frame 50), the upper part of the building had gone from −6.34 to +1.11 m/s on landing and then kept rising.
- **Overflow passes** are sized by the recent overflow count and loop, like the colours (they were sized by the contact capacity, every round of every sub-step).
- **Depenetration** is at most 1 m/s (it was 3; pieces made overlapping the slab they were crushing into popped out).

- **Rare bursts in mixed rubble (open).** In the lab collapse, about one full run in four throws a pile apart that was at rest: at frame 138 of one run, a 2-brick piece that had just woken went from rest to 2,800 m/s and 20,000 rad/s in one frame (575 MJ); in another, 24 MJ at frame 164. Runs are not repeatable (GPU atomics), so the lab's flight recorder (docs/LAB.md) keeps the read-backs before a burst. GPU test `stress` (boxes of 1–12 bricks, 60 chunks and a 288-brick slab piled between stumps, then pieces dropped on the sleeping pile) fails in some runs: a chunk thrown at about 15 m/s (seed 2: 22.5 m/s falling back at frame 340). Box-only and chunk-only piles pass. Over seeds 1–8 (software GPU, 900 frames each), runs over 15 m/s: 1 of 8 with the current solver (seed 4, 20.9 m/s), 0 of 8 with the three solver changes above switched off (`dbgBits` 7), 1 of 8 with only the energy-only limit off (seed 6, 20.9 m/s), 0 of 8 with only supports-first off; this morning's solver passed seeds 2 and 5. At one failure in eight these do not tell the changes apart. Also tried: restitution off, contact reduction off (no effect on the bursts), and dropping warm starts of contacts made while a body slept (GPU test `wakeup`: passes with and without it, not kept). `stress` is outside the default run until this is fixed.
- **Contact sag.** Stiffer contacts are still needed. A 20-high stack sags 7 mm (0.35 mm per contact), within budget but not the micron level of real masonry.
- **CPU reference.** The CPU engine does not have compound bodies, materials or restitution. It stays the reference for the box core only.
