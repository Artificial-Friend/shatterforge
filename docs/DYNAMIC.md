# The standing structure as a dynamic model (branch `test`, 10–11 November 2026)

The step, its rules and the solve were reworked on 11 November (docs/DYNPLAN.md, stages 0–2 and 6); this page describes them as they are now, and says where a number is the earlier build's.

**Question.** The standing building was analysed at rest after each change, a few frames late and at an unbounded cost (docs/STRUCTURE.md, Known limits: 50–370 ms at p95 in the page, 340 iterations for a tower on one column). Can it be stepped as a moving structure instead, in the step of a change?

**Answer.** Yes, with the solver the engine already has, and not with a GPU-style local solver.
- A dynamic step of the spring network is the static network with one more spring a cell. The multigrid of `solve_network` solves it within its usual tolerance in at most 16 iterations a step.
- A local solver (block Gauss-Seidel, a cell at a time: what vertex block descent is on a linear network, and what a GPU solves per colour) does not: on the 12-storey tower its forces are 15–50% wrong for the first 8 frames after a change, and 22–75% wrong 10 s after the weight is switched on.
- So the standing structure stays on the CPU, in the engine. Bonded bodies on the GPU are not the way to hold a building up.

**What it does not change.** The lab's `collapse` (the ground storey cut, the building falling as bodies) gives the same failed checks with it on and off. Those failures are in the falling pieces and the rubble (GPU contacts, and the engine breaking falling pieces a few frames late), which this does not touch. See "Lab" below.

## The model

The generalised-α step (Chung and Hulbert, 1993) on the network of docs/STRUCTURE.md, with the spectral radius at infinity `DYN_RHO` = 0.5 (α_m = 0, α_f = 1/3, γ = 5/6, β = 4/9). It is solved for the cells' move d = x' − x_p, x_p = x + h v + h² (1/2 − β) a:

```
(K + c_m M) d = ((f − α_f K x − α_m M a − C α_f v) / (1 − α_f) − C v_p − K x_p) ,   C = c_M M + c_K K
a' = d / (β h²),   v' = v_p + γ h a',   v_p = v + h (1 − γ) a
```

- K is the springs' stiffness, M the cells' mass and inertia (the brick's mass, a box's inertia), x, v, a the cells' displacements, velocities and accelerations (3 translations, 3 rotations).
- c_m M is a spring from each cell's centre to a fixed anchor. The dynamic step is therefore the static network plus one spring a cell and a changed right-hand side, and the rest of `analyse_warm` runs on it unchanged: the multigrid, the stones' beds, the joint and section checks.
- The solve starts from d = 0 and takes at most `DYN_IT` = 16 iterations. Its tolerance is measured against the loads, not against the step's own right-hand side (which is next to nothing near rest).
- **The first step from rest is implicit Euler** (α_m = α_f = 0, γ = β = 1). What a change lets go lands on cells their joints hold stiffly, a motion far faster than the step, and Euler damps it. Started with the acceleration the imbalance gives, the cell over a lost column left its first step at 4.4 m/s and 56 joints failed round it; it moves 0.2 mm.
- **Why second order.** The first build stepped by implicit Euler throughout, which damps every motion: a load let go peaked at 1.04 of its rest force where theory says 2. This step keeps what it resolves and damps what it cannot (a building's joints ring at kilohertz).
- **A cell's mass is its brick's.** It was its load over g: a load that is not a weight (a hit, a test's pull) then moved as mass. Rubble lying on a cell loads it and is not in its mass.

Against closed forms (`engine/src/dyn_closed.rs`; native, steps of 1/60 s and 1/240 s):

| Case | Theory | 1/60 s | 1/240 s | Gate (DYNPLAN) |
| --- | --- | --- | --- | --- |
| A 7.6 m concrete cantilever's weight let go: the tip's peak over its rest deflection | 2, less the damping | 1.877 | 1.906 | 1.8–2.0 |
| The same: its first period | 0.4072 s | +1.0% | +0.3% | 5% |
| The same: the moment at the held end, peak over rest | below 2 (not one mode's) | 1.779 | 1.783 | – |
| A 4.8 t block on a 7 m timber strip (one mass, one spring, 3 Hz), its weight let go: deflection and worst joint force over rest | 2 | 1.840, 1.827 | 1.867, 1.853 | 1.8–2.0 |
| The block arriving at the speed of a drop from h = 2 δ and 6 δ: peak over δ | 1 + √(1 + 2h/δ) = 3.236, 4.606 | −6.9%, −7.2% | −6.3%, −5.8% | 10% |
| A 4 m strip hinged at one end, let go: time to turn `DYN_TURN` | √(4 · 0.005 · L / 3g) = 52.1 ms | 50.0 ms (within a step) | 52.4 ms (+0.6%) | 10% |
| The cantilever ringing 10 s without the material's damping: energy over a period | constant | loses 0.61% at most, gains none | – | loses under 5%, never gains |
| One step of 100 s | the solve at rest | tip and worst joint within 0.006% | – | 0.1% |

- The energy at a step's end wobbles by 3.4 parts in 10,000 from one step to the next with the phase of the swing (the step conserves another quadratic form); over a period it only falls. The first step, Euler, takes 7% of what the sudden load put in.
- The cantilever at rest sags 61.5 mm at its last brick's centre; beam theory gives 62.3 mm. Before 11 November the solve at rest gave 53.1 mm (see "The solve").

## Plastic joints: bending that stays (13 November)

Until now a joint over its strength broke, so a part stood as built (bent by millimetres) or fell. A joint of a ductile material that **bending** takes over its strength now yields instead (`structure::Plastic`; the dynamic structure only).

- **Yield.** The moment it carried in the step that took it over, brought down to its strength, is its yield moment. From there it turns against `PL_H` = 2% of its elastic stiffness, about the turn it has made, and its ratio is reported as 0.999.
- **It keeps the turn.** When it turns back it is elastic again about where it is, and yields again only past what it had hardened to.
- **It breaks** when it has turned by its material's ductility, or if its ratio passes `PL_OVER` = 1.5 all the same.
- **What does not bend, breaks:** if the joint is over its strength without its moments (shear, tension, crushing), it fails as before.
- **Yielding joints do not wear.** They are at their strength because they yield, and their turning is counted against their ductility.

| Material | Ductility (rad) | |
| --- | --- | --- |
| Steel | 0.12 | bends |
| Reinforced concrete | 0.06 | bends a little, then breaks |
| Timber | 0.015 | hardly |
| Masonry, stone, glass | 0 | break |

- **The hardening is a game's stand-in, and says so.** At 2% a concrete joint carries 1.75 times its yield moment when it breaks; a reinforced section alone has 1.1 to 1.25. The rest stands for what a linear model has not: the tension a sagging slab or beam develops along itself (membrane and catenary action), which is what lets a real floor hang by a tenth of its span.
- **You asked for concrete that does not bend.** Plain concrete would not; the buildings' concrete is reinforced (its flexural strength is the reinforcement's), and a floor hanging over lost columns hangs by it. Set `ductility(CONCRETE)` to 0 and it breaks at once.
- **A joint's state changes after a step, not in it.** A load that lasts one step cannot make a joint turn; hits are therefore spread over `KICK_TIME` (below).

Against closed forms (`dyn_closed`, native, 1/60 s):

| Case | Theory | Result |
| --- | --- | --- |
| A 3.5 m concrete strip, a load on its end raised over 8 s to 1.1 times what takes its held joint to its strength; then taken off | the joint turns (1.1 − 1) / `PL_H` = 5 times its turn at yield, 0.0067 rad, and keeps it | keeps 0.0065 rad (−3.6%): the end is down 33.9 mm loaded (12.9 elastic), 21.0 mm unloaded; 1 joint yields, none fails |
| The same strip in masonry | breaks at 1 / 1.1 of the load, having bent only elastically | breaks at 0.92 of the load, 3.5 mm down; none yields |

In buildings (`dyn_tests`, native):

| Test | Result |
| --- | --- |
| `a_frame_that_lost_columns_hangs_bent` | the complex building built 5 storeys tall, the ground columns of two grid lines shot out: 737 joints fail and part comes down; what stands has sagged 80 mm on up to 79 yielding joints, worst joint 0.999, at rest after 110 steps. Solved at rest by the same cut: no sag |
| `a_hit_bends_steel_and_breaks_brick` | a sideways hit of 120 kN s on a ground column of that steel frame: 6 joints yield (20 fail); 60 kN s on the masonry building's wall: 93 joints fail, none yields; 3 kN s on the column: nothing |

**The range of the linear model is wider for it:** `DYN_TURN` = 0.03 rad and `DYN_MOVE` = 0.25 m (0.005 and 0.05 before), so that a part hanging on yielded joints is not taken to have left. At 0.03 rad a part is drawn 0.05% too long. A settling step may still move a cell only 0.005 rad and 5 cm.

**Hits are loads of the step** (DYNPLAN stage 5). A hit on the standing structure faster than `KICK_SPEED` = 1 m/s reaches the dynamic structure as its impulse over `KICK_TIME` = 0.1 s on the cell nearest it; the checks at rest that `impacts` ran for it are the solve at rest's only. Crushing at the contact is unchanged.

**Drawn where it is, without the view.** With the dynamic structure the engine writes the lots that have moved 2 cm from where they were built (`VIEW_BEND`; set_param 8 at 2), and the page draws them there at true size in their own colours; a brick whose cells are 5 mm apart is cut along them. **A part that leaves a sagged lot starts as a body with the structure's velocity there** (`view_motion`; DYNPLAN stage 4); it started at rest. It still starts where it was built: the bodies and what they land on are as built, and a part put down by its sag started inside what is under it (the lab's `collapse` had 101 energy gains with the shift, for 15). So a sagged part jumps back by its sag, centimetres, as it lets go. Not solved.

## The solve

Found by the closed-form tests and by comparing the game's solve with the same solved to 1e-12 (`structure::TOL_TEST`, or `SF_TOL=1e-12 cargo test ...`). All four change the solve at rest as well.

| What | Before | Now |
| --- | --- | --- |
| A weak spring to the ground (1e-7 of the cell's stiffest) on every cell's displacements and turns | the 7.6 m cantilever of 31 cells sagged 53.1 mm (theory 62.3), its worst joint read 1.152 | on the turns; on the displacements only of a cell its joints do not hold in some direction: 61.5 mm, 1.313 |
| The coarsest multigrid level | 30 sweeps of block Jacobi | factored (Cholesky), up to `DIRECT` = 64 cells: a structure that small is solved in one iteration |
| Coarse blocks | every cell of a bin, rigid | the cells of a bin joined by joints of at least `STRONG` = 1e-3 of what holds the weaker one: parts on open beds or softened joints move against their neighbours |
| Tolerance (r z over its start) | 1e-5 | 1e-7 |

The complex building's worst joint, standing and with its corner column shot out (the lab's `complexcol`; `dyn_tests::the_solve_is_converged_where_parts_stand_on_soft_joints` and `the_complex_building_comes_to_rest...`):

| | Standing | Column out | Iterations (5 solves at rest) |
| --- | --- | --- | --- |
| As it was (10 November) | 0.917 | 0.855 | 189 |
| The first three changes, 1e-5 | 0.448 | 0.889 | 93 |
| 1e-6 | 0.416 | 0.824 | 139 |
| 1e-7 (now) | 0.415 | 0.819 | 197 |
| 1e-8 | 0.415 | 0.817 | 260 |
| 1e-12 | 0.416 | 0.817 | 1,045 |
| Dynamic, now | 0.415 | 0.822 | 163 in 120 steps |

- **The 10 November numbers for this building were wrong by a factor of two standing** (0.917 for 0.416), and the "4% lower than at rest" in that day's limits compared two unconverged solves. With the solve converged, at rest and dynamic agree to 0.4%.
- Its arch under the loggia without its keystone: worst 0.76 and one joint lost at 1e-5, 0.51 and none from 1e-6 on.
- Coarsening by stiffness, the ground storey gone but one column, most iterations in a solve at rest (1e-7): 288 for 1,167 (4-storey frame), 240 for 780 (4-storey masonry), 174 for 712 (12-storey). On the sound complex building it costs: 95 iterations for 67.
- **Cost.** The complex building's analysis now costs 2.7 times the frame's a node (44.7 µs against 16.4), over phase 2's gate of 2 (`complex::tests::analysis_cost_a_node...`, out of the default run with this note). The frame's first solve: 40 iterations for 62 before.

## Rules added

- **Rest.** A structure whose cells' speeds, weighted by mass, are under `DYN_REST` = 0.5 mm/s, with no cell over `DYN_REST_CELL` = 5 cm/s and none watched, is calm. A calm structure **settles**: steps of `DYN_SETTLE_H` = 1 s from rest (implicit Euler), with the stones' beds found again from closed, until one converges with no cell out of balance by more than `DYN_SETTLED` = 3% of the largest load; then it rests, and its velocities are dropped. After `DYN_SETTLE_MAX` = 20 in a row that did not, it is not at rest: it goes on in steps of real time and tries again a second later. **No rest is forced** (13 November; see the next rule).
  - Without settling the complex building rested short of its equilibrium: worst joint 0.435 for 0.855 (as then solved).
  - A settling step that would move a cell out of the linear range has found a mechanism, a second of its fall in one step: it is refused, and the part falls in steps of real time.
  - Calm asks that the step converged (13 November). A step that stops at its cap has found the stiff motions and not the slow sag, so a structure whose steps all stop there moves slower than it should and looks calm. A step after one that stopped at its cap may take twice the iterations, up to `DYN_IT` × 2^`DYN_BOOST` = 128, and one that converges gives half back.
- **What rest leaves out of balance is not a force, up to what the tolerance can leave.** The solve stops at its tolerance; what each cell's loads and springs leave unbalanced when the structure rests (`Dyn::bias`) is taken off the loads of the steps that follow. Taken for real, the standing 12-storey tower never rested (9,600 iterations in 600 steps).
  - It is at most `DYN_BIAS` = 3% of the cell's own load, or `DYN_BIAS_FLOOR` = 0.5% of the structure's largest load if that is more (the standing tower's solve leaves up to 449 N on a cell, 0.39% of its largest load).
  - **Until 13 November it had no limit, and rests were forced** on structures that had not settled in 20 steps or whose settling step was refused. A rest forced out of balance then held every cell where it was: reported from the page as upper storeys that "just fly in the air" over a ground storey shot away. `dyn_tests::a_rest_writes_off_no_more_than_the_tolerance` (the complex building shot through from the west, 700 shots): 8 rests forced and a cell held against 4.65 times its own load before; none and 3% after.
- **At rest nothing is solved.** No velocities and no cell out of balance by more than `DYN_QUIET` = 0.1% of the largest load: the cells stay where they are.
- **Damping**, Rayleigh: in proportion to mass, `DYN_DAMP` = 1 /s, and to stiffness, `DYN_DAMP_K` = 1.6 ms. Together 4% of critical at 4 Hz, where they are least, 8% at 1 Hz and 6% at 10 Hz. Stiffness damping does not touch a part that moves as one piece, so a falling part falls. With mass damping alone the 4-storey frame, a corner column gone, rang at 25–80 mm/s for the 4 s tried.
- **A cell is known between steps by where it is** (its corners), not by its brick: a brick cut by a blast becomes new bricks, and its cells the blast did not touch keep their displacement and velocity. A fracture cell or free piece is known by its brick. A cell that is new is put where its neighbours hold it, by a solve of the new cells alone, and starts from rest. (By brick, every cell of a cut brick started again from no displacement beside neighbours 6 mm down.)
- **Leaving the structure.** The network is linear (small displacements): a part turned by t stretches by t² / 2 of its length for nothing, 1.25 × 10⁻⁵ at 0.005 rad. A cell that has turned `DYN_TURN` = 0.005 rad or moved `DYN_MOVE` = 5 cm from where it last rested is beyond that range, and is watched:
  - after `DYN_WATCH` = 0.15 s, a cell a whole range further out than when it crossed has left (from rest a part falls 5 cm in 0.1 s and 31 cm by 0.25 s). The joints between it and what stands are cut (ratio `DYN_CUT`), and the game detaches it as a falling piece;
  - one that after twice as long has not gone a quarter of a range further, or is slower than `DYN_REST_CELL` (a speed a falling part passes in 5 ms), has found a new resting place, and rests there. Without the speed rule the 4-storey masonry building left on one pier watched 3 or 4 cells creeping at 2–8 mm/s for 600 steps;
  - joints between two cells beyond the range, watched or gone, are not checked: what they show is the turning.
  - Measured from zero instead of from rest, the city's first dynamic step cut a timber floor loose: at rest it has turned 0.005 rad at its bearings under its own weight.
  - Taken to have left at once, the lab's `ball` failed (see "Lab"): cracked pieces by the holes, hanging by their softened joints, sag past the range and stay.
- **Rubble weighs between its lists.** The page sends the rubble lying on the structure once a second; the dynamic structure keeps the last list until the next (`rest_begin`). Weighing only in the step after its list, it was a load of some kN coming and going every second, and each time the building settled anew: the lab's `complexcol` had 97 of 99 structure steps at the cap and never settled; `dyn_tests::rubble_on_a_resting_building...` had 60 of 90 steps moving and 4 rests forced, now 3 and none.
- **A settling step's tolerance is measured against the loads**, as every step's. Against its own right-hand side, the little left out of balance, it asked for seven digits of next to nothing and stopped at the cap 20 times in a row.
- **Beds keep their state** between steps of real time: a stone's bed that opened or hinged stays so into the next step, and an open bed pressed together again closes. A settling step finds them again from closed.
- **A lot not solved yet is solved at rest first.** Stepped from no displacement, its weight would land on it all at once.
- **The step.** The time since the last structure step, between 1/120 s and `DYN_MAX_DT` = 0.1 s. A structure at rest that is changed now moves from now, however long ago its last step was.

## Where it is

- `engine/src/structure.rs`: `Dyn` (velocities, resting places, beds between steps), `analyse_dyn`, and `solve_network` with a cap on its iterations.
- `engine/src/game.rs`: `Game::dynamic`, `Game::dynst` (a `Warm` and a `Dyn` per lot), `structure_step` (each lot stepped on its own; a lot still moving stays active), `Game::moving`.
- `engine/src/wasm.rs`: `set_param(7, on)`, `structure_dyn(f)`.
- The page: setting "Dynamic structure" (on by default); a structure step that reports the structure still moving asks for the next; the HUD shows the sag and the fastest cell.
- The lab: `SF_STATIC=1 node game/lab.cjs ...` runs a scenario with it off.

## The engineer's view (key V)

Every brick is coloured by how near its worst joint or section is to its strength (blue unloaded, green at a third, yellow at 0.7 where joints begin to wear, red at strength), and the building is drawn where the structure's solution puts it, its displacements 1, 10, 100 or 1,000 times as large (setting "Displacements drawn"; a building sags millimetres).

- The engine writes it with each structure step when asked (`set_param(8, on)`; `Game::view_out`): each analysed brick's ratio, and every free cell with its centre, displacement and rotation. Switching it on has every lot analysed once.
- The page's copy of the engine takes it (`view_in`, `view_apply`) and remeshes the chunks of the bricks it names (`mesh::build_view`): faces are cut along a brick's cells, so that a slab drawn sagging bends, and each vertex moves by the mean of the cells that hold it.
- It needs the dynamic structure for the displacements; with it off the colours alone are drawn.
- A lot is written when the view is first asked for and when its step changed something; the page remeshes only the chunks of the bricks named. Before (reported from a real machine: the 8 × 8 city at 3 frames a second with the view on): every analysed lot was written at every step, the lots with furniture or rubble on their floors every second, and the page remeshed from the first chunk named to the last. Test `the_view_of_a_city_at_rest_is_written_once` (native): the first step writes 3,240,610 numbers, drawn in 640 ms (960 chunks); a step at rest writes nothing; a column shot out on one lot writes 13,674, drawn in 2–3 ms (3 chunks).
- Displacements are drawn at their true size by default (× 1): the world then differs from the plain one in colour only. Faces are cut along the cells only above × 1.
- 3 × 3 city: 7,539 bricks and 32,819 cells, 475 kB a full view. The 4-storey frame: 3,420 quads as built, 8,633 in the view.
- **What has left the structure has one colour**, violet (`VIEW_FREE`), off the scale of stress: falling pieces, crumbs, rubble. In their materials' colours the view seemed to switch off wherever a building came down (reported 13 November; page check `view`).

## The spike (`engine/src/structure/dyn_spike.rs`)

`cargo test --release spike_tower -- --ignored --nocapture` (also `spike_tower4`, `spike_complex`). Native, release, the build machine, one thread. The network is the one `analyse` solves, read through a test hook. Solvers: `Gs(n)` is n block Gauss-Seidel sweeps a step, `Mg(n)` n multigrid-preconditioned CG iterations, `Exact` the step solved to 10⁻¹⁶; "× 6" is 6 sub-steps a frame. Force error is the largest difference of any spring's force, over the largest spring force.

**12-storey concrete frame: 9,458 cells, 31,708 springs.**

Standing from the static solution, 3,600 frames:

| Solver | Drift (µm) | ms a frame |
| --- | --- | --- |
| Gs(1) | 0.045 | 1.26 |
| Gs(1) × 6 | 2.89 | 8.16 |
| Mg(1) | 0.043 | 10.88 |

Weight switched on at frame 0, from no displacement: force error against the static solution.

| Solver | Frame 1 | 6 | 30 | 120 | 600 | ms a frame |
| --- | --- | --- | --- | --- | --- | --- |
| Gs(2) | 0.990 | 0.952 | 0.676 | 0.438 | 0.753 | 2.05 |
| Gs(8) | 0.983 | 0.894 | 0.365 | 0.765 | 0.223 | 7.96 |
| Gs(1) × 6 | 0.963 | 0.652 | 0.715 | 0.461 | 0.610 | 7.75 |
| Gs(4) × 6 | 0.931 | 0.466 | 0.703 | 0.186 | 0.309 | 26.47 |
| Mg(1) | 0.836 | 0.455 | 0.224 | 0.100 | 0.054 | 12.18 |
| Mg(3) | 0.614 | 0.164 | 0.133 | 0.036 | 0.001 | 18.14 |
| Exact | 0.553 | 0.173 | 0.009 | 0.016 | 0.000 | 172.50 |

A ground-storey column cell taken out (the cell above moves 0.438 mm to its new resting place): force error against the exact dynamic step.

| Solver | Frame 1 | 2 | 4 | 8 | 15 | 30 | 60 | 120 | ms a frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gs(1) | 0.510 | 0.312 | 0.292 | 0.250 | 0.196 | 0.217 | 0.148 | 0.099 | 1.20 |
| Gs(16) | 0.216 | 0.247 | 0.223 | 0.167 | 0.153 | 0.039 | 0.056 | 0.050 | 13.86 |
| Gs(4) × 6 | 0.148 | 0.142 | 0.200 | 0.098 | 0.065 | 0.082 | 0.042 | 0.051 | 23.36 |
| Mg(2) | 0.120 | 0.104 | 0.143 | 0.058 | 0.052 | 0.029 | 0.016 | 0.023 | 17.78 |
| Mg(4) | 0.053 | 0.071 | 0.047 | 0.049 | 0.021 | 0.016 | 0.014 | 0.012 | 22.72 |

- The exact step's largest spring force peaks at 1.04 of the new static one: implicit Euler at 1/60 s damps the dynamic overshoot of a sudden column loss (2 in the textbook case) almost entirely.
- The 4-storey frame (3,306 cells) shows the same order: Gs(2) is still 0.456 wrong at frame 600, Mg(3) 0.000; Mg(4) costs 6.65 ms a frame.

**Why 16 iterations and not 4.** The force error above is against the largest force in the building. A joint beside the cut is judged on its own force: with 6 iterations the 4-storey frame's worst ratio in the first step after a corner column went was 0.467, where the converged step gives 0.335 (and 3 iterations gave 0.721 for three steps). With 12 it was 0.334. To its usual tolerance the solve needs 21 iterations in that step and 13–19 in the next three (`dyn_tests::a_column_gone_...`, with `DYN_IT` 3, 6, 12, 24, 60).

## Engine tests (`engine/src/dyn_tests.rs`, `cargo test --release dyn_ -- --nocapture`)

Native, release, the build machine, 11 November. The closed forms are under "The model".

| Test | Result |
| --- | --- |
| `a_standing_building_costs_no_iterations_and_does_not_drift` | 12-storey tower, 600 steps: 0 iterations, 0 failures, displacement unchanged (6.575 mm). 15.2 ms a step, all of it building the network |
| `a_column_gone_is_answered_within_the_steps_it_takes_to_move` | 4-storey frame, a corner column blasted out. At rest: 1 round, 40 iterations, worst joint left 0.333. Dynamic: at rest after 31 steps, worst 0.333; first steps 0.335, 0.331, 0.332, 0.337 |
| `a_tower_on_one_column_comes_down_in_bounded_steps` | see the table below |
| `the_complex_building_comes_to_rest_after_a_column_is_shot_out` | the lab's `complexcol`, a step every 4 frames. At rest: worst 0.415 → 0.819. Dynamic: 0.415 → 0.822, at rest again in the step after the last shot, 163 iterations in all, 0 joints failed either way |
| `the_solve_is_converged_where_parts_stand_on_soft_joints` | see "The solve" |
| `rubble_on_a_resting_building_does_not_keep_it_moving` | see "Rules added" |
| `a_city_is_stepped_lot_by_lot` | 3 × 3 city: first step 848 ms (9 lots at rest), then 0.14 ms a step at rest. A column shot out on lot 0: only that lot moves, at rest after 29 steps, slowest step 84 ms |
| `nothing_crumbles_where_the_solve_at_rest_stands` | 20 seeded cases (one to three 0.7 m blasts) on the five kinds: the solve at rest stands with its worst joint under 0.8 in 16, and the dynamic structure stands in all 16, worst joints within 0.002. |
| `the_same_cut_gives_the_same_fall` | the three buildings below, twice: the same to the joint. (At 1/30 s for 1/60 s: the test below) |
| `a_rest_writes_off_no_more_than_the_tolerance` | see "Rules added" |
| `the_same_joints_fail_at_half_the_frame_rate` (not in the default run) | joints failed and biggest piece at 1/60 and 1/30 s: 139 / 124 and 1,646 / 2,014; 278 / 279 and 1,901 / 1,820; 159 / 159 and 3,572 / 3,586. Gate 5%: met by the tower only |
| `a_building_warns_before_it_goes` (not in the default run) | first joint past 0.7 to first part gone: 0.05, 0.017, 0.017 s. Gate 0.3 s: not met, and wrong for this case (see "Limits") |
| `the_view_draws_the_structure_where_its_solution_puts_it` | see "The engineer's view" |

The ground storey blasted away but one column:

| Building | | Joints failed | Steps | Biggest falling piece (bricks) | Most iterations in a step | Slowest step (ms) | In all (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-storey concrete | at rest | 84 | 7 | 2,030 | 288 | 527 | 1,248 |
| | dynamic | 139 | 387 | 1,646 | 92 | 232 | 14,976 |
| 4-storey masonry | at rest | 130 | 5 | 2,037 | 245 | 462 | 883 |
| | dynamic | 278 | 28 | 1,901 | 128 | 302 | 2,115 |
| 12-storey concrete | at rest | 128 | 3 | 3,573 | 174 | 742 | 1,377 |
| | dynamic | 159 | 17 | 3,572 | 127 | 451 | 1,676 |

- (13 November's build, with plastic joints and steps of up to 128 iterations.) The 4-storey frame takes 387 steps to come to rest: parts hang and creep on yielded joints before they go.
- The slowest dynamic step is 44 to 65% of the slowest solve at rest, 232 to 451 ms: over DYNPLAN stage 3's gate of 150 ms.
- On 10 November (implicit Euler, the solve as it was) the dynamic 4-storey buildings broke into pieces of 1,317 and 800 bricks with 516 and 575 joints failed.

## Lab (software GPU)

docs/LAB.md, 11 November, has the runs and the tables. In short, `tools/compare.sh` on four scenarios, dynamic against at rest on the same build:

| | `ball` | `complexcol` | `keystone` | `nibble` |
| --- | --- | --- | --- | --- |
| Failed checks | none / none | none / none | none / none | 3 / 1 |
| Engine jobs p95 (ms) | 60 / 94 | 37 / 96 | 44 / 42 | 124 / 2,112 |
| Slowest engine job (ms) | 89 / 109 | 117 / 96 | 90 / 100 | 2,991 / 2,660 |
| Structure steps at the cap | 44 of 52 | 5 of 18 | 2 of 34 | 88 of 120 |
| Joints failed | 22 / 4 | 0 / 0 | 5 / 4 | 3,061 / 3,716 |

- Where the solve at rest stalls (`nibble`), the engine's p95 is a seventeenth and the structure's jobs take 9.9 s of the run for 103 s.
- `nibble` still fails two checks it passes at rest (frames drawn above and below what exists), and its slowest jobs are an `impacts` job (3.0 s) and a structure step of 2.1 s in which thousands of cells leave at once.
- Most moving steps stop at the cap of 16 iterations.
- `collapse`, `complex`, `city` and `mixed` (the building cut from the ground) fail the checks they failed before; their slowest job is `impacts`. The quick suite and the page checks are in docs/STATUS.md.

## Limits

- **The structure only.** Falling pieces are still rigid bodies on the GPU, broken by their plans and by the engine's impacts a few frames late. They do not bend, and a part that leaves bent becomes a body with its built shape. docs/DEFORM.md is the plan; `Dyn::free` and one closed-form test are its groundwork.
- **Small displacements.** Until a part leaves, displacements are small (the springs are small-turn springs). Joints of steel, reinforced concrete and timber yield and keep their bend ("Plastic joints"); nothing buckles.
- **How much stands is the model's strengths, in both modes.** Since 14 November the floors carry 2,000 N/m² besides themselves (timber floors 1,000) and concrete's shear strength is 3 MPa: the buildings stand at 0.15 (timber) to 0.64 (12-storey frame) of their worst joint's strength (docs/STRUCTURE.md, "The buildings with their floors loaded"). A 4-storey frame with its ground storey gone west of −8 m stands at 0.83; west of −6 m at 0.89, losing 3 joints dynamically; west of −4 m it comes down there (`dyn_tests::what_a_cut_leaves`, native).
- **Cost while moving and at rest.** Every step builds the network and its multigrid again: 15 ms for the 12-storey tower before the first iteration, also when it finds the structure at rest. Keeping them between steps that change nothing is not done (DYNPLAN stage 3).
- **Detached parts leave with the structure's velocity there** (`view_motion`, 13 November), not with its bend.
- **A hit is a load of the step** (an impulse over `KICK_TIME`), and it is the rigid stop's impulse: the GPU stops a body against the structure before the engine breaks what was hit, so a hit bounces first and breaks second (docs/LAB.md, 13 November, the ball). A hit that wakes a structure at rest is 2.8% too strong (docs/DEFORM.md, measured).
- **The frame rate changes which joints fail on the way down**, by 9 to 36% between 1/60 and 1/30 s, though not what falls (the piece within 2.3%).
- **No warning in a building already falling.** Left on one column, the first part is gone 0.02–0.05 s after the first joint passes 0.7: it is in free fall, 5 cm down in 0.1 s. Warning time exists where a building is near its limit and wears (seconds: `game::tests::a_building_left_near_its_limit...`), and that is where creaks and visible sag belong.
- **A building at its limit is not predictable.** The mixed building cut on two sides stands at 0.91–0.99 of strength, and whether wear brings its upper storeys down in the next 20 s turns on the solve's last digits (docs/LAB.md, 11 November).
- **Not on a real GPU yet.** Every timing here is native or the software GPU.

## Next

docs/DYNPLAN.md has the stages and where each stands.
