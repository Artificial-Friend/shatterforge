# The standing structure as a dynamic model (branch `test`, 10 November 2026)

**Question.** The standing building was analysed at rest after each change, a few frames late and at an unbounded cost (docs/STRUCTURE.md, Known limits: 50–370 ms at p95 in the page, 340 iterations for a tower on one column). Can it be stepped as a moving structure instead, in the step of a change?

**Answer.** Yes, with the solver the engine already has, and not with a GPU-style local solver.
- A dynamic step of the spring network is the static network with one more spring a cell. The multigrid of `solve_network` solves it within its usual tolerance in at most 16 iterations a step.
- A local solver (block Gauss-Seidel, a cell at a time: what vertex block descent is on a linear network, and what a GPU solves per colour) does not: on the 12-storey tower its forces are 15–50% wrong for the first 8 frames after a change, and 22–75% wrong 10 s after the weight is switched on.
- So the standing structure stays on the CPU, in the engine. Bonded bodies on the GPU are not the way to hold a building up.

**What it does not change.** The lab's `collapse` (the ground storey cut, the building falling as bodies) gives the same failed checks with it on and off. Those failures are in the falling pieces and the rubble (GPU contacts, and the engine breaking falling pieces a few frames late), which this does not touch. See "Lab" below.

## The model

Implicit Euler on the network of docs/STRUCTURE.md, for a step of h seconds:

```
(K + M / h²) x' = f + (M / h²) (x + h v),    v' = (x' − x) / h
```

- K is the springs' stiffness, M the cells' mass and inertia (a box's, from its size), x and v the cells' displacements and velocities (3 translations, 3 rotations).
- M / h² is a spring from each cell's centre to a fixed anchor (translation m / h², rotation I / h²). The dynamic step is therefore the static network plus one spring a cell and a changed right-hand side, and everything else of `analyse_warm` runs on it unchanged: the multigrid, the stones' beds, the joint and section checks.
- The solve starts from x + h v and takes at most `DYN_IT` = 16 iterations.

What follows from it:

| | At rest (before) | Dynamic |
| --- | --- | --- |
| A mechanism (a tower on one column) | a near-singular solve: 362–670 iterations, 1.0–1.6 s | K + M / h² is never singular: 16 iterations at most, and the part starts to move at the speed its weight gives it |
| A change | the new equilibrium in one solve, whenever it arrives | the forces rise over the frames the building takes to move |
| A building at rest | 0 iterations, but the network is built (31 ms, 12 storeys) | no step is asked for |
| Time | rounds of failures, as fast as the jobs come | a step is a step of time: wear, failures and movement share one clock |

## Rules added

- **Rest.** A step that needs no iteration found the structure at rest: the cells stay where they were. A step that converged with its fastest cell under `DYN_REST` = 0.5 mm/s ends the motion: velocities are dropped. (Without it the complex building, a corner column shot out, crept at 0.1–0.3 mm/s for the 8 s tried: the slow modes of stones on open and hinged beds, which implicit Euler hardly damps.)
- **Damping.** In proportion to mass, `DYN_DAMP` = 1 /s: a 1 Hz sway is damped at 8% of critical. A part in free fall loses 2% of its speed a step at 60 Hz for the tenth of a second before it leaves the structure.
- **Leaving the structure.** The network is linear (small displacements): a part turned by t stretches by t² / 2 of its length for nothing, 1.25 × 10⁻⁵ at 0.005 rad. A cell that has turned `DYN_TURN` = 0.005 rad or moved `DYN_MOVE` = 5 cm from where it last rested is beyond that range, and is watched:
  - after `DYN_WATCH` = 0.15 s, a cell a whole range further out than when it crossed has left (from rest a part falls 5 cm in 0.1 s and 31 cm by 0.25 s). The joints between it and what stands are cut (ratio `DYN_CUT`), and the game detaches it as a falling piece;
  - one that after twice as long has not gone a quarter of a range further has found a new resting place, and rests there;
  - joints between two cells beyond the range, watched or gone, are not checked: what they show is the turning.
  - Measured from zero instead of from rest, the city's first dynamic step cut a timber floor loose: at rest it has turned 0.005 rad at its bearings under its own weight.
  - Taken to have left at once, the lab's `ball` failed (see "Lab"): cracked pieces by the holes, hanging by their softened joints, sag past the range and stay.
  - At 0.01 rad and no watch the 4-storey frame left on one column broke into pieces of 1,242 bricks at most (1,171 joints failed), against 1,800 bricks (910 joints) at 0.005.
- **Beds keep their state.** A stone's bed that opened or hinged stays so into the next step, and an open bed pressed together again closes. Started closed at every step, the complex building never came to rest (lab `complexcol`: 70 steps in 280 frames, each re-opening them).
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
- Falling pieces and rubble are not part of the view: they are drawn as always, so a building brought down leaves it.
- 3 × 3 city: 7,539 bricks and 32,819 cells, 475 kB a full view. The 4-storey frame: 3,420 quads as built, 8,633 in the view.

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

## Engine tests (`engine/src/dyn_tests.rs`, `cargo test --release dyn_tests -- --nocapture`)

Native, release, the build machine.

| Test | Result |
| --- | --- |
| `a_standing_building_costs_no_iterations_and_does_not_drift` | 12-storey tower, 600 steps: 0 iterations, 0 failures, displacement unchanged (6.530 mm). 31–36 ms a step, all of it building the network |
| `a_column_gone_is_answered_within_the_steps_it_takes_to_move` | 4-storey frame, a corner column blasted out. At rest: 1 round, 29 iterations, worst joint left 0.332. Dynamic: at rest after 37 steps, worst 0.330; first steps 0.335, 0.332, 0.332, 0.334 with 16, 16, 14, 13 iterations |
| `a_tower_on_one_column_comes_down_in_bounded_steps` | see the table below |
| `the_complex_building_comes_to_rest_after_a_column_is_shot_out` | the lab's `complexcol`, a step every 4 frames. At rest: worst 0.917 → 0.855. Dynamic: 0.917 → 0.817, moving for 13 steps (0.9 s), 0 joints failed either way |
| `a_city_is_stepped_lot_by_lot` | 3 × 3 city: first step 704–990 ms (9 lots at rest), then 0.2 ms a step at rest. A column shot out on lot 0: only that lot moves, at rest after 95 steps, slowest step 31 ms |
| `the_view_draws_the_structure_where_its_solution_puts_it` | see "The engineer's view" |

The ground storey blasted away but one column:

| Building | | Joints failed | Steps | Biggest falling piece (bricks) | Most iterations in a step | Slowest step (ms) | In all (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4-storey concrete | at rest | 69 | 3 | 2,032 | 670 | 1,147 | 2,269 |
| | dynamic | 516 | 114 | 1,317 | 16 | 61 | 1,846 |
| 4-storey masonry | at rest | 128 | 3 | 2,011 | 457 | 1,026 | 1,673 |
| | dynamic | 575 | 116 | 800 | 16 | 74 | 1,325 |
| 12-storey concrete | at rest | 130 | 5 | 3,562 | 362 | 1,554 | 2,828 |
| | dynamic | 131 | 22 | 3,569 | 16 | 140 | 486 |

- The 12-storey tower comes down as at rest (131 joints against 130, the same piece), in a sixth of the time, and no step is longer than 140 ms where one took 1.6 s.
- The 4-storey buildings break up more: 516 and 575 joints against 69 and 128, in 1.9 s of steps. At rest, 64 joints fail a round, the worst first, and the column's go in the first rounds; stepped, the floors hang from one column until they are seen to fall, and crack meanwhile. Which is nearer the truth is not validated.

## Lab (software GPU)

| Scenario | Frames | With the dynamic structure | At rest (`SF_STATIC=1`) |
| --- | --- | --- | --- |
| `complexcol` | 300 | pass; 16 engine jobs, median 10.1 ms, max 52.4 ms | passes on phase 2 (docs/STATUS.md there) |
| `collapse` | 150 | fails: 2 frames drawn below what exists, 7 pieces moving too far, 2 rebounds, not settled; 57 engine jobs, median 6.1 ms, p95 299 ms, max 1,214 ms | the same four, the same counts; 54 jobs, median 5.7 ms, p95 323 ms, max 1,168 ms |
| `mixed` | 960 | fails the same five checks as on phase 2, the same counts | not run |
| `nibble` | 1,584 | fails 3 checks (4 frames below, 2 above, 5 jumps); engine jobs p95 107 ms, max 5,214 ms | fails 2 (2 below, 6 jumps); p95 1,629 ms, max 3,378 ms |

`ball` with the dynamic structure:

| | Checks | Joints failed | Pieces sent off | Structure jobs (of them at the cap) |
| --- | --- | --- | --- | --- |
| At rest | pass | 2 | 0 | 25 engine jobs in all |
| Dynamic, a cell beyond the range gone at once | fails: 1 piece moving further than its speed allows | 17 (14 of them cut) | 3 | 69 engine jobs in all |
| Dynamic, watched (`DYN_WATCH`) | pass | 15 (9 cut) | 4 | 66 (31) |
| The same with `DYN_IT` = 64 | pass | 17 (9 cut) | 5 | 83 (13) |

- Stepped, more comes away round the balls' holes than solved at rest. The count hardly changes with four times the iterations, so it is the model and not the cap: a piece let go by a hit drops onto its cracked, softened joints, and their forces pass what they carry at rest. Not validated.
- Half the structure's steps here stop at the cap (31 of 66): cracked joints are soft beside sound ones, and the multigrid converges slowly across them.

In `collapse` the building is cut from the ground at once and falls as bodies: the standing structure has little to answer, and the long engine jobs are the impacts on the falling pieces. The quick suite and the page checks are in docs/STATUS.md.

## Limits

- **The structure only.** Falling pieces are still rigid bodies on the GPU, broken by their plans and by the engine's impacts a few frames late.
- **Linear.** Displacements are small until a part leaves; nothing bends plastically or buckles.
- **Cost while moving.** Every step builds the network and its multigrid again: 31–36 ms for the 12-storey tower before the first iteration, 2.7 ms an iteration. Keeping them between steps that change nothing is not done.
- **A hinged bed stays hinged** until the lot is next solved at rest; only open beds close again.
- **Detached parts start from rest.** The cells' velocities are known when a part leaves, and are not yet given to the body on the GPU.
- **Steps that stop at the cap.** In a damaged building (cracked joints, fracture cells) half the steps of the lab's `ball` did not reach the solver's tolerance in 16 iterations, and 13 of 83 not in 64. Their forces are approximate, and a structure step reports it (`structure_dyn(7)`; the lab's timeline, `dyn.capped`).
- **A step is not bounded when much leaves at once.** The lab's `nibble` had one engine job of 5.2 s (3.4 s at most solved at rest), in a run where 3,053 cells were beyond the range at once; its p95 fell from 1,629 to 107 ms. It also draws 2 frames above what exists, which it does not at rest. Both open (docs/LAB.md, 10 November).
- **More fails than at rest** round damage (the lab's `ball`: 15 joints against 2) and in a building left on one column. Not validated against anything.
- **The complex building rests 4% lower** than at rest (0.817 against 0.855 at its worst joint) after the column shot: it stops under `DYN_REST`, and its beds keep the state they took on the way.
- **Not on a real GPU yet.** Every timing here is native or the software GPU.

## Next

1. A real machine: `collapse`, `mixed`, `nibble` and `city` with it on and off, and the benchmark of the city.
2. The network and its multigrid kept between steps.
3. The velocity of a part that leaves, to its body.
4. Falling pieces: more than one break plan a piece, and plans for landings on any side (docs/GPU.md, Breaking in the step of the hit). The spike says bonds solved on the GPU would not do it.
