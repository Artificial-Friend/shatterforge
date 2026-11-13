# Plan: the dynamic structure, better than solved at rest (branch `test`, 11 November 2026)

*"Where it stands" below is the plan's starting point (10 November's build); "Where each stage stands" is the state after the first pass through it.*

**Goal.** The standing structure stepped as a dynamic model (docs/DYNAMIC.md) becomes the only way it is solved, because it beats the solve at rest on everything a player notices: no stalls, a building that sags and then goes, the same cut giving the same fall, nothing crumbling without a visible cause.

**Order.** The engine's logic first (stages 1–6, all provable by native tests), then everything else (stage 7 on). A stage is done when its gates pass; a gate is a number.

**"Better than at rest" is measured, per scenario, on the same build:**

| Measure | Dynamic must be |
| --- | --- |
| Lab checks failed | no more than at rest, on any scenario |
| Slowest engine job | no slower than at rest |
| Engine jobs, p95 | at most a quarter of at rest, or under 20 ms (`tools/compare.cjs`) |
| Structure steps that stop at the iteration cap | under 5% |
| Closed-form dynamic tests (stage 1) | within the limits written there |

## Where it stands (docs/DYNAMIC.md, docs/LAB.md 10 November; software GPU and native)

| | Dynamic | At rest |
| --- | --- | --- |
| `nibble`, failed checks | 3 | 2 |
| `nibble`, engine jobs p95 / max (ms) | 107 / 5,214 | 1,629 / 3,378 |
| `ball`, joints failed | 15 | 2 |
| `ball`, structure steps at the cap | 31 of 66 | – |
| 12-storey tower on one column, slowest step (ms, native) | 140 | 1,554 |
| Peak force after a sudden column loss, over the new static force | 1.04 | – (theory for a load let go: 2) |
| `collapse`, `complex`, `city`, `mixed` | the same failed checks | the same |

## Where each stage stands (11 November; docs/DYNAMIC.md, docs/LAB.md of that day)

| Stage | Gate | Result | |
| --- | --- | --- | --- |
| 0 Scoreboard | `compare.sh` on `nibble`, `ball`, `complexcol`, `keystone`, tables in docs/LAB.md | done | **met** |
| | your six benchmark reports in `bench/` | none yet | open (yours) |
| 1 The step | the six closed-form tests | 1.88 and 1.84 (gate 1.8–2.0); period +1.0%; hinge within a step (+0.6% at 1/240 s); drop −7%; energy −0.6% a period, no gain; 100 s step 0.006% | **met** |
| | `dyn_tests` with their limits unchanged | pass | **met** |
| 2 The solver | steps at the cap under 5% in `ball` and `nibble` | 85% and 73% | **not met** |
| | forces at a capped step within 2% of the converged step, at the joint | not measured step by step; at rest the dynamic structure ends within 0.4% of the solve at rest, itself within 0.3% of 1e-12 | open |
| | (found on the way) the solve at rest was up to 2.2 times off on the complex building and 15% on a cantilever | fixed, tested | – |
| 3 No slow step | no step over 150 ms on the 12-storey tower | 198–263 ms | **not met** |
| | `nibble`, `ball`: slowest engine job below at rest's | `ball` 89 for 109 ms: met; `nibble` 2,991 for 2,660 ms: not (an `impacts` job; the structure's slowest step 2,055 ms) | **half** |
| | a step at rest under 1 ms | 15 ms (12-storey; a city's lot at rest 0.14 ms) | **not met** |
| 4 Leaving | `nibble`, `ball` fail no check that passes at rest | `ball` met; `nibble` draws 4 frames above and 2 below | **not met** |
| | the body's first velocity is its cells' | not started | open |
| | the hinged beam leaves when theory says | crosses the range at 50.0 ms for 52.1 (stage 1's test); the watch rule's 0.15 s after it is not tested against anything | half |
| 5 Hits in the step | all | not started; the step takes an impulse rightly (stage 1's dropped mass) | open |
| 6 The player | nothing crumbles: 20 seeded cases | 16 of 16 that stand at rest stand, worst joints within 0.001 | **met** |
| | the same cut, the same fall; 1/60 against 1/30 s within 5% | twice identical; the piece within 2.3%; the joints failed 9–36% apart | **half** |
| | 0.3 s of warning in the one-column cases | 0.02–0.05 s; the gate was wrong for a building in free fall | **not met**, gate to be rewritten for buildings that wear |
| | your eyes | – | open (yours) |

"Better than at rest", the table at the top, on the four scenarios compared:

| | `ball` | `complexcol` | `keystone` | `nibble` |
| --- | --- | --- | --- | --- |
| Failed checks no more than at rest | yes | yes | yes | no (3 for 1) |
| Slowest job no slower | yes | no (the first solve: 117 for 96 ms) | yes | no |
| p95 a quarter, or under 20 ms | no (60 for 94 ms) | no (37 for 96) | no (44 for 42) | yes (124 for 2,112) |
| Steps at the cap under 5% | no | no | no (6%) | no |

**13 November.** Reported from the page: parts held in the air, and the view off on fallen buildings. Both fixed (docs/LAB.md of that day). The first was this plan's stage 2 left open: steps that stop at the cap. A step may now take up to 128 iterations (steps at the cap in `ball`: 26 of 55, 47%, for 85%; `complexcol` 4 of 20), which makes stage 3 the first thing to do: no step over 150 ms is now 242–319 ms. It also showed that how much stands is a question for the model's loads and strengths, with three candidate changes listed there.

**The decision point is not reached.** The step and the model are right where they can be checked (stages 1 and 6's first gate, and the solve itself, which was not); the dynamic structure is not yet better than the solve at rest by this plan's own table. What stands between, in the order it should be done:

1. **Stage 3, the cost of a step that is not iterations**: the network and its multigrid kept between steps (15 ms a step at rest, most of a moving step), and a bound on what one step cuts and detaches (`nibble`'s 2 s step).
2. **Stage 2, the cap**: with the network kept, a step's time is its iterations, and more of them fit; or a stopping rule on the joints' forces. The gate (under 5% at the cap) may be the wrong one: what matters is the second gate, the force at the joint, which needs its test.
3. **Stage 4**, `nibble`'s frames drawn above and below, and the velocity to the body.
4. **Stage 5**, hits as loads of the step; it also removes the `impacts` job that is `nibble`'s slowest.
5. Stage 6's two open gates, rewritten: the frame rate against the joints failed, and warning for buildings that wear.

## Two kinds of test

| | Internal (run here, on every change) | Benchmarks (run by you, on the real GPU) |
| --- | --- | --- |
| What | `cargo test --release` (native), the GPU suite and the lab on the software GPU | the page's benchmark (key K) and the lab in the page (Copy JSON) |
| Good for | numbers that are computed: forces, iterations, counts of checks, which joints fail, determinism | frame time, steps a second, engine job times in the browser, how it looks |
| Not good for | wall-clock time in the page (software-GPU timings compare only with each other) | – |
| Where results go | docs/STATUS.md, generated | `bench/`, machine and date in the file name |

Every stage below names both. A stage whose benchmark is missing is not held up by it: the internal gates decide, and the benchmark confirms or reopens it.

## Stage 0. The scoreboard

So that "better" is one command and one table.

- `tools/compare.sh <scenario> [--frames n]`: runs a lab scenario with the dynamic structure and with `SF_STATIC=1`, and prints the table above for it (failed checks, engine jobs median / p95 / max, joints failed, pieces sent off, steps at the cap).
- The lab's report carries each engine job's time in the timeline (it has the structure jobs' `dyn` already), so the slowest job can be named.
- The page's benchmark report gains the structure's lines: steps, iterations, steps at the cap, slowest step, and whether the dynamic structure was on.
- **Internal gate:** `compare.sh nibble`, `ball`, `complexcol`, `keystone` run and their tables are in docs/LAB.md.
- **Your benchmark:** key K on the complex building, the 3 × 3 city and the 8 × 8 city, each with "Dynamic structure" on and off: six reports into `bench/`.

## Stage 1. The step itself is right

Today's step is implicit Euler: stable, but it damps nearly all of the motion (1.04 where theory says 2), so what the dynamic model adds is mostly timing, not dynamics.

- **Closed-form tests** (engine, native), as docs/STRUCTURE.md has for statics:

| Case | Theory | Gate |
| --- | --- | --- |
| A load let go on a column or a beam | peak force 2 × static | 1.8–2.0 |
| A cantilever's first period | 2π / (1.875² √(EI / ρAL⁴)) | within 5% |
| A beam hinged at one end, let go | reaches 0.005 rad at √(2 · 0.005 · 2L / 3g) | within 10% |
| A mass dropped from height h on a beam of static deflection δ | peak 1 + √(1 + 2h / δ) × static | within 10% |
| Energy, an undamped building left to ring for 10 s | constant | loses under 5% a period, never gains |
| A step of 100 s | the solve at rest | forces within 0.1% |

- **The integrator.** Replace implicit Euler by a second-order step with a set amount of damping (the generalised-α or Newmark family: the same matrix K + c M / h², another right-hand side), so that damping is the material's (`DYN_DAMP`, a few per cent of critical) and not the step's. Stay unconditionally stable.
- **Risk:** real dynamic overshoot fails more joints than today. That is the physics (a column lost suddenly is designed for with a factor of 2), and stage 6 decides how much of it the game keeps.
- **Internal gate:** the six tests pass; `dyn_tests` pass with their limits unchanged.
- **Your benchmark:** none.

## Stage 2. The solver converges in a damaged building

Half the steps in `ball` stop at 16 iterations, and 13 of 83 at 64: cracked joints are a millionth as stiff as sound ones beside them, and the multigrid's coarse cells straddle them.

- **Keep the network and its multigrid between steps** (also stage 3's cost): rebuild only the lot's part that changed.
- **Coarsen by stiffness:** a coarse cell does not join cells across a joint softer than its neighbours by a set factor (strength of connection), so soft joints stay visible on the coarse levels.
- **Hanging fragments out of the network:** a part held only by open or softened joints carries no load for the building; it is a body.
- **Measure before choosing:** an engine test that rebuilds the `ball` and `nibble` states natively (the store saved from the lab at a capped step) and counts iterations to tolerance.
- **Internal gates:** steps at the cap under 5% in `ball` and `nibble`; forces at a capped step within 2% of the converged step, at the joint and not only against the building's largest force.
- **Your benchmark:** none.

## Stage 3. No step is slow

- **Trace the 5.2 s job** in `nibble` (stage 0 names it): the solve, the cutting of 3,053 cells' joints, or the detaching.
- **A step's work is bounded whatever happens:** iterations (done), joints cut a step, pieces detached a step. What does not fit waits for the next step, and the part concerned stays watched.
- **No rebuild at rest or between unchanged steps** (31–36 ms for the 12-storey tower today before the first iteration).
- **Internal gates (native):** no step over 150 ms on the 12-storey tower in any `dyn_tests` case; `nibble` and `ball` slowest engine job below the solve at rest's; a step at rest under 1 ms.
- **Your benchmark:** key K on the 8 × 8 city with one building cut: engine job p95 and max in the report.

## Stage 4. Leaving the structure

- **The 2 frames drawn above what exists** in `nibble`: trace and fix, with a lab or page check.
- **Velocity to the body.** A part that leaves has its cells' velocities; the body on the GPU starts with their mean and spin. Today it starts from rest.
- **The watch rule against theory:** the hinged beam of stage 1 leaves at the time theory gives; a cracked piece hanging by softened joints (the `ball` case, as an engine test) does not leave.
- **Internal gates:** `nibble` and `ball` fail no check that passes at rest; a GPU or page test that the body's first velocity is the cells'.
- **Your benchmark:** the lab in the page, `nibble`, Copy JSON, dynamic on and off.

## Stage 5. Hits go into the step

Today a body landing on the standing structure reaches it as a load once a second, and as a separate analysis at rest in `impacts`.

- **Contact impulses as loads of the step** they happen in: J / h on the cells touched.
- **One path:** the `impacts` analysis of the standing structure goes; the dynamic step answers.
- **Internal gates:** the dropped-mass test of stage 1 through the game's interface; `ball` and `far` pass; `complexroom` passes.
- **Your benchmark:** key K (it fires three balls): steps a second and worst frame, on and off.

## Stage 6. What the player notices

The model is now right; this stage decides what the game shows of it. Each choice is a constant with its reason beside it.

- **Nothing crumbles without a cause.** A set of 20 seeded damage cases (shots, blasts, a column out) on each building kind where the solve at rest stands with its worst joint under 0.8: the dynamic structure stands too. Where real overshoot would fail a joint, the game may keep a share of it (a factor on the dynamic part of the force), written down as a game rule and not as physics.
- **The same cut, the same fall.** Determinism over 150 frames with the dynamic structure in every scenario of the quick suite, and across a change of frame rate (a step of 1/60 s and of 1/30 s give the same joints failed, within 5%).
- **A building warns.** Between a joint passing 0.7 and the collapse there is time and something to see: sag at true size near failure, and the creak events (joint id, ratio) that audio will use. Gate: in the one-column cases at least 0.3 s between the first joint over 0.7 and the first piece leaving.
- **Internal gates:** the three above.
- **Your benchmark:** your eyes. The same damage in both modes on the complex building and a 12-storey frame, and which looks like a building. This is the one gate only you can pass.

## Decision point

After stage 6 the table at the top is filled for all 12 scenarios.
- All rows pass: the solve at rest is kept only for a lot's first solve, the setting goes, and docs/STRUCTURE.md is rewritten around the dynamic step.
- A row fails and stays failing: it is written down (rule 5), and the default is decided on that row.

## Stage 7 on: everything else

In this order, each with its own plan when it starts.

1. **Falling pieces.** Crushing before the solve (the 602 t floor stopped by a 158 kg column stub, docs/LAB.md 9 November) and more than one break plan a piece. This is where `collapse`, `complex`, `city` and `mixed` fail, with either structure.
2. **The burst** in resting piles (one `stress` run in four).
3. **Scale on the real GPU:** the city benchmark at 3 × 3, 5 × 5, 8 × 8 with a collapse; lots analysed in parallel.
4. **The engineer's view:** falling pieces in it, the predictor ("what fails if I cut here").
5. **The engine's shape:** crates, scene API, Rust on wgpu, import.

## Not in this plan

- Bonded bodies on the GPU for the standing structure: the spike ruled them out (docs/DYNAMIC.md).
- The multigrid on the GPU: only if stage 3's gates cannot be met on the CPU.
- A real-world reference for how much should break: none is at hand. Stage 1's closed forms and stage 6's rules stand in for it, and say so.
