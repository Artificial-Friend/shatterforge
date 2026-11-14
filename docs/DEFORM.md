# Falling pieces that deform: where it stands and the plan (14 November 2026)

Not started as a feature. This is the preparation asked for on 14 November: what a falling piece is today, what
was measured to find out whether the engine's dynamic step can carry it, the options, the one recommended, and
stages with a gate each. The standing structure's bending is docs/DYNAMIC.md ("Plastic joints").

## What a falling piece is today

- **A rigid body on the GPU** (`gpu/gpu_world.js`): a compound of boxes and hulls with one position, one
  rotation, one velocity. It cannot change shape. A steel beam that comes off lands as straight as it left.
- **It breaks, it does not bend.** A hard contact reaches the engine through `Game::impacts` (game.rs): the
  material near the contact is crushed by the impact's energy (1/2 J v), and the piece is then checked *at rest*
  under its own inertia, every brick loaded by its mass times the piece's deceleration and held at the
  contact (`structure::analyse`). Joints over their strength crack or part; the piece splits into new rigid
  bodies. There is no time in that check and no yielding: a steel member either holds or parts.
- **What already bends:** the standing structure, in the dynamic step (`structure::analyse_dyn`): joints of
  steel, reinforced concrete and timber yield at their strength, turn, harden by `PL_H` and break at
  `ductility` (steel 0.12 rad, concrete 0.06, timber 0.015; masonry and glass 0, they break). A hit is an
  impulse over `KICK_TIME` (0.1 s). A part that leaves the structure (`DYN_TURN` 0.03 rad, `DYN_MOVE` 0.25 m)
  becomes a rigid body with the shape it was built with: **the bend it had is lost at that moment.**

So three things are missing: a piece bending when it lands, a piece keeping the bend it left with, and a hanging
part going over from "structure" to "piece" without a jump.

## What was measured (the groundwork, in the code since 14 November)

`Dyn::free` (structure.rs): cells with no way to a support are stepped too, held by their mass alone. Before, the
step fixed every cell that no support holds. Not used by the game. One closed-form test:

`dyn_closed::a_free_strip_hit_in_the_middle_leaves_with_the_impulse` (native, `cargo test --release a_free_strip
-- --nocapture`): a steel strip 6 m × 1 m × 0.25 m (12 cells, 1,884 kg), held by nothing, no gravity, hit in the
middle across its length with 0.94 kN s (to leave at 0.5 m/s), stepped 0.22 s at 1/60 s.

| | No damping | The structure's damping |
| --- | --- | --- |
| Share of the impulse the strip carries after 0.22 s | 1.028 | 0.871 |
| Expected | 1 (the step gives 37/36 = 1.028, below) | 0.846 × 37/36 = 0.870 |
| Distance gone | 85.5 mm | 78.5 mm |
| The middle ahead of the ends, at most | 0.38 mm | 0.38 mm |
| Joints failed | 0 | 0 |

What this says:

1. **The step carries a free body.** The same network, the same joints, the same solver; the piece leaves with
   the impulse and its middle goes first (it bends, elastically at this size of hit).
2. **Three things in the step are wrong for a piece in flight**, each found by this test:
   - *Damping on mass slows flight.* Rayleigh damping (`DYN_DAMP` = 1 /s) acts on a cell's whole velocity: a
     piece loses 13% of its speed in 0.17 s. For a standing structure the whole velocity is the vibration; for
     a piece it must be the velocity *within* the piece.
   - *New cells are put at rest by a solve at rest*, which has no answer for a body nothing holds (it ran its
     2,000 iterations and returned 10^27 m). Skipped with `Dyn::free`.
   - *Displacements are measured from where the piece was built.* The springs are small-turn springs and the
     leaving rule ends at 0.25 m and 0.03 rad: a free piece can be followed for a few centimetres of flight and
     no tumbling. The test stops at 86 mm for that reason.
3. **A hit on a structure at rest is 2.8% too strong** (37/36): the first step from rest is implicit Euler,
   which gives that step's force whole, and the next step counts the same acceleration again as the "old" one
   by 1 − γ = 1/6. A sixth of one of a hit's six steps. Small, known, and it applies to the standing structure's
   hits as well; the test holds the step to that number so that a change shows.

## The options

| | A. The engine steps the piece (recommended) | B. Rigid parts joined by GPU joints | C. Shape-matched soft body on the GPU |
| --- | --- | --- | --- |
| What it is | The GPU keeps flying the piece as one rigid body. When a contact is hard enough, the engine steps the piece's own network for the length of the impact, in the piece's frame, with the plastic joints that exist; the bent shape is written back to the body. | A piece of ductile material is several rigid bodies joined by hinge constraints with a yield torque, solved on the GPU with the contacts. | Each piece a cloud of particles pulled toward a best-fit rigid shape, with plasticity as a change of the rest shape. |
| Physics | The same model, numbers and tests as the standing structure. One calibration. | A new joint model on the GPU; its yield and break must be calibrated to agree with the engine's. | Not structural: no moments, no sections. Looks soft; a beam does not form a hinge. |
| Cost | Engine time only for pieces being hit hard (a few per frame), 10 to 300 cells each. | Every joint every substep, in the solver's colours: more constraints in the costliest stage. | High: particles for every piece. |
| Flight and tumbling | Free: the GPU does it, rigidly. | Free, and pieces flex in flight (mostly invisible). | Free. |
| What is hard | Two-way timing: the GPU stops the piece rigidly *in the step of the hit*, the engine bends it some frames later (the same lag the ball has: docs/LAB.md 13 November, "it bounces first and breaks second"). | Stiff joints between light bodies in a Jacobi-style solver: the known weak point of GPU rigid solvers (stretch, jitter). The mass-ratio limit in docs/GPU.md applies. | Does not fit the engine. |
| Risk | Medium, and the unknowns are measurable natively, without a GPU. | High: a second structural model, checkable only on a GPU. | Wrong tool. |

**Recommended: A.** It keeps one structural model. It can be built and gated natively, where I can run it
(closed forms, no GPU). It costs nothing for pieces that are not being hit. And it gives the visible result that
was asked for: a steel beam that lands across something stays bent; concrete cracks and hinges on its bars;
brick breaks. What A does *not* give is flexing in flight and a soft first contact; B would, at the price above.
If stage 3's gate fails (the lag is visible), the fallback is a narrow form of B: a single GPU hinge at the one
joint the engine says has yielded, not a general jointed-body system.

**What I do not know, and stage 1 and 3 are there to find out:** whether the bend arriving two to four frames
after the rigid stop reads as bending or as a pop (it cannot be judged on a software GPU at 4 s a step; it needs
your machine), and how many pieces per second a real collapse sends through the step (the lab can count it now,
before anything is built).

## Sizes: when does a piece bend at all?

Estimates from the engine's own numbers (not runs): a steel member drawn as a 0.25 m box is a rolled section,
78.5 kg/m, with a plastic moment of 250 MPa × 0.12 × 0.25³/4 m³ = **117 kN m**. A 6 m beam (471 kg) that lands
flat across one support at its middle has m g h to lose; if all of it went into one hinge, the hinge would turn
m g h / Mp:

| Fall | Energy | Hinge turn if one hinge took it all | |
| --- | --- | --- | --- |
| 0.5 m | 2.3 kJ | 0.02 rad (1°) | barely seen |
| 3 m | 13.9 kJ | 0.12 rad (7°) | steel's ductility here: at the limit |
| 10 m | 46 kJ | 0.39 rad (22°) | past it: it would break |

This is an upper bound (crushing and the other side take energy too), but it says two things. Bending will be seen
for falls of about 1 to 5 m and for pieces landing on pieces; and **steel's ductility of 0.12 rad, set for
joints of a standing frame, is low for a falling member**: real rolled steel folds through a radian without
parting. Stage 2 has to revisit that number with a test, in its own commit (rule 3).

## Stages

Each stage ends at a gate: a test with a number from theory, run natively unless it says otherwise. A stage
whose gate fails stops the plan there, with the result written down.

**0. Groundwork: done (14 November).** `Dyn::free`; the free strip test; the three faults above named.

**1. A piece stepped in its own frame.** The piece's rigid motion (the mean velocity and spin of its cells,
mass-weighted) is taken out before the step and put back after; the network sees only the deformation.
Damping acts on the velocity within the piece. Displacements are measured from the piece's own pose, so
flight and tumbling of any length cost nothing and the leaving rule does not apply.
- Gate a: the free strip keeps its impulse to 0.5% with the structure's damping on (0.871 today).
- Gate b: hit off-centre, linear *and* angular momentum to 1% (J and J × r).
- Gate c: the first free-free bending period of the strip against 2π / (4.730² √(E I / (ρ A L⁴))), within 5%
  (the cantilever's period is within that today, `dyn_closed`).
- Gate d: energy never rises without a load, over 10 s of free vibration.
- Also: count, in the lab's `collapse` and `ball`, the pieces per frame whose contacts pass a threshold, and
  their cells. That number decides stage 4's budget before stage 2 is written.

**2. Landing.** A piece's hard contacts (already reported: body, point, direction, impulse, approach speed)
become kicks on its network, held where it touches for the length of the impact. The check at rest in
`Game::impacts` stays for brittle pieces (masonry, glass, plain concrete), which is right for them and cheap;
ductile pieces (steel, reinforced concrete, timber) take the step.
- Gate a: a steel beam dropped flat across a support at mid-span: the hinge's turn against the rigid-plastic
  answer (energy in over Mp, less what the test measures as crushed), within 20%, at three heights.
- Gate b: the same beam of masonry breaks and no joint turns; of reinforced concrete cracks and turns up to its
  ductility.
- Gate c: momentum of the pieces after equals momentum before plus the contact impulses, to 2%.
- The ductility of steel in a falling member: measured against the table above, changed if wrong, own commit.

**3. The bent shape goes back to the body.** The piece's boxes are rewritten at their bent places (the body
keeps its id, centre of mass, velocity and spin; inertia recomputed); the mesh is redrawn from them. This is the
first stage that needs the GPU, and the page.
- Gate a (GPU test scene, software GPU): a body whose shape is rewritten at rest stays at rest; in flight keeps
  its momentum to 1%; no new overlap deeper than the solver's slop.
- Gate b (lab scenario `beamdrop`): a steel beam onto a concrete block from 3 m ends bent between 3° and 10°,
  rests, and is the same twice (determinism).
- Gate c (your machine, by eye, with the lab's Copy JSON under `bench/`): does it read as bending? This is the
  decision point for the fallback above.

**4. Budget.** Only pieces above a threshold of impulse and of size take the step; a cap on pieces per frame,
the rest take the cheap check; a bent piece that comes to rest is done. The debris budget (the Physics preset)
removes pieces oldest first and does not know about any of this.
- Gate: lab `collapse`, software GPU against software GPU: the structure and piece jobs' p95 no more than 20%
  above today's.

**5. Leaving bent.** A part that leaves the structure becomes a body with the shape it had when it left
(stage 3's write, at birth), with the velocities it has now (`view_motion` gives them already).
- Gate: `dyn_tests::a_frame_that_lost_columns_hangs_bent`, the part that falls after sagging 80 mm: its body's
  boxes are within 5 mm of the cells' last places, and the lab counts no jump at the hand-over.

## What this does not cover

- **A hit that goes on through what it breaks** (the ball that bounces before the floor gives: docs/LAB.md,
  13 November). It is the same lag as stage 3's risk and should be fixed first or together: it changes what a
  landing piece's contact impulse *is*.
- Crushing of a piece's sections into folds (a car body): that is shells, not beams and joints.
- Bending in flight (whip, vibration of a falling beam). Not visible at these sizes; option B would give it.
