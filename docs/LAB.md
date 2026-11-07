# Lab: scenario checks as data

The lab runs scripted destruction scenarios and reports what happened as numbers and pass/fail checks, not as screenshots. It runs the same way in a browser on a real GPU and headless on a software GPU.

## How it works

The lab drives the page's own frames on a **virtual clock** of 60 frames per second, however long each frame really takes. Each lab frame runs the page's real logic:
- engine jobs
- physics steps
- read-backs
- interpolation
- the instance lists handed to the renderer

Two kinds of work reach the page after a fixed number of frames, as they would on a real machine:
- read-backs, `readLag` frames after they were issued (default 2)
- engine jobs, `jobLag` frames after they were sent (default 4)

The engine runs inline, and the worker's time is simulated by `jobLag`.

The sub-steps are fixed at the default of 12, because adaptive sub-steps would make the runs depend on the machine. Settings are reset to their defaults for the run. The report lists them.

A software GPU and a fast one therefore go through the same sequence of frames. The only differences are the wall time and the small nondeterminism of GPU atomics. Each scenario starts from a fresh page load.

## Checks

Every frame, against what would be drawn:

| Check | What it catches |
| --- | --- |
| **Drawn volume below what exists.** The building mesh, debris and particles drawn add up to less than the static bricks and GPU bodies at any moment over the last few frames. | holes, missing parts, flicker |
| **Drawn volume above what exists.** | a part drawn twice (still in the building mesh and already as debris) |
| **Pieces that disappear and come back** within 30 frames | flicker of single pieces |
| **Pieces that move further than their speed allows** between two frames | jumps, wrong matching of pieces between read-backs |

At every read-back, against the physics:

| Check | Limit |
| --- | --- |
| Invalid poses (NaN) | 0 |
| Pieces below the ground (centre under −1 m) | 0 |
| Rebounds over half the impact speed. A piece falling faster than 2 m/s moves up at more than half that speed at the next read-back. | 0 (concrete restitution is 0.1) |
| Launches: pieces thrown up faster than 5 m/s | 0 |
| Energy gains: the kinetic plus height energy of the bodies present at both read-backs rises by more than 5% (+2 kJ), beyond the energy the page gave parts it cut free (below) | 0 |
| Roof fall, for cut scenarios | at least one storey |
| Volume lost, not to the tools: the building's volume less what stands, falls, lies as rubble or particles, and what the player's blasts took (`lost_m3` in the report says by what) | at most 2% (fines) |
| Body slots freed twice | 0 |
| Volume made: the opposite of volume lost; a part doubled | at most 0.5% |
| Crumbled share of a big part when it breaks: of a falling part of 50 m³ or more, the share of its volume that leaves as crumbs and particles (worst) | at most 0.25 |
| For `nibble`: joints cracked before the upper storeys fall (their standing volume down a tenth), seconds from the first failure to that fall, and the share of the upper storeys' volume that came down | at least 50 cracked, 2 s, half |

Also reported: the most rubble lying on the building's floors at once (t, from the weight the page sends the engine each second) and the joints cracked (brick and reinforced concrete crack first and hold; impacts refine bricks, so this counts the new joints too).
| Settled: everything asleep before the end | yes |

Pushes from outside are exempt for a while:
- after a cut or blast: 30 frames
- after a ball is fired: 120 frames

The rebound, launch and energy checks do not count during those windows.

**Energy given to parts cut free.** When the engine breaks a falling body, the parts cut free keep falling at their own velocity: the speed before the hit, less the energy the section that gave way took to crush. The GPU had already stopped the whole body rigidly by then (the engine answers a few frames later), so this energy comes from the page, not the solver. The lab counts it (`energy_given_by_cuts_J`, 58 MJ in the first 60 frames of `collapse`). The energy check allows for it while it spreads to the bodies those parts hit, used up as it shows and fading by 3% per read-back. Without it, the check counted the upper part (cut free at −5 m/s at frame 53) landing on the pieces under it as 0.6–0.9 MJ made from nothing. The rebound and launch checks are unchanged.

**Flight recorder.** The lab keeps every body's pose, velocity, spin and sleep state at the last 4 read-backs, and the last 3 frames of impact events. The first 2 bursts of a run (a launch, or an energy gain over 1 MJ, while quiet) keep a copy, under `bursts` in the report. A rare burst can then be read from any full run, without knowing beforehand where to trace.

## Report

Each scenario gives:
- the checks
- physics:
  - roof heights
  - dust as a share of the building
  - breaks
  - particles made and crushed
  - settle time
- performance:
  - GPU ms: median, 95th percentile and maximum, plus the stages of the slowest step
  - JS ms per frame
  - peak counts of bodies, awake bodies, contacts, particles and drawn instances
  - work counters (contacts × sub-steps, particles × sub-steps)
- the first 12 **events** of each kind, with frame numbers and details, for example a bounce event:
  - the body and its size
  - when the body was made
  - its vertical speed before and after
  - whether the page pushed it
- a **timeline** of the engine jobs: frame, operation, new bodies, breaks and static updates
- a **time series** every 10 frames
- with `--trace a:b`: the position, velocity and spin, at every read-back, of the bodies made in that window, plus the biggest body

## Scenarios

| Name | What |
| --- | --- |
| `collapse` | concrete frame, 4 storeys: the ground storey cut (36 blasts), collapse, settle (600 frames) |
| `ball` | concrete frame, 3 storeys: two steel balls into the facade (360) |
| `masonry` | masonry, 3 storeys: ground storey cut (600) |
| `timber` | mass timber, 3 storeys: ground storey cut (600) |
| `mixed` | the mixed building (brick, steel, concrete, timber), 5 storeys: ground storey cut (600) |
| `rubble` | concrete frame, 3 storeys: cut, then a blast into the rubble (720) |
| `nibble` | the mixed building, 5 storeys: its ground storey eaten away with the 0.7 m tool, shot by shot, as a player does: the west wall, the south and north fronts, the western columns, the east wall, the eastern columns (about 1,650 frames) |

Suites: `quick` (collapse, ball) and `full` (all seven).

## Running it

**In the browser (real GPU).** Open the settings panel (O) and press **Run lab checks, quick** or **full**. The page reloads once per scenario. At the end the panel shows the summary, and **Copy** gives the full JSON.

**Headless (software GPU).** From the repository root:

```
node game/lab.cjs quick                      # or full, or scenario names
node game/lab.cjs collapse --frames 150      # a shortened run
node game/lab.cjs collapse --trace 36:58     # trace the bodies made in frames 36-58
node game/lab.cjs ball --read-lag 3 --job-lag 6
node game/lab.cjs collapse --frames 150 --twice   # determinism: twice, compared frame by frame
```

The runner prints the summaries and writes `lab-results/<scenario>.json` and `all.json`. It exits with 1 when a check fails. It needs Playwright and `three` (set `SF_NODE_MODULES` to a `node_modules` folder that has `three`). On the software GPU a 4-storey collapse step takes about 4 s, so 150 frames take about 10 minutes.

## Determinism

A lab run is now deterministic: the same build gives the same run, frame for frame.

`--twice` runs each scenario twice in one browser and compares a digest of every frame. The digest covers:
- the poses and particles read back;
- the bodies, breaks, dust, bricks, joints, contacts and particles made;
- the awake count, the events and the jobs queued;
- the GPU's inputs and frame.

It reports the first frame where the runs part, and which of these differ. `tools/check.sh lab` runs the collapse twice for 150 frames.

Debugging switches:

| Switch | What it adds to the digest |
| --- | --- |
| `SF_DET_DUMP=1` | hashes of the GPU's body, solver and shape records, and of the contacts' colours as a set |
| `SF_DET_AT=f` | the raw records at frame f, from both runs, written to `det_raw.json` |
| `SF_DET_ROWS=1` | every frame's contacts with their colours, bodies and reduction flag |

Runs of 150–200 frames:

| When | Result |
| --- | --- |
| before | parted within the first frames |
| with the GPU fixes (docs/GPU.md, "Determinism") | parted at frames 91–127 |
| after the last fix (the reduction table's hash) | four runs of 200 frames agreed in every frame (two pairs, run side by side) |

Why runs differed:
- **Stats arrived when the browser mapped them**, so a frame saw this step's events, crushes and counts or the last one's, by wall time. The lab now waits for them every frame (`gpu.flushStats()`).
- **`readPose` read the velocities after the poses**, some steps later, and computed shapes' poses from JS records that a GPU break changes when its stats come back. It now copies the bodies, velocities and the GPU's own shape records in one read and computes the poses from those. (Copying the JS records at the read instead drew pieces of a just-broken body around its old centre: 6,709 jumps in `collapse --frames 150`; the lab's jump check caught it.)
- **The engine's maps had a seeded hasher.** It is now fixed (`crate::Map`).
- **GPU races.** Atomics' order fed choices and sums; see docs/GPU.md, "Determinism".

Determinism holds on one machine and browser. A different GPU or driver rounds differently, so its runs are deterministic among themselves but differ from these.

## What it found first

- **Particles flying across the scene.** A particle slot freed and reused while a read-back was in flight was matched by its new generation, so the old particle's position was blended into the new one: two crumbs moved 6 m and 15 m in one frame. The page now matches particles by their generation at the time the read-back was issued.
- **The building bouncing.** The upper part of the building (about 1,900 t) landed on rubble crushed off its own base and rebounded at up to 2.5 m/s, throwing the rubble up at 4 m/s. Light pieces wedged between a body hundreds of times heavier and the ground do not converge in a few solver iterations. Restitution, push limits, more sub-steps and more colours did not change it. Contacts now limit the mass ratio to 20: the heavier body responds as if at most 20 times heavier. The GPU test `sandwich` (a 7,700 t block landing at 5.7 m/s on 24 pieces of 11 t) went from a 2.52 m/s rebound to 0.34 m/s, and the 20-box stack is unchanged.
- **Pieces popping out.** Pieces cut from a falling part were made overlapping the slab they were crushing into (the engine answers a few frames after the hit), and the solver pushed them out at its 3 m/s limit. The limit is now 1 m/s.
- **The upper part rising after it landed.** A trace of frames 44–62 (`collapse --frames 70 --trace 44:62`) showed the upper part (1,554 bricks) going from −6.34 to +1.11 m/s on the stumps at frame 51 and then rising again (0.44, then 0.64 m/s); the roof rose 0.5 m over frames 60–90 of the full run. The mass-ratio limit was making momentum, and the body's supports were solved in the parallel overflow pass. Both are fixed (docs/GPU.md, "Still to do", and GPU test `stumps`). After the fixes the upper part lands from −6.0 to −0.56 and −0.17 m/s (`--frames 60 --trace 44:56`). On 150 frames: rebounds 7 → 4, launches 1 → 0, energy gains 6 → 2.
- **Open.** Pieces of 11–27 bricks next to the upper part are still thrown out at 2–4 m/s when it lands (frames 50–57), and one energy gain remains at frame 54, just after the upper part breaks.
- **Open (mixed).** With the ground storey cut, the mixed building comes down only partly in 600 frames: roof 22.39 → 19.9 m against the 3 m the check asks (software GPU, full length). With structure checks every frame (a bug: the rubble interval was 1 ms) it fell 5.6 m in 240 frames; it also shows 11 rebounds and 18 energy gains, of the same kind as the collapse's.
- **Open (irregular fracture).** With convex cells in the engine (docs/STRUCTURE.md), the collapse (150 frames) shows 1–4 frames of drawn volume below what exists, by 0.06–0.4 m³. With the page's former skin of cells it was 3–5. With cells off (`SF_NO_FRAC_CELLS=1`), there are none. When a hole appears, the lab lists the bodies and particles that exist but are not drawn (`undrawn` in the hole event).
- **Varies by run (collapse energy gains).** Full-length collapse runs today counted 2, 24 and 49 small energy gains (the last after irregular fracture: 4–250 kJ each from frame 235 on, no launches). The GPU's atomics make each run differ; a count from one run is not a trend.

## The foundation changes, compared (November 2026)

Each line is one collapse run of 150 frames on the software GPU (`node game/lab.cjs collapse --frames 150`), counted from its events. Runs vary (GPU atomics), so there are three of each. The raw lines are in `lab-results/cmp_*.txt`.

**Convex cells against boxes** (spring-network solver, no GPU breaks):

| Run | Hole frames | Jumps | Rebounds | Launches | Engine breaks | Roof (m) |
| --- | --- | --- | --- | --- | --- | --- |
| cells | 2 / 3 / 2 | 0 / 0 / 0 | 4 / 6 / 9 | 1 / 0 / 1 | 24 / 74 / 85 | 15.2 / 14.0 / 14.0 |
| boxes (`SF_NO_FRAC_CELLS=1`) | 0 / 0 / 0 | 0 / 0 / 31 | 12 / 11 / 21 | 2 / 2 / 8 | 86 / 67 / 94 | 13.6 / 15.1 / 12.8 |

Rebounds are counted by the check "rebounds over half the impact speed (while quiet)". With cells there are fewer, because spalled cells fall as particles. When they fell as chunk bodies, slivers 6 cm thick rebounded (9 and 19 rebounds in two runs).

**Breaking on the GPU** (cells on; docs/GPU.md):

| Run | Hole frames | Jumps | Rebounds | Launches | Engine breaks | GPU breaks | Roof (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first version | 1 / 0 / 1 | 0 / 1 / 0 | 1 / 4 / 1 | 1 / 0 / 1 | 10 / 2 / 3 | 25 / 35 / 27 | 15.9 / 15.9 / 15.9 |
| hits on a broken piece go to its parts | 0 / 0 / 0 | 0 / 0 / 0 | 5 / 2 / 1 | 5 / 0 / 0 | 2 / 2 / 18 | 26 / 26 / 25 | 15.9 / 15.8 / 15.9 |
| breaks batched into one job, parts not re-armed (kept) | 1 / 4 / 1 | 0 / 0 / 0 | 6 / 16 / 12 | 2 / 2 / 2 | 69 / 76 / 76 | 33 / 44 / 40 | 13.0 / 13.8 / 13.2 |
| off (`SF_NO_SNAP=1`), same build as the first | 1 / 4 / 1 | 11 / 5 / 0 | 13 / 29 / 2 | 2 / 6 / 0 | 106 / 115 / 24 | 0 | 11.5 / 12.4 / 15.9 |

What went wrong in the first two versions:
- A break on the GPU queued a job for the engine at the head of the line, and every part was armed with a plan of its own. Parts broke again as they kept landing, one job every 4 frames.
- The engine's answers to the hits (crushing, the part cut free) never had their turn. The building stopped coming down: the roof was at 15.9 m, against 11.5–13.8 m otherwise.

What the kept version does:
- All of a frame's breaks go in one job.
- The parts get no plan of their own; the engine's answer to the hit refines the break from there.
- No run of it has jumped. Runs without it jumped in two of three.

**Open: rebounds.** With every change, these runs count more rebounds than the last commit before the spring-network solver did (3 in its one 150-frame run; there have been 0–29 since). They come from pieces of 15–95 bricks landing on rubble at 2–5 m/s (`bounce` events). It is not yet known whether the cause is the solver's breaks (more pieces of that size) or contacts.

**Full-length collapse with all three changes** (600 frames, software GPU, `tools/check.sh lab`, 6 November):
- The building comes down much further: the roof falls 13.73 m (18.64 → 4.91 m). The last full run before this work fell 2.4 m.
- Energy gains are down to 0 (they were 49).
- 7 hole frames (they were 6).
- Rebounds are up to 29 (they were 2), and 16 pieces are thrown up faster than 5 m/s (there were 0).
  - At frame 53, a 3-box piece went from −2.0 to +12.2 m/s.
  - In the same frame, a spare body went from −2.7 to +6.8 m/s, just after its piece broke on the GPU.
- 55 pieces broke on the GPU.

**Open:** these launches. One of them is a body that a GPU break had just made.

## Rebounds and launches (7 November)

**How they were found.** Runs are now the same every time, so one bad bounce can be replayed and traced. `SF_TRACE_BODIES=42,95 node game/lab.cjs collapse --frames 66 --trace 50:65` records the chosen bodies every frame, straight from the GPU: pose, velocity, mass, and every contact with its normal, points, impulses and approach speeds (`btrace` in the report). In the first rebounds:
- A 2.3 t piece wedged in a pocket of the 138 t upper part went from −2.1 to +1.6 m/s.
- A 49 kg piece under a 773 kg one took 76,000 N s from the ground in one step.

Both are light pieces squeezed under heavy ones. GPU test `landing` reproduces it with two bodies (docs/GPU.md, "Shock propagation").

**Comparing changes.** One run cannot tell a trend from luck: the collapse is chaotic, and any change sends it another way. `SF_SEED` changes the page's random numbers (the scatter of crumbs), which gives another run of the same scenario. Each line below is three 150-frame collapse runs (seeds 0 / 7 / 8, software GPU), all with the slot fixes below. The raw lines are in `lab-results/cmp_{off,pin,sh,shp}_s*.txt`.

| Run | Rebounds | Launches | Energy gains | Jumps | Roof fall (m) |
| --- | --- | --- | --- | --- | --- |
| neither | 13 / 26 / 16 | 7 / 6 / 8 | 0 / 0 / 0 | 0 / 36 / 0 | 7.33 / 7.23 / 7.11 |
| pinching only (`SF_PINCH=1`) | 12 / 11 / 15 | 4 / 2 / 2 | 1 / 1 / 1 | 0 / 0 / 0 | 7.12 / 7.10 / 7.32 |
| shock propagation (kept) | 10 / 6 / 11 | 5 / 2 / 0 | 0 / 0 / 1 | 0 / 0 / 0 | 7.15 / 5.30 / 6.16 |
| both | 9 / 17 / 16 | 3 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 7.20 / 7.86 / 7.77 |

Shock propagation is kept; pinching stays off (docs/GPU.md, "Crushing on contact").

**Full length** (600 frames, `tools/check.sh lab`, software GPU, 7 November), against the run of 6 November:

| Check | 6 November | 7 November |
| --- | --- | --- |
| Rebounds | 23 | 17 |
| Launches | 7 | 11 |
| Energy gains | 11 | 0 |
| Jumps | 2 | 0 |
| Hole frames | 8 | 6 |

The building comes down much further: the roof goes from 18.64 to 4.29 m, against 10.65 m before. The launches are up over this longer run; the first are a slab and a sliver at frames 92–94 and spares later. They are open.

**Slot bugs found on the way** (docs/GPU.md, "Slots"):
- A stale engine answer about a GPU break broke a spare made later in the same slot. The slot was then freed twice, and two bodies shared it: 8 jumps of 1,000 m in a 150-frame run.
- The new check "body slots freed twice" (limit 0) counts it.
- The 36 jumps of seed 7 without either change are a spare thrown up about 1 m per frame at frame 142, a launch rather than a read-back fault (`read-back check` 0).

**Debugging switches** (`game/lab.cjs`):
- `SF_TRACE_BODIES` (with `--trace`)
- `SF_SEED`
- `SF_PINCH`
- `SF_PARK_DBG`: shapes drawn at a parked body, and every change to a body slot (`parked`, `slotLog` in the report)

## Parts vanishing, and the building that floated (7 November)

Reported from the playground (the mixed building, low settings, other devices and settings too): huge parts, even a floor, vanished for good while the building came down; and with more than half the ground storey removed the building stood, until a little more brought it all down at once.

**Vanishing.** The lab compared what is drawn with what exists, and both lost the same parts, so no check saw it. The new check "volume lost, not to the tools" compares with the building's volume instead. On the first runs (240 frames, software GPU) it found:

| Scenario | Lost | Of which by the engine's impacts | By particles crushed |
| --- | --- | --- | --- |
| `collapse` | 47.2% of 1,211 m³ | 489 m³ | 84 m³ |
| `mixed` | 8.5% of 1,542 m³ | 114 m³ | 17 m³ |

The engine removed crushed material beyond its per-impact budgets as dust: the crumbs (64 on low, 384 on high), and the crushed storey a falling part landed on. Engine test `crushed_material_over_the_budget_is_spilled_not_removed`: the 4-storey tower landing on its stumps with 64 crumbs removed 1,964 bricks, a whole storey, in one frame.

Now that material is sent as spill (id −2), and the page makes rubble particles of at most 0.4 m from it, whatever the setting. It is lost only when the 65,536 particles are all in use, and the HUD then says how much. After the fix, on the same runs:

| Scenario | Lost | By the engine's impacts | By particles crushed |
| --- | --- | --- | --- |
| `collapse` | 6.1% | 0 | 74 m³ |
| `mixed` | 1.6% | 0 | 24 m³ |
| `nibble` (full length) | 1.8% | 0 | 28 m³ |

**Full length** (`tools/check.sh lab`, 600 frames): `collapse` loses 17.4% of its 1,211 m³, all of it from particles crushed (210.6 m³; the engine 0, the cut 78.4 m³ as the tool's). The other checks are as before: 16 rebounds, 11 launches, 12 hole frames, 2 energy gains; the roof falls 10.5 m.

**Open: particles crushed.** What is still lost is rubble particles crushed to fines. Particles do not push bodies (docs/GPU.md, "Particles"), so a body landing on a heap goes through it, and the particles caught between it and the ground are removed. `collapse` fails the 2% limit on this alone (74 m³ in 240 frames).

**Floating.** In the new scenario `nibble`, the mixed building's ground storey is eaten away shot by shot. The engine had nothing between "under strength, unchanged" and "over strength, fails now". Now joints loaded past 70% of their strength wear and give way in time (docs/STRUCTURE.md, "Wear under sustained load"), and steel members have their rolled sections' strength instead of a solid box's.

`nibble`, full length (1,584 frames, software GPU):
- The first failures come at frame 230.
- 578 joints wear through and 757 crack before the upper storeys start to fall at frame 730 (8.3 s later).
- They come down over the next 3 s: 1,293 → 1,203 → 1,012 → 489 m³ standing at frames 600, 700, 800, 900.
- 66% of the upper storeys' volume comes down, the core stays.

Open in `nibble`, as in the other scenarios:
- 2 duplicate frames (static meshes updating a frame late);
- 5 jumps (particles on the upper floors, 1–4 cm over their limit);
- 3 rebounds;
- 8 energy gains.

## Landing parts, parts doubled, slow motion (7 November, the next day)

Reported from the playground: big parts burst into particles when they land; things fall in slow motion; parts of the upper storeys fall with nothing touching them.

**Seeing it.** `SF_SHOTS=40,90,140` renders the scene at those frames from outside the building and saves `<scenario>_f<frame>.png` in the output folder. `SF_SHOT_VIEW=x,y,z` moves the camera. The report now also says:
- where the particles came from (`particles_by`), and why the engine spilled bricks (`spill_why`);
- where each new body came loose (`made`: frame, job, lowest and highest point, volume);
- engine jobs that added volume (`gains`), and breaks whose parts hold more than their parent (`over_breaks`).

**Bursting.** See docs/STRUCTURE.md, "Falling parts landing". `mixed`, 150 frames:

| Rule | Worst crumbled share | Holes |
| --- | --- | --- |
| old masonry rule (`SF_OLD_MASONRY=1`) | 1.00 (a 277 m³ part at frame 101) | 6 frames |
| now | 0.14 | 0 |

**A part doubled.** In `mixed` the volume check went negative: −10.1%, 186 m³ more than the building. The GPU had broken a falling part by its plan in the very frame the engine's answer broke the whole part (body 7, frame 72). The engine's pieces held the spare's 21 slabs as well, and the spare stayed: a second copy, for good. Now, when the GPU's break arrives for a body the engine has already broken, the spare goes (`takeGpuSnaps`). The same holds when the GPU's break was known but not yet answered. The check "volume made" (limit 0.5%) catches it: 10.1% → 0.

**Slow motion.** Gravity is 9.81 m/s² (GPU tests `box` and `bounce` fall and bounce to the expected heights). The page steps 1/60 s per frame at most twice. A GPU too slow for 60 steps a second then runs the world slower than real time; in the real-machine bench (`bench/2026-11-05_linux-ryzen5-3600_nvidia-ampere`) a step takes 8.9 ms at the collapse's peak. Now:
- The HUD shows the rate ("time ×0.62") while anything moves.
- With the setting "real-time steps" on (default) and the sub-steps already at the adaptive floor (8, or the setting's if lower), a step covers 1/30 s with twice the sub-steps. The sub-step is the same length, so the accuracy is the same, and the per-step work (broad phase, pairs, colouring, about a fifth of a step) is halved.
- GPU tests with `DT2=1` (every scene in 1/30 s steps) and 16 sub-steps match 8 at 1/60: stack drift 15.5 mm both. At 12 sub-steps in 1/30 s steps the stack fell (430 mm), so a 1/30 s step never has fewer than twice the floor.
- Impacts from a 1/30 s step reach the engine with that step's length, since force is impulse over the step.

**Parts falling far above a cut.** See docs/STRUCTURE.md, "Wear under sustained load": steel no longer wears.

**Full length after these changes** (software GPU; `tools/check.sh lab`, plus `nibble` and `mixed` run in full):

| Scenario | Rebounds | Launches | Volume lost | Crumbled share (worst) | Roof or upper storeys |
| --- | --- | --- | --- | --- | --- |
| `collapse`, 600 frames | 7 (16 the day before) | 1 (11) | 4.6% (17.4%), all particles crushed | 0.32 (over 0.25: open) | roof 18.64 → 15.38 m (→ 8.16 the day before: less of the stumps' storey crumbles) |
| `mixed`, 600 frames | 48 | 17 | 16.5%, all particles crushed (254 m³) | 0.14 | roof 22.39 → 9.71 m |
| `nibble`, 1,584 frames | 1 | 0 | 1.3% | 0 | 69% of the upper storeys down, 8 s after the first failure, 765 joints cracked before |

Open:
- Rubble particles crushed under heavy bodies are removed (one-way particles). This is what fails `collapse` and `mixed` on volume.
- `collapse`'s worst crumbled share is 0.32.
- Rebounds and launches in `mixed`.
