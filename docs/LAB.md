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

Suites: `quick` (collapse, ball) and `full` (all six).

## Running it

**In the browser (real GPU).** Open the settings panel (O) and press **Run lab checks, quick** or **full**. The page reloads once per scenario. At the end the panel shows the summary, and **Copy** gives the full JSON.

**Headless (software GPU).** From the repository root:

```
node game/lab.cjs quick                      # or full, or scenario names
node game/lab.cjs collapse --frames 150      # a shortened run
node game/lab.cjs collapse --trace 36:58     # trace the bodies made in frames 36-58
node game/lab.cjs ball --read-lag 3 --job-lag 6
```

The runner prints the summaries and writes `lab-results/<scenario>.json` and `all.json`. It exits with 1 when a check fails. It needs Playwright and `three` (set `SF_NODE_MODULES` to a `node_modules` folder that has `three`). On the software GPU a 4-storey collapse step takes about 4 s, so 150 frames take about 10 minutes.

## What it found first

- **Particles flying across the scene.** A particle slot freed and reused while a read-back was in flight was matched by its new generation, so the old particle's position was blended into the new one: two crumbs moved 6 m and 15 m in one frame. The page now matches particles by their generation at the time the read-back was issued.
- **The building bouncing.** The upper part of the building (about 1,900 t) landed on rubble crushed off its own base and rebounded at up to 2.5 m/s, throwing the rubble up at 4 m/s. Light pieces wedged between a body hundreds of times heavier and the ground do not converge in a few solver iterations. Restitution, push limits, more sub-steps and more colours did not change it. Contacts now limit the mass ratio to 20: the heavier body responds as if at most 20 times heavier. The GPU test `sandwich` (a 7,700 t block landing at 5.7 m/s on 24 pieces of 11 t) went from a 2.52 m/s rebound to 0.34 m/s, and the 20-box stack is unchanged.
- **Pieces popping out.** Pieces cut from a falling part were made overlapping the slab they were crushing into (the engine answers a few frames after the hit), and the solver pushed them out at its 3 m/s limit. The limit is now 1 m/s.
- **The upper part rising after it landed.** A trace of frames 44–62 (`collapse --frames 70 --trace 44:62`) showed the upper part (1,554 bricks) going from −6.34 to +1.11 m/s on the stumps at frame 51 and then rising again (0.44, then 0.64 m/s); the roof rose 0.5 m over frames 60–90 of the full run. The mass-ratio limit was making momentum, and the body's supports were solved in the parallel overflow pass. Both are fixed (docs/GPU.md, "Still to do", and GPU test `stumps`). After the fixes the upper part lands from −6.0 to −0.56 and −0.17 m/s (`--frames 60 --trace 44:56`). On 150 frames: rebounds 7 → 4, launches 1 → 0, energy gains 6 → 2.
- **Open.** Pieces of 11–27 bricks next to the upper part are still thrown out at 2–4 m/s when it lands (frames 50–57), and one energy gain remains at frame 54, just after the upper part breaks.
- **Open (mixed).** With the ground storey cut, the mixed building comes down only partly in 600 frames: roof 22.39 → 19.9 m against the 3 m the check asks (software GPU, full length). With structure checks every frame (a bug: the rubble interval was 1 ms) it fell 5.6 m in 240 frames; it also shows 11 rebounds and 18 energy gains, of the same kind as the collapse's.
- **Open (irregular fracture).** With bricks on fracture surfaces drawn as Voronoi cells (docs/STRUCTURE.md), the collapse (150 frames) shows 3–5 frames of drawn volume below what exists, by 0.07–0.4 m³: crumbs not yet drawn for one or two frames. With it off (`SF_NO_FRAC_CELLS=1`), none. When a hole appears, the lab now lists the bodies and particles that exist but are not drawn (`undrawn` in the hole event).
- **Varies by run (collapse energy gains).** Full-length collapse runs today counted 2, 24 and 49 small energy gains (the last after irregular fracture: 4–250 kJ each from frame 235 on, no launches). The GPU's atomics make each run differ; a count from one run is not a trend.
