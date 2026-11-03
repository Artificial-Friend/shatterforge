# Validation and benchmark

## Structural model against theory (`cargo test --release validation -- --nocapture`)

Each case builds a small structure, analyses it under its own weight (plus a given load for columns) and compares the worst stress ratio with the textbook value.

| Case | Theory | Engine error |
| --- | --- | --- |
| Concrete cantilever strip 0.25 m deep, 3 / 5 / 7 m long | σ = 3ρgL²/h (M = wL²/2, section bh²/6) | 0.0 % |
| Concrete cantilever 0.5 m deep, 5 m | same | 0.0 % |
| Timber cantilever 0.25 m deep, 5 m | same | 0.0 % |
| Simple span on 0.25 m bearings, 6 m clear | σ = 3ρg(c² + bc)/h (midspan moment with an even bearing pressure; c half the clear span, b the bearing width) | −7.7 % |
| Simple span on 0.25 m bearings, 12 m clear, concrete 0.25 m / timber 0.5 m deep | same | −0.5 % |
| Columns 0.5 m square: concrete 5,000 kN, masonry 2,000 kN, timber 1,000 kN | σ = (P + W)/A against the compressive strength | −0.4 to −0.7 % |

The suite found three faults, which are fixed:

- **Bending was checked at the wrong place.** It was checked only at cell boundaries (bricks are cut into cells of up to 2 m), up to 1 m from the support face where the moment peaks. Cantilever stresses came out 26–56 % low. Cells are now also cut at the edges of the load-bearing supports a brick rests on.
- **Moments funnelled through column-width cells.** A bay's moment went through cells as wide as a column. In a plate, the moment spreads sideways about as far as its lever arm, so the section is now the face widened by the arm on each side, up to the plate's width.
- **Spurious links across side joints.** Side joints linked cells that do not touch the shared face. Cells are now paired by grid lookup on the face only. This also made the tower analysis 3× faster (103 ms native for 9,458 cells).

**Known limit.** The model ends a span's moment at the bearing faces (as two cantilevers meeting at midspan). This misses the b·c term: about b/c low, a few percent for column-width bearings. It would be about 25 % for 1 m bearing pads. A flow model cannot tell a beam resting on a pad from one clamped into it.

**Not validated yet.**
- Two-way slab moments: the effective-width rule is a reasonable approximation, not a calibrated one.
- Punching shear at columns.
- Impact crater sizes and crush energies.
- Collapse speed against real collapses.

These need measured data: crater tests, and video of real collapses.

## Benchmark (settings → Run benchmark)

The page reloads and runs the same scripted destruction every time, with a fixed seed for every random number:

1. A fixed view from outside the tower.
2. The ground storey is cut by 36 fixed blasts (radius 2.5 m on a 4 m grid, at 1.8 m height). The script waits until the engine has applied them all.
3. The tower drops and crushes down.
4. Three steel balls (0.8 m, 120 m/s) are fired at fixed simulated times: 3, 5 and 7 s.
5. It stops after 12 s of simulated time.

Every simulated second the report adds a row: wall time, fps, physics steps/s, GPU ms per step, bodies, awake bodies, contacts, breaks, bricks to dust and roof height. At the end it gives:
- the real-time factor (simulated time / wall time)
- frame time (median, 95th percentile, worst)
- the worst GPU step and the last GPU stage times
- peak bodies and contacts
- the totals

The script is repeatable but not bit-identical: impacts are processed in the worker as the results arrive, so the order of events can differ slightly between runs.
