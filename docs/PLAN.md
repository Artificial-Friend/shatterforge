# Plan: the engine's foundations (November 2026)

Three foundation changes, in this order, before scaling to a city. Each keeps the engine's interface (`analyse`, `structure_step`, `impacts`, the box outputs to the page) so the page, the GPU solver and the lab carry over. Each ends with the full checks (CLAUDE.md) and a report.

## 1. Structural solver: rigid-body-spring network

**Why.** The current load-flow model sends load to the ground like current through resistors. Its limits:
- **No stiffness.** Steel draws no more load than brick of the same size.
- **No moment balance.** Moments are patched on with an overturning check and a flat-slab check.
- **Fragile buildings.** Every building stands at 85–94% of its strength, so each honest physics change knocked one over.

**What.**
- Each cell of the existing node graph (bricks cut into cells of at most 2 m) becomes a rigid body with 6 degrees of freedom.
- Each link (a cell face inside a brick, or a joint face between bricks) becomes a spring at the face centre:
  - normal stiffness E A / L, shear stiffness G A / L;
  - bending stiffness E I / L about the two in-face axes, torsion stiffness G J / L;
  - two materials in series.
- Moduli: concrete 30 GPa, brick 5, steel 200, glass 70, timber 11 (a joist floor 1, from its joists' I), with G = 0.4 E.
- Gravity, point loads (rubble, impacts) and sideways loads are all one load case, solved by preconditioned conjugate gradients (6 × 6 block Jacobi). Ground bricks are fixed.

**Checks, per joint** (sub-links summed about the joint's face) **and per cell face inside a brick:**
- normal stress N / A ± bending M / S, against tension (the flexural strength where bending dominates) and compression;
- shear (V / A, plus torsion) against cohesion + 0.4 × compression (Mohr–Coulomb: precompressed masonry is stronger in shear).
- Connection factors and the residual strength of cracked joints stay as they are.

**What goes away:** the overturning check (moments come out of the solve), the flat-slab bay check (bending comes out of the solve), and the plate-width spread.

**Validation, as engine tests against closed-form results:**
- a cantilever (root moment w L² / 2);
- a simply supported span (w L² / 8);
- a column under its own weight (N);
- a portal frame under a sideways load;
- a slab on an off-centre column (tips);
- the four buildings standing under their own weight;
- the behaviour tests (one column, rubble on floors, sideways hits) with physics-derived limits.

## 2. Primitive: convex cells

**Why.** Everything is an axis-aligned box on a 0.125 m grid; irregular fracture is a skin of cells over boxes. Breaks follow the grid, non-box architecture is impossible, and the skin's bookkeeping caused the short holes in the lab.

**What.**
- A brick may carry a convex cell shape: one of the hull library's cells (16 Voronoi patterns of a box, 63 cells), scaled into its box.
- The engine knows its true volume and centroid (mass) and its faces.
- Joints between cells of one fractured brick are planar faces of the pattern; joints to neighbours stay on the box faces, clipped to the cells that reach them.
- Fracture splits a brick into its cells as engine bricks, in place of the page-side skin. The structure, the meshes, the outputs and the GPU then agree on one geometry: the page draws and collides what the engine says, and the spall bookkeeping on the page goes away.

## 3. Same-frame break decisions on the GPU

**Why.** The engine answers an impact about 4 frames late. The inherit kernel, the energy given back to parts cut free, timing holes and part of the rebound problem all come from working around that delay.

**What.**
- The GPU decides in the step of the hit whether a falling piece breaks. Its contact impulses are compared with the piece's weakest joints, which the engine sends with the piece: per joint, its face, strength and the cells on either side.
- If it breaks, the GPU splits the body there and then, at that weak joint, into the two bodies the engine pre-computed for it.
- The engine then refines the break when its answer arrives. The pieces are already moving apart, from the right place.
- The building's own structure check stays asynchronous.

## Later

Determinism, solver stability work (mass ratios, continuous collision, islands), scaling (streaming, level of detail, a check per building), authoring of imported meshes, effects.
