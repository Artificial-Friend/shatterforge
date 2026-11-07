# Structure: what holds the building up, and what breaks it

The engine (`engine/src/structure.rs`, `game.rs`) checks the standing building as a rigid-body-spring network. Numbers below come from the engine tests (`cd engine && cargo test --release -- --nocapture --test-threads 1`), native, on the cloud machine.

## The model

Each brick is cut into cells of at most 2 m across x and z, and also at the edges of whatever bears on it. Each cell is a rigid body with 6 unknowns: 3 displacements and 3 rotations.

Each link becomes a spring at the centre of its face. A link is a face between two cells of one brick, or a sub-face of a joint between two bricks. The spring has these stiffnesses, with the two halves (cell centre to face) in series:

| Stiffness | Value |
| --- | --- |
| normal | E A / L |
| shear | G A / L, with G = 0.4 E |
| bending, about both in-face axes | E I / L |
| torsion | G J / L |

Young's moduli (`modulus_of`):

| Material | E |
| --- | --- |
| concrete | 30 GPa |
| brick | 5 GPa |
| steel | 200 GPa |
| glass | 70 GPa |
| timber | 11 GPa |
| a timber joist floor | 1 GPa (from its joists' I) |

The ground storey's base cells are fixed. Gravity, rubble lying on floors and impact forces form one load case; a sideways hit is a load case of its own.

The solve is conjugate gradients, preconditioned by a multigrid V-cycle:
- **Levels.** Cells are grouped into rigid blocks of 3, 6, 12… m. A spring between two blocks is the same spring with its arms measured from the blocks' centres.
- **Smoothing.** Damped block Jacobi (each cell's 6 × 6 block inverted, ω = 0.8).
- **Finest level.** The stiffness is applied straight from the springs: 15 numbers per spring instead of a 6 × 6 block, because the solve is limited by memory.
- **Warm start.** The standing building's next check starts from its last solution (`structure::Warm`).

| Test | Result |
| --- | --- |
| `intact_tower_stands`, 12-storey concrete tower | 9,458 cells, 62 iterations, 184 ms. Before the multigrid: 1,552 iterations with block Jacobi alone, 1.2 s. Before applying the finest level from its springs: 468 ms. |
| `a_second_check_starts_from_the_first`, the same tower checked again | 0 iterations, 31 ms |
| same test, the ground storey blasted away but one column | 337 iterations warm against 340 cold. The first round of failures is identical (301 joints, worst 14.17) |

The tolerance is r·z reduced 10⁵ times from the start. With 10⁸, every test printed the same ratios, in 104 iterations instead of 62 for the tower.

## Checks

Each joint is checked as one section, its sub-faces summed about the joint's face centre. Each face inside a brick is checked as the brick's own section. The worst of:
- **Tension** at the extreme fibre, N / A + M / S, against the tensile strength. Where bending makes most of it, the limit blends towards the flexural strength.
- **Compression** at the other fibre.
- **Shear** (V / A, plus torsion) against cohesion + 0.4 × compression (Mohr–Coulomb: a brick wall under load is stronger in shear than mortar alone).

What happens over strength:
- A joint fails: reinforced concrete cracks first, and brick cracks first (see below).
- A face inside a brick cracks the brick there.

Connection factors and the residual strength of cracked joints are as before.

What the solve replaced: the scalar load flow, the overturning check at each bearing level, the flat-slab bay check for rubble, and the plate-width spread. Moments, tipping and bay bending now come out of the solve.

## Validation against closed-form results

From `validation.rs`. Each row compares the engine's ratio with the ratio from theory.

| Case | Engine | Theory | |
| --- | --- | --- | --- |
| cantilever 3 m, 0.25 m deep, concrete (root moment w L² / 2) | 0.212 | 0.212 | −0.0% |
| cantilever 5 m, 0.25 m, concrete | 0.588 | 0.589 | −0.1% |
| cantilever 7 m, 0.25 m, concrete | 1.152 | 1.154 | −0.2% |
| cantilever 5 m, 0.50 m, concrete | 0.294 | 0.294 | −0.0% |
| cantilever 5 m, 0.25 m, timber | 0.074 | 0.074 | −0.1% |
| column 0.5 m square, 5,000 kN, concrete | 0.667 | 0.669 | −0.4% |
| column 0.5 m square, 2,000 kN, brick | 0.800 | 0.806 | −0.7% |
| column 0.5 m square, 1,000 kN, timber | 0.191 | 0.191 | −0.4% |

For spans, the slab is bonded to its bearings, so the engine's result must lie between 0.6 and 1.6 of the clamped end moment, w L² / 12 with L the distance between bearing centres:

| Case | Engine | Clamped |
| --- | --- | --- |
| span 6 m clear, 0.25 m, concrete | 0.097 | 0.153 |
| span 12 m clear, 0.25 m, concrete | 0.464 | 0.589 |
| span 12 m clear, 0.50 m, timber | 0.057 | 0.037 |

Other tests:
- `a_slab_on_one_off_centre_column_tips_and_a_centred_one_holds`: a 6 m slab on one 0.5 m column.
  - Column at a corner: worst 9.03, so it tips.
  - Column centred: 0.70, so it holds.
- `cantilever_as_one_brick`: a 7.5 m cantilever cracks at 1.32; a 2 m one stands at 0.09.
- `simply_supported_span`: a 7.5 m span stands at 0.18; a 22.5 m span fails at 1.87.

## The buildings under their own weight

Test `each_kind_of_building_stands_at_its_height`:

| Building | Worst joint | Under the load flow |
| --- | --- | --- |
| concrete frame, 12 storeys | 39% | 85–94% |
| masonry, 5 storeys | 83% | |
| mass timber, 6 storeys | 15% | |
| mixed, 5 storeys | 58% | 85% |

What the change needed:
- **Timber strengths.** Timber is now 21 MPa compression, 14 MPa tension and 4 MPa shear (C24 timber, about 21 / 14 / 4 MPa). It was 5 / 5 / 1. The cantilever validation came out at 4 times its theory with the old values.
- **Steel on brick is a hinge.** It sits on a padstone. When bonded, the beam's end rotation bent the brick pier under it in the mixed building to 1.31 of strength.
- **A timber joist floor spans one way, along its joists** (x). Across them, the boards' bending and the deck's twisting are a twentieth of a solid plate's. As a plate bending both ways, a 3.25 m bay under 40 kPa came out at a third of its strength. With soft boards but the deck still twisting like a solid plate, it came out at 0.68: torsion carried a third of the load to the walls along the bay's sides.

## Rubble weighs on floors

Once a second, the page sends the weight of the debris at rest to the engine with a structure check:
- bodies slower than 0.5 m/s, and particles asleep;
- binned in 0.5 m cells;
- setting "Rubble weighs on floors".

The engine puts each load on the cell of the standing brick under it.

| Test | Load | Result |
| --- | --- | --- |
| `rubble_lying_on_a_floor_weighs_on_it`, concrete flat slab, 0.25 m, 6.4 m bay | 3 kPa | 0 failures |
| same | 60 kPa | first round worst 1.97, 26 failures, comes down |
| `the_mixed_buildings_timber_floor_gives_way_under_rubble_and_its_brick_cracks`, timber joist bay, 3.25 m between wall and beam | 3 kPa | 0.59, the building's own worst joint elsewhere |
| same | 40 kPa | first round 1.55 at a joist section, comes down |

## Sideways hits

The horizontal part of an impact force on the standing building is a load case of its own, and it fails joints in shear and tension.

Test `a_sideways_hit_knocks_bricks_out_of_a_wall`: a 1 t piece at 6 m/s into a masonry pier. Joints fail sideways; the mortar cracks first and the pier still stands, and a second hit where it is cracked knocks bricks out.

## Brick cracks before it breaks

A joint with brick on either side (brick to brick, brick on a slab or lintel) cracks the first time it fails:
- it keeps bearing in compression, with no tension and 0.4 of its shear left;
- the second failure parts it.

In the standing building, a crack through a merged brick pier parts it straight away. When held, its halves cracked again round after round. In a falling piece, brick parts at once.

Cracked joints are drawn as dark seams 2 cm wide along the joint (test `a_cracked_joint_shows_as_seams`). The HUD counts the joints cracked and holding.

## The mixed building (kind 3)

Five storeys on the 22.5 m footprint, with a concrete core and stair:
- **Ground storey:** a shop with brick walls.
  - On the south and north fronts, three shop windows between brick piers, each under a steel beam 0.375 m deep.
  - Four steel columns inside, at (±7.5, ±7.5) m.
- **First floor:** 0.25 m concrete.
- **Upper storeys:**
  - Load-bearing brick walls, 0.375 m thick, with 1.5 × 1.75 m windows under steel lintels.
  - Steel beams along both axes at ±7.5 m, on the columns. Timber joist floors run over them between the walls and the core.
  - Concrete floors inside the core.
- **Roof:** timber, with a brick parapet.

Choices that stay from the load-flow model:

| Choice | Physical reason |
| --- | --- |
| Brick walls merge into full-height piers | |
| Brick under and over windows, and the parapet, carry only their own weight (infill: their springs are a fiftieth as stiff) | as the masonry building's spandrels |
| Steel in a brick pocket bears only vertically; its sides are a fiftieth as stiff | no bond at the sides |
| A timber floor's seam with another material is a hinge (no bending) | joists bear on a ledger or in pockets |
| A timber floor bends at 2.4 MPa over its 0.25 m depth | 50 × 225 mm joists at 400 mm, 24 MPa: 25 kNm per metre |

## Convex cells (irregular fracture)

A brick on a fracture surface is split, in the engine, into the convex cells of one of 16 fracture patterns (`hull.rs`). These are Voronoi cells of 2–6 seeds that tile the brick exactly. One or two of its cells spall off and fall as crumbs. A break shows jagged edges instead of a step of boxes.

Every cell is a brick of its own (`Bricks::hull`, its pattern cell mapped into the box of the brick it came from), so the structure, the meshes, the outputs and the GPU agree on one geometry:
- **Mass and centroid** are the cell's real ones (`vol_share`, `centroid`).
- **Joints.**
  - Between two cells of one brick, a joint is their shared face (axis 3), a polygon at any angle.
  - With a neighbour, a joint is where the cell's polygon on its box face overlaps the neighbour's (`contact_poly`). Its area is exact, and the cells' faces on a box face tile it.
- **Structure.** A cell is one node at its centroid. A polygon face is a section of its own: area, second moments and extreme fibres from the polygon, and a frame along its normal for a face at an angle.
- **Mesh.** The engine draws a cell's faces, except those shared with a cell still joined to it.
- **Outputs.** The box records carry the hull id (code / 512). The page collides each cell as a hull shape of the GPU's library, which is the same library: `node gpu/test_hulls.cjs` checks all 63 cells' boxes, volumes and centroids against the GPU's, to 3e-16.
- **Spalled cells** fall as particles, drawn as their own cell.

The page's separate skin of cells over whole bricks, and its bookkeeping, are gone.

Which bricks:
- A brick is marked (`F_FRAC`) when a joint of it breaks or a neighbour is crushed away.
- Only bricks of 0.25–2 m a side are marked, and only standing ones. A falling piece's parts are placed by the GPU one for one.
- At most 1,500 bricks are split; beyond that a brick stays a box (`frac_max`). The cut of the lab's collapse marks thousands, and as cells they ran the GPU out of shapes.
- The pattern and the spalled cells come from the brick's index (`brick::frac_cells`).
- Removing a spalled cell marks nothing more, so a fracture surface does not spread by itself.

Tests:

| Test | What | Result |
| --- | --- | --- |
| `the_patterns_tile_the_cube` | each pattern's volumes add up to the cube; its faces on each cube face add up to that face; shared faces match from both sides | to 1e-9 |
| `a_fractured_brick_splits_into_its_convex_cells` | a 1 × 1 × 0.5 m brick between four neighbours, every pattern: the cells fill it, share each old joint's face exactly, and are joined on every shared face | |
| `a_cell_draws_its_faces_but_those_it_shares` | the mesh of a split 1 m brick encloses 1 m³, outward | 1 m³ to 1e-4 |
| `a_pier_carries_its_load_through_a_fractured_brick` | a 1 m concrete pier under 2,000 kN, its middle block split by each pattern: it stands. On a face at an angle the shear is at most σ / 2, which bounds the ratio | whole 0.070; through the cells at most 0.249 (bound 0.26) |
| `a_blast_leaves_a_fracture_surface` | a blast in a masonry facade | 16 standing bricks split into 37 cells, every cell joined, hull ids in the output |

## Wear under sustained load (7 November)

A building eaten away bit by bit stood unchanged while every joint was under strength, however close, and then one more cut failed it all at once: it seemed to float, then to be triggered. Concrete and masonry under a sustained load above about 70-75% of their short-term strength fail in time (creep rupture). Now:
- Each structure check carries the time since the last one (the page's virtual clock in the lab).
- A joint or a face inside a brick loaded past SUSTAINED (0.7) of its strength wears by `dt × WEAR_RATE × x²`, where x is how far past 0.7 it is (0 at 0.7, 1 at full strength). Worn through, it fails as if over strength: reinforced concrete and brick crack first, the rest part. A crack's wear starts again from zero.
- A cracked joint is half as stiff, and wear softens it further, to 0.4 of that when worn through. Its load moves to its neighbours, which may then wear in turn.
- While any joint is wearing, the page checks every 250 ms.

Game time is compressed (real creep rupture takes hours): at 90% a joint wears through in about 1.5 s, at 80% in about 6 s, at 75% in about 24 s.

| Engine test | Result |
| --- | --- |
| `a_building_left_near_its_limit_wears_and_comes_down_over_time`, concrete frame, 4 storeys, ground storey cut row by row from the west until the worst joint is past 0.75 (the row at x = −4.4 m cracks 125 joints, which hold; worst 1.00) | without wear: 0 failures in 40 checks, standing for ever; with wear (checks 0.5 s apart): failures in checks 2, 3, 5, 6, 24, 32 … 74, 141 joints, 82 cracked |

Deflection stays small in the linear solve: the largest displacement in that test is 10.5 mm before and 11.4 mm after the wear. The sag is not drawn; the cracks are, as before.

## Steel as rolled sections (7 November)

A steel member was its solid box: a 0.25 m column or beam had 625 cm² of steel, 6 times a 254 × 146 universal beam's 47 cm², and as much more strength, stiffness and weight. A steel member now has the section's share of the box (`STEEL_AREA` 0.16 of area and weight, `STEEL_SHEAR` 0.10, `STEEL_FLEX` 0.12 of the plastic modulus, `STEEL_STIFF` 0.17 of the second moment). The mixed building's intact worst joint goes from 0.58 to 0.56, and all the engine tests pass unchanged.

Exploring (`explore_the_mixed_building_cut_away`, ignored by default) what the mixed building stands on once its ground storey is cut all at once:

| Cut | Failures | Bricks fell | Worst joint left | Left beyond the core, above the ground storey |
| --- | --- | --- | --- | --- |
| all but the core | 4,038 | 2,634 | 0.70 | 84 t |
| the west half | 927 | 1,878 | 0.17 | nothing |
| the perimeter walls | 3,638 | 2,534 | 0.74 | 73 t |

## Known limits

- **Solve time.** In the lab (collapse, 150 frames, software GPU, `engine_ms`), engine jobs in the page took 4–8 ms median, 50–370 ms at p95 and up to 810 ms. The load flow answered in a few ms. The building's check stays asynchronous; breaking of falling pieces now starts on the GPU (docs/GPU.md).
- **A building on a mechanism** converges slowly: 340 iterations for the tower on one column. A warm start does not help there.
