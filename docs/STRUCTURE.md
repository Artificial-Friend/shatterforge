# Structure: what holds the building up, and what breaks it

The engine (`engine/src/structure.rs`, `game.rs`) checks the standing building as a load-flow network over its bricks, cut into cells of at most 2 m. Loads flow to the ground like current through resistors, and each joint's force is checked against its strength in compression, tension, shear and bending. Numbers below come from the engine tests (`cd engine && cargo test --release -- --nocapture`).

## Overturning (no building on one column)

The flow carries every load down some path, but it has no moment balance.

At each bearing level (a storey's floor line), each connected part above has a weight W acting at its load centroid. The supports bearing at that level hold it only if one of these is true:
- the centroid lies inside their convex hull;
- the centroid lies outside by no more than they can carry in bending, e = Σ f_flex b d² / 6 / W.

Beyond that, every support joint at that level fails. The part comes loose, and the rigid-body solver on the GPU tips it.

Test `a_slab_on_one_off_centre_column_tips_and_a_centred_one_holds`:
- 6 m slab on one 0.5 m column, at a corner: tips. Its weight acts 3.9 m off the column, which can bend at most 250 kNm (e = 1.2 m).
- The same slab with the column centred: holds.

## Loads at their points

Rubble and impact forces load the cell they land on. Before, they were spread over the whole brick, and a floor slab is one 22 m brick.

## Rubble weighs on floors

Once a second, the page sends the weight of the debris at rest to the engine with a structure check:
- bodies slower than 0.5 m/s, and particles asleep;
- binned in 0.5 m cells;
- setting "Rubble weighs on floors".

The engine puts each load on the standing brick under it.

Loaded floor bays are checked as in flat-slab design. In each direction, the bay's total static moment q L² / 8 must be carried by the sections over its supports and at midspan together: 2 m_p, with m_p = f_flex d² / 6. A free edge is checked as q L² / 2 against m_p.

How L is measured:
- between the nearest supports that cross the load's line (walls, beams);
- posts within 6 m count only on a side with no crossing support (a flat plate on columns).

A bay over its capacity cracks along its support lines and comes down.

The cell graph alone underestimated a bay's bending by about 3 times, because the moment splits over parallel paths. With 60 kPa on one bay it came out at 0.70 of strength.

| Test | Load | Result |
| --- | --- | --- |
| `rubble_lying_on_a_floor_weighs_on_it`, concrete flat slab (0.25 m, 6.4 m bay) | 3 kPa | panel ratio 0.20, holds |
| same | 60 kPa | panel ratio 1.45, comes down |
| `the_mixed_buildings_timber_floor_gives_way_under_rubble_and_its_brick_cracks`, timber joist floor (3.25 m between wall and beam) | 3 kPa | 0.12, holds |
| same | 40 kPa | comes down |

## Sideways hits

The horizontal part of an impact force on the standing building is a load case of its own. It flows to the ground in the hit's main direction and fails joints in shear and tension on the way. Before, only the downward part loaded the structure, and a sideways hit only crushed what it touched.

Test `a_sideways_hit_knocks_bricks_out_of_a_wall`: a 1 t piece at 6 m/s into a masonry pier. Over two hits, 16 joints failed and 6 and 11 cracked.

## Brick cracks before it breaks

A joint with brick on either side (brick to brick, brick on a slab or lintel) cracks the first time it fails:
- it keeps bearing in compression, with no tension and 0.4 of its shear left;
- the second failure parts it.

In the standing building:
- a crack through a merged brick pier parts it straight away (held, its halves cracked again round after round);
- in a falling piece, brick parts at once.

Cracked joints, brick and reinforced concrete alike, are drawn as dark seams 2 cm wide along the joint (test `a_cracked_joint_shows_as_seams`). The HUD counts the joints cracked and holding.

## The mixed building (kind 3)

Five storeys on the 22.5 m footprint, with a concrete core and stair:
- **Ground storey:** a shop. Brick walls. On the south and north fronts, three shop windows between brick piers, each under a steel beam 0.375 m deep. Four steel columns inside at (±7.5, ±7.5) m.
- **First floor:** 0.25 m concrete.
- **Upper storeys:**
  - Load-bearing brick walls, 0.375 m thick, with 1.5 × 1.75 m windows under steel lintels.
  - Steel beams along both axes at ±7.5 m, on the columns. Timber joist floors span 3.25 m between beams, walls and the core.
  - Concrete floors inside the core.
- **Roof:** timber, with a brick parapet.

It stands at 85% of strength at its worst joint (test `each_kind_of_building_stands_at_its_height`).

What it took, in a model where every material is equally stiff:

| Choice | Physical reason | Without it |
| --- | --- | --- |
| Brick walls merge into full-height piers | | merged as long horizontal bands, they carried the load sideways round the building's corners: 21 times the corner joints' strength |
| Brick under and over windows, and the parapet, carry only their own weight | as the masonry building's spandrels | |
| Steel in a brick pocket bears only vertically; its sides conduct like infill | no bond at the sides | they drew the wall's load sideways into the lintel ends |
| A timber floor's seam with another material is a hinge (no bending) | joists bear on a ledger or in pockets | |
| A timber floor bends at 2.4 MPa over its 0.25 m depth | 50 × 225 mm joists at 400 mm, 24 MPa: 25 kNm per metre | as solid timber (20 MPa), a bay carried 300 kPa of rubble |
| Shop windows stand between brick piers | glass fronts on slender posts under one transfer beam were too soft a support | |

## Irregular fracture

A brick on a fracture surface is drawn and collided as the Voronoi cells of one of the 16 fracture patterns, which tile it exactly. One or two of its cells spall off and fall as crumbs. A break shows jagged edges instead of a step of boxes, on both the piece that falls and the part left standing.

Which bricks:
- The engine marks a brick as on a fracture surface (`F_FRAC`) when a joint of it breaks or a neighbour is crushed away.
- Only bricks of 0.25–2 m a side are marked. Bigger ones are refined near a break, and smaller ones are crumbs already.
- The mark goes with the brick's box to the page (code bit 256), and the engine's mesh leaves marked bricks out.

Which cells:
- The pattern and the spalled cells come from the brick's index (`brick::frac_cells`, the same hash on the page). The worker's engine, the page's copy and the page agree without being told.
- A brick that already spalled keeps the same cells when its piece breaks again. Spalling again made its material twice.

On the page:
- Standing fractured bricks are static hull shapes on the GPU and instanced cell meshes. Each job's cells are shown with that job's chunk meshes. Shown as they stood at the moment, a later job's cells appeared before its debris did: holes of up to 0.6 m³.
- At most 1,500 standing bricks become cells, and only while there is shape capacity to spare. Beyond that a brick stays whole, drawn as one cube cell. The cut in the lab's collapse marks thousands of bricks, and as cells and chunk bodies they ran the GPU world out of shapes: the building did not fall, with 5,813 bodies awake and 13 s GPU steps.
- New pieces made from the standing structure take cells for their fractured bricks. A piece cut from a moving piece does not spall again: the GPU places such pieces one for one.
- Spalled cells fall as particles. All crumbs (particles) are drawn as cells scaled to their volume; their physics is a sphere as before.

Engine test `a_blast_leaves_a_fracture_surface`: a blast in a masonry facade marks 16 standing bricks, all 0.25–2 m. The spalled cells are always one or two, never all, the same every time.

Open: in the lab's collapse (150 frames, software GPU), 3–5 frames show drawn volume below what exists by 0.07–0.4 m³: crumbs not yet drawn for one or two frames. The same run with cell fracture off (`SF_NO_FRAC_CELLS=1 node game/lab.cjs collapse --frames 150`) has none.

## Known limits

- **Every material equally stiff.** The flow's stiffness has no Young's modulus, so steel draws no more load than brick of the same size. Tried: stiffness by modulus (steel 6.7, brick 0.17 of concrete), and shear deformation of deep members in series with bending. Each one put the 12-storey concrete tower over strength under its own weight (105–111%) and the masonry building at 147%. Not kept.
- **Bay bending.** The cell graph's bending inside a bay is too low (above); the flat-slab check covers floors under resting loads only.
