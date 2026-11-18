# The Quarter: the demo's level (15 November 2026)

One level. Nine lots, three by three, 40 m apart, each a building of its own kind and material, modelled on a
real building or a real type, member by member, on the engine's 0.125 m grid. They share the ground and nothing
else; the engine solves each lot on its own (`sim::Tiles`). The code is `game/src/quarter.rs` (the layout) and one
file per building; `game/src/kit.rs` is what they are built with.

```text
  north (+z)
  temple                 office tower             tenement
  aqueduct               market hall              mill, chimney, water tower
  white villa            glass house              barn
  west (-x)                                       east (+x)
```

The player starts on the crossing south-east of the market hall, looking north-west across it to the tower.

## The buildings

Numbers from `cargo test --release -p sf-demo -- --nocapture` (each building's own test: built, nothing but its
loose contents detached, the first structure step at rest and a second of the dynamic structure failing nothing).
"Worst joint" is the most loaded joint's load over its strength under the building's own weight and its floors'
loads.

| Lot | What | Modelled on | Materials and structure | Bricks (free pieces) | Worst joint |
| --- | --- | --- | --- | --- | --- |
| SW | The white villa (`savoye.rs`) | Le Corbusier's Villa Savoye, Poissy, 1931 | 5 x 5 concrete pilotis 0.25 m square at 4.75 m; flat slabs; white rendered masonry and a ribbon window round the box (unglazed on the terrace); dark green ground storey with a half round of glass; the ramp in four flights off a concrete spine wall; solarium screens on the roof; furniture and a car, loose | 304 (34) | 0.57 (0.61 turned) |
| S | The glass house (`farnsworth.rs`) | Mies van der Rohe's Farnsworth House, Plano, 1951 | 8 white steel columns outside two slabs 23.5 x 8.5 m (steel channels, cross beams every 2 m, a 0.125 m concrete deck); glass from floor to roof between steel mullions; a timber core; the lower terrace on the house's columns; travertine steps on steel strings; furniture, loose | 196 (2) | 0.31 |
| SE | The barn (`barn.rs`) | The Barley Barn, Cressing Temple, Essex, about 1220 (aisled) | 6 timber frames: arcade posts on stone pads, tie beams, braces, collars; arcade plates and wall plates; rafters every 1.5 m at 4 in 3 from the aisles' eaves to a ridge 10.75 m up, boards and tiles; weatherboards; bales and a cart, loose | 344 (228) | 0.05 |
| W | The aqueduct (`aqueduct.rs`) | The Aqua Claudia, Rome, AD 52, at half size | 5 semicircular arches of 3 m on piers 1.5 m square and 6 m tall, 9 wedge stones each, laid dry (beds with no tension); stepped abutments; the channel on top: floor, walls, cover slabs | 143 (47) | 0.77 (beds: where the thrust passes; they do not wear) |
| Middle | The market hall (`complex.rs`) | A type: a braced steel hall behind a stone arcade (the engine's "complex building" since phase 2) | Steel frame of two storeys with diagonal braces, concrete floors, a pitched timber roof on a steel ridge, a stone arcade and loggia, a brick wing turned 30 degrees, 171 pieces of furniture, loose | 798 (395) | 0.58 |
| E | The mill (`tower.rs`, kind 3), its chimney and the water tower (`works.rs`) | A type: an English brick mill of about 1890 with its stack; a riveted steel tank on a braced tower, about 1900 | Mill: 4 storeys of brick in running bond round steel columns and beams, a concrete first floor, timber floors above. Chimney: a square brick stack 24 m on a plinth, 2.5 m to 1.75 m in four stages, hollow. Tower: 4 steel legs on a 5 m square, 15 m in four tiers with struts and a diagonal on each face, an eight-sided tank 6.5 m across and 4.5 m high (empty) | mill 712, chimney 144, tower 113 (73) | 0.51, 0.09, 0.09 |
| NW | The temple (`temple.rs`) | The Hephaisteion, Athens, about 450 BC (6 x 13 columns) | A platform of three steps; 38 marble columns of four tapering drums, an echinus and an abacus (eight-sided: a convex piece has at most 16 corners), laid dry; architrave, frieze and cornice in blocks from column to column; a cella of coursed blocks with two porches and a door; a frieze of triglyphs; gables; a timber roof at 1 in 4, rafters every 1.25 m bearing on the cella's walls; door leaves; a bronze statue, loose | 1,021 (414) | 0.67 (beds) |
| N | The office tower (`tower.rs`, kind 0) | A type: a 1960s concrete-frame office block | 10 storeys of 3.75 m on 3 x 3 bays of 7.5 m: columns, flat slabs, a core with a stair, precast cladding and a glass band, partitions | 1,866 | 0.63 |
| NE | The tenement (`tower.rs`, kind 1) | A type: a load-bearing brick block | 4 storeys: masonry piers and facade, a concrete core and floors | 732 | 0.47 |

The whole: **6,373 bricks, 1,193 of them free pieces, 16,570 joints**, built in 0.2 to 0.5 s; the first structure
step 0.8 to 1.6 s natively, then 3 to 20 ms a step with every lot quiet (test `quarter::the_quarter_is_what_it_should_be_and_stands`).
Left alone for 10 s of structure checks no joint wears (test `nothing_in_the_quarter_wears_left_alone`; a joint
past 0.7 of its strength wears and gives way in time).

## What was learnt building it

- **A joint's load depends on how the building is turned, by up to a third.** The villa as designed stood at
  0.69; turned half round on its lot, at 0.79 and wearing. The solve converges either way (77 to 102 iterations
  to 1e-7); the cells the bricks are cut into differ with the turn (1,375 to 1,414 nodes), and a small stiff
  joint that takes load it was not meant to take reads differently in each. The fix was in the building, not the
  solver: such joints were removed (the ramp's spine wall kept 0.125 m clear of the slab edges, the ground
  storey's walls 0.125 m under the floor, the ramps clear of the well's sides). After: 0.57 to 0.61 in all four
  turns. This is worth knowing for any building made for the engine: **keep stiff members from touching what they
  are not meant to carry, and check a design in all four turns** (`savoye::tests::explore_the_villas_worst_joints`,
  `kit::worst_joints`).
- Bare stone does not span: the glass house's travertine treads, 3.25 m long and 0.125 m thick, broke under
  themselves at 2.36 of masonry's bending strength. They are travertine on a steel tray now (concrete in the
  engine's terms).
- A flat roof on a steep one needs a face, not an edge: rafters bear on plates whose tops are cut to the
  rafters' underside (`temple.rs`, `barn.rs`), because the engine joins what shares a face.

## Its lab scenarios

`node game/lab/lab.cjs quarter` (the suite) or one by name; `docs/LAB.md` has the checks. Each acts in a
building's own frame (`qblast`, `qball`, `qcut`), so a building moved or turned on its lot keeps its scenario.

| Scenario | What is done | What must happen |
| --- | --- | --- |
| `quarter` | Nothing, for 10 s | The loose contents settle, nothing fails |
| `qtemple` | The three columns at the south-east corner and the next along the front shot out | It settles |
| `qaqueduct` | The keystone of the middle arch shot out from both sides | It settles (15 s: the stones of a dry arch go one after another) |
| `qvilla` | Three rows of its pilotis shot out, fifteen columns | **It holds**: the roof no more than 0.1 m lower |
| `qglass` | The four north columns cut under the floor | The roof at least 1 m lower |
| `qbarn` | Eight arcade posts shot out, four of each row | **It holds** |
| `qtower` | Three of the water tower's four legs shot out | **It holds** |
| `qchimney` | The chimney's foot blown out on one side (the west) | Its top at least 10 m lower (15 s). **It fails**: the stack tips west onto the mill's eaves, 3.4 m off and 16 m up, and stays propped there, 20 degrees over (docs/LAB.md, 19 November) |
| `qmill`, `qoffice` | Their ground storeys cut | The roof at least 3 m and 2 m lower |
| `qballhall`, `qballglass`, `qballtemple`, `qballoffice` | The steel ball through the market hall, the glass house end to end, the temple's cella, and down through the office tower's floors | It goes through what it has the energy for and does not come back |

The results of each on the current build are in docs/STATUS.md.

### Cuts they carry

Three of the buildings stand after cuts that were meant to bring them down, and their scenarios now say so and
hold them to it (the roof may not come down more than 0.1 m). Each was tried with less first (lab, 18 November,
software GPU, 900 frames):

| Building | The cut | What fails | What carries it |
| --- | --- | --- | --- |
| The white villa | 15 of its pilotis, three rows from the entrance's end, at 1.5 m (one row: no joint fails) | 30 joints, 11 more wearing after 15 s | The box hangs from the rows left: its walls and slabs are one frame |
| The barn | 8 arcade posts, the four inner ones of each row, at 3.5 m (one row: no joint fails) | 10 joints | The arcade plates and the ties take the roof to the gable frames and the aisle walls |
| The water tower | 3 of its 4 legs at 2 m (two legs: no joint fails) | 8 joints | The fourth leg, the bracing, and the riser pipe under the tank's middle; the tank is empty |

The chimney is a fourth: cut from the east, where it could fall into the open, it stands on the loose blocks
of its own foot (r = 1.5 and 2 m; docs/LAB.md, 19 November).

Whether the real buildings would stand is not claimed. A villa on a third of its columns, a barn without its
arcade and a tank on one leg and a pipe would at least be condemned; here they are carried by joints whose
strengths were set for the whole building's material, not measured for these members, and by a model of small
displacements (docs/DYNAMIC.md, "Small displacements"), in which a slender member under load cannot buckle. What would bring each down in the
game has not been found, and until 18 November the barn's scenario passed for the wrong reason: its shots
started inside the posts' line and broke nothing.

## The buildings that are not in the game

The single towers of five kinds, the complex building alone and the test city of n x n lots (kinds 0 to 5) were
the game's levels until 15 November. They are no longer offered: the game has the Quarter. Their generators stay
(`tower.rs`, `complex.rs`, `city.rs`) because the Quarter is built from three of them, the engine's 101 tests run on
them (`engine/src/lib.rs` takes them as fixtures), and sixteen lab scenarios with a history of numbers act on them
(`collapse`, `ball`, `masonry`, ...: docs/LAB.md). Removing them would remove what the engine is checked against.

## Left out, and why

- The villa's spiral stair (it would pass through two slabs as a curved body: many small free pieces for a detail).
- Water in the water tower's tank: the engine has no liquids. It stands empty, and falls lighter than it would.
- Fluting and the raking cornice of the temple; the barn's porches. Geometry only; no new structure.
