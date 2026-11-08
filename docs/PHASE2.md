# Phase 2: free geometry (convex pieces at any angle, with beam-theory joints)

**Goal.** A building can be made of convex pieces at any angle, not only boxes on the 0.125 m grid: diagonal braces and rafters, sloped slabs and roofs, rotated wings, arches and vaults of wedge stones, curved facades of segments. They stand, carry load and break by the same structural model as the boxes, at about the same cost per step.

**Gate.** A complex building (irregular plan, double-height spaces, a braced steel frame, a sloped roof on rafters, an arcade of masonry arches, room contents) stands under its own weight, collapses gradually when cut, and passes the lab's checks, with the engine tests below matching closed-form results.

## What already exists

The structure is already a rigid-body-spring network (docs/STRUCTURE.md): every cell is a rigid body with 6 unknowns, every face a spring with normal, shear, bending and torsion stiffness from the materials and the face's section, solved by preconditioned conjugate gradients with a multigrid. Joints of convex cells are polygons with their own frames (`geo_of`), and the springs work in that frame. So beam theory at the joints is there; what is missing is geometry:
- every brick is an axis-aligned box on the tick grid (`x y z`, `sx sy sz`), or one of the hull library's 63 fixed cells mapped into such a box;
- joints are found as shared box faces on the grid (`jaxis` 0-2), or as faces of one fractured brick's cells (`jaxis` 3);
- cracks cut bricks along grid planes;
- the GPU knows the boxes and the library's hulls only.

## Steps

Each step ends with the checks in CLAUDE.md and a note in this file.

1. **Free pieces in the store.** A brick may be a free convex piece: up to 16 corners and 12 faces (the GPU's hull limits), a centre on a 1 mm grid and a rotation. Free pieces keep their data in a side table, so box bricks keep their 25 B. Volume, centroid and inertia come from the polytope. Engine tests: mass properties of a rotated box, a wedge, a prism.
   *Done (9 November, branch phase-2).* `Bricks::add_piece(centre, rotation, points, ...)` makes the hull of the points (`hull::convex`: the planes through three points with every point on one side, then the polytope of those planes), refused if flat, beyond the GPU's hull limits or longer than a brick's box (255 ticks, 31.9 m). The brick's `hull` is `HULL_FREE` (255) and the piece (`Piece`: centre in mm, rotation as an f32 quaternion, corners in its own frame as f32, face loops, volume and centroid) sits in `Bricks::pieces`, found by `piece_of`; its box is the tick box that holds its corners, so the spatial index and queries work on it unchanged. `hull::moments` gives volume, centroid and inertia tensor of any convex polytope (tetrahedra from the corners' mean, second moments V/20 (sum x_i x_j + S_i S_j)); `Bricks::shape`, `volume`, `centroid` and `inertia` work for boxes, cells and pieces alike. The engine mesh draws a piece's faces (not one lying on the ground). Save and load carry the pieces.
   Engine tests (`brick::piece_tests`, `cargo test --release`): a 2 x 1 x 0.5 m box turned 0.7 rad about (1, 2, 3) has volume 1 m^3 and inertia R diag(1.25, 4.25, 5)/12 R^T within 1e-6; a right wedge (legs 1.2 and 0.6 m, 2 m long) has V = abL/2, centroid (a/3, b/3, L/2), Ixy = L a^2 b^2/72 and the closed-form diagonal, and the same turned 90 degrees; a hexagonal prism (side 0.4 m, 1.5 m) has V = 3 sqrt 3/2 s^2 H and Iyy = 5 V s^2/12; a 20-sided prism, flat points and a 40 m piece are refused; a cube turned 45 degrees on a cube joins nothing by its box (0 joints), its mesh is six outward faces enclosing 1 m^3, and it survives a save and load.
   Not yet: a free piece joins nothing (step 2), the page and the GPU do not get them (step 5), and no building makes them, so nothing that runs changes.

2. **Joints between any pieces.** Two faces join where they are coplanar and opposed (within 1 degree and 5 mm) and their polygons overlap: the joint is the overlap polygon, with its normal and frame. Box to box stays on the fast grid path. Engine tests: a rotated box on a box, wedge to wedge, a brace end on a column face; areas against hand results.
3. **The structure solver over free pieces.** A long free piece (a 6 m brace) is cut at build time into segments of at most 2 m by planes across its length, so it bends as the boxes' cells do. Links come from the joint polygons. Engine tests against closed-form results:
   - an inclined strut carries P / sin(theta);
   - a braced portal frame under a sideways load is stiffer than the unbraced one, by the ratio of the closed-form frame;
   - a three-hinged arch of wedge stones under a uniform load pushes out at its feet with H = w L^2 / 8 f;
   - a sloped rafter roof pushes out on its walls (no tie) and does not with a tie.
4. **Breaking free pieces.** A crack cuts a piece by the plane of the failing section into two convex pieces, joints clipped exactly. Crushing and spalling as for boxes. Engine tests: a brace snaps where its moment peaks; an arch fails by four hinges when one foot moves out.
5. **To the page and the GPU.** Free pieces go out as hulls: a runtime hull table on the GPU (the library's slots, then pieces uploaded as they are made, same limits), drawn and collided like the chunks. Meshes: each face minus what touching pieces cover. GPU tests: a rotated hull rests flat on a box; a wedge stack stands.
6. **A complex building and its scenarios.** A generator (seeded) for the building in the gate, with room contents as loose bodies. Lab scenarios: cut the ground storey, take out a corner column, blast a room, knock out an arch's keystone; checks for volume, gradual collapse, contents, settling, and the spread over several seeds.
7. **Cost.** Build time, analysis time and memory per brick against the boxes-only buildings, and the benchmark on real machines (bench/).

## Limits written down

- A free piece is convex; anything else is several pieces.
- Curves are segments (an arch of 9 to 15 stones, a curved facade of 1-2 m panels).
- Joints form only between faces that meet flat: two pieces touching at an edge or a corner rest on each other through the GPU's contacts once broken loose, but are not joined while standing.
