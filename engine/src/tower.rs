//! Real-scale office tower (docs/SPEC.md scale standard), built as adaptive bricks.
//!
//! Each storey is painted into a tick-resolution grid of element ids (slab, columns, walls, glass, stairs,
//! partitions; later paint wins, so openings are carved by painting empty). The grid is then merged greedily
//! into the largest boxes of one element: a 22.5 m slab is one brick, a wall between two openings is one
//! brick. Bricks split on demand when damaged.
//!
//! Layout (ticks of 0.125 m; x east, z north, y up; footprint centred on the origin):
//! - 3 x 3 bays of 7.5 m (180 ticks); storey 3.75 m (30): slab 0.25 m (2) + 3.5 m clear (28)
//! - columns 0.625 m square on the 7.5 m grid; exterior wall 0.25 m: spandrel, glass band, header, mullions
//! - core in the centre bay: concrete walls, a switch-back stair 1.25 m wide (0.25 m rise and going, landing at
//!   half height), stair opening in each slab above, door to the corridor
//! - 1.75 m corridor ring around the core; offices 5.5 m deep behind 0.125 m partitions with 0.875 x 2.125 m doors
//! - ground floor: open lobby, 1.75 m double entrance; roof: parapet and a stair penthouse with a roof door
use crate::brick::{mat, Bricks, F_INFILL};
use alloc::vec;
use alloc::vec::Vec;

pub const HALF: i32 = 90;        // footprint half width (11.25 m)
pub const STOREY: i32 = 30;      // 3.75 m
pub const SLAB_T: i32 = 2;       // 0.25 m
const W: i32 = 2 * HALF;

#[derive(Clone, Copy)]
pub struct Element { pub material: u8, pub paint: u8, pub infill: bool }

/// Element ids are per kind (a storey's slab, columns, ...); 0 is empty.
pub mod el {
    pub const SLAB: u16 = 1;
    pub const COLUMN: u16 = 2;
    pub const SPANDREL: u16 = 3;
    pub const GLASS: u16 = 4;
    pub const HEADER: u16 = 5;
    pub const MULLION: u16 = 6;
    pub const CORE: u16 = 7;
    pub const STAIR: u16 = 8;
    pub const LANDING: u16 = 9;
    pub const PARTITION: u16 = 10;
    pub const PARAPET: u16 = 11;
    pub const STAIR2: u16 = 12;   // alternate steps, so neighbouring steps stay separate bricks
    // the mixed building (kind 3)
    pub const BRICK: u16 = 13;     // load-bearing brick wall, laid in running bond
    pub const SCOL: u16 = 14;      // steel column
    pub const BEAM: u16 = 15;      // steel beam under a floor
    pub const LINTEL: u16 = 16;    // steel lintel over an opening, steel transfer beam over the shopfront
    pub const TFLOOR: u16 = 17;    // timber floor (boards on joists, as one plate)
    pub const SILL: u16 = 18;      // brick under a window: carries its own weight only (as the masonry spandrels)
    pub const COUNT: usize = 19;
}
/// What the tower is built of: 0 a reinforced-concrete frame, 1 load-bearing masonry piers and facade with a concrete
/// core and floors,
/// 2 mass timber (floors, posts, core and stairs of timber),
/// 3 mixed (see `paint_mixed`): brick walls, steel columns and beams, a concrete first floor, timber floors above.
pub fn element_in(e: u16, kind: u8) -> Element {
    use el::*;
    let mut k = element(e);
    match kind {
        // (the core and parapet stay concrete: masonry spanning the core doors and the parapet between piers would
        // need lintels and beams the layout does not have)
        1 => { if matches!(e, COLUMN | SPANDREL | HEADER | MULLION) { k.material = mat::MASONRY; } }
        2 => { if matches!(e, SLAB | COLUMN | CORE | STAIR | STAIR2 | LANDING | PARAPET | SPANDREL | HEADER | MULLION) { k.material = mat::TIMBER; } }
        _ => {}
    }
    k
}
pub fn element(e: u16) -> Element {
    use el::*;
    let (material, paint, infill) = match e {
        SLAB => (mat::CONCRETE, 0, false),
        COLUMN => (mat::CONCRETE, 1, false),
        // precast cladding hung on the slab edges: carries its own weight only (non-structural)
        SPANDREL | HEADER | MULLION => (mat::CONCRETE, 2, true),
        PARAPET => (mat::CONCRETE, 2, false),
        GLASS => (mat::GLASS, 0, true),
        BRICK => (mat::MASONRY, 6, false),
        SILL => (mat::MASONRY, 6, true),
        SCOL | BEAM | LINTEL => (mat::STEEL, 7, false),
        TFLOOR => (mat::TIMBER, 8, false),
        CORE => (mat::CONCRETE, 3, false),
        STAIR | STAIR2 | LANDING => (mat::CONCRETE, 4, false),
        _ => (mat::TIMBER, 5, true),   // partitions: stud walls
    };
    Element { material, paint, infill }
}

/// One storey's paint grid: W x STOREY x W element ids.
struct Storey { g: Vec<u16>, y0: i32, kind: u8 }
impl Storey {
    fn new(y0: i32) -> Self { Storey { g: vec![0; (W * STOREY * W) as usize], y0, kind: 0 } }
    /// Paint the box [x0,x1) x [y0,y1) x [z0,z1) in world ticks (clipped to this storey).
    fn fill(&mut self, x: (i32, i32), y: (i32, i32), z: (i32, i32), e: u16) {
        let (xa, xb) = ((x.0 + HALF).max(0), (x.1 + HALF).min(W));
        let (ya, yb) = ((y.0 - self.y0).max(0), (y.1 - self.y0).min(STOREY));
        let (za, zb) = ((z.0 + HALF).max(0), (z.1 + HALF).min(W));
        for yy in ya..yb { for zz in za..zb { for xx in xa..xb { self.g[((yy * W + zz) * W + xx) as usize] = e; } } }
    }
    fn at(&self, x: i32, y: i32, z: i32) -> u16 { self.g[((y * W + z) * W + x) as usize] }
    /// Greedy merge into boxes of one element (x first, then z, then y), at most 255 ticks per side.
    fn emit(&self, out: &mut Bricks, structure: u16) {
        let mut done = vec![false; self.g.len()];
        let idx = |x: i32, y: i32, z: i32| ((y * W + z) * W + x) as usize;
        for y in 0..STOREY { for z in 0..W { for x in 0..W {
            let e = self.at(x, y, z);
            if e == 0 || done[idx(x, y, z)] { continue; }
            let free = |x: i32, y: i32, z: i32| self.at(x, y, z) == e && !done[idx(x, y, z)];
            let (mut x1, mut y1, mut z1) = (x + 1, y + 1, z + 1);
            if e == el::BRICK {
                // brick walls merge upward first, into piers of the full height they have: as long horizontal bands
                // (merged along the wall first) they carried the load sideways, round the corners of the building
                while y1 < STOREY && free(x, y1, z) { y1 += 1; }
                // (and only over columns of exactly that height: a band under the windows must not take the feet
                // of the piers beside it)
                let same = |xx: i32, zz: i32| (y..y1).all(|yy| free(xx, yy, zz)) && !(y1 < STOREY && free(xx, y1, zz));
                while x1 < W && x1 - x < 255 && same(x1, z) { x1 += 1; }
                'zb: while z1 < W && z1 - z < 255 { for xx in x..x1 { if !same(xx, z1) { break 'zb; } } z1 += 1; }
            } else {
            while x1 < W && x1 - x < 255 && free(x1, y, z) { x1 += 1; }
            'zl: while z1 < W && z1 - z < 255 { for xx in x..x1 { if !free(xx, y, z1) { break 'zl; } } z1 += 1; }
            'yl: while y1 < STOREY { for zz in z..z1 { for xx in x..x1 { if !free(xx, y1, zz) { break 'yl; } } } y1 += 1; }
            }
            for yy in y..y1 { for zz in z..z1 { for xx in x..x1 { done[idx(xx, yy, zz)] = true; } } }
            let ek = element_in(e, self.kind);
            if e == el::BRICK && RUNNING_BOND {
                // brick laid in courses: units 1 m long (8 ticks) and 0.75 m high (6), each course offset by half a
                // unit (running bond), on the world grid so courses line up across pieces of wall. A wall cracks
                // along its mortar joints: between these units.
                let along = if x1 - x >= z1 - z { 0 } else { 2 };
                let (wx, wy, wz) = (x - HALF, y + self.y0, z - HALF);
                let (ex, ey, ez) = (x1 - HALF, y1 + self.y0, z1 - HALF);
                let mut yy = wy;
                while yy < ey {
                    let c = yy.div_euclid(COURSE_H);
                    let y2 = ((c + 1) * COURSE_H).min(ey);
                    let off = if c % 2 == 0 { 0 } else { UNIT / 2 };
                    let (a0, a1) = if along == 0 { (wx, ex) } else { (wz, ez) };
                    let mut a = a0;
                    while a < a1 {
                        let next = ((a - off).div_euclid(UNIT) + 1) * UNIT + off;
                        let a2 = next.min(a1);
                        let (mn, sz) = if along == 0 { ([a, yy, wz], [a2 - a, y2 - yy, ez - wz]) } else { ([wx, yy, a], [ex - wx, y2 - yy, a2 - a]) };
                        out.add(mn, sz, ek.material, ek.paint, structure);
                        a = a2;
                    }
                    yy = y2;
                }
                continue;
            }
            let b = out.add([x - HALF, y + self.y0, z - HALF], [x1 - x, y1 - y, z1 - z], ek.material, ek.paint, structure);
            if ek.infill { out.flags[b as usize] |= F_INFILL; }
            if ek.material == mat::CONCRETE && !ek.infill { out.flags[b as usize] |= crate::brick::F_REINF; }
            if matches!(e, el::STAIR | el::STAIR2 | el::LANDING) { out.flags[b as usize] |= crate::brick::F_MONO; }
        } } }
    }
}

/// Grid lines of the 7.5 m column grid.
const GRID: [i32; 4] = [-90, -30, 30, 90];
/// Core: the centre bay, walls 2 ticks inside the grid lines; corridor ring 14 ticks wide outside it.
const CORE_H: i32 = 30;
const CORRIDOR: i32 = CORE_H + 14;
/// Stair: two flights of 10 ticks (1.25 m) in the east half of the core.
const STAIR_X: (i32, i32) = (8, 18);
const STAIR_X2: (i32, i32) = (18, 28);
/// Stair opening in the slab above.
const HOLE: ((i32, i32), (i32, i32)) = ((8, 28), (-20, 2));

/// brick units of the mixed building: 1 m long, 0.75 m courses (in ticks)
const UNIT: i32 = 8;
/// lay brick walls unit by unit (else merged into piers like the other masonry, splitting on the course grid when they crack)
const RUNNING_BOND: bool = false;
const COURSE_H: i32 = 6;

/// The mixed building (kind 3), 5 storeys by default, on the same 22.5 m footprint and concrete core:
/// - ground storey: a shop. Brick walls; on the south and north fronts three shop windows between brick piers,
///   each under a steel beam; four steel columns inside at (+-7.5, +-7.5) m
/// - first floor: a 0.25 m concrete slab
/// - upper storeys: load-bearing brick walls 0.375 m thick all round, with 1.5 x 1.75 m windows every 3.75 m
///   under steel lintels; floors of timber (0.25 m) on steel beams along both axes at +-7.5 m (on the columns
///   where they cross), so the timber spans 3.75 m between beams, walls and the core; a timber roof with a brick
///   parapet; timber stud partitions as in the other kinds
fn paint_mixed(s: &mut Storey, k: i32, storeys: i32) {
    use el::*;
    let y0 = s.y0;
    let floor = y0 + SLAB_T;
    let top = y0 + STOREY;
    let roof = k == storeys;
    s.fill((-HALF, HALF), (y0, floor), (-HALF, HALF), if k <= 1 { SLAB } else { TFLOOR });
    // inside the concrete core the floor is concrete too (the stair landings and the lobby)
    s.fill((-CORE_H, CORE_H), (y0, floor), (-CORE_H, CORE_H), SLAB);
    if k > 0 { s.fill(HOLE.0, (y0, floor), HOLE.1, 0); }
    const T: i32 = 3;   // brick wall thickness (0.375 m)
    if roof {
        for (x, z) in [((-HALF, HALF), (-HALF, -HALF + T)), ((-HALF, HALF), (HALF - T, HALF)), ((-HALF, -HALF + T), (-HALF, HALF)), ((HALF - T, HALF), (-HALF, HALF))] {
            s.fill(x, (floor, floor + 6), z, SILL);   // (a parapet carries only itself)
        }
        core_walls(s, floor, top, true);
        return;
    }
    let south = ((-HALF, HALF), (-HALF, -HALF + T));
    let north = ((-HALF, HALF), (HALF - T, HALF));
    let west = ((-HALF, -HALF + T), (-HALF + T, HALF - T));
    let east = ((HALF - T, HALF), (-HALF + T, HALF - T));
    // beams under the floor above (timber floors: storeys 2 and up), in this storey's top 2 ticks
    let beams = k + 1 >= 2;
    let wall_top = top;
    if k == 0 {
        // shop: brick walls all round; on the south and north fronts three shop windows between brick piers
        // (2 m at the corners and between windows), each under a steel beam 0.375 m deep bearing 0.25 m on the
        // piers. (Glass shopfronts on slender posts under one transfer beam were too soft a path: the walls above
        // sent their load round the corners into the side walls.)
        for (x, z) in [south, north, west, east] { s.fill(x, (floor, wall_top), z, BRICK); }
        for (a, b) in [(-74, -38), (-22, 22), (38, 74)] {
            for zz in [south.1, north.1] {
                s.fill((a - 2, b + 2), (top - 3, top), zz, LINTEL);
                s.fill((a, b), (floor, top - 3), zz, 0);
            }
            s.fill((a, b), (floor, top - 3), (-HALF + 1, -HALF + 2), GLASS);
            s.fill((a, b), (floor, top - 3), (HALF - 2, HALF - 1), GLASS);
        }
        // the shop door, in the middle window
        s.fill((-7, 7), (floor, floor + 20), (-HALF, -HALF + T), 0);
    } else {
        // brick walls with windows under steel lintels
        for (x, z) in [south, north, west, east] { s.fill(x, (floor, wall_top), z, BRICK); }
        for c in [-75, -45, -15, 15, 45, 75] {
            let (a, b) = (c - 6, c + 6);
            let (sill, head) = (floor + 7, floor + 21);
            // the brick over the window (to the floor above) sits on its lintel and, like the brick under the
            // window, is no load path between the piers; so the lintel carries only that, between the piers
            s.fill((a, b), (head + 2, top), south.1, SILL); s.fill((a, b), (head + 2, top), north.1, SILL);
            s.fill(west.0, (head + 2, top), (a, b), SILL); s.fill(east.0, (head + 2, top), (a, b), SILL);
            s.fill((a, b), (head, head + 2), south.1, LINTEL); s.fill((a, b), (head, head + 2), north.1, LINTEL);
            s.fill(west.0, (head, head + 2), (a, b), LINTEL); s.fill(east.0, (head, head + 2), (a, b), LINTEL);
            // the brick under the window carries only itself (it is not a load path from pier to pier)
            s.fill((a, b), (floor, sill), south.1, SILL); s.fill((a, b), (floor, sill), north.1, SILL);
            s.fill(west.0, (floor, sill), (a, b), SILL); s.fill(east.0, (floor, sill), (a, b), SILL);
            s.fill((a, b), (sill, head), south.1, 0); s.fill((a, b), (sill, head), north.1, 0);
            s.fill(west.0, (sill, head), (a, b), 0); s.fill(east.0, (sill, head), (a, b), 0);
            s.fill((a, b), (sill, head), (-HALF + 1, -HALF + 2), GLASS); s.fill((a, b), (sill, head), (HALF - 2, HALF - 1), GLASS);
            s.fill((-HALF + 1, -HALF + 2), (sill, head), (a, b), GLASS); s.fill((HALF - 2, HALF - 1), (sill, head), (a, b), GLASS);
        }
        // timber stud partitions round a corridor, as in the other kinds
        let c = CORRIDOR;
        s.fill((-HALF + T, HALF - T), (floor, top), (c, c + 1), PARTITION);
        s.fill((-HALF + T, HALF - T), (floor, top), (-c - 1, -c), PARTITION);
        for &d in &[-40, -20, 10, 34] {
            s.fill((d, d + 7), (floor, floor + 17), (c, c + 1), 0);
            s.fill((d, d + 7), (floor, floor + 17), (-c - 1, -c), 0);
        }
    }
    core_walls(s, floor, top, false);
    // steel columns at (+-7.5, +-7.5) m, 2 ticks square, up to the beams (or the slab)
    for &cx in &[-60, 60] { for &cz in &[-60, 60] {
        s.fill((cx - 1, cx + 1), (floor, top), (cz - 1, cz + 1), SCOL);
    } }
    if beams {
        // steel beams 0.25 m square under the timber floor above, along x at z = +-7.5 m and along z at x = +-7.5 m,
        // bearing on the brick walls at their ends (through the wall's thickness), on the columns where they
        // cross
        for &cz in &[-60, 60] { s.fill((-HALF, HALF), (top - 2, top), (cz - 1, cz + 1), BEAM); }
        for &cx in &[-60, 60] { s.fill((cx - 1, cx + 1), (top - 2, top), (-HALF, HALF), BEAM); }
    }
    stair(s, floor);
}

fn paint_storey(s: &mut Storey, k: i32, storeys: i32) {
    use el::*;
    if s.kind == 3 { paint_mixed(s, k, storeys); return; }
    let y0 = s.y0;
    let floor = y0 + SLAB_T;           // top of this storey's slab
    let top = y0 + STOREY;           // underside of the next slab
    let roof = k == storeys;         // penthouse storey on the roof
    // slab (with the stair opening above the flight from the storey below)
    s.fill((-HALF, HALF), (y0, floor), (-HALF, HALF), SLAB);
    if k > 0 { s.fill(HOLE.0, (y0, floor), HOLE.1, 0); }
    if roof {
        // parapet around the roof edge, 1 m high
        for (x, z) in [((-HALF, HALF), (-HALF, -HALF + 2)), ((-HALF, HALF), (HALF - 2, HALF)), ((-HALF, -HALF + 2), (-HALF, HALF)), ((HALF - 2, HALF), (-HALF, HALF))] {
            s.fill(x, (floor, floor + 8), z, PARAPET);
        }
        // penthouse: the core walls one storey up, roofed, with a door out to the roof
        core_walls(s, floor, top, true);
        return;
    }
    // exterior wall: spandrel 0.875 m, glass band to 3.0 m, header to the slab above; mullions every 1.875 m
    let ring = [((-HALF, HALF), (-HALF, -HALF + 2)), ((-HALF, HALF), (HALF - 2, HALF)), ((-HALF, -HALF + 2), (-HALF + 2, HALF - 2)), ((HALF - 2, HALF), (-HALF + 2, HALF - 2))];
    for (x, z) in ring {
        s.fill(x, (floor, floor + 7), z, SPANDREL);
        s.fill(x, (floor + 7, floor + 24), z, GLASS);
        s.fill(x, (floor + 24, top), z, HEADER);
    }
    for m in (-HALF..=HALF).step_by(15) {
        let (a, b) = (m - 1, m + 1);
        s.fill((a, b), (floor + 7, floor + 24), (-HALF, -HALF + 2), MULLION);
        s.fill((a, b), (floor + 7, floor + 24), (HALF - 2, HALF), MULLION);
        s.fill((-HALF, -HALF + 2), (floor + 7, floor + 24), (a, b), MULLION);
        s.fill((HALF - 2, HALF), (floor + 7, floor + 24), (a, b), MULLION);
    }
    // glass is one tick thick, on the outer face of the wall line
    s.fill((-HALF + 1, HALF - 1), (floor + 7, floor + 24), (-HALF + 1, -HALF + 2), 0);
    s.fill((-HALF + 1, HALF - 1), (floor + 7, floor + 24), (HALF - 2, HALF - 1), 0);
    s.fill((-HALF + 1, -HALF + 2), (floor + 7, floor + 24), (-HALF + 1, HALF - 1), 0);
    s.fill((HALF - 2, HALF - 1), (floor + 7, floor + 24), (-HALF + 1, HALF - 1), 0);
    if k == 0 {
        // lobby entrance: a 1.75 m x 2.5 m opening in the south facade
        s.fill((-7, 7), (floor, floor + 20), (-HALF, -HALF + 2), 0);
    }
    // core walls with the stair door
    core_walls(s, floor, top, false);
    // offices: a corridor ring around the core; the north and south bands run the full width (they own the
    // corners), the east and west bands sit between them. Every office has a door onto the corridor.
    if k > 0 {
        let c = CORRIDOR;
        // corridor walls; the north and south ones run to the facade and close off the east/west bands
        s.fill((-HALF + 2, HALF - 2), (floor, top), (c, c + 1), PARTITION);
        s.fill((-HALF + 2, HALF - 2), (floor, top), (-c - 1, -c), PARTITION);
        s.fill((c, c + 1), (floor, top), (-c, c), PARTITION);
        s.fill((-c - 1, -c), (floor, top), (-c, c), PARTITION);
        // partitions between offices
        for &p in &[-30, 0, 30] {
            s.fill((p, p + 1), (floor, top), (c + 1, HALF - 2), PARTITION);
            s.fill((p, p + 1), (floor, top), (-HALF + 2, -c - 1), PARTITION);
        }
        s.fill((c + 1, HALF - 2), (floor, top), (0, 1), PARTITION);
        s.fill((-HALF + 2, -c - 1), (floor, top), (0, 1), PARTITION);
        // doors, 0.875 x 2.125 m
        for &d in &[-40, -20, 10, 34] {
            s.fill((d, d + 7), (floor, floor + 17), (c, c + 1), 0);
            s.fill((d, d + 7), (floor, floor + 17), (-c - 1, -c), 0);
        }
        for &d in &[-25, 15] {
            s.fill((c, c + 1), (floor, floor + 17), (d, d + 7), 0);
            s.fill((-c - 1, -c), (floor, floor + 17), (d, d + 7), 0);
        }
    }
    // columns on the grid (inside the exterior wall at the perimeter); painted last so they win
    for &gx in &GRID { for &gz in &GRID {
        // (0.625 m square; the masonry building's brick piers 1.125 m since 14 Nov. Brick is a sixth as stiff as
        // the concrete core beside it: at 0.625 m the piers of a 5-storey building shortened enough against the core
        // to load the top floor to 0.68 of its strength under its own weight, 0.50 at 0.875 m; with what a floor
        // carries and concrete's shear at 3 MPa, 0.79 at 0.875 m and 0.54 at 1.125 m: dyn_tests' standing margins)
        let w = if s.kind == 1 { 9 } else { 5 };
        let cx = if gx == -HALF { -HALF + 2 } else if gx == HALF { HALF - 2 - w } else { gx - w / 2 };
        let cz = if gz == -HALF { -HALF + 2 } else if gz == HALF { HALF - 2 - w } else { gz - w / 2 };
        if gx.abs() == CORE_H && gz.abs() == CORE_H { continue; }   // the core corners are core wall
        s.fill((cx, cx + w), (floor, top), (cz, cz + w), COLUMN);
    } }
    // the stair: flight A north along x in STAIR_X (8 steps), landing, flight B south in STAIR_X2 (7 steps)
    stair(s, floor);
}

fn core_walls(s: &mut Storey, floor: i32, top: i32, penthouse: bool) {
    use el::*;
    let c = CORE_H;
    s.fill((-c, c), (floor, top), (-c, -c + 2), CORE);
    s.fill((-c, c), (floor, top), (c - 2, c), CORE);
    s.fill((-c, -c + 2), (floor, top), (-c, c), CORE);
    s.fill((c - 2, c), (floor, top), (-c, c), CORE);
    // door from the corridor (south wall), next to the foot of flight A
    s.fill((STAIR_X.0, STAIR_X.0 + 7), (floor, floor + 17), (-c, -c + 2), 0);
    // west door into the open core lobby
    s.fill((-c, -c + 2), (floor, floor + 17), (-4, 3), 0);
    if penthouse { s.fill((-c, c), (top - SLAB_T, top), (-c, c), SLAB); }
}

fn stair(s: &mut Storey, floor: i32) {
    use el::*;
    // each step is a 0.5 m deep block (its top 0.25 m above the previous): a cast stair with a waist,
    // so there is headroom under the flight above
    for k in 1..=8 {
        let t = floor + 2 * k;
        let z = -24 + 2 * (k - 1);
        s.fill(STAIR_X, ((t - 4).max(floor), t), (z, z + 2), if k % 2 == 0 { STAIR } else { STAIR2 });
    }
    let land = floor + 16;
    s.fill((STAIR_X.0, STAIR_X2.1), (land - 4, land), (-8, 2), LANDING);
    for j in 1..=7 {
        let t = land + 2 * j;
        let z = -8 - 2 * j;
        s.fill(STAIR_X2, (t - 4, t), (z, z + 2), if j % 2 == 0 { STAIR } else { STAIR2 });
    }
}

/// Build the tower: `storeys` storeys plus the roof with its penthouse. Returns the brick store with joints.
pub fn build(storeys: i32, structure: u16) -> Bricks { build_kind(storeys, structure, 0) }
pub fn build_kind(storeys: i32, structure: u16, kind: u8) -> Bricks {
    // kind 4: the complex building, of boxes and free pieces at any angle (crate::complex); its size is its own
    // (the complex building: as designed, two storeys, up to 3 asked; taller if more are)
    if kind == 4 { return if storeys > 3 { crate::complex::build_tall(storeys as usize) } else { crate::complex::build() }; }
    // kind 5: the city, `storeys` lots a side (crate::city)
    if kind == 5 { return crate::city::build(storeys); }
    let mut b = Bricks::new();
    for k in 0..=storeys {
        let mut s = Storey::new(k * STOREY);
        s.kind = kind;
        paint_storey(&mut s, k, storeys);
        // flight B of the storey below reaches 2 ticks into this storey's slab level: nothing to add, the slab
        // is solid there (outside the opening)
        s.emit(&mut b, structure);
    }
    b.build_joints();
    b.log.clear();
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::time::Instant;

    #[test] fn tower_gate() {
        let t = Instant::now();
        let mut b = build(12, 1);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let n = b.alive_count();
        std::println!("tower 12 storeys: {n} bricks, {} joints, {:.1} B/brick records+joints, {:.1} B/brick with index, built in {ms:.0} ms",
            b.live_joints, b.bytes_per_brick(), b.total_bytes_per_brick());
        assert!(b.total_bytes_per_brick() < 100.0, "phase 1 gate: under 100 B a brick");
        // nothing floats: every brick is connected to the ground
        assert!(b.detached().is_empty(), "detached pieces in a fresh tower");
        // incremental joints equal a rebuild
        let mut r = b.clone(); r.build_joints(); assert_eq!(b.joint_set(), r.joint_set());
        // a hit in a slab refines and keeps the joint graph exact
        b.refine_sphere([0.0, (3 * STOREY + 1) as f32, 60.0], 6.0, 1);
        let mut r = b.clone(); r.build_joints(); assert_eq!(b.joint_set(), r.joint_set());
    }

    #[test] fn dimensions_follow_the_scale_standard() {
        let b = build(2, 1);
        // storey height, slab, door and stair rise straight from the bricks
        // walking surfaces up the stair: step tops and the floor (slab top) of each storey
        let mut steps: std::vec::Vec<i32> = (0..b.len() as u32).filter(|&i| b.paint[i as usize] == 4 && b.size(i)[2] == 2).map(|i| b.tbox(i).max[1]).collect();
        for k in 0..=2 { steps.push(k * STOREY + SLAB_T); }
        steps.sort(); steps.dedup();
        let rises: std::vec::Vec<i32> = steps.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(rises.iter().all(|&r| r == 2), "stair rises are 0.25 m: {rises:?}");
        let slab = (0..b.len() as u32).find(|&i| b.paint[i as usize] == 0 && b.mat[i as usize] == mat::CONCRETE).unwrap();
        assert_eq!(b.size(slab)[1], SLAB_T);
    }
}
