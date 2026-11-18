//! The white villa: Le Corbusier's Villa Savoye (Poissy, 1931), to its published dimensions on the 0.125 m grid.
//!
//! - A grid of 5 x 5 concrete pilotis 4.75 m apart, 0.25 m square. On them, 3 m up, the living floor: a box
//!   19.75 x 22 m (the slab runs 1.25 m past the end rows north and south), walls white, a ribbon window 1.125 m
//!   high round all four sides at 1 m above the floor.
//! - Under the box, set back a bay on three sides, the ground storey: dark green walls, and at the south end a
//!   half round of glass (eight panes, a 4.5 m radius: the turning circle of the car it was drawn for), the
//!   entrance in its middle.
//! - Up the middle, the ramp: two flights a storey between the hall and the living floor, two more outside from
//!   the terrace to the roof, 1.25 m wide, either side of a spine wall, with a landing at the north end.
//! - The living floor: rooms along the west side and in the north-east; the south-east quarter is the terrace,
//!   open to the sky inside the box's walls (their ribbon unglazed there), glass between it and the living room.
//! - On the roof, the solarium: screen walls 2 m high, one a half round, one straight with a window.
//! The spiral stair is left out.
use crate::brick::{mat, Bricks, F_INFILL};
use crate::kit::*;
use alloc::vec::Vec;

/// the column grid (m), and the levels: the living floor's slab (underside, top), the roof slab (underside, top)
pub const GRID: [f64; 5] = [-9.5, -4.75, 0.0, 4.75, 9.5];
pub const F1: (f64, f64) = (3.0, 3.25);
pub const F2: (f64, f64) = (6.25, 6.5);
/// the slabs' edges (the walls stand outside them, 0.25 m thick)
const EX: f64 = 9.625;
const EZ: f64 = 10.75;
/// the ramp's well in the slabs: x, z
const WELL_X: (f64, f64) = (-1.375, 1.375);
const WELL_Z: (f64, f64) = (-4.5, 6.25);
/// the window: sill and head above the floor
const SILL: f64 = 1.0;
const HEAD: f64 = 2.125;

#[derive(Default, Clone, Debug)]
pub struct Parts { pub pilotis: Vec<u32>, pub columns: Vec<u32>, pub slabs: Vec<u32>, pub ramps: Vec<Vec<u32>>, pub glass: Vec<u32>, pub round: Vec<u32>, pub solarium: Vec<u32>, pub contents: Vec<u32> }

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut p = Parts::default();
    let mut sid: u16 = 100;
    let col = |b: &mut Bricks, x: f64, z: f64, y0: f64, y1: f64| bx(b, [x - 0.125, y0, z - 0.125], [x + 0.125, y1, z + 0.125], mat::CONCRETE, P_COLUMN);
    // ---- the pilotis and the columns above them (the two in the ramp's well are the spine wall)
    for &x in &GRID { for &z in &GRID {
        let in_well = x == 0.0 && z > WELL_Z.0 && z < WELL_Z.1;
        if in_well { continue; }
        p.pilotis.push(col(&mut b, x, z, 0.0, F1.0));
        // (no column stands in the open terrace but on its edges: the south-east quarter's inner one is left out)
        let terrace_inner = x == 4.75 && z == -4.75;
        if !terrace_inner { p.columns.push(col(&mut b, x, z, F1.1, F2.0)); }
    } }
    // ---- the slabs, round the ramp's well; the roof is open over the terrace too, but for a strip along its two
    // outer edges (the frame the box's walls hang on there)
    let slab = |b: &mut Bricks, x0: f64, x1: f64, z0: f64, z1: f64, y: (f64, f64)| bx(b, [x0, y.0, z0], [x1, y.1, z1], mat::CONCRETE, P_SLAB);
    for &(x0, x1, z0, z1) in &[(-EX, WELL_X.0, -EZ, EZ), (WELL_X.1, EX, -EZ, EZ), (WELL_X.0, WELL_X.1, -EZ, WELL_Z.0), (WELL_X.0, WELL_X.1, WELL_Z.1, EZ)] { p.slabs.push(slab(&mut b, x0, x1, z0, z1, F1)); }
    for &(x0, x1, z0, z1) in &[(-EX, WELL_X.0, -EZ, EZ), (WELL_X.0, WELL_X.1, WELL_Z.1, EZ), (WELL_X.1, EX, 0.0, EZ), (WELL_X.0, EX, -EZ, -9.25), (8.25, EX, -9.25, 0.0), (WELL_X.0, WELL_X.1, -9.25, WELL_Z.0)] { p.slabs.push(slab(&mut b, x0, x1, z0, z1, F2)); }
    // ---- the box's walls, outside the slabs' edges: white rendered masonry below the window (hung on the floor's
    // edge, from its underside), the ribbon (glass between steel mullions every 2.375 m; open on the terrace), and a
    // concrete band above it to the roof's top
    let (wy0, sill, head, top) = (F1.0, F1.1 + SILL, F1.1 + HEAD, F2.1);
    let mut wall = |b: &mut Bricks, axis: usize, at: (f64, f64), a0: f64, a1: f64, open: (f64, f64)| {
        let mk = |a: f64, c: f64, y0: f64, y1: f64| if axis == 0 { ([a, y0, at.0], [c, y1, at.1]) } else { ([at.0, y0, a], [at.1, y1, c]) };
        for q in run_spans(a0, a1, 4.75) { let (lo, hi) = mk(q.0, q.1, wy0, sill); infill(b, lo, hi, mat::MASONRY, P_WHITE); }
        for q in run_spans(a0, a1, 4.75) { let (lo, hi) = mk(q.0, q.1, head, top); bx(b, lo, hi, mat::CONCRETE, P_WHITE); }
        // mullions on the 2.375 m grid from the wall's start; glass between them where the wall is not the terrace's
        let mut a = a0;
        while a < a1 - 1e-9 {
            let c = (a + 2.375).min(a1);
            let (lo, hi) = mk(a, a + 0.125, sill, head); bx(b, lo, hi, mat::STEEL, P_WHITE);
            let mid = 0.5 * (a + c);
            if !(mid > open.0 && mid < open.1) && c - a > 0.25 { let (lo, hi) = mk(a + 0.125, c, sill, head); p.glass.push(infill(b, lo, hi, mat::GLASS, 0)); }
            a = c;
        }
    };
    // south and north (along x, the corners theirs), east and west (along z, between them)
    wall(&mut b, 0, (-EZ - 0.25, -EZ), -EX - 0.25, EX + 0.25, (WELL_X.1, 99.0));
    wall(&mut b, 0, (EZ, EZ + 0.25), -EX - 0.25, EX + 0.25, (99.0, 99.0));
    wall(&mut b, 2, (EX, EX + 0.25), -EZ, EZ, (-99.0, 0.0));
    wall(&mut b, 2, (-EX - 0.25, -EX), -EZ, EZ, (99.0, 99.0));
    // ---- the ground storey: walls on the second column lines east and west and under the north edge, dark green;
    // garage doors in the east wall, a strip of windows in the west one, a door in the north one
    let g = |b: &mut Bricks, lo: P3, hi: P3| bx(b, lo, hi, mat::MASONRY, P_GREEN);
    // (inside the columns' lines and 0.125 m clear of the columns: against them, the walls took the floor's load
    // off the columns through their sides, 2.4 times what brick on concrete holds in shear)
    // And they stop 0.125 m under the floor: the frame carries the house, the ground storey's walls only themselves.
    // (Built up to the slab they shared its load with the columns, and where a cross wall's end met the concrete
    // band over the windows the joint stood at 0.61 to 0.77 depending on how the villa is turned: over 0.7 it wears.)
    let (gx, gz0, gz1, gh) = (4.5, -4.75, 9.25, F1.0 - 0.125);
    // east: three garage doors 2.5 m wide and 2.25 m high, piers between
    {
        let mut z = gz0;
        for k in 0..3 { let d0 = -4.0 + 3.25 * k as f64; g(&mut b, [gx - 0.25, 0.0, z], [gx, gh, d0]); g(&mut b, [gx - 0.25, 2.25, d0], [gx, gh, d0 + 2.5]); z = d0 + 2.5; }
        g(&mut b, [gx - 0.25, 0.0, z], [gx, gh, gz1]);
    }
    // west: three windows 0.75 m high at 1.5 m, 2.5 m long, piers between them
    g(&mut b, [-gx, 0.0, gz0], [-gx + 0.25, 1.5, gz1]);
    // (the band over the windows is a concrete lintel on the piers: of brick over one 9 m window it stood at 0.61 to
    // 0.68 of its strength, and as concrete over that window it hung on the cross wall's end at 0.89 to 1.0)
    bx(&mut b, [-gx, 2.25, gz0], [-gx + 0.25, gh, gz1], mat::CONCRETE, P_GREEN);
    for w in [gz0, -2.0, 0.5, 1.25, 3.75, 4.5, 6.25, gz1].chunks(2) { g(&mut b, [-gx, 1.5, w[0]], [-gx + 0.25, 2.25, w[1]]); }
    for w in [-2.0, 0.5, 1.25, 3.75, 4.5, 6.25].chunks(2) { p.glass.push(infill(&mut b, [-gx + 0.125, 1.5, w[0]], [-gx + 0.25, 2.25, w[1]], mat::GLASS, 0)); }
    // north: a service door 1 m wide and 2.125 m high at x = 3
    g(&mut b, [-gx + 0.25, 0.0, gz1 - 0.25], [2.5, gh, gz1]);
    g(&mut b, [2.5, 2.125, gz1 - 0.25], [3.5, gh, gz1]);
    g(&mut b, [3.5, 0.0, gz1 - 0.25], [gx - 0.25, gh, gz1]);
    // inside: a wall across at z = 6.5 (the garage and the servants' rooms behind it), a door in it
    g(&mut b, [-gx + 0.25, 0.0, 6.5], [-3.0, gh, 6.75]);
    g(&mut b, [-3.0, 2.125, 6.5], [-2.0, gh, 6.75]);
    g(&mut b, [-2.0, 0.0, 6.5], [-0.125, gh, 6.75]);
    g(&mut b, [0.125, 0.0, 6.5], [gx - 0.25, gh, 6.75]);
    // the half round of glass at the south end: eight panes about (0, -4.75), their middles at 4.5 m; the two
    // in the middle are the entrance (a pane each, their heads at 2.25 m, open under them)
    let n = 8;
    for k in 0..n {
        let a0 = PI + PI * k as f64 / n as f64;
        let a1 = PI + PI * (k + 1) as f64 / n as f64;
        let am = 0.5 * (a0 + a1);
        let half = 4.5 * crate::kit::sin(PI / (2 * n) as f64) / crate::kit::cos(PI / (2 * n) as f64);
        let c = [4.5 * cos(am), 0.0, gz0 + 4.5 * sin(am)];
        // (a pane's own frame: its width along x, its thickness along z; turned so its thickness points outward)
        let turn = -(am - PI / 2.0) + PI;
        let door = k == n / 2 - 1 || k == n / 2;
        // (the glass goes up to the floor: the panes over the entrance hang from it)
        let (y0, y1) = if door { (2.25, F1.0) } else { (0.0, F1.0) };
        sid += 1;
        let i = turned(&mut b, [c[0], 0.5 * (y0 + y1), c[2]], turn, [half, 0.5 * (y1 - y0), 0.0625], mat::GLASS, 0, sid);
        b.flags[i as usize] |= F_INFILL;
        p.round.push(i);
    }
    // ---- the ramp: a spine wall up the middle from the ground to 1 m above the roof, the landings at its north
    // end, four flights (two inside, two outside above them)
    let rise = F1.1 / 2.0;
    let (rz0, rz1) = (WELL_Z.0, 5.0);
    // (concrete: the flights are carried by it along one edge; of brick it held them at 2.0 of its strength)
    // (clear of the slabs' edges at the well's ends, 0.125 m: touching them on its 0.25 m end face it took their
    // load there, at 0.69 to 0.93 of the joint's strength depending on how the villa is turned)
    run(&mut b, [-0.125, 0.0, rz0 + 0.125], [0.125, F2.1 + 1.0, rz1 - 0.125], 1, 3.125, mat::CONCRETE, P_WHITE);
    // the wall the landings bear on, at the well's north end (clear of the slabs too)
    run(&mut b, [WELL_X.0 + 0.125, 0.0, 5.875], [WELL_X.1 - 0.125, F2.1 + 1.0, WELL_Z.1 - 0.125], 1, 3.125, mat::CONCRETE, P_WHITE);
    for storey in 0..2 {
        let y0 = storey as f64 * F1.1;
        // the landing, half a storey up
        bx(&mut b, [WELL_X.0, y0 + rise - 0.25, rz1], [WELL_X.1, y0 + rise, 5.875], mat::CONCRETE, P_STAIR);
        // up along +z on the east side of the spine, back along -z on the west side
        // (1.125 m wide, 0.125 m clear of the well's sides: against a slab's edge near its head a flight hung on it
        // there by a sliver of contact, at 0.6 to 0.73)
        for &(xs, za, zb, ya, yb) in &[(0.6875, rz0, rz1, y0, y0 + rise), (-0.6875, rz1, rz0, y0 + rise, y0 + 2.0 * rise)] {
            // (0.3125 m thick: a flight is a cantilever off the spine wall along one edge, and at 0.25 m its joint
            // there stood at 0.58 to 0.68)
            let (f, h) = ([xs, ya - 0.15625, za], [xs, yb - 0.15625, zb]);
            let a = unit(sub(h, f));
            let up = [0.0, 1.0, 0.0];
            let nn = unit(sub(up, mul(a, dot(up, a))));
            let dz = if zb > za { 1.0 } else { -1.0 };
            // (its ends cut upright: the foot against the hall's floor edge or the landing, the head against the
            // landing or the floor's edge in the well)
            p.ramps.push(member(&mut b, f, h, nn, [1.0, 0.0, 0.0], 0.3125, 1.125, [0.0, 0.0, dz], [0.0, 0.0, dz], 2.0, mat::CONCRETE, P_STAIR, &mut sid));
        }
    }
    // ---- the living floor: glass between the rooms and the open terrace and well; partitions
    let (fy0, fy1) = (F1.1, F2.0);
    let mut pane = |b: &mut Bricks, lo: P3, hi: P3| p.glass.push(infill(b, lo, hi, mat::GLASS, 0));
    // the living room's glass wall to the terrace (along z = 0, x from the well to the east wall), in four panes
    // (north of the columns on that line, under the roof's edge)
    for q in run_spans(WELL_X.1 + 0.125, EX, 2.125) { pane(&mut b, [q.0, fy0, 0.125], [q.1, fy1, 0.25]); }
    // round the well: west side, north side, and the east side north of the terrace
    for q in run_spans(WELL_Z.0, WELL_Z.1, 2.75) { pane(&mut b, [WELL_X.0 - 0.125, fy0, q.0], [WELL_X.0, fy1, q.1]); }
    for q in run_spans(0.125, WELL_Z.1, 3.125) { pane(&mut b, [WELL_X.1, fy0, q.0], [WELL_X.1 + 0.125, fy1, q.1]); }
    // (the well's north side is the landings' wall)
    // partitions (white stud walls, 0.125 m): the west side in four rooms with a corridor along the well, doors 0.875 m
    let part = |b: &mut Bricks, lo: P3, hi: P3| infill(b, lo, hi, mat::TIMBER, P_WHITE);
    let cx = -3.0;   // the corridor's west wall
    for &z in &[-5.5, -0.5, 4.5] { part(&mut b, [-EX, fy0, z], [cx, fy1, z + 0.125]); }
    {
        // the corridor wall with a door to each room
        let mut z = -EZ;
        for &d in &[-8.5, -3.5, 1.5, 7.0] { part(&mut b, [cx, fy0, z], [cx + 0.125, fy1, d]); part(&mut b, [cx, fy0 + 2.125, d], [cx + 0.125, fy1, d + 0.875]); z = d + 0.875; }
        part(&mut b, [cx, fy0, z], [cx + 0.125, fy1, EZ]);
    }
    // the corridor's south end is closed to the terrace side by the box's wall; the kitchen in the north-east
    part(&mut b, [WELL_X.1 + 0.125, fy0, 6.25], [5.0, fy1, 6.375]);
    part(&mut b, [6.0, fy0, 6.25], [EX, fy1, 6.375]);
    // ---- the solarium on the roof: a half round screen (six panels about (-5, 5), radius 3 m, open to the south
    // east) and a straight screen with a window along the north edge of the well's strip
    for k in 0..6 {
        let am = PI / 2.0 + PI * (k as f64 + 0.5) / 6.0;
        let half = 3.0 * crate::kit::sin(PI / 12.0) / crate::kit::cos(PI / 12.0);
        let c = [-5.0 + 3.0 * cos(am), F2.1 + 1.0, 5.0 + 3.0 * sin(am)];
        sid += 1;
        p.solarium.push(turned(&mut b, c, -(am - PI / 2.0) + PI, [half, 1.0, 0.125], mat::MASONRY, P_WHITE, sid));
    }
    for &(x0, x1, y0, y1) in &[(-1.375, 1.5, 0.0, 2.0), (1.5, 3.5, 0.0, 0.75), (1.5, 3.5, 1.5, 2.0), (3.5, 9.625, 0.0, 2.0)] {
        for q in run_spans(x0, x1, 3.5) { p.solarium.push(bx(&mut b, [q.0, F2.1 + y0, 8.0], [q.1, F2.1 + y1, 8.25], mat::MASONRY, P_WHITE)); }
    }
    // ---- furniture, loose
    let mut obj = |b: &mut Bricks, parts: &[(P3, P3, u8, u8)]| { sid += 1; for &(lo, hi, m, paint) in parts { loose(b, &mut p.contents, lo, hi, m, paint, sid); } };
    let f = F1.1;
    // the living room (north-east): a long table, six chairs, two armchairs, a low table
    obj(&mut b, &[([3.0, f + 0.625, 2.0], [5.5, f + 0.75, 3.0], mat::TIMBER, P_WOOD), ([3.125, f, 2.125], [3.25, f + 0.625, 2.875], mat::STEEL, P_STEEL), ([5.25, f, 2.125], [5.375, f + 0.625, 2.875], mat::STEEL, P_STEEL)]);
    for k in 0..3 { for &(z, back) in &[(1.375, 1.375), (3.125, 3.5)] {
        let x = 3.25 + 0.75 * k as f64;
        obj(&mut b, &[([x, f + 0.375, z], [x + 0.5, f + 0.5, z + 0.5], mat::TIMBER, P_WOOD), ([x + 0.125, f, z + 0.125], [x + 0.375, f + 0.375, z + 0.375], mat::STEEL, P_STEEL), ([x, f + 0.5, back], [x + 0.5, f + 1.0, back + 0.125], mat::TIMBER, P_WOOD)]);
    } }
    for &x in &[6.75, 8.0] { obj(&mut b, &[([x, f + 0.25, 4.0], [x + 0.75, f + 0.375, 4.75], mat::TIMBER, P_DARKWOOD), ([x + 0.125, f, 4.125], [x + 0.625, f + 0.25, 4.625], mat::STEEL, P_STEEL), ([x, f + 0.375, 4.75], [x + 0.75, f + 0.875, 4.875], mat::TIMBER, P_DARKWOOD)]); }
    obj(&mut b, &[([7.0, f + 0.375, 2.5], [8.25, f + 0.5, 3.25], mat::GLASS, 0), ([7.5, f, 2.75], [7.75, f + 0.375, 3.0], mat::STEEL, P_STEEL)]);
    // the bedrooms (west): a bed in each of three, a desk in the fourth
    for &z in &[-9.5, -4.5, 0.5] { obj(&mut b, &[([-9.0, f, z], [-7.0, f + 0.25, z + 1.5], mat::TIMBER, P_WOOD), ([-9.0, f + 0.25, z], [-7.0, f + 0.5, z + 1.5], mat::TIMBER, P_PARTITION)]); }
    obj(&mut b, &[([-9.0, f + 0.625, 6.0], [-7.5, f + 0.75, 6.875], mat::TIMBER, P_WOOD), ([-8.875, f, 6.125], [-8.75, f + 0.625, 6.75], mat::STEEL, P_STEEL), ([-7.75, f, 6.125], [-7.625, f + 0.625, 6.75], mat::STEEL, P_STEEL)]);
    // on the terrace: a concrete table (Le Corbusier's, built in) and two planters
    bx(&mut b, [5.0, f + 0.625, -6.0], [7.5, f + 0.75, -5.0], mat::CONCRETE, P_WHITE);
    bx(&mut b, [6.125, f, -5.625], [6.375, f + 0.625, -5.375], mat::CONCRETE, P_WHITE);
    for &x in &[2.0, 8.5] { obj(&mut b, &[([x, f, -9.0], [x + 0.75, f + 0.5, -8.25], mat::CONCRETE, P_WHITE), ([x + 0.125, f + 0.5, -8.875], [x + 0.625, f + 1.0, -8.375], mat::TIMBER, P_GREEN)]); }
    // in the garage: a car, loose (a body on four wheels; steel)
    obj(&mut b, &[([1.875, 0.25, -2.5], [3.625, 0.875, 1.75], mat::STEEL, P_OXIDE), ([2.0, 0.875, -1.5], [3.5, 1.375, 0.75], mat::GLASS, 0),
        ([1.875, 0.0, -2.0], [2.125, 0.25, -1.375], mat::TIMBER, P_DARKWOOD), ([3.375, 0.0, -2.0], [3.625, 0.25, -1.375], mat::TIMBER, P_DARKWOOD), ([1.875, 0.0, 0.625], [2.125, 0.25, 1.25], mat::TIMBER, P_DARKWOOD), ([3.375, 0.0, 0.625], [3.625, 0.25, 1.25], mat::TIMBER, P_DARKWOOD)]);
    b.build_joints();
    b.log.clear();
    (b, p)
}
/// a0..a1 in equal spans of at most `max`, on the grid
fn run_spans(a0: f64, a1: f64, max: f64) -> Vec<(f64, f64)> {
    let n = (((a1 - a0) / max) - 1e-9).max(0.0) as usize + 1;
    let at = |k: usize| if k == 0 { a0 } else if k == n { a1 } else { tk(a0 + (a1 - a0) * k as f64 / n as f64) as f64 * T };
    (0..n).map(|k| (at(k), at(k + 1))).filter(|q| q.1 > q.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    /// (by hand: cargo test --release -p sf-demo explore_the_villas -- --ignored --nocapture)
    #[test] #[ignore] fn explore_the_villas_worst_joints() {
        let b = build();
        worst_joints(&b, 6);
        for q in 1u8..4 { let mut t = Bricks::new(); t.place(&b, [-320, 0, -320], q, 0); t.build_joints(); std::println!("turned {q}:"); worst_joints(&t, 4); }
    }
    #[test] fn the_white_villa_is_what_it_should_be_and_stands() {
        let (b, p) = build_parts();
        assert_eq!(p.pilotis.len(), 23); assert_eq!(p.columns.len(), 22);
        assert_eq!(p.ramps.len(), 4);
        assert_eq!(p.round.len(), 8); assert!(p.solarium.len() >= 10);
        let r = stands("the white villa", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
