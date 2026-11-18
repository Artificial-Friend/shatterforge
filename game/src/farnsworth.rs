//! The glass house: Mies van der Rohe's Farnsworth House (Plano, Illinois, 1951), to its published dimensions
//! on the 0.125 m grid.
//!
//! - Two slabs, floor and roof, 23.5 x 8.5 m (77 x 28 ft), held between eight white steel columns that stand
//!   outside their long edges, 6.75 m (22 ft) apart, the slabs running 1.625 m past the end columns. The floor is
//!   1.625 m above the ground (the house stands on a flood plain), the room 2.875 m high.
//! - Each slab: steel channels round its edge, steel cross beams every 2 m between them, a 0.125 m concrete deck
//!   on the beams (travertine on the floor).
//! - The east 16.75 m is the room, glass from floor to roof on all four sides (panes between steel mullions, the
//!   door in the west wall); the west 6.75 m is the open porch.
//! - In the room a core of timber panels (two bathrooms, the kitchen along its north side, the fireplace on its
//!   south side, its flue through the roof), a wardrobe, and a few pieces of furniture, loose.
//! - South of the house, lower (0.75 m) and set 6.75 m to the west, the terrace, 16.75 x 6.75 m, of the same
//!   construction, its north edge on the house's columns; travertine steps from the ground to the terrace and
//!   from the terrace to the porch, on one line between the columns.
use crate::brick::{mat, Bricks};
use crate::kit::*;
use alloc::vec::Vec;

/// the slabs (m): half length, half width, the columns along x
pub const HL: f64 = 11.75;
pub const HW: f64 = 4.25;
pub const COLS: [f64; 4] = [-10.125, -3.375, 3.375, 10.125];
/// the floor's top, the roof's underside and top
pub const FLOOR: f64 = 1.625;
pub const CEIL: f64 = 4.5;
pub const ROOF: f64 = 5.0;
/// where the room begins (its west glass wall)
pub const ROOM_W: f64 = -5.0;
/// the terrace: its x range, its z range, its top
pub const TER_X: (f64, f64) = (-18.5, -1.75);
pub const TER_Z: (f64, f64) = (-11.25, -4.5);
pub const TER_TOP: f64 = 0.75;
const TER_COLS: [f64; 3] = [-16.875, -10.125, -3.375];

#[derive(Default, Clone, Debug)]
pub struct Parts { pub columns: Vec<u32>, pub terrace_columns: Vec<u32>, pub glass: Vec<u32>, pub floor: Vec<u32>, pub roof: Vec<u32>, pub core: Vec<u32>, pub contents: Vec<u32>, pub steps: Vec<u32> }

/// A slab of the house's construction between x0..x1 and z0..z1 with its top at `top`: channels 0.25 m wide and
/// 0.5 m deep round the edge, cross beams 0.125 x 0.375 m every 2 m (on the grid of whole 2 m from x = 0), a 0.125 m
/// concrete deck on them. Returns the deck's bricks.
fn slab(b: &mut Bricks, x0: f64, x1: f64, z0: f64, z1: f64, top: f64, deck_paint: u8) -> Vec<u32> {
    let (bot, beam_top) = (top - 0.5, top - 0.125);
    // the long channels, cut at the columns' lines so that each length is a member of its own
    for &(za, zb) in &[(z0, z0 + 0.25), (z1 - 0.25, z1)] { run(b, [x0, bot, za], [x1, top, zb], 0, 6.75, mat::STEEL, P_WHITE); }
    for &(xa, xb) in &[(x0, x0 + 0.25), (x1 - 0.25, x1)] { bx(b, [xa, bot, z0 + 0.25], [xb, top, z1 - 0.25], mat::STEEL, P_WHITE); }
    let mut xs = Vec::new();
    let mut x = (floor_to(x0 + 0.25, 2.0)) + 2.0;
    while x < x1 - 0.25 - 0.5 { if x > x0 + 0.25 + 0.5 { xs.push(x); } x += 2.0; }
    for &x in &xs { bx(b, [x, bot, z0 + 0.25], [x + 0.125, beam_top, z1 - 0.25], mat::STEEL, P_WHITE); }
    // the deck: a panel from beam centre to beam centre
    let mut edges = alloc::vec![x0 + 0.25]; edges.extend(xs.iter().copied()); edges.push(x1 - 0.25);
    let mut out = Vec::new();
    for w in edges.windows(2) { out.push(bx(b, [w[0], beam_top, z0 + 0.25], [w[1], top, z1 - 0.25], mat::CONCRETE, deck_paint)); }
    out
}
fn floor_to(x: f64, m: f64) -> f64 { crate::math::floor64(x / m) * m }

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut p = Parts::default();
    let mut sid: u16 = 100;
    // ---- the house: eight columns outside the slabs' long edges, from the ground to the roof's top
    for &x in &COLS { for &s in &[-1.0f64, 1.0] {
        let (za, zb) = if s < 0.0 { (-HW - 0.25, -HW) } else { (HW, HW + 0.25) };
        p.columns.push(bx(&mut b, [x - 0.125, 0.0, za], [x + 0.125, ROOF, zb], mat::STEEL, P_WHITE));
    } }
    p.floor = slab(&mut b, -HL, HL, -HW, HW, FLOOR, P_TRAVERTINE);
    p.roof = slab(&mut b, -HL, HL, -HW, HW, ROOF, P_WHITE);
    // ---- the glass: panes 0.125 m thick standing on the floor's channels, under the roof's; mullions at the
    // columns and midway between them, and at the corners
    let (gy0, gy1) = (FLOOR, CEIL);
    // (a mullion is the 0.125 m from its x on; a pane runs from there to the next mullion's x)
    let mut mull: Vec<f64> = alloc::vec![ROOM_W];
    for k in 0..=5 { let x = -3.375 + 3.375 * k as f64; if x < HL - 0.5 { mull.push(x); } }
    mull.push(HL - 0.125);
    for &s in &[-1.0f64, 1.0] {
        let (za, zb) = if s < 0.0 { (-HW + 0.125, -HW + 0.25) } else { (HW - 0.25, HW - 0.125) };
        for &x in &mull { bx(&mut b, [x, gy0, za], [x + 0.125, gy1, zb], mat::STEEL, P_WHITE); }
        for w in mull.windows(2) { p.glass.push(infill(&mut b, [w[0] + 0.125, gy0, za], [w[1], gy1, zb], mat::GLASS, 0)); }
    }
    // the east wall: two panes about a mullion; the west wall: a pane each side of the door (2 m wide, open)
    for &(x, door) in &[(HL - 0.125, false), (ROOM_W, true)] {
        // (between the long walls' glass; a mullion is the 0.125 m from its z on)
        let zs: &[f64] = if door { &[-HW + 0.25, -1.0, 1.0, HW - 0.25] } else { &[-HW + 0.25, 0.0, HW - 0.25] };
        for (k, w) in zs.windows(2).enumerate() {
            if door && k == 1 {
                // the door's frame: a head at 2.25 m and the glass above it
                bx(&mut b, [x, gy0 + 2.25, w[0]], [x + 0.125, gy0 + 2.375, w[1] + 0.125], mat::STEEL, P_WHITE);
                p.glass.push(infill(&mut b, [x, gy0 + 2.375, w[0]], [x + 0.125, gy1, w[1] + 0.125], mat::GLASS, 0));
                continue;
            }
            let (a, c) = (w[0] + if k > 0 { 0.125 } else { 0.0 }, w[1]);
            p.glass.push(infill(&mut b, [x, gy0, a], [x + 0.125, gy1, c], mat::GLASS, 0));
        }
        for &z in &zs[1..zs.len() - 1] { bx(&mut b, [x, gy0, z], [x + 0.125, if door { gy0 + 2.25 } else { gy1 }, z + 0.125], mat::STEEL, P_WHITE); }
    }
    // ---- the core: timber panels 0.125 m thick to 0.375 m under the roof (the roof does not bear on it), 7.25 x
    // 3.5 m, north of the middle; a cross wall between the two bathrooms and the plant room between them
    let (cx0, cx1, cz0, cz1, ch) = (1.0, 8.25, -0.5, 3.0, CEIL - 0.375);
    let wall = |b: &mut Bricks, lo: P3, hi: P3| infill(b, lo, hi, mat::TIMBER, P_WOOD);
    // south and north faces in three panels each, the ends, two cross walls
    for &(za, zb) in &[(cz0, cz0 + 0.125), (cz1 - 0.125, cz1)] { for w in [cx0, 3.25, 6.0, cx1].windows(2) { p.core.push(wall(&mut b, [w[0], FLOOR, za], [w[1], ch, zb])); } }
    // (the bathrooms' doors: in the ends, 0.75 m wide and 2 m high, towards the south)
    for &(xa, xb) in &[(cx0, cx0 + 0.125), (cx1 - 0.125, cx1)] {
        p.core.push(wall(&mut b, [xa, FLOOR, cz0 + 0.125], [xb, ch, cz0 + 0.5]));
        p.core.push(wall(&mut b, [xa, FLOOR + 2.0, cz0 + 0.5], [xb, ch, cz0 + 1.25]));
        p.core.push(wall(&mut b, [xa, FLOOR, cz0 + 1.25], [xb, ch, cz1 - 0.125]));
    }
    for &x in &[3.25, 6.0] { p.core.push(wall(&mut b, [x, FLOOR, cz0 + 0.125], [x + 0.125, ch, cz1 - 0.125])); }
    // the fireplace in the south face: a travertine hearth, and its steel flue from the core through the roof
    bx(&mut b, [3.75, FLOOR, cz0 - 0.5], [5.5, FLOOR + 0.125, cz0], mat::MASONRY, P_TRAVERTINE);
    // (the flue and the vent stack stand on the floor and pass through holes in the roof deck: they are joined to
    // the floor only)
    // the kitchen: a steel counter along the core's north face
    bx(&mut b, [cx0 + 0.25, FLOOR, cz1], [cx1 - 0.25, FLOOR + 0.875, cz1 + 0.625], mat::STEEL, P_STEEL);
    // the wardrobe: a free-standing teak cabinet towards the east end
    {
        sid += 1; let s = sid;
        loose(&mut b, &mut p.contents, [9.25, FLOOR, -2.0], [9.875, FLOOR + 1.75, 1.5], mat::TIMBER, P_WOOD, s);
    }
    // ---- furniture, loose: a bed, a dining table with four chairs, two lounge chairs and a low table, a desk
    let mut obj = |b: &mut Bricks, parts: &[(P3, P3, u8, u8)]| { sid += 1; for &(lo, hi, m, paint) in parts { loose(b, &mut p.contents, lo, hi, m, paint, sid); } };
    let f = FLOOR;
    // the bed (east of the core): frame and mattress
    obj(&mut b, &[([8.75, f, -3.5], [10.75, f + 0.25, -2.0], mat::TIMBER, P_WOOD), ([8.75, f + 0.25, -3.5], [10.75, f + 0.5, -2.0], mat::TIMBER, P_PARTITION)]);
    // the dining table (west of the core, by the door) and its chairs
    obj(&mut b, &[([-3.5, f + 0.625, 1.25], [-1.5, f + 0.75, 2.25], mat::TIMBER, P_WOOD), ([-3.375, f, 1.375], [-3.25, f + 0.625, 2.125], mat::STEEL, P_STEEL), ([-1.75, f, 1.375], [-1.625, f + 0.625, 2.125], mat::STEEL, P_STEEL)]);
    for &(x, z) in &[(-3.125, 0.625), (-2.25, 0.625), (-3.125, 2.375), (-2.25, 2.375)] {
        let back = if z < 1.0 { (z, z + 0.125) } else { (z + 0.375, z + 0.5) };
        obj(&mut b, &[([x, f + 0.375, z], [x + 0.5, f + 0.5, z + 0.5], mat::TIMBER, P_WOOD), ([x + 0.125, f, z + 0.125], [x + 0.375, f + 0.375, z + 0.375], mat::STEEL, P_STEEL), ([x, f + 0.5, back.0], [x + 0.5, f + 1.0, back.1], mat::TIMBER, P_WOOD)]);
    }
    // the sitting area south of the core: two lounge chairs and a low glass table
    for &x in &[2.5, 4.5] {
        obj(&mut b, &[([x, f + 0.25, -3.25], [x + 0.75, f + 0.375, -2.5], mat::TIMBER, P_DARKWOOD), ([x + 0.125, f, -3.125], [x + 0.625, f + 0.25, -2.625], mat::STEEL, P_STEEL), ([x, f + 0.375, -3.25], [x + 0.75, f + 0.875, -3.125], mat::TIMBER, P_DARKWOOD)]);
    }
    obj(&mut b, &[([3.375, f + 0.375, -2.25], [4.375, f + 0.5, -1.5], mat::GLASS, 0), ([3.75, f, -2.0], [4.0, f + 0.375, -1.75], mat::STEEL, P_STEEL)]);
    // the desk at the east end
    obj(&mut b, &[([10.0, f + 0.625, 0.5], [11.0, f + 0.75, 2.25], mat::TIMBER, P_WOOD), ([10.125, f, 0.625], [10.25, f + 0.625, 2.125], mat::STEEL, P_STEEL), ([10.75, f, 0.625], [10.875, f + 0.625, 2.125], mat::STEEL, P_STEEL)]);
    // ---- the terrace: six short columns outside its long edges, the same slab
    // (its north edge bears on the house's two western south columns, which stand against it; a column of its own
    // at its west end)
    for (k, &x) in TER_COLS.iter().enumerate() { for &s in &[-1.0f64, 1.0] {
        if s > 0.0 && k > 0 { continue; }
        let (za, zb) = if s < 0.0 { (TER_Z.0 - 0.25, TER_Z.0) } else { (TER_Z.1, TER_Z.1 + 0.25) };
        p.terrace_columns.push(bx(&mut b, [x - 0.125, 0.0, za], [x + 0.125, TER_TOP, zb], mat::STEEL, P_WHITE));
    } }
    slab(&mut b, TER_X.0, TER_X.1, TER_Z.0, TER_Z.1, TER_TOP, P_TRAVERTINE);
    // ---- the steps, 3.5 m wide, on one line: ground to terrace (six risers of 0.125 m), terrace to porch (seven)
    let (sx0, sx1) = (-8.5, -5.0);
    for k in 0..6 {
        let z = TER_Z.0 - 0.375 * (6 - k) as f64;
        p.steps.push(bx(&mut b, [sx0, 0.0, z], [sx1, 0.125 * (k + 1) as f64, z + 0.375], mat::MASONRY, P_TRAVERTINE));
    }
    // (the upper flight: travertine treads on two steel strings, which bear on the terrace's north channel and the
    // house's south channel)
    let run_z = (TER_Z.1 + 0.25 - 0.375 * 7.0, -HW);
    for &x in &[sx0, sx1 - 0.125] {
        let mut pts = Vec::new();
        for &dx in &[0.0, 0.125] { pts.extend_from_slice(&[[x + dx, TER_TOP, run_z.0], [x + dx, TER_TOP + 0.25, run_z.0], [x + dx, FLOOR, run_z.1], [x + dx, FLOOR - 0.375, run_z.1], [x + dx, TER_TOP, run_z.0 + 0.375]]); }
        sid += 1; piece(&mut b, &pts, mat::STEEL, P_WHITE, sid);
    }
    for k in 0..7 {
        let (z, y) = (run_z.0 + 0.375 * k as f64, TER_TOP + 0.125 * (k + 1) as f64);
        if z + 0.375 > -HW + 1e-9 { break; }
        // (travertine on a steel tray, as built: as bare stone a tread 3.25 m long broke under itself, 2.36 of its strength)
        p.steps.push(bx(&mut b, [sx0 + 0.125, y - 0.125, z], [sx1 - 0.125, y, z + 0.375], mat::CONCRETE, P_TRAVERTINE));
    }
    b.build_joints();
    b.log.clear();
    (b, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_glass_house_is_what_it_should_be_and_stands() {
        let (b, p) = build_parts();
        assert_eq!(p.columns.len(), 8); assert_eq!(p.terrace_columns.len(), 4);
        assert!(p.glass.len() >= 14, "{} panes", p.glass.len());
        let r = stands("the glass house", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
