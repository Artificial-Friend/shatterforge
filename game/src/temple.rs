//! The temple: a Doric temple of marble on the plan of the Hephaisteion (Athens, about 450 BC), six columns by
//! thirteen, on the 0.125 m grid (columns 2.5 m apart for its 2.58 m, 1 m thick at the foot for its 1.02 m).
//!
//! - A platform of three steps (0.375 m each), 15.5 x 33 m at the ground, 14 x 31.5 m at the top.
//! - 34 columns round it and two in each porch: a shaft of four drums tapering from 1 m to 0.75 m (eight-sided:
//!   the engine's convex pieces have at most 16 corners), a spreading echinus and a square abacus, 5.5 m in all.
//! - On them the entablature: architrave blocks from column to column, a frieze with its triglyphs, a cornice that
//!   projects.
//! - The cella inside: walls of large blocks in courses, a porch at each end between the walls' ends with two
//!   columns, a door 2 m wide and 4 m high from the front porch; in it the statue on its base (loose).
//! - A gable at each end and a timber roof at 1 in 4: rafters every 1.25 m from the cornice to the ridge, bearing on
//!   the cella's walls on the way, boards and tiles on them.
//! Stones meet on beds that hold no tension (`structure::mortar_bed`: drum on drum, the gable's stones): it stands
//! by its weight.
use crate::brick::{mat, Bricks};
use crate::kit::*;
use alloc::vec::Vec;

pub const BAY: f64 = 2.5;
/// columns across and along, the platform's top, the columns' height (shaft, echinus, abacus)
pub const NX: usize = 6;
pub const NZ: usize = 13;
pub const BASE: f64 = 1.125;
const SHAFT: f64 = 5.0;
const COL_TOP: f64 = BASE + SHAFT + 0.5;
/// the entablature's courses (m above the columns): architrave, frieze, cornice
const ARCH: f64 = 0.875;
const FRIEZE: f64 = 0.75;
const CORNICE: f64 = 0.25;
pub const EAVE: f64 = COL_TOP + ARCH + FRIEZE + CORNICE;
/// the roof's slope (rise over run) and where the rafters' feet stand on the cornice (m from the middle)
const SLOPE: f64 = 0.25;
const FOOT: f64 = 6.75;
/// the cella: its walls' middle (x), half thickness, where the walls end (z) and where its cross walls stand
const CELLA_X: f64 = 3.75;
const WALL: f64 = 0.375;
const CELLA_Z: f64 = 11.25;
const CROSS_Z: f64 = 8.375;
const WALL_TOP: f64 = BASE + 12.0 * 0.625;

#[derive(Default, Clone, Debug)]
pub struct Parts { pub columns: Vec<Vec<u32>>, pub architrave: Vec<u32>, pub frieze: Vec<u32>, pub triglyphs: Vec<u32>, pub cornice: Vec<u32>, pub gables: Vec<u32>, pub rafters: Vec<Vec<u32>>, pub boards: Vec<Vec<u32>>, pub walls: Vec<u32>, pub statue: Vec<u32>, pub platform: Vec<u32> }

fn axis(k: usize, n: usize) -> f64 { (k as f64 - (n - 1) as f64 / 2.0) * BAY }
/// A column on the platform at (x, z): four tapering drums, the echinus, the abacus. Each stone a member of its own.
fn column(b: &mut Bricks, sid: &mut u16, x: f64, z: f64) -> Vec<u32> {
    let mut out = Vec::new();
    let r = |y: f64| 0.5 - 0.125 * y / SHAFT;   // (to the flats, y above the platform)
    let frustum = |b: &mut Bricks, sid: &mut u16, y0: f64, y1: f64, r0: f64, r1: f64| {
        let mut pts = Vec::new();
        for q in ngon(x, z, r0, 8) { pts.push([q[0], BASE + y0, q[1]]); }
        for q in ngon(x, z, r1, 8) { pts.push([q[0], BASE + y1, q[1]]); }
        *sid += 1; piece(b, &pts, mat::MASONRY, P_MARBLE, *sid)
    };
    for k in 0..4 { let (y0, y1) = (k as f64 * SHAFT / 4.0, (k + 1) as f64 * SHAFT / 4.0); out.push(frustum(b, sid, y0, y1, r(y0), r(y1))); }
    out.push(frustum(b, sid, SHAFT, SHAFT + 0.25, 0.375, 0.5625));
    out.push(bx(b, [x - 0.625, BASE + SHAFT + 0.25, z - 0.625], [x + 0.625, COL_TOP, z + 0.625], mat::MASONRY, P_MARBLE));
    out
}

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut p = Parts::default();
    let mut sid: u16 = 100;
    let (ax, az) = (axis(NX - 1, NX), axis(NZ - 1, NZ));   // the corner columns' axes: 6.25, 15
    let st = |b: &mut Bricks, lo: P3, hi: P3| bx(b, lo, hi, mat::MASONRY, P_MARBLE);
    // ---- the platform: three courses, each 0.375 m wider all round than the one on it, of blocks two bays long
    for c in 0..3 {
        let out = 0.75 + 0.375 * (2 - c) as f64;
        let (y0, y1) = (0.375 * c as f64, 0.375 * (c + 1) as f64);
        let xe = [-ax - out, -BAY, BAY, ax + out];
        let mut ze = alloc::vec![-az - out]; for k in 0..6 { ze.push(-12.5 + 5.0 * k as f64); } ze.push(az + out);
        for wx in xe.windows(2) { for wz in ze.windows(2) { p.platform.push(st(&mut b, [wx[0], y0, wz[0]], [wx[1], y1, wz[1]])); } }
    }
    // ---- the columns round the platform, and two in each porch between the cella's wall ends
    for i in 0..NX { for j in 0..NZ {
        if i > 0 && i < NX - 1 && j > 0 && j < NZ - 1 { continue; }
        p.columns.push(column(&mut b, &mut sid, axis(i, NX), axis(j, NZ)));
    } }
    for &z in &[-10.0, 10.0] { for &x in &[-1.25, 1.25] { p.columns.push(column(&mut b, &mut sid, x, z)); } }
    // ---- the entablature: blocks from column axis to column axis, 1 m wide over the columns (the corners are the
    // fronts'); the cornice 0.375 m further out
    let (y0, y1, y2, y3) = (COL_TOP, COL_TOP + ARCH, COL_TOP + ARCH + FRIEZE, EAVE);
    let mut xs = alloc::vec![-ax - 0.5]; for k in 1..NX - 1 { xs.push(axis(k, NX)); } xs.push(ax + 0.5);
    let mut zs = alloc::vec![-az + 0.5]; for k in 1..NZ - 1 { zs.push(axis(k, NZ)); } zs.push(az - 0.5);
    for &s in &[-1.0f64, 1.0] {
        let (za, zb) = if s < 0.0 { (-az - 0.5, -az + 0.5) } else { (az - 0.5, az + 0.5) };
        for w in xs.windows(2) {
            p.architrave.push(st(&mut b, [w[0], y0, za], [w[1], y1, zb]));
            p.frieze.push(st(&mut b, [w[0], y1, za], [w[1], y2, zb]));
        }
        let (xa, xb) = if s < 0.0 { (-ax - 0.5, -ax + 0.5) } else { (ax - 0.5, ax + 0.5) };
        for w in zs.windows(2) {
            p.architrave.push(st(&mut b, [xa, y0, w[0]], [xb, y1, w[1]]));
            p.frieze.push(st(&mut b, [xa, y1, w[0]], [xb, y2, w[1]]));
        }
    }
    // the triglyphs: a block 0.5 m wide standing 0.125 m proud of the frieze over every column and every bay's
    // middle (they carry themselves only)
    for k in 0..=2 * (NZ - 1) { let z = -az + 1.25 * k as f64; for &s in &[-1.0f64, 1.0] {
        let (xa, xb) = if s < 0.0 { (-ax - 0.625, -ax - 0.5) } else { (ax + 0.5, ax + 0.625) };
        p.triglyphs.push(infill(&mut b, [xa, y1, z - 0.25], [xb, y2, z + 0.25], mat::MASONRY, P_MARBLE));
    } }
    for k in 0..=2 * (NX - 1) { let x = -ax + 1.25 * k as f64; for &s in &[-1.0f64, 1.0] {
        let (za, zb) = if s < 0.0 { (-az - 0.625, -az - 0.5) } else { (az + 0.5, az + 0.625) };
        p.triglyphs.push(infill(&mut b, [x - 0.25, y1, za], [x + 0.25, y2, zb], mat::MASONRY, P_MARBLE));
    } }
    // the cornice: a ring of slabs, 1.375 m wide (0.375 m out, the frieze's width in); the fronts' take the corners
    {
        let (o, i) = (0.875, 0.5);
        let mut cx = alloc::vec![-ax - o]; for k in 1..NX - 1 { cx.push(axis(k, NX)); } cx.push(ax + o);
        let mut cz = alloc::vec![-az + i]; for k in 1..NZ - 1 { cz.push(axis(k, NZ)); } cz.push(az - i);
        for &s in &[-1.0f64, 1.0] {
            let (za, zb) = if s < 0.0 { (-az - o, -az + i) } else { (az - i, az + o) };
            for w in cx.windows(2) { p.cornice.push(st(&mut b, [w[0], y2, za], [w[1], y3, zb])); }
            let (xa, xb) = if s < 0.0 { (-ax - o, -ax + i) } else { (ax - i, ax + o) };
            for w in cz.windows(2) { p.cornice.push(st(&mut b, [xa, y2, w[0]], [xb, y3, w[1]])); }
        }
    }
    // ---- the cella: side walls of blocks 2.5 m long in courses of 0.625 m, from porch end to porch end
    for &s in &[-1.0f64, 1.0] {
        let (xa, xb) = if s < 0.0 { (-CELLA_X - WALL, -CELLA_X + WALL) } else { (CELLA_X - WALL, CELLA_X + WALL) };
        p.walls.extend(coursed(&mut b, [xa, BASE, -CELLA_Z], [xb, WALL_TOP, CELLA_Z], 2, 2.5, 0.625, mat::MASONRY, P_MARBLE));
    }
    // the cross walls: the front one with the door (2 m wide, 4 m high, a lintel 4.5 m long over it), the back one whole
    let (ix, door_h) = (CELLA_X - WALL, 4.0);
    for &s in &[-1.0f64, 1.0] {
        let (za, zb) = if s < 0.0 { (-CROSS_Z - WALL, -CROSS_Z + WALL) } else { (CROSS_Z - WALL, CROSS_Z + WALL) };
        if s < 0.0 {
            for &(xa, xb) in &[(-ix, -1.0), (1.0, ix)] { p.walls.extend(coursed(&mut b, [xa, BASE, za], [xb, BASE + door_h, zb], 0, 2.5, 0.625, mat::MASONRY, P_MARBLE)); }
            for &(xa, xb) in &[(-ix, -2.25), (-2.25, 2.25), (2.25, ix)] { p.walls.push(st(&mut b, [xa, BASE + door_h, za], [xb, BASE + door_h + 0.625, zb])); }
            p.walls.extend(coursed(&mut b, [-ix, BASE + door_h + 0.625, za], [ix, WALL_TOP, zb], 0, 2.5, 0.625, mat::MASONRY, P_MARBLE));
        } else {
            p.walls.extend(coursed(&mut b, [-ix, BASE, za], [ix, WALL_TOP, zb], 0, 2.5, 0.625, mat::MASONRY, P_MARBLE));
        }
    }
    // the door's two leaves, of timber, 1 m wide and 4 m high, standing open into the cella
    for &s in &[-1.0f64, 1.0] { let (xa, xb) = if s < 0.0 { (-1.125, -1.0) } else { (1.0, 1.125) }; infill(&mut b, [xa, BASE, -CROSS_Z + WALL], [xb, BASE + door_h, -CROSS_Z + WALL + 1.0], mat::TIMBER, P_DARKWOOD); }
    // the porches' beams: an architrave across each, from wall to wall over the two columns
    for &z in &[-10.0, 10.0] { for w in [-ix, -1.25, 1.25, ix].windows(2) { p.architrave.push(st(&mut b, [w[0], y0, z - 0.5], [w[1], y1, z + 0.5])); } }
    // ---- the gables: a wall of stones on the cornice at each end, its top the rafters' underside
    let under = |x: f64| EAVE - 0.125 + (FOOT - x.abs()) * SLOPE;
    for &s in &[-1.0f64, 1.0] {
        let z = s * az;
        for w in [-6.25, -3.75, -1.25, 1.25, 3.75, 6.25].windows(2) {
            let mut poly = alloc::vec![[w[0], EAVE], [w[1], EAVE]];
            if under(w[1]) > EAVE + 1e-6 { poly.push([w[1], under(w[1])]); }
            if w[0] < 0.0 && w[1] > 0.0 { poly.push([0.0, under(0.0)]); }
            if under(w[0]) > EAVE + 1e-6 { poly.push([w[0], under(w[0])]); }
            sid += 1;
            p.gables.push(prism_z(&mut b, &poly, z - 0.25, z + 0.25, mat::MASONRY, P_MARBLE, sid));
        }
    }
    // ---- the roof. A timber plate on each cella wall, its top the rafters' underside; rafters every 1.25 m, their
    // feet level on the cornice, their heads meeting at the ridge; boards and tiles between them
    for &s in &[-1.0f64, 1.0] {
        let (xa, xb) = (s * (CELLA_X - WALL), s * (CELLA_X + WALL));
        let poly = [[xa, WALL_TOP], [xb, WALL_TOP], [xb, under(xb)], [xa, under(xa)]];
        let mut z = -CELLA_Z;
        while z < CELLA_Z - 1e-9 { sid += 1; prism_z(&mut b, &poly, z, z + 2.5, mat::TIMBER, P_DARKWOOD, sid); z += 2.5; }
    }
    let up = [0.0, 1.0, 0.0];
    let cosr = 1.0 / crate::math::sqrt64(1.0 + SLOPE * SLOPE);
    let depth = 0.25 * cosr;   // (0.125 m above and below its axis, measured upright)
    let nr = (2.0 * az / 1.25 + 0.5) as usize + 1;
    for &s in &[-1.0f64, 1.0] {
        let a = unit([-s, SLOPE, 0.0]);
        let nn = unit(sub(up, mul(a, dot(up, a))));
        for k in 0..nr {
            let z = -az + 1.25 * k as f64;
            let (f, h) = ([s * FOOT, EAVE, z], [0.0, EAVE + FOOT * SLOPE, z]);
            p.rafters.push(member(&mut b, f, h, nn, [0.0, 0.0, 1.0], depth, 0.125, up, [1.0, 0.0, 0.0], 3.75, mat::TIMBER, P_DARKWOOD, &mut sid));
        }
        // the tiles, on boards: 0.0625 m, from above the rafters' feet to the ridge, between neighbouring rafters
        let lift = depth / 2.0 + 0.03125;
        for k in 0..nr - 1 {
            let zm = -az + 1.25 * k as f64 + 0.625;
            let (f, h) = ([s * FOOT, EAVE, zm], [0.0, EAVE + FOOT * SLOPE, zm]);
            let (bf, bh) = (add(f, mul(nn, lift)), add(h, mul(nn, lift)));
            let t_top = -(depth / 2.0) * nn[1] / a[1];
            let start = add(bf, mul(a, t_top));
            p.boards.push(member(&mut b, start, bh, nn, [0.0, 0.0, 1.0], 0.0625, 1.25, mul(a, -1.0), [1.0, 0.0, 0.0], 3.75, mat::TIMBER, P_TILE, &mut sid));
        }
    }
    // ---- the statue: a base of two blocks in the cella, and on it a figure of bronze (four loose parts, one body)
    st(&mut b, [-1.25, BASE, 3.5], [1.25, BASE + 0.375, 6.0]);
    st(&mut b, [-0.875, BASE + 0.375, 3.875], [0.875, BASE + 1.0, 5.625]);
    {
        sid += 1; let s = sid; let y = BASE + 1.0;
        loose(&mut b, &mut p.statue, [-0.5, y, 4.375], [-0.125, y + 1.25, 5.0], mat::STEEL, P_OXIDE, s);
        loose(&mut b, &mut p.statue, [0.125, y, 4.375], [0.5, y + 1.25, 5.0], mat::STEEL, P_OXIDE, s);
        loose(&mut b, &mut p.statue, [-0.625, y + 1.25, 4.25], [0.625, y + 2.5, 5.125], mat::STEEL, P_OXIDE, s);
        loose(&mut b, &mut p.statue, [-0.25, y + 2.5, 4.375], [0.25, y + 3.0, 4.875], mat::STEEL, P_OXIDE, s);
    }
    b.build_joints();
    b.log.clear();
    (b, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    /// (by hand: cargo test --release -p sf-demo explore_the_temples -- --ignored --nocapture)
    #[test] #[ignore] fn explore_the_temples_worst_joints() { let b = build(); worst_joints(&b, 8); }
    #[test] fn the_temple_is_what_it_should_be_and_stands() {
        let (b, p) = build_parts();
        assert_eq!(p.columns.len(), 2 * NX + 2 * (NZ - 2) + 4);
        assert!(p.columns.iter().all(|c| c.len() == 6));
        assert_eq!(p.gables.len(), 10); assert_eq!(p.triglyphs.len(), 2 * 25 + 2 * 11);
        assert_eq!(p.rafters.len(), 50); assert_eq!(p.boards.len(), 48);
        let r = stands("the temple", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
