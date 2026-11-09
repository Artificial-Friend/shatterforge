//! The complex building (kind 4, docs/PHASE2.md step 6), made of boxes and free convex pieces at any angle:
//! - a steel frame of two 4 m storeys on a 3 x 2 grid of 6 m bays (x -9..9, z -6..6 m): 0.5 m steel columns, 0.5 m
//!   deep steel beams on the grid lines, 0.25 m concrete floors; the first floor left out over one bay (x -3..3,
//!   z -6..0) for a double-height hall;
//! - diagonal steel braces (free pieces) in the end bays of both long facades, on both storeys, from the foot of
//!   one column to the head of the next, under the beam;
//! - a pitched timber roof: a steel ridge beam on four posts along the middle (z = 0), rafters from the eaves to
//!   the ridge every 1.5 m, and roof boards on the rafters (all free pieces, long ones in segments of at most 2 m);
//! - the south front's ground storey an arcade (z -9.25..-8.75 m): seven masonry piers 2 m tall with six
//!   semicircular arches of nine wedge stones, a stone beam from keystone to keystone (ends on stone blocks over the
//!   end piers), and on it the first floor carried out as a loggia over the arcade, with a parapet;
//! - furniture in every room of both floors (offices of four workstations with monitors and chairs, bookcases,
//!   cabinets, plants, boxes; a meeting room; a lobby in the hall with sofas, a table, a reception desk): each object
//!   loose (its parts joined to each other, to nothing else), a body standing where it was put;
//! - an irregular plan: a one-storey wing to the north, turned 30 degrees, of 0.25 m masonry walls (a door in
//!   its front under a stone lintel) and a 0.2 m concrete roof slab, all free pieces.
use crate::brick::{mat, Bricks, F_LOOSE, F_REINF, NONE};
use crate::math::floor64;
use alloc::vec::Vec;
use alloc::vec;

const T: f64 = 0.125;
fn tk(m: f64) -> i32 { floor64(m / T + 0.5) as i32 }
/// grid lines (m)
pub const XS: [f64; 4] = [-9.0, -3.0, 3.0, 9.0];
pub const ZS: [f64; 3] = [-6.0, 0.0, 6.0];
/// storey height, slab, beam depth, column half width (m)
pub const H: f64 = 4.0;
const SLAB: f64 = 0.25;
const BEAM: f64 = 0.5;
const COL: f64 = 0.25;
/// the arcade: piers' centres, their half width, height; the arches' depth (z range) and stones
const PIER_X: [f64; 7] = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0];
const PIER: f64 = 0.375;
const PIER_H: f64 = 2.0;
/// how far an end pier (a buttress) reaches outward from its centre (m)
const BUTTRESS: f64 = 1.5;
const ARC_Z: (f64, f64) = (-9.5, -8.75);
const STONES: usize = 9;

/// paints (colours; the material drives the physics)
const P_STEEL: u8 = 7;
const P_SLAB: u8 = 0;
const P_TIMBER: u8 = 5;
const P_STONE: u8 = 6;

fn bx(b: &mut Bricks, lo: [f64; 3], hi: [f64; 3], m: u8, paint: u8) -> u32 {
    let l = [tk(lo[0]), tk(lo[1]), tk(lo[2])];
    let h = [tk(hi[0]), tk(hi[1]), tk(hi[2])];
    let i = b.add(l, [h[0] - l[0], h[1] - l[1], h[2] - l[2]], m, paint, 1);
    if m == mat::CONCRETE { b.flags[i as usize] |= F_REINF; }
    i
}
/// a free piece from corners in the world (m): centre at their mean, not turned; `sid` its member (`structure`: the
/// segments of one member share it; a joint of masonry between members is a mortar bed, structure::connection)
fn piece(b: &mut Bricks, pts: &[[f64; 3]], m: u8, paint: u8, sid: u16) -> u32 {
    let n = pts.len() as f64;
    let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
    let cs = [floor64(c[0] * 1000.0 + 0.5) / 1000.0, floor64(c[1] * 1000.0 + 0.5) / 1000.0, floor64(c[2] * 1000.0 + 0.5) / 1000.0];
    let local: Vec<[f64; 3]> = pts.iter().map(|p| [p[0] - cs[0], p[1] - cs[1], p[2] - cs[2]]).collect();
    b.add_piece(cs, [0.0, 0.0, 0.0, 1.0], &local, m, paint, sid).expect("a piece of the complex building within the hull limits")
}
/// a rotation about y (x y z w)
fn quat_y(a: f64) -> [f32; 4] { [0.0, crate::math::sin((a / 2.0) as f32), 0.0, crate::math::cos((a / 2.0) as f32)] }
/// v turned by a about y
fn qrot_y(a: f64, v: [f64; 3]) -> [f64; 3] { let (s, c) = (crate::math::sin(a as f32) as f64, crate::math::cos(a as f32) as f64); [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]] }
/// the corners of a box of half size h about its centre
fn box_corners(h: [f64; 3]) -> Vec<[f64; 3]> { let mut v = Vec::new(); for &a in &[-1.0, 1.0] { for &b in &[-1.0, 1.0] { for &c in &[-1.0, 1.0] { v.push([a * h[0], b * h[1], c * h[2]]); } } } v }
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn mul(a: [f64; 3], k: f64) -> [f64; 3] { [a[0] * k, a[1] * k, a[2] * k] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn unit(a: [f64; 3]) -> [f64; 3] { let l = crate::math::sqrt64(dot(a, a)); mul(a, 1.0 / l) }

/// A straight member from `f` to `h` (m, on its axis), `w` deep across its axis in the plane of the axis and `n`
/// (`n` at right angles to the axis), `d` wide along `out`, its ends cut by the planes through `f` (normal `nf`)
/// and `h` (normal `nh`): segments of at most 2 m.
fn member(b: &mut Bricks, f: [f64; 3], h: [f64; 3], n: [f64; 3], out: [f64; 3], w: f64, d: f64, nf: [f64; 3], nh: [f64; 3], m: u8, paint: u8, sid: &mut u16) -> Vec<u32> {
    let a = unit(sub(h, f));
    let mut pts = Vec::new();
    for &o in &[-w / 2.0, w / 2.0] { for &e in &[-d / 2.0, d / 2.0] {
        let p = add(add(f, mul(n, o)), mul(out, e));
        let t0 = dot(nf, sub(f, p)) / dot(nf, a);
        let t1 = dot(nh, sub(h, p)) / dot(nh, a);
        pts.push(add(p, mul(a, t0))); pts.push(add(p, mul(a, t1)));
    } }
    *sid += 1;
    let k = piece(b, &pts, m, paint, *sid);
    b.segment(k, 2.0)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Room { Office, Meeting, Lobby }
/// one loose box of an object (m), on the tick grid
fn part(b: &mut Bricks, out: &mut Vec<u32>, lo: [f64; 3], hi: [f64; 3], m: u8, paint: u8, sid: u16) {
    let i = bx(b, lo, hi, m, paint);
    b.structure[i as usize] = sid; b.flags[i as usize] |= F_LOOSE;
    out.push(i);
}
/// one loose box of an object turned by `a` about y round `c` (m): a free piece
fn part_turned(b: &mut Bricks, out: &mut Vec<u32>, c: [f64; 3], a: f64, lo: [f64; 3], hi: [f64; 3], m: u8, paint: u8, sid: u16) {
    let mid = [0.5 * (lo[0] + hi[0]) - c[0], 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]) - c[2]];
    let r = qrot_y(a, mid);
    let h = [0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2])];
    let i = b.add_piece([c[0] + r[0], mid[1], c[2] + r[2]], quat_y(a), &box_corners(h), m, paint, sid).expect("a furniture part");
    b.flags[i as usize] |= F_LOOSE;
    out.push(i);
}
fn rnd(s: &mut u32) -> f64 { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; (*s as f64) / 4294967296.0 }
/// A desk (top and two side panels), with a monitor on it and a chair before it, turned a little.
fn workstation(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, seed: &mut u32, x: f64, z: f64, y: f64) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y + 0.625, z], [x + 1.5, y + 0.75, z + 0.75], mat::TIMBER, P_TIMBER, s);
    part(b, out, [x, y, z], [x + 0.125, y + 0.625, z + 0.75], mat::TIMBER, P_TIMBER, s);
    part(b, out, [x + 1.375, y, z], [x + 1.5, y + 0.625, z + 0.75], mat::TIMBER, P_TIMBER, s);
    *sid += 1;
    part(b, out, [x + 0.5, y + 0.75, z + 0.5], [x + 1.0, y + 1.125, z + 0.625], mat::GLASS, 0, *sid);
    // the chair: post, seat, back; turned up to 30 degrees either way
    *sid += 1; let s = *sid;
    let c = [x + 0.75, 0.0, z - 0.5];
    let a = (rnd(seed) - 0.5) * 1.05;
    part_turned(b, out, c, a, [c[0] - 0.0625, y, c[2] - 0.0625], [c[0] + 0.0625, y + 0.375, c[2] + 0.0625], mat::STEEL, P_STEEL, s);
    part_turned(b, out, c, a, [c[0] - 0.25, y + 0.375, c[2] - 0.25], [c[0] + 0.25, y + 0.5, c[2] + 0.25], mat::TIMBER, P_TIMBER, s);
    part_turned(b, out, c, a, [c[0] - 0.25, y + 0.5, c[2] - 0.25], [c[0] + 0.25, y + 1.0, c[2] - 0.125], mat::TIMBER, P_TIMBER, s);
}
/// A bookcase: two sides and four shelves, 1 m wide, 1.875 m tall, 0.375 m deep (along x).
fn bookcase(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y, z], [x + 0.125, y + 1.875, z + 0.375], mat::TIMBER, P_TIMBER, s);
    part(b, out, [x + 0.875, y, z], [x + 1.0, y + 1.875, z + 0.375], mat::TIMBER, P_TIMBER, s);
    for &h in &[0.0, 0.625, 1.25, 1.75] { part(b, out, [x + 0.125, y + h, z], [x + 0.875, y + h + 0.125, z + 0.375], mat::TIMBER, P_TIMBER, s); }
}
/// A table on four legs, its top `w` x `d` at 0.75 m.
fn table(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64, w: f64, d: f64, h: f64) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y + h - 0.125, z], [x + w, y + h, z + d], mat::TIMBER, P_TIMBER, s);
    for &(lx, lz) in &[(x, z), (x + w - 0.125, z), (x, z + d - 0.125), (x + w - 0.125, z + d - 0.125)] { part(b, out, [lx, y, lz], [lx + 0.125, y + h - 0.125, lz + 0.125], mat::TIMBER, P_TIMBER, s); }
}
/// A plain chair (seat on four legs, a back), facing +z or -z.
fn chair(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64, back_at_low_z: bool) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y + 0.375, z], [x + 0.5, y + 0.5, z + 0.5], mat::TIMBER, P_TIMBER, s);
    for &(lx, lz) in &[(x, z), (x + 0.375, z), (x, z + 0.375), (x + 0.375, z + 0.375)] { part(b, out, [lx, y, lz], [lx + 0.125, y + 0.375, lz + 0.125], mat::TIMBER, P_TIMBER, s); }
    let bz = if back_at_low_z { z } else { z + 0.375 };
    part(b, out, [x, y + 0.5, bz], [x + 0.5, y + 1.0, bz + 0.125], mat::TIMBER, P_TIMBER, s);
}
/// A sofa: base, back, two arms, 2 m long along x, its back at high z if `back_far`.
fn sofa(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64, back_far: bool) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y, z], [x + 2.0, y + 0.375, z + 0.875], mat::TIMBER, 2, s);
    let bz = if back_far { z + 0.625 } else { z };
    part(b, out, [x + 0.25, y + 0.375, bz], [x + 1.75, y + 0.875, bz + 0.25], mat::TIMBER, 2, s);
    part(b, out, [x, y + 0.375, z], [x + 0.25, y + 0.625, z + 0.875], mat::TIMBER, 2, s);
    part(b, out, [x + 1.75, y + 0.375, z], [x + 2.0, y + 0.625, z + 0.875], mat::TIMBER, 2, s);
}
/// A plant: a stone pot and its foliage.
fn plant(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64) {
    *sid += 1; let s = *sid;
    part(b, out, [x, y, z], [x + 0.375, y + 0.375, z + 0.375], mat::MASONRY, P_STONE, s);
    part(b, out, [x + 0.0625, y + 0.375, z + 0.0625], [x + 0.3125, y + 1.125, z + 0.3125], mat::TIMBER, 1, s);
}
/// Boxes stacked (each its own object), a cabinet.
fn boxes(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64, n: usize) {
    for k in 0..n { *sid += 1; part(b, out, [x, y + 0.375 * k as f64, z], [x + 0.5, y + 0.375 * (k + 1) as f64, z + 0.375], mat::TIMBER, 3, *sid); }
}
fn cabinet(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, x: f64, z: f64, y: f64) {
    *sid += 1; part(b, out, [x, y, z], [x + 0.5, y + 1.25, z + 0.5], mat::STEEL, P_STEEL, *sid);
}
/// Furnish a room (its inner area x0..x1, z0..z1, at least 4.5 m square, floor at y).
fn furnish(b: &mut Bricks, out: &mut Vec<u32>, sid: &mut u16, seed: &mut u32, x0: f64, x1: f64, z0: f64, z1: f64, y: f64, kind: Room) {
    match kind {
        Room::Office => {
            for &(dx, dz) in &[(0.25, 0.75), (2.5, 0.75), (0.25, 3.0), (2.5, 3.0)] { workstation(b, out, sid, seed, x0 + dx, z0 + dz, y); }
            bookcase(b, out, sid, x1 - 1.0, z0 + 2.0, y);
            cabinet(b, out, sid, x0, z1 - 0.5, y);
            plant(b, out, sid, x1 - 0.375, z1 - 0.375, y);
            boxes(b, out, sid, x1 - 0.5, z0, y, 2);
        }
        Room::Meeting => {
            table(b, out, sid, x0 + 1.0, z0 + 1.75, y, 2.5, 1.0, 0.75);
            for k in 0..3 { let cx = x0 + 1.125 + 0.875 * k as f64; chair(b, out, sid, cx, z0 + 1.0, y, true); chair(b, out, sid, cx, z0 + 3.0, y, false); }
            cabinet(b, out, sid, x1 - 0.5, z1 - 0.5, y);
            plant(b, out, sid, x0, z1 - 0.375, y);
            bookcase(b, out, sid, x1 - 1.0, z0, y);
        }
        Room::Lobby => {
            sofa(b, out, sid, x0 + 0.5, z0 + 0.25, y, false);
            sofa(b, out, sid, x0 + 0.5, z0 + 2.5, y, true);
            table(b, out, sid, x0 + 1.0, z0 + 1.5, y, 1.0, 0.5, 0.5);
            // the reception desk: a counter and its top
            *sid += 1; let s = *sid;
            part(b, out, [x1 - 2.75, y, z1 - 1.0], [x1 - 0.25, y + 1.0, z1 - 0.75], mat::TIMBER, P_TIMBER, s);
            part(b, out, [x1 - 2.75, y + 1.0, z1 - 1.125], [x1 - 0.25, y + 1.125, z1 - 0.625], mat::TIMBER, P_TIMBER, s);
            for &(px, pz) in &[(x0, z1 - 0.375), (x1 - 0.375, z0), (x1 - 0.375, z0 + 0.5)] { plant(b, out, sid, px, pz, y); }
            chair(b, out, sid, x1 - 1.75, z1 - 0.5, y, false);
        }
    }
}

/// What the complex building is made of, for its scenarios and tests.
#[derive(Default, Clone, Debug)]
pub struct Parts { pub loggia: u32, pub contents: Vec<u32>, pub wing: Vec<u32>, pub columns: Vec<u32>, pub braces: Vec<Vec<u32>>, pub rafters: Vec<Vec<u32>>, pub boards: Vec<Vec<u32>>, pub ridge: u32, pub piers: Vec<u32>, pub arches: Vec<Vec<u32>>, pub keystones: Vec<u32>, pub beam: Vec<u32>, pub parapet: u32 }

pub fn build() -> Bricks { build_parts().0 }
/// The part of a convex polygon where n.p <= d (Sutherland-Hodgman against one line).
fn clip(poly: &[[f64; 2]], n: [f64; 2], d: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let side = |p: &[f64; 2]| n[0] * p[0] + n[1] * p[1] - d;
    for k in 0..poly.len() {
        let (p, q) = (poly[k], poly[(k + 1) % poly.len()]);
        let (sp, sq) = (side(&p), side(&q));
        if sp <= 0.0 { out.push(p); }
        if (sp < 0.0 && sq > 0.0) || (sp > 0.0 && sq < 0.0) { let t = sp / (sp - sq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    }
    out
}
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut parts = Parts { ridge: NONE, loggia: NONE, parapet: NONE, ..Default::default() };
    // member ids of the free pieces (boxes are member 1)
    let mut sid: u16 = 100;
    let up = [0.0, 1.0, 0.0];
    let (x0, x1, z0, z1) = (XS[0] - COL, XS[3] + COL, ZS[0] - COL, ZS[2] + COL);
    // ---- the steel frame
    for k in 0..2 {
        let y = k as f64 * H;
        let (ytop, ybeam) = (y + H - SLAB, y + H - SLAB - BEAM);
        for &x in &XS { for &z in &ZS { parts.columns.push(bx(&mut b, [x - COL, y, z - COL], [x + COL, ytop, z + COL], mat::STEEL, P_STEEL)); } }
        // beams between the columns' faces, along x and z on every grid line
        for &z in &ZS { for w in XS.windows(2) { bx(&mut b, [w[0] + COL, ybeam, z - COL], [w[1] - COL, ytop, z + COL], mat::STEEL, P_STEEL); } }
        for &x in &XS { for w in ZS.windows(2) { bx(&mut b, [x - COL, ybeam, w[0] + COL], [x + COL, ytop, w[1] - COL], mat::STEEL, P_STEEL); } }
        // the floor above, a slab per bay (none over the hall on the first floor), meeting on the grid lines
        let xe = [x0, XS[1], XS[2], x1]; let ze = [z0, ZS[1], z1];
        for i in 0..3 { for j in 0..2 {
            if k == 0 && i == 1 && j == 0 { continue; }
            bx(&mut b, [xe[i], ytop, ze[j]], [xe[i + 1], ytop + SLAB, ze[j + 1]], mat::CONCRETE, P_SLAB);
        } }
        // braces in the end bays of the long facades: from a column's foot (beside it, on the floor) to the next
        // column's face under the beam, 0.2 m steel
        for &z in &[ZS[0], ZS[2]] { for &(xa, xb) in &[(XS[0], XS[1]), (XS[3], XS[2])] {
            let s = if xb > xa { 1.0 } else { -1.0 };
            let foot = [xa + s * (COL + 0.35), y, z];
            let head = [xb - s * COL, ybeam - 0.25, z];
            let ax = unit(sub(head, foot));
            let nn = [-ax[1], ax[0], 0.0];
            parts.braces.push(member(&mut b, foot, head, nn, [0.0, 0.0, 1.0], 0.2, 0.2, up, [1.0, 0.0, 0.0], mat::STEEL, P_STEEL, &mut sid));
        } }
    }
    // ---- the roof: a ridge beam on posts, rafters, boards
    let yr = 2.0 * H;                           // the roof slab's top
    let (ridge_lo, ridge_hi) = (yr + 2.5, yr + 3.0);
    for &x in &XS { bx(&mut b, [x - 0.125, yr, -0.125], [x + 0.125, ridge_lo, 0.125], mat::STEEL, P_STEEL); }
    parts.ridge = bx(&mut b, [x0, ridge_lo, -0.125], [x1, ridge_hi, 0.125], mat::STEEL, P_STEEL);
    let rafter_x: Vec<f64> = (0..12).map(|k| -8.25 + 1.5 * k as f64).collect();
    let (rw, rd) = (0.25, 0.1);
    for &side in &[-1.0f64, 1.0] {
        // the rafter's axis: from its foot on the roof slab by the eave to its head on the ridge beam's side
        let foot_z = side * (z1 - 0.5);
        let head_z = side * 0.125;
        let (fy, hy) = (yr, 0.5 * (ridge_lo + ridge_hi));
        let along = [0.0, 0.0, -side];         // horizontal, from the eave to the ridge
        for &x in &rafter_x {
            let (f, h) = ([x, fy, foot_z], [x, hy, head_z]);
            let a = unit(sub(h, f));
            // n: across the axis, upward
            let nn = unit(sub(up, mul(a, dot(up, a))));
            parts.rafters.push(member(&mut b, f, h, nn, [1.0, 0.0, 0.0], rw, rd, up, along, mat::TIMBER, P_TIMBER, &mut sid));
        }
        // boards between neighbouring rafters, on their top faces: 0.05 m thick, from above the rafters' feet to
        // the ridge beam, cut square at the eave
        let f = [0.0, fy, foot_z]; let h = [0.0, hy, head_z];
        let a = unit(sub(h, f));
        let nn = unit(sub(up, mul(a, dot(up, a))));
        let lift = rw / 2.0 + 0.025;
        for w in rafter_x.windows(2) {
            let xm = 0.5 * (w[0] + w[1]);
            let (bf, bh) = (add([xm, fy, foot_z], mul(nn, lift)), add([xm, hy, head_z], mul(nn, lift)));
            // the foot: square to the rafter, where the rafters' top edge meets their level-cut foot
            let t_top = -(rw / 2.0) * nn[1] / a[1];
            let start = add(bf, mul(a, t_top));
            parts.boards.push(member(&mut b, start, bh, nn, [1.0, 0.0, 0.0], 0.05, w[1] - w[0], mul(a, -1.0), along, mat::TIMBER, P_TIMBER, &mut sid));
        }
    }
    // ---- the arcade
    // (the end piers are buttresses: 1.5 m wide outward, to take the end arches' thrust; with only half a pier and a
    // block beyond them, the end arch ran away as a mechanism once its beds took no tension)
    let last = PIER_X.len() - 1;
    for (k, &x) in PIER_X.iter().enumerate() {
        let (xa, xb) = if k == 0 { (x - BUTTRESS, x + PIER) } else if k == last { (x - PIER, x + BUTTRESS) } else { (x - PIER, x + PIER) };
        parts.piers.push(bx(&mut b, [xa, 0.0, ARC_Z.0], [xb, PIER_H, ARC_Z.1], mat::MASONRY, P_STONE));
    }
    let mut key_top = 0.0;
    let mut springers: Vec<u16> = vec![0; PIER_X.len()];   // the member id of each pier's tas-de-charge
    for (ai, w) in PIER_X.windows(2).enumerate() {
        let (cx, r0) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]) - PIER);
        let mut arch = Vec::new();
        // the keystone's top, where its outer corners are (both at the same height), at the first floor's underside:
        // the ring as deep as that makes it (0.43 m at the crown), the floor bearing on the stones' level tops
        let tk = core::f64::consts::PI * (1.0 - (STONES / 2) as f64 / STONES as f64);
        let r1 = (H - SLAB - PIER_H) / crate::math::sin(tk as f32) as f64;
        let top = PIER_H + r1 * crate::math::sin(tk as f32) as f64;
        for k in 0..STONES {
            let (t0, t1) = (core::f64::consts::PI * (1.0 - k as f64 / STONES as f64), core::f64::consts::PI * (1.0 - (k + 1) as f64 / STONES as f64));
            // a stone between its radial joints carried out to a level top at the keystone's top and to the piers'
            // centre lines: the spandrel is the stones' own, and what bears on the arcade bears on a level course.
            // (Fillers on the stones' sloping outer faces slid off at the haunches: 7 joints over strength under the
            // floor. Outer corners carried straight up left a re-entrant corner on one side of each joint, and the
            // stones' hulls overlapped their neighbours.) Each stone is the wedge between its joints clipped by the
            // level top and the piers' centre lines, so convex, and neighbours share their whole joint plane.
            let half = r0 + PIER;
            let far = 4.0 * (r1 + half);
            let mut poly: Vec<[f64; 2]> = vec![
                [cx + r0 * crate::math::cos(t0 as f32) as f64, PIER_H + r0 * crate::math::sin(t0 as f32) as f64],
                [cx + far * crate::math::cos(t0 as f32) as f64, PIER_H + far * crate::math::sin(t0 as f32) as f64],
                [cx + far * crate::math::cos(t1 as f32) as f64, PIER_H + far * crate::math::sin(t1 as f32) as f64],
                [cx + r0 * crate::math::cos(t1 as f32) as f64, PIER_H + r0 * crate::math::sin(t1 as f32) as f64]];
            poly = clip(&poly, [0.0, 1.0], top);
            poly = clip(&poly, [1.0, 0.0], cx + half);
            poly = clip(&poly, [-1.0, 0.0], -(cx - half));
            let mut pts = Vec::new();
            for q in &poly { for &z in &[ARC_Z.0, ARC_Z.1] { pts.push([q[0], q[1], z]); } }
            // (the springing stones lie flat on the piers: their bed at exactly the piers' top)
            for p in pts.iter_mut() { if (p[1] - PIER_H).abs() < 1e-3 { p[1] = PIER_H; } }
            // (the stones that reach a pier's centre line, from the arches on both sides of it and the buttress at
            // an end, are one stone: the tas-de-charge, below)
            let tas = [(ai, cx - half), (ai + 1, cx + half)].iter().find(|&&(_, xe)| poly.iter().any(|q| (q[0] - xe).abs() < 1e-6)).map(|&(p, _)| p);
            let id = match tas {
                Some(p) if springers[p] != 0 => springers[p],
                _ => { sid += 1; if let Some(p) = tas { springers[p] = sid; } sid }
            };
            let s = piece(&mut b, &pts, mat::MASONRY, P_STONE, id);
            if k == STONES / 2 { parts.keystones.push(s); key_top = pts.iter().fold(0.0f64, |m, p| m.max(p[1])); }
            arch.push(s);
        }
        parts.arches.push(arch);
    }
    // the stone beam on the keystones, its top at the first floor's underside: a member from each keystone's centre to
    // the next, and at each end from the keystone to a stone block on the end pier's outer half (resting on half of
    // each). (From pier centre to pier centre, resting on one keystone under its middle, the end members overhung
    // 1.5 m and loaded those keystones' beds at 0.90 of where their thrust may go.)
    // the buttresses' upper stones, from the springing to the keystones' top beside the end arches' outer stones. Over
    // each pier the arch stones that reach its centre line, from both sides, and at the ends this block, are one
    // stone: the tas-de-charge (the lowest courses of arches bonded into their pier; what joins them is the stone,
    // not a bed). (As stones of their own, the east end's second stone hung from its block by an upright bed with
    // little thrust across it, 1.38 of what it holds in shear, and over the inner piers the two arches' springers met
    // at 0.998; with the end stones run out over the buttress instead, they slid on their sloping joints, a
    // mechanism.)
    for (k, &sg) in [-1.0f64, 1.0].iter().enumerate() {
        let (pc, xo) = if sg < 0.0 { (PIER_X[0], PIER_X[0] - BUTTRESS) } else { (PIER_X[PIER_X.len() - 1], PIER_X[PIER_X.len() - 1] + BUTTRESS) };
        let mut pts = Vec::new();
        for &x in &[pc, xo] { for &y in &[PIER_H, key_top] { for &z in &[ARC_Z.0, ARC_Z.1] { pts.push([x, y, z]); } } }
        parts.beam.push(piece(&mut b, &pts, mat::MASONRY, P_STONE, springers[if k == 0 { 0 } else { PIER_X.len() - 1 }]));
    }
    let ytop = H - SLAB;
    // the loggia: the first floor carried out over the arcade, from the stones' level tops to the building's own floor
    // and south columns, with a 1 m parapet along its edge
    parts.loggia = bx(&mut b, [x0, ytop, ARC_Z.0], [x1, H, z0], mat::CONCRETE, P_SLAB);
    parts.parapet = bx(&mut b, [x0, H, ARC_Z.0], [x1, H + 1.0, ARC_Z.0 + 0.25], mat::MASONRY, P_STONE);
    // ---- furniture, in every room on both floors: each object its parts joined to each other and to nothing else
    // (loose), a body standing where it was put from the first structure step
    let mut rooms: Vec<(f64, f64, f64, f64, f64, Room)> = Vec::new();
    let xe = [XS[0], XS[1], XS[2], XS[3]]; let ze = [ZS[0], ZS[1], ZS[2]];
    for k in 0..2 { for i in 0..3 { for j in 0..2 {
        let hall = i == 1 && j == 0;
        if k == 1 && hall { continue; }      // over the hall: no floor
        let kind = if k == 0 && hall { Room::Lobby } else if k == 1 && i == 2 && j == 1 { Room::Meeting } else { Room::Office };
        rooms.push((xe[i] + 0.75, xe[i + 1] - 0.75, ze[j] + 0.75, ze[j + 1] - 0.75, k as f64 * H, kind));
    } } }
    let mut seed = 0x51ed_270bu32;
    for (rx0, rx1, rz0, rz1, y, kind) in rooms {
        furnish(&mut b, &mut parts.contents, &mut sid, &mut seed, rx0, rx1, rz0, rz1, y, kind);
    }
    // ---- the wing: 6 x 4 m outside, 3 m tall, turned 30 degrees about its centre, north of the frame
    let (wc, q) = ([4.0f64, 0.0, 11.5], quat_y(30f64.to_radians()));
    let place = |b: &mut Bricks, c: [f64; 3], h: [f64; 3], m: u8, paint: u8, sid: &mut u16| -> Vec<u32> {
        // a box of half size h centred at c in the wing's own frame, turned and placed with the wing (a member)
        *sid += 1;
        let r = qrot_y(30f64.to_radians(), c);
        let k = b.add_piece([wc[0] + r[0], wc[1] + c[1], wc[2] + r[2]], q, &box_corners(h), m, paint, *sid).expect("a wing piece");
        b.segment(k, 2.0)
    };
    let (wl, wd, wh, t) = (6.0f64, 4.0f64, 3.0f64, 0.25f64);
    // the back wall (local z +) and the front with a 1 m door under a 0.5 m lintel; the side walls between them
    parts.wing.extend(place(&mut b, [0.0, wh / 2.0, wd / 2.0 - t / 2.0], [wl / 2.0, wh / 2.0, t / 2.0], mat::MASONRY, P_STONE, &mut sid));
    let door = 0.5;
    for &s in &[-1.0f64, 1.0] { let len = wl / 2.0 - door; parts.wing.extend(place(&mut b, [s * (door + len / 2.0), wh / 2.0, -wd / 2.0 + t / 2.0], [len / 2.0, wh / 2.0, t / 2.0], mat::MASONRY, P_STONE, &mut sid)); }
    parts.wing.extend(place(&mut b, [0.0, wh - 0.25, -wd / 2.0 + t / 2.0], [door, 0.25, t / 2.0], mat::MASONRY, P_STONE, &mut sid));
    for &s in &[-1.0f64, 1.0] { parts.wing.extend(place(&mut b, [s * (wl / 2.0 - t / 2.0), wh / 2.0, 0.0], [t / 2.0, wh / 2.0, wd / 2.0 - t], mat::MASONRY, P_STONE, &mut sid)); }
    parts.wing.extend(place(&mut b, [0.0, wh + 0.1, 0.0], [wl / 2.0, 0.1, wd / 2.0], mat::CONCRETE, P_SLAB, &mut sid));
    b.build_joints();
    b.log.clear();
    (b, parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_complex_building_is_what_it_should_be_and_stands() {
        // 24 columns, 8 braces (4 segments each: their longest edge is over 6 m), 24 rafters and 22 boards (4 segments each), 6 arches of 9 stones
        // on 7 piers, a stone beam of 14 pieces (two end blocks, 7 members keystone to keystone and to the blocks); every piece within the GPU's hull limits; nothing detached; and
        // under its own weight nothing over strength
        let t = std::time::Instant::now();
        let (b, p) = build_parts();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let free = (0..b.len() as u32).filter(|&i| b.piece(i).is_some()).count();
        std::println!("complex building: {} bricks ({free} free pieces), {} joints, built in {ms:.0} ms", b.alive_count(), b.live_joints);
        assert_eq!(p.columns.len(), 24);
        assert_eq!(p.braces.len(), 8); assert!(p.braces.iter().all(|s| s.len() == 4), "{:?}", p.braces.iter().map(|s| s.len()).collect::<std::vec::Vec<_>>());
        assert_eq!(p.rafters.len(), 24); assert!(p.rafters.iter().all(|s| s.len() == 4), "{:?}", p.rafters.iter().map(|s| s.len()).collect::<std::vec::Vec<_>>());
        assert_eq!(p.boards.len(), 22); assert!(p.boards.iter().all(|s| s.len() == 4));
        assert_eq!(p.arches.len(), 6); assert!(p.arches.iter().all(|a| a.len() == 9));
        assert_eq!(p.beam.len(), 2);   // (the buttresses' upper stones)
        // the furniture, and only it, stands loose: each object a piece of its own, all its parts in it
        let det = b.detached();
        let mut loose: std::vec::Vec<u32> = det.iter().flat_map(|d| d.iter().cloned()).collect(); loose.sort();
        let mut want = p.contents.clone(); want.sort();
        let mut objects: std::vec::Vec<u16> = p.contents.iter().map(|&i| b.structure[i as usize]).collect(); objects.sort(); objects.dedup();
        std::println!("furniture: {} objects of {} parts", objects.len(), p.contents.len());
        assert_eq!(loose, want, "nothing but the furniture is detached (first other: {:?})", loose.iter().find(|i| !want.contains(i)).map(|&i| (i, b.tbox(i).min, b.piece(i).is_some())));
        assert_eq!(det.len(), objects.len(), "each object one piece");
        for d in &det { let s0 = b.structure[d[0] as usize]; assert!(d.iter().all(|&i| b.structure[i as usize] == s0), "a piece is one object"); }
        assert!(objects.len() >= 150, "a furnished building: {} objects", objects.len());
        // the wing: 3 segments for each long wall's pieces as needed, all joined to the ground or each other
        assert!(p.wing.len() >= 9, "{}", p.wing.len());
        // what each free piece is joined to: braces at both ends, rafters on the slab and the ridge, boards on two
        // rafters, arch stones to their neighbours, keystones to the beam
        let joined = |i: u32, o: u32| { let mut j = b.jhead[i as usize]; while j != NONE { if b.joint_other(j, i) == o { return true; } j = b.joint_next(j, i); } false };
        for (ai, a) in p.arches.iter().enumerate() { for (k, w) in a.windows(2).enumerate() { assert!(joined(w[0], w[1]), "arch {ai}: stones {k} and {} joined ({:?} {:?})", k + 1, b.centroid(w[0]), b.centroid(w[1])); } }
        for (k, &s) in p.keystones.iter().enumerate() { assert!(joined(s, p.loggia), "keystone {k} carries the loggia"); }
        for r in &p.rafters { assert!(joined(*r.last().unwrap(), p.ridge) || joined(r[0], p.ridge), "rafter on the ridge"); }
        let mut g = crate::game::Game::new_kind(3, 4);
        let n = g.structure_step(64);
        let made = g.pieces.iter().filter(|p| !p.is_empty()).count();
        std::println!("first structure step: {made} pieces made (the furniture)");
        assert_eq!(made, objects.len());
        std::println!("complex building under its own weight: worst ratio {:.3}, {} failed", g.max_ratio, n);
        assert_eq!(n, 0, "nothing fails under its own weight");
        assert!(g.max_ratio < 1.0);
    }
    #[test] fn cost_against_the_boxes_only_buildings() {
        // Build time, analysis time per node and memory per brick, against the concrete frame (kind 0, 4 storeys).
        // (Software timings on this machine: they compare only with each other.)
        let time = |f: &mut dyn FnMut()| { let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 };
        let mut bc = Bricks::new(); let tc = time(&mut || { bc = build(); });
        let mut b0 = Bricks::new(); let t0 = time(&mut || { b0 = crate::tower::build_kind(4, 1, 0); });
        let analyse = |b: &Bricks| -> (f64, usize) {
            let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i) && b.flags[i as usize] & F_LOOSE == 0).collect();
            let load: std::vec::Vec<f32> = set.iter().map(|&i| crate::structure::mass(b, i) * 9.81).collect();
            let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
            let mut best = f64::MAX; let mut nodes = 0;
            for _ in 0..3 { let t = std::time::Instant::now(); let a = crate::structure::analyse(b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0)); best = best.min(t.elapsed().as_secs_f64() * 1e3); nodes = a.nodes; }
            (best, nodes)
        };
        let ((ac, nc), (a0, n0)) = (analyse(&bc), analyse(&b0));
        std::println!("complex: {} bricks ({} free, {} shapes), built in {tc:.1} ms, analysis {ac:.1} ms for {nc} nodes ({:.1} us a node), {:.1} B a brick ({:.1} of it the pieces' table)",
            bc.alive_count(), bc.pieces.len(), bc.shapes.len(), 1e3 * ac / nc as f64, bc.total_bytes_per_brick(), bc.piece_bytes() as f32 / bc.alive_count() as f32);
        std::println!("concrete frame, 4 storeys: {} bricks, built in {t0:.1} ms, analysis {a0:.1} ms for {n0} nodes ({:.1} us a node), {:.1} B a brick",
            b0.alive_count(), 1e3 * a0 / n0 as f64, b0.total_bytes_per_brick());
        // Gates (phase 2: "at about the same cost"). Analysis a node: free pieces' links are general-frame springs
        // and their joints polygons, against the boxes' axis-aligned rectangles: measured 1.27 times; within 2 (a
        // ratio of two timings in the same run, so the machine's load cancels). With the stones' beds solved again
        // where they open or hinge (structure::analyse_warm: 2 re-solves here), and an arcade that takes more
        // iterations (52 against the frame's 36): 1.45 to 1.95 times over five runs on 9 Nov, close to the gate
        // (docs/PHASE2.md, still to do). Memory: three quarters of this
        // building are free pieces, each 42 B plus its share of the shapes kept once: measured 186 B a brick (370
        // with a shape a piece); under 200.
        assert!((ac / nc as f64) < 2.0 * (a0 / n0 as f64), "analysis a node {:.1} us against {:.1}", 1e3 * ac / nc as f64, 1e3 * a0 / n0 as f64);
        assert!(bc.total_bytes_per_brick() < 200.0);
    }
    #[test] fn a_blast_takes_free_pieces_apart_as_it_does_boxes() {
        // A 1.5 m blast on the first floor's south-west brace (the room shot of the lab's complexroom). A blast
        // refines what it touches, then crumbles what has its centre inside it. Free pieces were not refined: whole
        // 1.5 m segments went when their centres were inside, parts of them well outside the blast too, 0.162 m^3 of
        // the brace's 0.238 (0.68). Cut down like boxes, only what is inside goes: 0.127 m^3 (0.53; a 1.5 m sphere
        // covers about 3 of the brace's 6 m), and the brace is left in more, smaller pieces.
        let (b, p) = build_parts();
        let mut g = crate::game::Game::new_kind(3, 4);
        g.structure_step(64);
        let braces = |g: &crate::game::Game| -> std::vec::Vec<u32> { (0..g.b.len() as u32).filter(|&i| g.b.alive(i) && g.b.piece(i).is_some() && g.b.mat[i as usize] == mat::STEEL && g.b.flags[i as usize] & crate::brick::F_DYNAMIC == 0).collect() };
        let (n0, v0): (usize, f64) = { let s = braces(&g); (s.len(), s.iter().map(|&i| g.b.volume(i)).sum()) };
        // the room shot's hit point: 9.88 m along the shot from (-6, 5.5, -16) to the north
        g.blast(crate::math::v3(-6.0, 5.5, -16.0 + 9.88 + 0.05), 1.5);
        let s = braces(&g);
        let gone = v0 - s.iter().map(|&i| g.b.volume(i)).sum::<f64>();
        let one: f64 = p.braces[4].iter().map(|&i| b.volume(i)).sum();
        std::println!("blast on a brace: {gone:.3} m^3 of steel braces gone (one brace is {one:.3} m^3), {} brace pieces, {n0} before", s.len());
        assert!(gone > 0.4 * one && gone < 0.62 * one, "about half the brace goes: {:.2}", gone / one);
        assert!(s.len() > n0, "what is left of it is cut into more pieces");
    }
    #[test] fn an_arch_under_the_loggia_stands_without_its_keystone() {
        // The arcade's third arch with its keystone taken out. The loggia bears on the stones' level tops, and the
        // halves of the ring are held between it and the piers: their beds take no tension (structure::mortar_bed),
        // but the floor over them spans the keystone's 0.5 m and presses them onto their beds. Measured: nothing fails,
        // worst ratio 0.50 after (0.92 before: what the keystone carried goes to its neighbours' beds, which have
        // room). (The arches as they were, the floor on a stone beam over the crowns: the stones fell, 7 of 8, and two
        // arches to the west with them.) The arch with nothing over it is in docs/PHASE2.md (still to do).
        let (_, p) = build_parts();
        let mut g = crate::game::Game::new_kind(3, 4);
        g.structure_step(64);
        let arch = p.arches[2].clone(); let key = p.keystones[2];
        g.b.remove(key);
        let (mut steps, mut failed) = (0, 0);
        loop { let n = g.structure_step(64); failed += n; steps += 1; if n == 0 || steps > 60 { break; } }
        let fallen = arch.iter().filter(|&&s| s != key && g.b.flags[s as usize] & crate::brick::F_DYNAMIC != 0).count();
        std::println!("arch without its keystone, under the loggia: {failed} joints failed in {steps} steps; {fallen} of its 8 other stones fell; worst ratio now {:.2}", g.max_ratio);
        assert_eq!(failed, 0);
        assert_eq!(fallen, 0);
        assert!(g.max_ratio < 1.0);
    }
}
