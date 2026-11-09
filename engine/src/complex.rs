//! The complex building (kind 4, docs/PHASE2.md step 6), made of boxes and free convex pieces at any angle:
//! - a steel frame of two 4 m storeys on a 3 x 2 grid of 6 m bays (x -9..9, z -6..6 m): 0.5 m steel columns, 0.5 m
//!   deep steel beams on the grid lines, 0.25 m concrete floors; the first floor left out over one bay (x -3..3,
//!   z -6..0) for a double-height hall;
//! - diagonal steel braces (free pieces) in the end bays of both long facades, on both storeys, from the foot of
//!   one column to the head of the next, under the beam;
//! - a pitched timber roof: a steel ridge beam on four posts along the middle (z = 0), rafters from the eaves to
//!   the ridge every 1.5 m, and roof boards on the rafters (all free pieces, long ones in segments of at most 2 m);
//! - in front (z -9.5..-9 m), an arcade: seven masonry piers 3 m tall with six semicircular arches of nine wedge
//!   stones, and a stone beam from keystone to keystone.
//! - room contents on the first floor: desks, cabinets and a chair turned in each bay, loose (joined to nothing:
//!   bodies resting on the floor from the first structure step);
//! - an irregular plan: a one-storey wing to the north, turned 30 degrees, of 0.25 m masonry walls (a door in
//!   its front under a stone lintel) and a 0.2 m concrete roof slab, all free pieces.
use crate::brick::{mat, Bricks, F_LOOSE, F_REINF, NONE};
use crate::math::floor64;
use alloc::vec::Vec;

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
/// the arcade: piers' centres, their half width, height; the arches' ring and depth (z range)
const PIER_X: [f64; 7] = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0];
const PIER: f64 = 0.375;
const PIER_H: f64 = 3.0;
const RING: f64 = 0.375;
const ARC_Z: (f64, f64) = (-9.5, -9.0);
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

/// What the complex building is made of, for its scenarios and tests.
#[derive(Default, Clone, Debug)]
pub struct Parts { pub contents: Vec<u32>, pub wing: Vec<u32>, pub columns: Vec<u32>, pub braces: Vec<Vec<u32>>, pub rafters: Vec<Vec<u32>>, pub boards: Vec<Vec<u32>>, pub ridge: u32, pub piers: Vec<u32>, pub arches: Vec<Vec<u32>>, pub keystones: Vec<u32>, pub beam: Vec<u32> }

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut parts = Parts { ridge: NONE, ..Default::default() };
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
    for &x in &PIER_X { parts.piers.push(bx(&mut b, [x - PIER, 0.0, ARC_Z.0], [x + PIER, PIER_H, ARC_Z.1], mat::MASONRY, P_STONE)); }
    let mut key_top = 0.0;
    for w in PIER_X.windows(2) {
        let (cx, r0) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]) - PIER);
        let r1 = r0 + RING;
        let mut arch = Vec::new();
        for k in 0..STONES {
            let (t0, t1) = (core::f64::consts::PI * (1.0 - k as f64 / STONES as f64), core::f64::consts::PI * (1.0 - (k + 1) as f64 / STONES as f64));
            let mut pts = Vec::new();
            for &t in &[t0, t1] { for &r in &[r0, r1] { for &z in &[ARC_Z.0, ARC_Z.1] {
                pts.push([cx + r * crate::math::cos(t as f32) as f64, PIER_H + r * crate::math::sin(t as f32) as f64, z]);
            } } }
            // (the springing stones lie flat on the piers: their bed at exactly the piers' top)
            for p in pts.iter_mut() { if (p[1] - PIER_H).abs() < 1e-3 { p[1] = PIER_H; } }
            sid += 1;
            let s = piece(&mut b, &pts, mat::MASONRY, P_STONE, sid);
            if k == STONES / 2 { parts.keystones.push(s); key_top = pts.iter().fold(0.0f64, |m, p| m.max(p[1])); }
            arch.push(s);
        }
        parts.arches.push(arch);
    }
    // the stone beam on the keystones: a 0.5 m deep member from each keystone's centre to the next (two segments),
    // resting on half of each (from pier centre to pier centre, resting on one keystone under its middle, the end ones
    // overhung 1.5 m and loaded that keystone's bed at 0.90 of where its thrust may go)
    let keys: Vec<f64> = PIER_X.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    for w in keys.windows(2) {
        let mut pts = Vec::new();
        for &x in &[w[0], w[1]] { for &y in &[key_top, key_top + 0.5] { for &z in &[ARC_Z.0, ARC_Z.1] { pts.push([x, y, z]); } } }
        sid += 1;
        let k = piece(&mut b, &pts, mat::MASONRY, P_STONE, sid);
        parts.beam.extend(b.segment(k, 2.0));
    }
    // ---- room contents on the first floor, in every bay with a floor: two desks, a cabinet, a chair turned 25 degrees
    let yf = H;
    for &cx in &[-6.0f64, 0.0, 6.0] { for &cz in &[-3.0f64, 3.0] {
        if cx == 0.0 && cz < 0.0 { continue; }    // the hall: no floor
        for &dx in &[-1.25f64, 1.25] { parts.contents.push(bx(&mut b, [cx + dx - 0.75, yf, cz - 0.375], [cx + dx + 0.75, yf + 0.75, cz + 0.375], mat::TIMBER, P_TIMBER)); }
        parts.contents.push(bx(&mut b, [cx - 0.25, yf, cz + 1.75], [cx + 0.25, yf + 1.75, cz + 2.25], mat::STEEL, P_STEEL));
        let q = quat_y(25f64.to_radians());
        sid += 1;
        let chair = b.add_piece([cx, yf + 0.45, cz - 1.5], q, &box_corners([0.25, 0.45, 0.25]), mat::TIMBER, P_TIMBER, sid).expect("a chair");
        parts.contents.push(chair);
    } }
    for &i in &parts.contents { b.flags[i as usize] |= F_LOOSE; }
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
        // on 7 piers, a stone beam of 10 segments (5 members, keystone to keystone); every piece within the GPU's hull limits; nothing detached; and
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
        assert_eq!(p.beam.len(), 10);
        // the contents, and only they, stand loose: each a piece of its own
        let det = b.detached();
        let mut loose: std::vec::Vec<u32> = det.iter().flat_map(|d| d.iter().cloned()).collect(); loose.sort();
        let mut want = p.contents.clone(); want.sort();
        assert_eq!(det.len(), 20, "detached pieces: the 20 loose contents, each on its own");
        assert_eq!(loose, want, "nothing else is detached (first other: {:?})", loose.iter().find(|i| !want.contains(i)).map(|&i| (i, b.tbox(i).min, b.piece(i).is_some())));
        // the wing: 3 segments for each long wall's pieces as needed, all joined to the ground or each other
        assert!(p.wing.len() >= 9, "{}", p.wing.len());
        // what each free piece is joined to: braces at both ends, rafters on the slab and the ridge, boards on two
        // rafters, arch stones to their neighbours, keystones to the beam
        let joined = |i: u32, o: u32| { let mut j = b.jhead[i as usize]; while j != NONE { if b.joint_other(j, i) == o { return true; } j = b.joint_next(j, i); } false };
        for a in &p.arches { for w in a.windows(2) { assert!(joined(w[0], w[1]), "arch stones joined"); } }
        for (k, &s) in p.keystones.iter().enumerate() { assert!(p.beam.iter().any(|&m| joined(s, m)), "keystone {k} carries the beam"); }
        for r in &p.rafters { assert!(joined(*r.last().unwrap(), p.ridge) || joined(r[0], p.ridge), "rafter on the ridge"); }
        let mut g = crate::game::Game::new_kind(3, 4);
        let n = g.structure_step(64);
        std::println!("first structure step: {} pieces made (the contents)", g.pieces.iter().filter(|p| !p.is_empty()).count());
        assert_eq!(g.pieces.iter().filter(|p| !p.is_empty()).count(), 20);
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
        // ratio of two timings in the same run, so the machine's load cancels). Memory: three quarters of this
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
    #[test] fn an_arch_without_its_keystone_falls() {
        // The arcade's third arch with its keystone taken out: the two halves of the ring lean in on nothing, and their
        // beds open (no tension: structure::mortar_bed): the stones come down, structure step by step, and the arches
        // to the west, whose thrust it balanced, follow. Measured: 7 of its 8 other stones fall, and 4 and 5 of the
        // first and second arches'; the three to the east stand. With the stones' joints as the stone itself (0.3 MPa
        // tension), nothing failed: each half stood as a cantilever off its pier, at 0.78 of strength.
        let (_, p) = build_parts();
        let mut g = crate::game::Game::new_kind(3, 4);
        g.structure_step(64);
        let arch = p.arches[2].clone(); let key = p.keystones[2];
        g.b.remove(key);
        let (mut steps, mut failed) = (0, 0);
        loop { let n = g.structure_step(64); failed += n; steps += 1; if n == 0 || steps > 60 { break; } }
        let fallen = arch.iter().filter(|&&s| s != key && g.b.flags[s as usize] & crate::brick::F_DYNAMIC != 0).count();
        let beam_up = p.beam.iter().filter(|&&b| g.b.alive(b) && g.b.flags[b as usize] & crate::brick::F_DYNAMIC == 0).count();
        let others: std::vec::Vec<usize> = p.arches.iter().map(|a| a.iter().filter(|&&s| g.b.flags[s as usize] & crate::brick::F_DYNAMIC != 0).count()).collect();
        std::println!("arch without its keystone: {failed} joints failed in {steps} steps; {fallen} of its 8 other stones fell; {beam_up} of the beam's 10 segments standing; stones fallen per arch {others:?}; worst ratio now {:.2}", g.max_ratio);
        assert!(fallen >= 6, "the arch comes down");
        assert!(others[3..].iter().all(|&n| n == 0), "the arches east of it stand");
    }
}
