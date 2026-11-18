//! What the quarter's buildings are put together with: boxes on the 0.125 m grid, free convex pieces at any angle,
//! straight members cut at their ends, prisms, loose contents. Lengths in metres (x east, y up, z north).
//!
//! Paints 9 to 15 are finishes of their own, whatever the material (engine/src/mesh.rs `colour`): 9 marble, 10 white
//! (render on masonry, paint on steel), 11 travertine and sandstone, 12 dark weathered timber, 13 red oxide,
//! 14 roof tile, 15 dark green.
use crate::brick::{mat, Bricks, F_INFILL, F_LOOSE, F_REINF};
use crate::math::floor64;
use alloc::vec::Vec;

pub const T: f64 = 0.125;
pub fn tk(m: f64) -> i32 { floor64(m / T + 0.5) as i32 }
pub const P_SLAB: u8 = 0;
pub const P_COLUMN: u8 = 1;
pub const P_FACADE: u8 = 2;
pub const P_CORE: u8 = 3;
pub const P_STAIR: u8 = 4;
pub const P_PARTITION: u8 = 5;
pub const P_BRICK: u8 = 6;
pub const P_STEEL: u8 = 7;
pub const P_WOOD: u8 = 8;
pub const P_MARBLE: u8 = 9;
pub const P_WHITE: u8 = 10;
pub const P_TRAVERTINE: u8 = 11;
pub const P_DARKWOOD: u8 = 12;
pub const P_OXIDE: u8 = 13;
pub const P_TILE: u8 = 14;
pub const P_GREEN: u8 = 15;

pub type P3 = [f64; 3];
pub fn add(a: P3, b: P3) -> P3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: P3, b: P3) -> P3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn mul(a: P3, k: f64) -> P3 { [a[0] * k, a[1] * k, a[2] * k] }
pub fn dot(a: P3, b: P3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: P3, b: P3) -> P3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn unit(a: P3) -> P3 { let l = crate::math::sqrt64(dot(a, a)); mul(a, 1.0 / l) }
pub fn sin(a: f64) -> f64 { crate::math::sin(a as f32) as f64 }
pub fn cos(a: f64) -> f64 { crate::math::cos(a as f32) as f64 }
pub const PI: f64 = core::f64::consts::PI;

/// A box from `lo` to `hi` (m, snapped to the grid), member 1. Concrete is reinforced.
pub fn bx(b: &mut Bricks, lo: P3, hi: P3, m: u8, paint: u8) -> u32 {
    let l = [tk(lo[0]), tk(lo[1]), tk(lo[2])];
    let h = [tk(hi[0]), tk(hi[1]), tk(hi[2])];
    debug_assert!(h[0] > l[0] && h[1] > l[1] && h[2] > l[2], "an empty box: {lo:?} {hi:?}");
    let i = b.add(l, [h[0] - l[0], h[1] - l[1], h[2] - l[2]], m, paint, 1);
    if m == mat::CONCRETE { b.flags[i as usize] |= F_REINF; }
    i
}
/// A box that carries only itself (glazing, partitions, cladding): `F_INFILL`.
pub fn infill(b: &mut Bricks, lo: P3, hi: P3, m: u8, paint: u8) -> u32 { let i = bx(b, lo, hi, m, paint); b.flags[i as usize] |= F_INFILL; i }
/// A long box in pieces of at most `max` m along `axis` (a brick is at most 255 ticks a side, and a wall or beam
/// in one brick breaks as one).
pub fn run(b: &mut Bricks, lo: P3, hi: P3, axis: usize, max: f64, m: u8, paint: u8) -> Vec<u32> {
    let mut out = Vec::new();
    let n = (((hi[axis] - lo[axis]) / max) - 1e-9).max(0.0) as usize + 1;
    for k in 0..n {
        let (mut l, mut h) = (lo, hi);
        l[axis] = tk(lo[axis] + (hi[axis] - lo[axis]) * k as f64 / n as f64) as f64 * T;
        h[axis] = if k + 1 == n { hi[axis] } else { tk(lo[axis] + (hi[axis] - lo[axis]) * (k + 1) as f64 / n as f64) as f64 * T };
        if h[axis] > l[axis] { out.push(bx(b, l, h, m, paint)); }
    }
    out
}
/// A wall of cut stone or brick laid in courses `course` m high of units `unit` m long, every other course set off
/// by half a unit (running bond), from `lo` to `hi`, along `axis` (0 or 2). Openings are left by the caller: lay the
/// wall in parts.
pub fn coursed(b: &mut Bricks, lo: P3, hi: P3, axis: usize, unit: f64, course: f64, m: u8, paint: u8) -> Vec<u32> {
    let mut out = Vec::new();
    let mut y = lo[1];
    let mut c = (lo[1] / course + 0.5) as i64;
    while y < hi[1] - 1e-9 {
        let y2 = (y + course).min(hi[1]);
        let off = if c % 2 == 0 { 0.0 } else { unit / 2.0 };
        // (units on the world's grid of the wall's line, so courses bond across parts of one wall)
        let mut a = lo[axis];
        while a < hi[axis] - 1e-9 {
            let next = (floor64((a - off) / unit + 1e-9) + 1.0) * unit + off;
            let a2 = next.min(hi[axis]);
            let (mut l, mut h) = ([lo[0], y, lo[2]], [hi[0], y2, hi[2]]);
            l[axis] = a; h[axis] = a2;
            if tk(a2) > tk(a) && tk(y2) > tk(y) { out.push(bx(b, l, h, m, paint)); }
            a = a2;
        }
        y = y2; c += 1;
    }
    out
}
/// A free piece from corners in the world (m): centre at their mean, not turned; `sid` its member (the segments of
/// one member share it; a joint of masonry between members is a mortar bed, structure::connection).
pub fn piece(b: &mut Bricks, pts: &[P3], m: u8, paint: u8, sid: u16) -> u32 {
    let n = pts.len() as f64;
    let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
    let cs = [floor64(c[0] * 1000.0 + 0.5) / 1000.0, floor64(c[1] * 1000.0 + 0.5) / 1000.0, floor64(c[2] * 1000.0 + 0.5) / 1000.0];
    let local: Vec<P3> = pts.iter().map(|p| [p[0] - cs[0], p[1] - cs[1], p[2] - cs[2]]).collect();
    let i = b.add_piece(cs, [0.0, 0.0, 0.0, 1.0], &local, m, paint, sid).expect("a piece within the hull limits");
    if m == mat::CONCRETE { b.flags[i as usize] |= F_REINF; }
    i
}
/// An upright prism: the polygon `poly` (x, z; convex) from `y0` to `y1`.
pub fn prism(b: &mut Bricks, poly: &[[f64; 2]], y0: f64, y1: f64, m: u8, paint: u8, sid: u16) -> u32 {
    let mut pts = Vec::new();
    for q in poly { pts.push([q[0], y0, q[1]]); pts.push([q[0], y1, q[1]]); }
    piece(b, &pts, m, paint, sid)
}
/// A prism along z: the polygon `poly` (x, y; convex) from `z0` to `z1`.
pub fn prism_z(b: &mut Bricks, poly: &[[f64; 2]], z0: f64, z1: f64, m: u8, paint: u8, sid: u16) -> u32 {
    let mut pts = Vec::new();
    for q in poly { pts.push([q[0], q[1], z0]); pts.push([q[0], q[1], z1]); }
    piece(b, &pts, m, paint, sid)
}
/// A prism along x: the polygon `poly` (z, y; convex) from `x0` to `x1`.
pub fn prism_x(b: &mut Bricks, poly: &[[f64; 2]], x0: f64, x1: f64, m: u8, paint: u8, sid: u16) -> u32 {
    let mut pts = Vec::new();
    for q in poly { pts.push([x0, q[1], q[0]]); pts.push([x1, q[1], q[0]]); }
    piece(b, &pts, m, paint, sid)
}
/// A regular polygon of `n` sides about (cx, cz), `r` to its flats, a flat facing +x.
pub fn ngon(cx: f64, cz: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
    let rc = r / cos(PI / n as f64);
    (0..n).map(|k| { let a = (2.0 * k as f64 + 1.0) * PI / n as f64; [cx + rc * cos(a), cz + rc * sin(a)] }).collect()
}
/// A straight member from `f` to `h` (m, on its axis), `w` deep across its axis along `n` (at right angles to the
/// axis), `d` wide along `out`, its ends cut by the planes through `f` (normal `nf`) and `h` (normal `nh`); in
/// segments of at most `seg` m.
pub fn member(b: &mut Bricks, f: P3, h: P3, n: P3, out: P3, w: f64, d: f64, nf: P3, nh: P3, seg: f64, m: u8, paint: u8, sid: &mut u16) -> Vec<u32> {
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
    b.segment(k, seg)
}
/// A member square to its own axis at both ends.
pub fn strut(b: &mut Bricks, f: P3, h: P3, n: P3, w: f64, d: f64, seg: f64, m: u8, paint: u8, sid: &mut u16) -> Vec<u32> {
    let a = unit(sub(h, f));
    let nn = unit(sub(n, mul(a, dot(n, a))));
    let out = cross(a, nn);
    member(b, f, h, nn, out, w, d, a, a, seg, m, paint, sid)
}
/// A rotation about y (x y z w), and a point turned by it.
pub fn quat_y(a: f64) -> [f32; 4] { [0.0, crate::math::sin((a / 2.0) as f32), 0.0, crate::math::cos((a / 2.0) as f32)] }
pub fn rot_y(a: f64, v: P3) -> P3 { let (s, c) = (sin(a), cos(a)); [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]] }
pub fn box_corners(h: P3) -> Vec<P3> { let mut v = Vec::new(); for &a in &[-1.0, 1.0] { for &b in &[-1.0, 1.0] { for &c in &[-1.0, 1.0] { v.push([a * h[0], b * h[1], c * h[2]]); } } } v }
/// A box of half size `h` centred at `c`, turned by `a` about y: a free piece.
pub fn turned(b: &mut Bricks, c: P3, a: f64, h: P3, m: u8, paint: u8, sid: u16) -> u32 {
    let i = b.add_piece(c, quat_y(a), &box_corners(h), m, paint, sid).expect("a turned box");
    if m == mat::CONCRETE { b.flags[i as usize] |= F_REINF; }
    i
}
/// One loose box of an object (its parts joined to each other and to nothing else: a body standing where it is put).
pub fn loose(b: &mut Bricks, out: &mut Vec<u32>, lo: P3, hi: P3, m: u8, paint: u8, sid: u16) {
    let i = bx(b, lo, hi, m, paint);
    b.structure[i as usize] = sid; b.flags[i as usize] |= F_LOOSE;
    out.push(i);
}
/// The bricks of `b` as a simulation of one lot (for a building's own tests).
pub fn sim_of(b: Bricks) -> crate::sim::Sim {
    let top = (0..b.len() as u32).filter(|&i| b.alive(i)).map(|i| b.tbox(i).max[1]).max().unwrap_or(0);
    crate::sim::Sim::from_bricks(b, crate::sim::Tiles { tile: 0, n: 1, slices: top / crate::sim::BAND + 2 })
}
/// The joints nearest their strength under the building's own weight (at rest), worst first: for a building's tests.
#[cfg(test)]
pub fn worst_joints(b: &Bricks, n: usize) {
    extern crate std;
    use crate::brick::{F_ALIVE, F_DYNAMIC};
    let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.flags[i as usize] & (F_ALIVE | F_DYNAMIC | F_LOOSE) == F_ALIVE).collect();
    let load: Vec<f32> = set.iter().map(|&i| crate::structure::weight(b, i)).collect();
    let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
    let a = crate::structure::analyse(b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
    let mut r = a.ratios.clone();
    r.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let r3 = |v: [f64; 3]| [(v[0] * 1e3 + 0.5).floor() / 1e3, (v[1] * 1e3 + 0.5).floor() / 1e3, (v[2] * 1e3 + 0.5).floor() / 1e3];
    for &(j, ratio) in r.iter().take(n) {
        let (p, q) = (b.ja[j as usize], b.jb[j as usize]);
        let (pa, qa) = (b.box_m(p), b.box_m(q));
        std::println!("  {ratio:.3} (bed: {}): centre {:?} half {:?} (material {}, paint {}) to centre {:?} half {:?} (material {}, paint {})", a.beds.contains(&j), r3(pa.0), r3(pa.1), b.mat[p as usize], b.paint[p as usize], r3(qa.0), r3(qa.1), b.mat[q as usize], b.paint[q as usize]);
    }
    std::println!("  worst inside a brick: {:?}; {} iterations, {} nodes", a.worst_in, a.iterations, a.nodes);
}
/// A building's own check: every brick within the brick store's limits, nothing but its loose contents detached,
/// and under its own weight and its floors' loads nothing over its strength. Returns the worst joint's ratio.
#[cfg(test)]
pub fn stands(name: &str, b: &Bricks) -> f32 {
    extern crate std;
    let det = b.detached();
    let bad = det.iter().filter(|d| d.iter().any(|&i| b.flags[i as usize] & F_LOOSE == 0)).count();
    if bad > 0 {
        for d in det.iter().filter(|d| d.iter().any(|&i| b.flags[i as usize] & F_LOOSE == 0)).take(6) {
            let i = d[0]; let (lo, hi) = b.box_m(i);
            std::println!("  detached: {} bricks, first {:?}..{:?} material {} paint {}", d.len(), lo, hi, b.mat[i as usize], b.paint[i as usize]);
        }
    }
    assert_eq!(bad, 0, "{name}: parts held by nothing");
    let free = (0..b.len() as u32).filter(|&i| b.alive(i) && b.piece(i).is_some()).count();
    let mut g = sim_of(b.clone());
    let t = std::time::Instant::now();
    let failed = g.structure_step(64);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
    for i in 0..b.len() as u32 { if !b.alive(i) { continue; } let t = b.tbox(i); for k in 0..3 { lo[k] = lo[k].min(t.min[k]); hi[k] = hi[k].max(t.max[k]); } }
    std::println!("{name}: {} bricks ({free} free pieces), {} joints, {:?}..{:?} m; first structure step {ms:.0} ms, {failed} failed, worst ratio {:.3}",
        b.alive_count(), b.live_joints, lo.map(|x| x as f64 * T), hi.map(|x| x as f64 * T), g.max_ratio);
    let say = |g: &crate::sim::Sim| for &(p, q, r) in g.last_failed.iter().take(24) {
        let (a, c) = (b.box_m(p), b.box_m(q));
        let r2 = |v: [f64; 3]| [(v[0] * 1e3 + 0.5).floor() / 1e3, (v[1] * 1e3 + 0.5).floor() / 1e3, (v[2] * 1e3 + 0.5).floor() / 1e3];
        std::println!("  failed at {r:.2}: {:?}..{:?} (material {}, paint {}) to {:?}..{:?} (material {}, paint {})", r2(a.0), r2(a.1), b.mat[p as usize], b.paint[p as usize], r2(c.0), r2(c.1), b.mat[q as usize], b.paint[q as usize]);
    };
    if failed > 0 { say(&g); }
    assert_eq!(failed, 0, "{name}: joints over their strength under its own weight");
    // and with the dynamic structure, a second of it: nothing fails, nothing leaves
    let mut g = sim_of(b.clone());
    g.dynamic = true;
    let mut f2 = 0;
    for _ in 0..12 { g.struct_dt = 1.0 / 12.0; f2 += g.structure_step(64); }
    std::println!("  dynamic, 1 s: {f2} failed, worst ratio {:.3}", g.max_ratio);
    if f2 > 0 { say(&g); }
    assert_eq!(f2, 0, "{name}: joints fail in the dynamic structure's first second");
    g.max_ratio
}
