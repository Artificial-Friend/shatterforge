//! The aqueduct: a stretch of a Roman arcade of cut stone, laid dry, at half the size of the Aqua Claudia's (Rome,
//! AD 52: arches of 6 m on piers 3.3 m square): five arches of 3 m span on piers 1.5 m square and 6 m tall, 4.5 m
//! apart, an abutment at each end, and on top the channel (the specus): a floor, two walls, cover slabs.
//!
//! The arches are built as the complex building's arcade is (game/src/complex.rs, where the reasons are): nine wedge
//! stones each, carried out to a level top at the keystone's height and to the piers' centre lines; over each pier the
//! stones of both arches that reach its line are one stone (the tas-de-charge). Stones meet on beds that hold no
//! tension (`structure::mortar_bed`).
use crate::brick::{mat, Bricks};
use crate::kit::*;
use alloc::vec;
use alloc::vec::Vec;

/// arches, the piers' spacing, half width and height, the arcade's half depth, the stones in an arch
pub const ARCHES: usize = 5;
pub const PITCH: f64 = 4.5;
pub const PIER: f64 = 0.75;
pub const PIER_H: f64 = 6.0;
pub const HALF_D: f64 = 0.75;
pub const STONES: usize = 9;
/// the level top of the arches above the springing, and how far an abutment reaches beyond its pier's centre
pub const RING_TOP: f64 = 2.25;
pub const ABUT: f64 = 3.0;

#[derive(Default, Clone, Debug)]
pub struct Parts { pub piers: Vec<Vec<u32>>, pub arches: Vec<Vec<u32>>, pub keystones: Vec<u32>, pub abutments: Vec<u32>, pub channel: Vec<u32>, pub covers: Vec<u32> }

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
pub fn pier_x(k: usize) -> f64 { (k as f64 - ARCHES as f64 / 2.0) * PITCH }

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut p = Parts::default();
    let mut sid: u16 = 100;
    let st = |b: &mut Bricks, lo: P3, hi: P3| bx(b, lo, hi, mat::MASONRY, P_TRAVERTINE);
    let np = ARCHES + 1;
    let top = PIER_H + RING_TOP;
    // ---- the piers: blocks 0.75 m high, the full section each; the end ones run out as abutments, stepped
    for k in 0..np {
        let x = pier_x(k);
        let mut col = Vec::new();
        for c in 0..(PIER_H / 0.75 + 0.5) as usize {
            let (y0, y1) = (0.75 * c as f64, 0.75 * (c + 1) as f64);
            let (xa, xb) = if k == 0 { (x - ABUT + 0.375 * (c / 2) as f64, x + PIER) } else if k == np - 1 { (x - PIER, x + ABUT - 0.375 * (c / 2) as f64) } else { (x - PIER, x + PIER) };
            col.push(st(&mut b, [xa, y0, -HALF_D], [xb, y1, HALF_D]));
        }
        p.piers.push(col);
    }
    // ---- the arches
    let mut key_top = 0.0;
    let mut springers: Vec<u16> = vec![0; np];
    for ai in 0..ARCHES {
        let (xa, xb) = (pier_x(ai), pier_x(ai + 1));
        let (cx, r0) = (0.5 * (xa + xb), 0.5 * (xb - xa) - PIER);
        let mut arch = Vec::new();
        let tk = PI * (1.0 - (STONES / 2) as f64 / STONES as f64);
        let r1 = RING_TOP / sin(tk);
        for k in 0..STONES {
            let (t0, t1) = (PI * (1.0 - k as f64 / STONES as f64), PI * (1.0 - (k + 1) as f64 / STONES as f64));
            let half = r0 + PIER;
            let far = 4.0 * (r1 + half);
            let mut poly: Vec<[f64; 2]> = vec![
                [cx + r0 * cos(t0), PIER_H + r0 * sin(t0)], [cx + far * cos(t0), PIER_H + far * sin(t0)],
                [cx + far * cos(t1), PIER_H + far * sin(t1)], [cx + r0 * cos(t1), PIER_H + r0 * sin(t1)]];
            poly = clip(&poly, [0.0, 1.0], top);
            poly = clip(&poly, [1.0, 0.0], cx + half);
            poly = clip(&poly, [-1.0, 0.0], -(cx - half));
            let mut pts = Vec::new();
            for q in &poly { for &z in &[-HALF_D, HALF_D] { pts.push([q[0], q[1], z]); } }
            for q in pts.iter_mut() { if (q[1] - PIER_H).abs() < 1e-3 { q[1] = PIER_H; } }
            let tas = [(ai, cx - half), (ai + 1, cx + half)].iter().find(|&&(_, xe)| poly.iter().any(|q| (q[0] - xe).abs() < 1e-6)).map(|&(pp, _)| pp);
            let id = match tas {
                Some(pp) if springers[pp] != 0 => springers[pp],
                _ => { sid += 1; if let Some(pp) = tas { springers[pp] = sid; } sid }
            };
            let s = piece(&mut b, &pts, mat::MASONRY, P_TRAVERTINE, id);
            if k == STONES / 2 { p.keystones.push(s); key_top = pts.iter().fold(0.0f64, |m, q| m.max(q[1])); }
            arch.push(s);
        }
        p.arches.push(arch);
    }
    // the abutments' upper stones, from the springing to the arches' level top, one stone with the end arches' springers
    for (k, &sg) in [-1.0f64, 1.0].iter().enumerate() {
        let pk = if k == 0 { 0 } else { np - 1 };
        let (pc, xo) = (pier_x(pk), pier_x(pk) + sg * (ABUT - 0.375 * 4.0));
        let mut pts = Vec::new();
        for &x in &[pc, xo] { for &y in &[PIER_H, key_top] { for &z in &[-HALF_D, HALF_D] { pts.push([x, y, z]); } } }
        p.abutments.push(piece(&mut b, &pts, mat::MASONRY, P_TRAVERTINE, springers[pk]));
    }
    // ---- the channel on the arches' level top: a floor of slabs 0.25 m thick, two walls 0.25 m thick and 1 m high in
    // blocks, cover slabs across; lengths of 2.25 m (half a bay), joints over the keystones and the piers
    let (x0, x1) = (pier_x(0) - 1.5, pier_x(np - 1) + 1.5);
    let mut x = x0;
    let mut n = 0;
    while x < x1 - 1e-9 {
        let x2 = (x + 2.25).min(x1);
        p.channel.push(st(&mut b, [x, top, -HALF_D], [x2, top + 0.25, HALF_D]));
        // (the walls in running bond with the floor: set off by half a length)
        let (wa, wb) = (if n == 0 { x } else { x - 1.125 }, if x2 >= x1 - 1e-9 { x2 } else { x2 - 1.125 });
        for &(za, zb) in &[(-HALF_D, -HALF_D + 0.25), (HALF_D - 0.25, HALF_D)] { p.channel.push(st(&mut b, [wa, top + 0.25, za], [wb, top + 1.25, zb])); }
        p.covers.push(st(&mut b, [x, top + 1.25, -HALF_D], [x2, top + 1.375, HALF_D]));
        x = x2; n += 1;
    }
    b.build_joints();
    b.log.clear();
    (b, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_aqueduct_is_what_it_should_be_and_stands() {
        let (b, p) = build_parts();
        assert_eq!(p.piers.len(), ARCHES + 1); assert_eq!(p.arches.len(), ARCHES);
        assert!(p.arches.iter().all(|a| a.len() == STONES));
        assert_eq!(p.keystones.len(), ARCHES);
        let r = stands("the aqueduct", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
