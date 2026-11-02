//! Render mesh of a brick store: every brick face minus the parts covered by touching neighbours (exact, from
//! the joint graph), as quads. Faces resting on the ground are dropped.
use crate::brick::{mat, Bricks, NONE, TICK};
use alloc::vec::Vec;

/// Interleaved vertices (position m, normal, colour) and triangle indices.
#[derive(Default)]
pub struct Mesh { pub v: Vec<f32>, pub i: Vec<u32>, pub quads: usize }

type Rect = [i32; 4]; // u0, v0, u1, v1 (half-open)

fn subtract(rs: &mut Vec<Rect>, c: Rect) {
    let mut out = Vec::with_capacity(rs.len() + 4);
    for &r in rs.iter() {
        if c[0] >= r[2] || c[2] <= r[0] || c[1] >= r[3] || c[3] <= r[1] { out.push(r); continue; }
        // up to four pieces of r outside c
        if c[1] > r[1] { out.push([r[0], r[1], r[2], c[1]]); }
        if c[3] < r[3] { out.push([r[0], c[3], r[2], r[3]]); }
        let (v0, v1) = (r[1].max(c[1]), r[3].min(c[3]));
        if c[0] > r[0] { out.push([r[0], v0, c[0], v1]); }
        if c[2] < r[2] { out.push([c[2], v0, r[2], v1]); }
    }
    *rs = out;
}

pub fn colour(b: &Bricks, i: u32) -> [f32; 3] {
    let iu = i as usize;
    let base = match (b.mat[iu], b.paint[iu]) {
        (mat::GLASS, _) => [0.36, 0.50, 0.58],
        (mat::TIMBER, _) => [0.86, 0.84, 0.79],
        (_, 0) => [0.62, 0.61, 0.58],   // slab
        (_, 1) => [0.56, 0.56, 0.54],   // column
        (_, 2) => [0.79, 0.77, 0.71],   // facade
        (_, 3) => [0.67, 0.65, 0.61],   // core
        (_, 4) => [0.58, 0.57, 0.55],   // stair
        _ => [0.7, 0.7, 0.7],
    };
    // a little per-brick variation so large surfaces are not flat colour
    let h = (i.wrapping_mul(2654435761) >> 24) as f32 / 255.0;
    let k = 0.96 + 0.06 * h;
    [base[0] * k, base[1] * k, base[2] * k]
}

pub fn build(b: &Bricks) -> Mesh { build_filtered(b, |_, _| true) }

/// Mesh of the live bricks that pass `keep` (a chunk of the store).
pub fn build_filtered(b: &Bricks, keep: impl Fn(&Bricks, u32) -> bool) -> Mesh {
    let mut m = Mesh::default();
    let mut rects: Vec<Rect> = Vec::new();
    for i in 0..b.len() as u32 {
        if !b.alive(i) || !keep(b, i) { continue; }
        let tb = b.tbox(i);
        let col = colour(b, i);
        for ax in 0..3 {
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            for side in 0..2 {
                // the ground hides bottom faces at y = 0
                if ax == 1 && side == 0 && tb.min[1] <= 0 { continue; }
                rects.clear();
                rects.push([tb.min[u], tb.min[w], tb.max[u], tb.max[w]]);
                let mut j = b.jhead[i as usize];
                while j != NONE && !rects.is_empty() {
                    let ju = j as usize;
                    // joint axis matches and the neighbour is on this side: a = min side, b = max side
                    if b.jaxis[ju] as usize == ax && ((side == 1 && b.ja[ju] == i) || (side == 0 && b.jb[ju] == i)) {
                        let o = b.tbox(b.joint_other(j, i));
                        subtract(&mut rects, [o.min[u], o.min[w], o.max[u], o.max[w]]);
                    }
                    j = b.joint_next(j, i);
                }
                let plane = if side == 1 { tb.max[ax] } else { tb.min[ax] };
                let mut n = [0.0f32; 3]; n[ax] = if side == 1 { 1.0 } else { -1.0 };
                for r in &rects {
                    let base = (m.v.len() / 9) as u32;
                    for (cu, cw) in [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])] {
                        let mut p = [0i32; 3]; p[ax] = plane; p[u] = cu; p[w] = cw;
                        m.v.extend_from_slice(&[p[0] as f32 * TICK, p[1] as f32 * TICK, p[2] as f32 * TICK, n[0], n[1], n[2], col[0], col[1], col[2]]);
                    }
                    // (u, w, axis) is right-handed, so u x w = +axis: counter-clockwise seen from +axis
                    if side == 1 { m.i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]); }
                    else { m.i.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]); }
                    m.quads += 1;
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn two_touching_bricks_hide_the_shared_face() {
        let mut b = Bricks::new();
        b.add([0, 0, 0], [4, 2, 2], mat::CONCRETE, 0, 0);
        b.add([4, 0, 0], [4, 2, 2], mat::CONCRETE, 0, 0);
        b.build_joints();
        let m = build(&b);
        // 2 bricks x 6 faces - 2 bottoms on the ground - 2 shared faces
        assert_eq!(m.quads, 8);
        // partial cover leaves the uncovered strip
        let mut c = Bricks::new();
        c.add([0, 2, 0], [8, 2, 2], mat::CONCRETE, 0, 0);
        c.add([2, 0, 0], [4, 2, 2], mat::CONCRETE, 0, 0);
        c.build_joints();
        let m = build(&c);
        // top brick: top, 4 sides, bottom minus middle = 2 strips -> 7; lower brick: 4 sides + top covered -> 4
        assert_eq!(m.quads, 11);
    }
    #[test] fn tower_mesh() {
        let t = std::time::Instant::now();
        let b = crate::tower::build(12, 1);
        let m = build(&b);
        std::println!("tower mesh: {} quads, {} vertices, {:.1} ms", m.quads, m.v.len() / 9, t.elapsed().as_secs_f64() * 1e3);
    }
}
