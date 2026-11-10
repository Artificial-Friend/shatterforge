//! Render mesh of a brick store: every brick face minus the parts covered by touching neighbours (exact, from
//! the joint graph), as quads. Faces resting on the ground are dropped.
use crate::brick::{mat, Bricks, TBox, NONE, TICK};
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
        (mat::MASONRY, _) => [0.60, 0.36, 0.28],   // brick
        (mat::STEEL, _) => [0.36, 0.40, 0.44],
        (mat::TIMBER, 8) => [0.62, 0.46, 0.30],   // timber floor
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

/// The structure as the engineer sees it (docs/DYNAMIC.md): each brick coloured by how near its worst joint or
/// section is to its strength, and the building drawn where the structure's solution puts it, its displacements
/// `scale` times as large (a building sags millimetres). The cells are those of the structure's last step.
#[derive(Default)]
pub struct View {
    pub on: bool,
    /// displacements drawn this many times as large (0: not drawn)
    pub scale: f32,
    /// colour by load / strength
    pub stress: bool,
    pub ratio: crate::Map<u32, f32>,
    pub cells: crate::Map<u32, Vec<ViewCell>>,
}
/// A cell of a brick: its corners in x and z (ticks), its centre (m), displacement (m) and rotation (rad).
#[derive(Clone, Copy)]
pub struct ViewCell { pub lo: [i32; 2], pub hi: [i32; 2], pub c: [f32; 3], pub u: [f32; 3], pub r: [f32; 3] }
impl View {
    /// load / strength as a colour: blue (unloaded) to green (a third) to yellow (0.7, where joints begin to
    /// wear) to red (at strength)
    pub fn tint(r: f32) -> [f32; 3] {
        let l = |a: [f32; 3], b: [f32; 3], t: f32| { let t = t.clamp(0.0, 1.0); [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t] };
        let (c0, c1, c2, c3) = ([0.27, 0.42, 0.70], [0.30, 0.68, 0.36], [0.93, 0.80, 0.22], [0.86, 0.16, 0.13]);
        if r < 0.35 { l(c0, c1, r / 0.35) } else if r < 0.7 { l(c1, c2, (r - 0.35) / 0.35) } else { l(c2, c3, (r - 0.7) / 0.3) }
    }
    /// where the structure's solution puts point p (m) of brick i, `scale` times its displacement: the mean over
    /// the brick's cells that hold p (a point on the face between two cells belongs to both, so the brick stays
    /// in one piece)
    pub fn moved(&self, i: u32, p: [f32; 3]) -> [f32; 3] {
        if self.scale == 0.0 { return p; }
        let Some(cs) = self.cells.get(&i) else { return p };
        let (tx, tz) = (p[0] / TICK, p[2] / TICK);
        let (mut d, mut n) = ([0.0f32; 3], 0.0f32);
        for pass in 0..2 {
            for c in cs {
                // (second pass, a point outside every cell: a convex cell's or a free piece's own corner, the one cell)
                if pass == 0 && (tx < c.lo[0] as f32 - 0.01 || tx > c.hi[0] as f32 + 0.01 || tz < c.lo[1] as f32 - 0.01 || tz > c.hi[1] as f32 + 0.01) { continue; }
                let q = [p[0] - c.c[0], p[1] - c.c[1], p[2] - c.c[2]];
                d[0] += c.u[0] + c.r[1] * q[2] - c.r[2] * q[1];
                d[1] += c.u[1] + c.r[2] * q[0] - c.r[0] * q[2];
                d[2] += c.u[2] + c.r[0] * q[1] - c.r[1] * q[0];
                n += 1.0;
            }
            if n > 0.0 { break; }
        }
        if n == 0.0 { return p; }
        [p[0] + self.scale * d[0] / n, p[1] + self.scale * d[1] / n, p[2] + self.scale * d[2] / n]
    }
    /// brick i's cell boundaries along x and z (ticks), for cutting its faces where it bends
    fn cuts(&self, i: u32) -> (Vec<i32>, Vec<i32>) {
        let (mut xs, mut zs) = (Vec::new(), Vec::new());
        if self.scale != 0.0 { if let Some(cs) = self.cells.get(&i) { for c in cs { xs.push(c.lo[0]); xs.push(c.hi[0]); zs.push(c.lo[1]); zs.push(c.hi[1]); } } }
        xs.sort_unstable(); xs.dedup(); zs.sort_unstable(); zs.dedup();
        (xs, zs)
    }
}

/// Mesh of the live bricks that pass `keep` (a chunk of the store).
pub fn build_filtered(b: &Bricks, keep: impl Fn(&Bricks, u32) -> bool) -> Mesh { build_view(b, keep, None) }
/// As `build_filtered`, drawn as `view` shows the structure.
pub fn build_view(b: &Bricks, keep: impl Fn(&Bricks, u32) -> bool, view: Option<&View>) -> Mesh {
    let view = view.filter(|v| v.on);
    let mut m = Mesh::default();
    let mut rects: Vec<Rect> = Vec::new();
    let mut cut: Vec<Rect> = Vec::new();
    for i in 0..b.len() as u32 {
        if !b.alive(i) || b.flags[i as usize] & crate::brick::F_DYNAMIC != 0 || !keep(b, i) { continue; }
        let tb = b.tbox(i);
        let v0 = m.v.len();
        let col = match view { Some(v) if v.stress => View::tint(v.ratio.get(&i).copied().unwrap_or(0.0)), _ => colour(b, i) };
        let cuts = view.map(|v| v.cuts(i));
        // (a brick's vertices, once it is meshed: moved to where the structure puts them)
        let finish = |m: &mut Mesh| { if let Some(v) = view { let mut o = v0; while o < m.v.len() { let q = v.moved(i, [m.v[o], m.v[o + 1], m.v[o + 2]]); m.v[o] = q[0]; m.v[o + 1] = q[1]; m.v[o + 2] = q[2]; o += 9; } } };
        if let Some(cl) = b.cell(i) { cell_faces(b, i, cl, col, &mut m); finish(&mut m); continue; }
        if let Some(p) = b.piece(i) { piece_faces(p, col, &mut m); finish(&mut m); continue; }
        // cracked joints of this brick (axis, plane, overlap box): drawn as dark seams where they meet its faces
        let mut cracks: Vec<(usize, i32, TBox)> = Vec::new();
        let mut j = b.jhead[i as usize];
        while j != NONE {
            let ju = j as usize;
            // (seams on the grid's joints: a cracked joint to a free piece or between cells, axis 4 or 3, indexed past
            // the box's three axes and panicked when a box's joint to an arch stone cracked; complex test
            // an_arch_without_its_keystone_falls)
            if b.jdmg[ju] & 1 != 0 && b.jaxis[ju] < 3 {
                let a = b.jaxis[ju] as usize;
                let o = b.tbox(b.joint_other(j, i));
                let mut ov = TBox { min: [0; 3], max: [0; 3] };
                for d in 0..3 { ov.min[d] = tb.min[d].max(o.min[d]); ov.max[d] = tb.max[d].min(o.max[d]); }
                let plane = if b.ja[ju] == i { tb.max[a] } else { tb.min[a] };
                cracks.push((a, plane, ov));
            }
            j = b.joint_next(j, i);
        }
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
                    // (a convex cell covers only part of the face: the face stays, and the cell's own face, coplanar and
                    // facing the other way, is hidden by it)
                    if b.jaxis[ju] as usize == ax && ((side == 1 && b.ja[ju] == i) || (side == 0 && b.jb[ju] == i)) && b.hull[b.joint_other(j, i) as usize] == 0 {
                        let o = b.tbox(b.joint_other(j, i));
                        subtract(&mut rects, [o.min[u], o.min[w], o.max[u], o.max[w]]);
                    }
                    j = b.joint_next(j, i);
                }
                let plane = if side == 1 { tb.max[ax] } else { tb.min[ax] };
                let mut n = [0.0f32; 3]; n[ax] = if side == 1 { 1.0 } else { -1.0 };
                // (the view: a face is cut along the brick's cell boundaries, so that a slab drawn sagging bends)
                if let Some((xs, zs)) = &cuts {
                    for (d, lines) in [(0usize, xs), (2usize, zs)] {
                        if d == ax { continue; }
                        let along_u = d == u;
                        for &c in lines.iter() {
                            cut.clear();
                            for &r in rects.iter() {
                                let (lo, hi) = if along_u { (r[0], r[2]) } else { (r[1], r[3]) };
                                if c > lo && c < hi { if along_u { cut.push([r[0], r[1], c, r[3]]); cut.push([c, r[1], r[2], r[3]]); } else { cut.push([r[0], r[1], r[2], c]); cut.push([r[0], c, r[2], r[3]]); } } else { cut.push(r); }
                            }
                            core::mem::swap(&mut rects, &mut cut);
                        }
                    }
                }
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
                    // seams of cracked joints across this face rect: a dark strip 2 cm wide inside this brick along the
                    // joint's edge, a few mm proud of the face
                    for &(a, cp, ov) in &cracks {
                        if a == ax || ov.min[ax] > plane || ov.max[ax] < plane { continue; }
                        let (lo_w, hi_w, along) = if a == u { (ov.min[w].max(r[1]), ov.max[w].min(r[3]), w) } else { (ov.min[u].max(r[0]), ov.max[u].min(r[2]), u) };
                        if hi_w <= lo_w { continue; }
                        let inward = if cp == tb.max[a] { -CRACK_W } else { CRACK_W };
                        let base = (m.v.len() / 9) as u32;
                        let off = n[ax] * 0.004;
                        for (ca, cl) in [(0.0f32, lo_w), (inward, lo_w), (inward, hi_w), (0.0, hi_w)] {
                            let mut p = [0.0f32; 3];
                            p[ax] = plane as f32 * TICK + off; p[a] = cp as f32 * TICK + ca; p[along] = cl as f32 * TICK;
                            m.v.extend_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2], 0.13, 0.11, 0.10]);
                        }
                        // (two-sided: the strip's winding depends on its axes)
                        m.i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3, base, base + 2, base + 1, base, base + 3, base + 2]);
                        m.quads += 1;
                    }
                }
            }
        }
        finish(&mut m);
    }
    m
}
/// The faces of a convex cell: all of them but those it shares with a cell still joined to it, and its bottom on
/// the ground. (Its faces on its box's sides are drawn even where a neighbour covers them: inside the material,
/// facing away from the viewer.)
fn cell_faces(b: &Bricks, i: u32, cl: &crate::hull::Cell, col: [f32; 3], m: &mut Mesh) {
    let (c, h) = b.box_m(i);
    let ground = b.tbox(i).min[1] <= 0;
    for f in &cl.faces {
        if f.other == 3 && ground { continue; }
        if f.other >= 6 {
            let mut j = b.jhead[i as usize]; let mut hidden = false;
            while j != NONE { if b.jaxis[j as usize] == 3 { if let Some(o) = b.cell(b.joint_other(j, i)) { if o.slot == f.other - 6 { hidden = true; break; } } } j = b.joint_next(j, i); }
            if hidden { continue; }
        }
        let poly = cl.face_in(f, c, h);
        let (_, _, n) = crate::hull::polygon(&poly);
        let base = (m.v.len() / 9) as u32;
        for p in &poly { m.v.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, n[0] as f32, n[1] as f32, n[2] as f32, col[0], col[1], col[2]]); }
        for t in 1..poly.len() as u32 - 1 { m.i.extend_from_slice(&[base, base + t, base + t + 1]); }
        m.quads += 1;
    }
}
/// The faces of a free piece, all of them but one lying on the ground (facing down at y = 0). (Faces against its
/// neighbours: with its joints, docs/PHASE2.md step 5.)
fn piece_faces(p: crate::brick::PieceRef, col: [f32; 3], m: &mut Mesh) {
    for (n, poly) in p.world_faces() {
        if n[1] < -0.999 && poly.iter().all(|q| q[1].abs() < 1e-4) { continue; }
        let base = (m.v.len() / 9) as u32;
        for q in &poly { m.v.extend_from_slice(&[q[0] as f32, q[1] as f32, q[2] as f32, n[0] as f32, n[1] as f32, n[2] as f32, col[0], col[1], col[2]]); }
        for t in 1..poly.len() as u32 - 1 { m.i.extend_from_slice(&[base, base + t, base + t + 1]); }
        m.quads += 1;
    }
}
/// width of a drawn crack seam (m)
const CRACK_W: f32 = 0.02;

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
    #[test] fn a_cracked_joint_shows_as_seams() {
        // two bricks side by side on the ground, their joint cracked: 8 faces as before, plus a seam on each of
        // the three visible faces of each brick that meet the joint (top, front, back)
        let mut b = Bricks::new();
        b.add([0, 0, 0], [4, 2, 2], mat::MASONRY, 0, 0);
        b.add([4, 0, 0], [4, 2, 2], mat::MASONRY, 0, 0);
        b.build_joints();
        b.jdmg[b.jhead[0] as usize] = 1;
        assert_eq!(build(&b).quads, 8 + 6);
    }
    #[test] fn a_cell_draws_its_faces_but_those_it_shares() {
        // a 1 m brick on the ground split into its cells: each cell draws its faces except the ones shared with
        // a cell still there and its bottom on the ground; the faces drawn enclose the cells' volume (divergence
        // theorem: the sum over faces of area x centroid . normal / 3 is the volume) once the shared faces of a
        // removed cell show
        let mut b = Bricks::new();
        let i = b.add([0, 0, 0], [8, 8, 8], mat::MASONRY, 0, 0);
        b.build_joints();
        let (cells, _) = b.shatter(i);
        let m = build(&b);
        let faces: usize = cells.iter().map(|&c| b.cell(c).unwrap().faces.len()).sum();
        let shared: usize = (0..b.ja.len()).filter(|&j| b.ja[j] != NONE && b.jaxis[j] == 3).count();
        let bottoms = cells.iter().filter(|&&c| b.cell(c).unwrap().side(3).is_some()).count();
        assert_eq!(m.quads, faces - 2 * shared - bottoms);
        // every triangle faces outward: the mesh's signed volume (with the ground faces left out, measured from y = 0)
        // is the brick's volume
        let mut vol = 0.0f64;
        for t in m.i.chunks(3) {
            let p = |k: u32| { let o = k as usize * 9; [m.v[o] as f64, m.v[o + 1] as f64, m.v[o + 2] as f64] };
            let (a, c, d) = (p(t[0]), p(t[1]), p(t[2]));
            vol += (a[0] * (c[1] * d[2] - c[2] * d[1]) - a[1] * (c[0] * d[2] - c[2] * d[0]) + a[2] * (c[0] * d[1] - c[1] * d[0])) / 6.0;
        }
        assert!((vol - 1.0).abs() < 1e-4, "outward faces enclose 1 m^3: {vol}");
    }
    #[test] fn tower_mesh() {
        let t = std::time::Instant::now();
        let b = crate::tower::build(12, 1);
        let m = build(&b);
        std::println!("tower mesh: {} quads, {} vertices, {:.1} ms", m.quads, m.v.len() / 9, t.elapsed().as_secs_f64() * 1e3);
    }
}
