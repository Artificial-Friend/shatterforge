//! The phase-1 playground: a real-scale tower in the brick store, meshed per storey, a walker, and blasts
//! that refine the hit, crumble what is inside the blast and detach what is no longer connected to the
//! ground. Changes come out as a flat log for the GPU physics (static shapes to update, new debris bodies).
use crate::brick::{Bricks, TICK};
use crate::math::*;
use crate::mesh::{self, Mesh};
use crate::player::Player;
use crate::tower;
use alloc::vec::Vec;

pub struct Game {
    pub b: Bricks,
    pub player: Player,
    pub chunks: Vec<Mesh>,
    pub out: Vec<f32>,
    pub dirty: (i32, i32),
}

/// Bricks belong to the mesh chunk (storey band) holding their min corner.
fn chunk_of(y: i32) -> i32 { y.div_euclid(tower::STOREY) }

impl Game {
    pub fn new(storeys: i32) -> Self {
        let b = tower::build(storeys, 1);
        let n = (storeys + 2) as usize;
        let mut g = Game { b, player: Player::new(v3(0.0, 0.26, -14.0)), chunks: (0..n).map(|_| Mesh::default()).collect(), out: Vec::new(), dirty: (0, n as i32 - 1) };
        for k in 0..n as i32 { g.remesh(k); }
        g
    }

    pub fn remesh(&mut self, k: i32) {
        if k < 0 || k as usize >= self.chunks.len() { return; }
        let (y0, y1) = (k * tower::STOREY, (k + 1) * tower::STOREY);
        self.chunks[k as usize] = mesh::build_filtered(&self.b, |b, i| { let y = b.y[i as usize] as i32; y >= y0 && y < y1 });
    }

    fn push_box(&mut self, i: u32) {
        let t = self.b.tbox(i);
        self.out.extend_from_slice(&[t.min[0] as f32, t.min[1] as f32, t.min[2] as f32, (t.max[0] - t.min[0]) as f32, (t.max[1] - t.min[1]) as f32, (t.max[2] - t.min[2]) as f32, self.b.mat[i as usize] as f32]);
    }

    /// All live bricks: [n, (index, x, y, z, sx, sy, sz, material) ...] in ticks.
    pub fn dump(&mut self) {
        self.out.clear();
        self.out.push(0.0);
        let mut n = 0;
        for i in 0..self.b.len() as u32 { if self.b.alive(i) { self.out.push(i as f32); self.push_box(i); n += 1; } }
        self.out[0] = n as f32;
    }

    /// Blast at c (m) with radius r (m). Output log in `out`:
    /// [n_updates, (index, alive, x, y, z, sx, sy, sz, material) ..., n_bodies, (n_boxes, (x, y, z, sx, sy, sz, material) ...) ...]
    pub fn blast(&mut self, c: V3, r: f32) -> usize {
        let ct = [c.x / TICK, c.y / TICK, c.z / TICK];
        let rt = r / TICK;
        self.b.log.clear();
        self.b.refine_sphere(ct, rt + 1.0, 2);
        // what is inside the blast crumbles into loose pieces (each its own body)
        let mut near = Vec::new();
        let lo = [floor(ct[0] - rt) as i32, floor(ct[1] - rt) as i32, floor(ct[2] - rt) as i32];
        self.b.query(lo, [lo[0] + 2 * rt as i32 + 2, lo[1] + 2 * rt as i32 + 2, lo[2] + 2 * rt as i32 + 2], false, &mut near);
        let mut bodies: Vec<Vec<u32>> = Vec::new();
        for &i in &near {
            let t = self.b.tbox(i);
            let m = [(t.min[0] + t.max[0]) as f32 * 0.5, (t.min[1] + t.max[1]) as f32 * 0.5, (t.min[2] + t.max[2]) as f32 * 0.5];
            let d2 = (m[0] - ct[0]) * (m[0] - ct[0]) + (m[1] - ct[1]) * (m[1] - ct[1]) + (m[2] - ct[2]) * (m[2] - ct[2]);
            if d2 <= rt * rt { bodies.push(alloc::vec![i]); }
        }
        let crumbs: Vec<u32> = bodies.iter().map(|v| v[0]).collect();
        for &i in &crumbs { self.b.remove(i); }
        // whatever lost its connection to the ground falls as rigid pieces
        let pieces = self.b.detached();
        // write the log: body boxes first (bricks still readable), then removals
        let mut body_out = Vec::new();
        core::mem::swap(&mut self.out, &mut body_out);
        self.out.clear();
        let mut nb = 0;
        for p in bodies.iter().chain(pieces.iter()) {
            self.out.push(p.len() as f32);
            for &i in p { self.push_box(i); }
            nb += 1;
        }
        for p in &pieces { for &i in p { self.b.remove(i); } }
        let body_part = core::mem::take(&mut self.out);
        self.out = body_out; self.out.clear();
        let mut log = core::mem::take(&mut self.b.log);
        log.sort_unstable(); log.dedup();
        self.out.push(log.len() as f32);
        let (mut ylo, mut yhi) = (i32::MAX, i32::MIN);
        for &i in &log {
            let t = self.b.tbox(i);
            ylo = ylo.min(t.min[1]); yhi = yhi.max(t.max[1]);
            self.out.push(i as f32);
            self.out.push(if self.b.alive(i) { 1.0 } else { 0.0 });
            self.push_box(i);
        }
        self.out.push(nb as f32);
        self.out.extend_from_slice(&body_part);
        // remesh the storeys whose bricks changed or touch a changed brick
        if ylo <= yhi {
            let (k0, k1) = (chunk_of(ylo - tower::STOREY), chunk_of(yhi + 1));
            for k in k0..=k1 { self.remesh(k); }
            self.dirty = (k0.max(0), k1.min(self.chunks.len() as i32 - 1));
        } else { self.dirty = (0, -1); }
        nb
    }

    /// Ray from o along unit d (m): distance to the first brick, or a negative value.
    pub fn raycast(&mut self, o: V3, d: V3, max: f32) -> f32 {
        match self.b.raycast([o.x / TICK, o.y / TICK, o.z / TICK], [d.x, d.y, d.z], max / TICK) { Some((_, t, _)) => t * TICK, None => -1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn blast_through_a_slab_edge_drops_debris() {
        let mut g = Game::new(3);
        let q0: usize = g.chunks.iter().map(|m| m.quads).sum();
        // a blast in the first-floor slab, next to the stair opening
        let t = std::time::Instant::now();
        let nb = g.blast(v3(-2.0, 3.875, -2.0), 0.6);
        std::println!("blast: {nb} bodies, log {} floats, {:.2} ms, chunks remeshed {:?}", g.out.len(), t.elapsed().as_secs_f64() * 1e3, g.dirty);
        assert!(nb > 10, "crumbs");
        let q1: usize = g.chunks.iter().map(|m| m.quads).sum();
        assert!(q1 > q0, "the crater adds faces");
        let mut r = g.b.clone(); r.build_joints(); assert_eq!(g.b.joint_set(), r.joint_set());
        assert!(g.b.detached().is_empty());
    }
    #[test] fn raycast_hits_the_facade() {
        let mut g = Game::new(2);
        let t = g.raycast(v3(5.0, 1.0, -30.0), v3(0.0, 0.0, 1.0), 100.0);
        assert!((t - (30.0 - 11.25)).abs() < 0.01, "{t}");
    }
}
