//! The playground: a real-scale tower in the brick store, meshed per storey, a walker, blasts, the load-path
//! check that makes overloaded joints fail, and falling pieces that crush and crack on impact.
//!
//! Bricks of falling pieces stay in the store (flag DYNAMIC, `body` = piece id) with their internal joints,
//! in the structure's own tick frame; the GPU moves them as one rigid body. On a hard impact the piece is
//! crushed around the contact (energy) and its joints are checked against the inertial loads of the impact
//! (load flow); it breaks into the pieces that remain connected.
//!
//! Output logs (`out`), all boxes in ticks of the structure frame:
//!   static changes  [n, (brick, alive, x, y, z, sx, sy, sz, material) ...]
//!   new bodies      [n, (piece id or -1, n_boxes, (x, y, z, sx, sy, sz, material) ...) ...]
use crate::brick::{Bricks, F_ALIVE, F_DYNAMIC, NONE, TICK};
use crate::math::*;
use crate::mesh::{self, Mesh};
use crate::player::Player;
use crate::structure;
use crate::tower;
use alloc::vec;
use alloc::vec::Vec;

pub struct Game {
    pub b: Bricks,
    pub player: Player,
    pub chunks: Vec<Mesh>,
    pub out: Vec<f32>,
    pub dirty: (i32, i32),
    /// bricks of each falling piece (empty = retired)
    pub pieces: Vec<Vec<u32>>,
    /// last structural check: worst joint stress / strength
    pub max_ratio: f32,
    pub failed_total: usize,
}

fn chunk_of(y: i32) -> i32 { y.div_euclid(tower::STOREY) }

impl Game {
    pub fn new(storeys: i32) -> Self {
        let b = tower::build(storeys, 1);
        let n = (storeys + 2) as usize;
        let mut g = Game { b, player: Player::new(v3(0.0, 0.26, -14.0)), chunks: (0..n).map(|_| Mesh::default()).collect(),
                           out: Vec::new(), dirty: (0, n as i32 - 1), pieces: Vec::new(), max_ratio: 0.0, failed_total: 0 };
        for k in 0..n as i32 { g.remesh(k); }
        g
    }

    pub fn remesh(&mut self, k: i32) {
        if k < 0 || k as usize >= self.chunks.len() { return; }
        let (y0, y1) = (k * tower::STOREY, (k + 1) * tower::STOREY);
        self.chunks[k as usize] = mesh::build_filtered(&self.b, |b, i| { let y = b.y[i as usize] as i32; y >= y0 && y < y1 });
    }

    fn push_box(out: &mut Vec<f32>, b: &Bricks, i: u32) {
        let t = b.tbox(i);
        out.extend_from_slice(&[t.min[0] as f32, t.min[1] as f32, t.min[2] as f32, (t.max[0] - t.min[0]) as f32, (t.max[1] - t.min[1]) as f32, (t.max[2] - t.min[2]) as f32, b.mat[i as usize] as f32]);
    }

    /// All static bricks: [n, (index, x, y, z, sx, sy, sz, material) ...] in ticks.
    pub fn dump(&mut self) {
        self.out.clear();
        self.out.push(0.0);
        let mut n = 0;
        for i in 0..self.b.len() as u32 {
            if self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE { self.out.push(i as f32); Self::push_box(&mut self.out, &self.b, i); n += 1; }
        }
        self.out[0] = n as f32;
    }

    fn new_piece(&mut self, bricks: Vec<u32>) -> u32 {
        let id = self.pieces.len() as u32;
        for &i in &bricks { self.b.flags[i as usize] |= F_DYNAMIC; self.b.body[i as usize] = id; }
        self.pieces.push(bricks);
        id
    }

    /// Write the static change log (from the store's log) and the new bodies; remesh what changed.
    fn emit(&mut self, crumbs: &[u32], pieces: &[u32]) {
        let mut out = Vec::new();
        let mut log = core::mem::take(&mut self.b.log);
        log.sort_unstable(); log.dedup();
        // a brick now in a piece is gone from the static world
        let stat: Vec<u32> = log.into_iter().filter(|&i| (i as usize) < self.b.len()).collect();
        out.push(stat.len() as f32);
        let (mut ylo, mut yhi) = (i32::MAX, i32::MIN);
        for &i in &stat {
            let t = self.b.tbox(i);
            ylo = ylo.min(t.min[1]); yhi = yhi.max(t.max[1]);
            let alive = self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE;
            out.extend_from_slice(&[i as f32, if alive { 1.0 } else { 0.0 }]);
            Self::push_box(&mut out, &self.b, i);
        }
        out.push((crumbs.len() + pieces.len()) as f32);
        for &i in crumbs { out.extend_from_slice(&[-1.0, 1.0]); Self::push_box(&mut out, &self.b, i); }
        for &p in pieces {
            out.push(p as f32); out.push(self.pieces[p as usize].len() as f32);
            for k in 0..self.pieces[p as usize].len() { let i = self.pieces[p as usize][k]; Self::push_box(&mut out, &self.b, i); }
        }
        self.out = out;
        if ylo <= yhi {
            let (k0, k1) = (chunk_of(ylo - tower::STOREY), chunk_of(yhi + 1));
            for k in k0..=k1 { self.remesh(k); }
            self.dirty = (k0.max(0), k1.min(self.chunks.len() as i32 - 1));
        } else { self.dirty = (0, -1); }
    }

    /// Detach everything no longer connected to the ground as pieces; returns their ids.
    fn detach(&mut self) -> Vec<u32> {
        let groups = self.b.detached();
        let mut ids = Vec::new();
        for g in groups {
            for &i in &g { self.b.log.push(i); }
            ids.push(self.new_piece(g));
        }
        ids
    }

    /// Blast at c (m) with radius r (m): refine, crumble what is inside, detach what lost the ground.
    pub fn blast(&mut self, c: V3, r: f32) -> usize {
        let ct = [c.x / TICK, c.y / TICK, c.z / TICK];
        let rt = r / TICK;
        self.b.log.clear();
        self.b.refine_sphere(ct, rt + 1.0, 2);
        let mut near = Vec::new();
        let lo = [floor(ct[0] - rt) as i32, floor(ct[1] - rt) as i32, floor(ct[2] - rt) as i32];
        self.b.query(lo, [lo[0] + 2 * rt as i32 + 2, lo[1] + 2 * rt as i32 + 2, lo[2] + 2 * rt as i32 + 2], false, &mut near);
        let mut crumbs = Vec::new();
        for &i in &near {
            let t = self.b.tbox(i);
            let m = [(t.min[0] + t.max[0]) as f32 * 0.5, (t.min[1] + t.max[1]) as f32 * 0.5, (t.min[2] + t.max[2]) as f32 * 0.5];
            let d2 = (m[0] - ct[0]) * (m[0] - ct[0]) + (m[1] - ct[1]) * (m[1] - ct[1]) + (m[2] - ct[2]) * (m[2] - ct[2]);
            if d2 <= rt * rt { crumbs.push(i); }
        }
        for &i in &crumbs { self.b.remove(i); }
        let pieces = self.detach();
        let n = crumbs.len() + pieces.len();
        self.emit(&crumbs, &pieces);
        n
    }

    /// Load-path check of the standing structure under its own weight: joints over strength fail, and what
    /// is no longer held up detaches. Call repeatedly while it reports failures (progressive collapse).
    /// At most `max_fail` joints fail per call, the most overloaded first.
    pub fn structure_step(&mut self, max_fail: usize) -> usize {
        self.b.log.clear();
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE).collect();
        let load: Vec<f32> = set.iter().map(|&i| structure::mass(&self.b, i) * 9.81).collect();
        let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
        let a = structure::analyse(&self.b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
        self.max_ratio = a.max_ratio;
        let work = self.apply_failures(&a, max_fail);
        self.failed_total += work.len();
        let pieces = if work.is_empty() { Vec::new() } else { self.detach() };
        self.emit(&[], &pieces);
        work.len()
    }

    /// Apply the worst `max` failures of an analysis: failed joints are removed, cracked bricks are split at
    /// the failing section and the new joint across the cut is left out. Returns how many were applied.
    fn apply_failures(&mut self, a: &structure::Analysis, max: usize) -> Vec<u32> {
        let mut work: Vec<(f32, u32, Option<usize>)> = a.failed.iter().map(|&(j, r)| (r, j, None)).collect();
        for (k, c) in a.cracks.iter().enumerate() { work.push((c.ratio, NONE, Some(k))); }
        work.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(core::cmp::Ordering::Equal));
        work.truncate(max);
        let mut done = Vec::new();
        let mut cracked: Vec<u32> = Vec::new();
        for &(_, j, c) in &work {
            if let Some(k) = c {
                let c = &a.cracks[k];
                if cracked.contains(&c.brick) { continue; }
                let t = self.b.tbox(c.brick);
                if c.at <= t.min[c.axis] || c.at >= t.max[c.axis] { continue; }
                cracked.push(c.brick);
                let n = self.b.split(c.brick, c.axis, c.at);
                let mut jj = self.b.jhead[n as usize];
                while jj != NONE { if self.b.joint_other(jj, n) == c.brick { self.b.remove_joint(jj); break; } jj = self.b.joint_next(jj, n); }
                done.push(c.brick);
            } else {
                if self.b.ja[j as usize] == NONE { continue; }
                let (x, y) = (self.b.ja[j as usize], self.b.jb[j as usize]);
                self.b.remove_joint(j);
                self.b.log.push(x); self.b.log.push(y);
                done.push(j);
            }
        }
        done
    }

    /// A falling piece hits something. p: contact point (m, structure frame), d: direction the piece was
    /// pushed back by the contact (unit, structure frame), impulse (N s) over `dt`, approach speed (m/s).
    /// Crushes the piece around the contact and cracks it where the impact loads exceed joint strength.
    /// Output: [n_bodies, (piece id or -1, n_boxes, boxes ...) ...]; the hit piece is retired when it changed.
    pub fn impact(&mut self, piece: u32, p: V3, d: V3, impulse: f32, approach: f32, dt: f32) -> usize {
        self.out.clear();
        let pid = piece as usize;
        if pid >= self.pieces.len() || self.pieces[pid].is_empty() { self.out.push(0.0); return 0; }
        let mut set = core::mem::take(&mut self.pieces[pid]);
        let pt = [p.x / TICK, p.y / TICK, p.z / TICK];
        let total_mass: f32 = set.iter().map(|&i| structure::mass(&self.b, i)).sum();
        // 1. crushing: the impact energy 1/2 J v crushes material near the contact
        let nearest = set.iter().copied().min_by(|&x, &y| dist2(&self.b, x, pt).partial_cmp(&dist2(&self.b, y, pt)).unwrap()).unwrap();
        let energy = 0.5 * impulse * approach;
        let vol = energy / structure::crush_energy(self.b.mat[nearest as usize]);
        let r = cbrt(3.0 * vol / (4.0 * core::f32::consts::PI)).min(1.5);
        let mut crumbs = Vec::new();
        if r > 0.12 {
            let rt = r / TICK;
            let near: Vec<u32> = set.iter().copied().filter(|&i| sphere_hits(&self.b, i, pt, rt + 1.0)).collect();
            let mut created = Vec::new();
            self.b.refine_bricks(near, pt, rt + 1.0, 2, &mut created);
            set.extend_from_slice(&created);
            for k in (0..set.len()).rev() {
                let i = set[k];
                if dist2(&self.b, i, pt) <= rt * rt && set.len() > 1 { crumbs.push(i); set.swap_remove(k); }
            }
            for &i in &crumbs { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
        }
        // 2. cracking: every brick decelerates with the piece; its inertia flows to the contact
        let mut cracked = 0;
        if !set.is_empty() {
            let accel = impulse / (total_mass * dt);
            let load: Vec<f32> = set.iter().map(|&i| structure::mass(&self.b, i) * accel).collect();
            // the bricks at the contact take the load out
            let rt = (r / TICK).max(2.0) + 2.0;
            let mut sink: Vec<bool> = set.iter().map(|&i| dist2(&self.b, i, pt) <= rt * rt).collect();
            if !sink.iter().any(|&s| s) { let k = (0..set.len()).min_by(|&x, &y| dist2(&self.b, set[x], pt).partial_cmp(&dist2(&self.b, set[y], pt)).unwrap()).unwrap(); sink[k] = true; }
            let before = self.b.len();
            let a = structure::analyse(&self.b, &set, &load, &sink, -d);
            cracked = self.apply_failures(&a, usize::MAX).len();
            // bricks split off by cracks belong to the piece too
            for i in before..self.b.len() { set.push(i as u32); }
        }
        let changed = !crumbs.is_empty() || cracked > 0;
        if !changed { self.pieces[pid] = set; self.out.push(0.0); return 0; }
        // 3. the remaining bricks split into connected pieces
        let groups = components(&self.b, &set);
        let mut ids = Vec::new();
        for g in groups { ids.push(self.new_piece(g)); }
        self.out.push((crumbs.len() + ids.len()) as f32);
        for &i in &crumbs { self.out.extend_from_slice(&[-1.0, 1.0]); Self::push_box(&mut self.out, &self.b, i); }
        for &id in &ids {
            self.out.push(id as f32); self.out.push(self.pieces[id as usize].len() as f32);
            for k in 0..self.pieces[id as usize].len() { let i = self.pieces[id as usize][k]; Self::push_box(&mut self.out, &self.b, i); }
        }
        crumbs.len() + ids.len()
    }

    /// Ray from o along unit d (m): distance to the first static brick, or a negative value.
    pub fn raycast(&mut self, o: V3, d: V3, max: f32) -> f32 {
        match self.b.raycast([o.x / TICK, o.y / TICK, o.z / TICK], [d.x, d.y, d.z], max / TICK) { Some((_, t, _)) => t * TICK, None => -1.0 }
    }
}

fn centre(b: &Bricks, i: u32) -> [f32; 3] { let t = b.tbox(i); [(t.min[0] + t.max[0]) as f32 * 0.5, (t.min[1] + t.max[1]) as f32 * 0.5, (t.min[2] + t.max[2]) as f32 * 0.5] }
fn dist2(b: &Bricks, i: u32, p: [f32; 3]) -> f32 { let c = centre(b, i); (c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) + (c[2] - p[2]) * (c[2] - p[2]) }
fn sphere_hits(b: &Bricks, i: u32, c: [f32; 3], r: f32) -> bool {
    let t = b.tbox(i); let mut d2 = 0.0;
    for k in 0..3 { let v = c[k].max(t.min[k] as f32).min(t.max[k] as f32) - c[k]; d2 += v * v; }
    d2 < r * r
}
fn cbrt(x: f32) -> f32 { if x <= 0.0 { 0.0 } else { let mut y = x.max(1e-3); for _ in 0..40 { y = y - (y * y * y - x) / (3.0 * y * y); } y } }
/// Connected groups of `set` through joints.
fn components(b: &Bricks, set: &[u32]) -> Vec<Vec<u32>> {
    let mut idx = vec![NONE; b.len()];
    for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
    let mut seen = vec![false; set.len()];
    let mut out = Vec::new();
    for s in 0..set.len() {
        if seen[s] { continue; }
        let mut g = Vec::new(); let mut st = vec![s as u32]; seen[s] = true;
        while let Some(k) = st.pop() {
            let i = set[k as usize]; g.push(i);
            let mut j = b.jhead[i as usize];
            while j != NONE { let o = b.joint_other(j, i); let l = idx[o as usize]; if l != NONE && !seen[l as usize] { seen[l as usize] = true; st.push(l); } j = b.joint_next(j, i); }
        }
        out.push(g);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    fn bodies_in(out: &[f32]) -> std::vec::Vec<(i32, usize)> {
        let nu = out[0] as usize; let mut k = 1 + nu * 9; let nb = out[k] as usize; k += 1;
        let mut v = std::vec::Vec::new();
        for _ in 0..nb { let id = out[k] as i32; let n = out[k + 1] as usize; v.push((id, n)); k += 2 + n * 7; }
        v
    }
    #[test] fn blast_through_a_slab_edge_drops_debris() {
        let mut g = Game::new(3);
        let q0: usize = g.chunks.iter().map(|m| m.quads).sum();
        let nb = g.blast(v3(-2.0, 3.875, -2.0), 0.6);
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
    #[test] fn losing_columns_brings_the_tower_down() {
        // cut the ground storey's columns and walls at 1.6 m, except the core: the core alone cannot carry
        // the floors (shear at the slab-core joints), so the structure check fails joints until the upper
        // tower detaches
        let mut g = Game::new(6);
        assert_eq!(g.structure_step(64), 0, "intact tower stands");
        let mut x = -11.6f32;
        while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
            if !(x.abs() < 4.4 && z.abs() < 4.4) { g.blast(v3(x, 1.6, z), 0.7); }
            z += 0.8; } x += 0.8; }
        let mut rounds = 0; let mut biggest = 0;
        while rounds < 200 {
            let n = g.structure_step(64);
            for (id, sz) in bodies_in(&g.out) { if id >= 0 { biggest = biggest.max(sz); } }
            rounds += 1;
            if n == 0 { break; }
        }
        std::println!("after cutting the perimeter: {} joints failed in {rounds} rounds, biggest falling piece {biggest} bricks, worst ratio now {:.2}", g.failed_total, g.max_ratio);
        assert!(biggest > 300, "the floors above come down");
    }
    #[test] fn a_falling_slab_cracks_on_impact() {
        // a free piece: a 4 m x 4 m slab of 1 m x 0.25 m planks joined edge to edge, hitting the ground on a corner
        let mut g = Game::new(1);
        let base = g.b.len() as u32;
        for i in 0..4 { for k in 0..4 { g.b.add([200 + i * 8, 100, k * 8], [8, 2, 8], crate::brick::mat::CONCRETE, 0, 2); } }
        let mut set = std::vec::Vec::new();
        for i in base..g.b.len() as u32 { set.push(i); }
        for &i in &set { let t = g.b.tbox(i); let mut c = std::vec::Vec::new(); g.b.query(t.min, t.max, true, &mut c); for o in c { if o > i && set.contains(&o) { let (p, q) = (g.b.tbox(i), g.b.tbox(o)); if let Some((ax, area, low)) = crate::brick::face_contact(&p, &q) { if low { g.b.add_joint(i, o, ax, area, 0); } else { g.b.add_joint(o, i, ax, area, 0); } } } } }
        let id = g.new_piece(set);
        // 16 x 0.6 t = 9.6 t landing on a corner at 8 m/s
        let m: f32 = g.pieces[id as usize].iter().map(|&i| structure::mass(&g.b, i)).sum();
        let n = g.impact(id, v3(25.0, 12.5, 0.0), v3(0.0, 1.0, 0.0), m * 8.0, 8.0, 1.0 / 60.0);
        std::println!("slab impact: mass {m:.0} kg -> {n} bodies (crumbs + pieces)");
        assert!(n > 1, "it breaks");
    }
}
