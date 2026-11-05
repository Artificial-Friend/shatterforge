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
    /// weight resting on the structure (rubble lying on floors), for the next structure_step: (x, y, z m, N)
    pub rest: Vec<[f32; 4]>,
    /// last structure_step: the resting weight it found a floor for (N)
    pub rest_applied: f32,
    /// joints and bricks failed by sideways hits (all impacts calls)
    pub side_failed: usize,
    /// worst floor panel under resting loads at the last structure_step (load / capacity)
    pub panel_ratio: f32,
    /// joints cracked (reinforced concrete, masonry) and still holding
    pub cracked_total: usize,
    pub player: Player,
    pub chunks: Vec<Mesh>,
    pub out: Vec<f32>,
    pub dirty: (i32, i32),
    /// bricks of each falling piece (empty = retired)
    pub pieces: Vec<Vec<u32>>,
    /// last structural check: worst joint stress / strength
    pub max_ratio: f32,
    pub failed_total: usize,
    /// input records for `impacts`
    pub inp: Vec<f32>,
    /// the last piece analysed on impact: worst ratio and [failed joints, cracks, contact bricks, bricks]
    pub piece_ratio: f32,
    pub piece_info: [f32; 4],
    pub dbg: Vec<f32>,
    caps: Vec<f32>,
    /// bricks removed as dust (rubble beyond the budget)
    pub dust: usize,
    /// build meshes (off in a worker that only simulates)
    pub mesh: bool,
    /// the last section cut: [contacts at, plane, capacity MN, crush energy per metre MJ, required force MN, squeezed]
    pub cut_dbg: [f32; 6],
    /// settings: crumbs and rubble bricks kept as bodies per impacts call, rubble fragments kept, and how deep a
    /// section that gives way is crushed (m; more = a crushed storey absorbs more of the fall)
    pub crumb_keep: usize,
    pub rubble_keep: usize,
    pub crush_depth: f32,
}

/// fragments of fewer bricks than this count as rubble; at most `rubble_keep` (default RUBBLE_KEEP) of them per
/// impacts call stay as bodies
const RUBBLE_MIN: usize = 48;
const RUBBLE_KEEP: usize = 64;
/// crumbs kept as bodies per impacts call (the rest of the crushed material goes as dust)
const CRUMB_KEEP: usize = 256;
/// how deep a section that gives way is crushed before what is beyond it moves freely (m)
const CRUSH_DEPTH: f32 = 0.5;

fn chunk_of(y: i32) -> i32 { y.div_euclid(tower::STOREY) }

impl Game {
    pub fn new(storeys: i32) -> Self { Self::new_kind(storeys, 0) }
    /// A tower of `storeys` storeys built as `kind` (see `tower::element_in`).
    pub fn new_kind(storeys: i32, kind: u8) -> Self {
        let b = tower::build_kind(storeys, 1, kind);
        let n = (storeys + 2) as usize;
        let mut g = Game { b, rest: Vec::new(), rest_applied: 0.0, side_failed: 0, panel_ratio: 0.0, cracked_total: 0, player: Player::new(v3(0.0, 0.26, -14.0)), chunks: (0..n).map(|_| Mesh::default()).collect(),
                           out: Vec::new(), dirty: (0, n as i32 - 1), pieces: Vec::new(), max_ratio: 0.0, failed_total: 0, inp: Vec::new(), piece_ratio: 0.0, piece_info: [0.0; 4], dbg: Vec::new(), caps: Vec::new(), dust: 0, mesh: true, cut_dbg: [0.0; 6], crumb_keep: CRUMB_KEEP, rubble_keep: RUBBLE_KEEP, crush_depth: CRUSH_DEPTH };
        for k in 0..n as i32 { g.remesh(k); }
        g
    }

    pub fn remesh(&mut self, k: i32) {
        if !self.mesh { return; }
        if k < 0 || k as usize >= self.chunks.len() { return; }
        let (y0, y1) = (k * tower::STOREY, (k + 1) * tower::STOREY);
        self.chunks[k as usize] = mesh::build_filtered(&self.b, |b, i| { let y = b.y[i as usize] as i32; y >= y0 && y < y1 });
    }

    /// A brick as [x, y, z, sx, sy, sz (ticks), material + 16 x paint].
    fn push_box(out: &mut Vec<f32>, b: &Bricks, i: u32) {
        let t = b.tbox(i);
        out.extend_from_slice(&[t.min[0] as f32, t.min[1] as f32, t.min[2] as f32, (t.max[0] - t.min[0]) as f32, (t.max[1] - t.min[1]) as f32, (t.max[2] - t.min[2]) as f32, (b.mat[i as usize] as u32 + 16 * b.paint[i as usize] as u32) as f32]);
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
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE).collect();
        sever_infill(&mut self.b, &set);
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
        // the crumb budget holds for blasts too: an even sample of the crumbs fly, the rest is dust
        let keep = self.crumb_keep;
        if crumbs.len() > keep { let n = crumbs.len(); let kept: Vec<u32> = (0..keep).map(|k| crumbs[k * n / keep]).collect(); self.dust += n - keep; crumbs = kept; }
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
        let mut load: Vec<f32> = set.iter().map(|&i| structure::mass(&self.b, i) * 9.81).collect();
        // rubble lying on the structure weighs on the brick under it: a floor heaped with debris carries it, and
        // fails when it is too much
        let rest = core::mem::take(&mut self.rest);
        self.rest_applied = 0.0;
        let mut point = Vec::new();
        if !rest.is_empty() {
            let mut idx = vec![NONE; self.b.len()];
            for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
            for r in &rest {
                if let Some(i) = self.brick_below(v3(r[0], r[1], r[2]), 4.0) { let k = idx[i as usize]; if k != NONE { point.push((k, r[0] / TICK, r[2] / TICK, r[3])); self.rest_applied += r[3]; } }
            }
        }
        let _ = &mut load;
        let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
        let mut a = structure::analyse_at(&self.b, &set, &load, &point, &sink, v3(0.0, -1.0, 0.0));
        if !point.is_empty() {
            let pts: Vec<(u32, f32, f32, f32)> = point.iter().map(|&(k, x, z, f)| (set[k as usize], x, z, f)).collect();
            let (cracks, worst) = structure::panels(&self.b, &pts);
            self.panel_ratio = worst;
            a.max_ratio = a.max_ratio.max(worst);
            a.cracks.extend(cracks);
        }
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
                // reinforced concrete and masonry crack and hold on (rebar; mortar in compression); others separate
                let reinf = self.b.flags[c.brick as usize] & crate::brick::F_REINF != 0 || self.b.mat[c.brick as usize] == crate::brick::mat::MASONRY;
                let n = self.b.split(c.brick, c.axis, c.at);
                let mut jj = self.b.jhead[n as usize];
                while jj != NONE {
                    if self.b.joint_other(jj, n) == c.brick {
                        // reinforced concrete cracks but the steel holds the two halves: the joint stays, cracked
                        if reinf { self.b.jdmg[jj as usize] = self.b.jdmg[jj as usize].max(1); self.cracked_total += 1; } else { self.b.remove_joint(jj); }
                        break;
                    }
                    jj = self.b.joint_next(jj, n);
                }
                done.push(c.brick);
            } else {
                if self.b.ja[j as usize] == NONE { continue; }
                let (x, y) = (self.b.ja[j as usize], self.b.jb[j as usize]);
                // reinforced on both sides and not yet cracked: it cracks; cracked already: it separates
                let reinf = (self.b.flags[x as usize] & self.b.flags[y as usize]) & crate::brick::F_REINF != 0;
                // brick laid in mortar (to brick, or bedded on a slab or a lintel): the mortar cracks first (the joint
                // still bears, without tension), then parts
                let masonry = self.b.mat[x as usize] == crate::brick::mat::MASONRY || self.b.mat[y as usize] == crate::brick::mat::MASONRY;
                if (reinf || masonry) && self.b.jdmg[j as usize] == 0 { self.b.jdmg[j as usize] = 1; self.b.log.push(x); self.b.log.push(y); self.cracked_total += 1; done.push(j); continue; }
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

    /// A frame's hard contacts, both sides (`inp`, 11 floats each): body (piece id, or -1 for the standing
    /// structure), contact point (m, tower frame), unit direction the contact pushes this body (tower frame),
    /// normal impulse J (N s), approach speed v (m/s), material of the other side (-1 unknown).
    ///
    /// The impact energy 1/2 J v crushes the same volume on both sides, V = E / (c_self + c_other), so concrete
    /// on concrete crushes half each and glass on concrete crushes the glass. The standing structure takes
    /// the contact force J/dt as a load in its structural check; a piece takes its inertia, drawn to all of
    /// its contacts. Output: the static changes and new static pieces (as `emit`), then
    /// `[n_broken, (parent, n_children, crush energy, (id or -1, hit, n, boxes 7 each)...)...]`. The crush energy
    /// (J) is what crushing the section that gave way took: it slows the parts cut free. `hit` is 1 for a part that
    /// holds a contact (it was stopped) and 0 for a part cut free of every contact (it keeps falling).
    pub fn impacts(&mut self, dt: f32) -> usize {
        let inp = core::mem::take(&mut self.inp);
        struct Hit { who: i32, p: [f32; 3], d: V3, j: f32, v: f32, other: i32 }
        let hits: Vec<Hit> = inp.chunks_exact(11).map(|r| Hit {
            who: r[0] as i32, p: [r[1] / TICK, r[2] / TICK, r[3] / TICK], d: v3(r[4], r[5], r[6]), j: r[7], v: r[8], other: r[9] as i32,
        }).collect();
        let ce = |m: i32| if m < 0 { 0.0 } else { structure::crush_energy(m as u8) };
        // the energy divides by the other side's crush energy: the weaker material takes most of it
        // (steel on concrete crushes the concrete, concrete on concrete half each): V = E c_o / (c_s (c_s + c_o))
        let crush_r = |e: f32, cs: f32, co: f32| { let co = if co > 0.0 { co } else { cs }; cbrt(3.0 * e * co / (cs * (cs + co)) / (4.0 * core::f32::consts::PI)).min(1.5) };
        self.b.log.clear();
        // ---- the standing structure
        let mut scrumbs = Vec::new();
        let mut extra: Vec<(u32, f32, [f32; 3])> = Vec::new();
        let mut side: Vec<(u32, V3, [f32; 3])> = Vec::new();
        for h in hits.iter().filter(|h| h.who < 0) {
            let mut near = Vec::new();
            let lo = [floor(h.p[0]) as i32 - 3, floor(h.p[1]) as i32 - 3, floor(h.p[2]) as i32 - 3];
            self.b.query(lo, [lo[0] + 7, lo[1] + 7, lo[2] + 7], true, &mut near);
            let Some(&first) = near.iter().min_by(|&&x, &&y| dist2(&self.b, x, h.p).partial_cmp(&dist2(&self.b, y, h.p)).unwrap()) else { continue };
            let cs = structure::crush_energy(self.b.mat[first as usize]);
            let r = crush_r(0.5 * h.j * h.v, cs, ce(h.other));
            let rt = r / TICK;
            if r > 0.12 {
                self.b.refine_sphere(h.p, rt + 1.0, 2);
                let mut near = Vec::new();
                let lo = [floor(h.p[0] - rt) as i32 - 1, floor(h.p[1] - rt) as i32 - 1, floor(h.p[2] - rt) as i32 - 1];
                let w = 2 * rt as i32 + 4;
                self.b.query(lo, [lo[0] + w, lo[1] + w, lo[2] + w], false, &mut near);
                for &i in &near { if dist2(&self.b, i, h.p) <= rt * rt { scrumbs.push(i); } }
                for &i in &scrumbs { if self.b.flags[i as usize] & F_ALIVE != 0 { self.b.remove(i); } }
            }
            // the contact force loads the bricks around the contact: its downward part with the weight, its sideways
            // part as a load case of its own (a piece swinging into a wall pushes it over or knocks bricks out)
            let f = h.j / dt * (-h.d.y).max(0.0);
            let lat = v3(h.d.x, 0.0, h.d.z) * (h.j / dt);
            if f <= 0.0 && lat.len() <= 0.0 { continue; }
            let rr = rt.max(2.0) + 3.0;
            let mut near = Vec::new();
            let lo = [floor(h.p[0] - rr) as i32, floor(h.p[1] - rr) as i32, floor(h.p[2] - rr) as i32];
            let w = 2 * rr as i32 + 2;
            self.b.query(lo, [lo[0] + w, lo[1] + w, lo[2] + w], true, &mut near);
            near.retain(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && sphere_hits(&self.b, i, h.p, rr));
            if near.is_empty() { continue; }
            if f > 0.0 { let share = f / near.len() as f32; for &i in &near { extra.push((i, share, h.p)); } }
            if lat.len() > 0.0 { let share = 1.0 / near.len() as f32; for &i in &near { side.push((i, lat * share, h.p)); } }
        }
        scrumbs.sort_unstable(); scrumbs.dedup();
        // crumb budget: beyond it, crushed material goes as dust (an even sample of the crumbs stays)
        let thin = |v: &mut Vec<u32>, keep: usize, dust: &mut usize| { if v.len() > keep { let n = v.len(); let kept: Vec<u32> = (0..keep).map(|k| v[k * n / keep]).collect(); *dust += n - keep; *v = kept; } };
        thin(&mut scrumbs, self.crumb_keep / 2, &mut self.dust);
        // what is left of the budgets for the falling pieces (shared by all of this call's pieces)
        let mut crumb_room = self.crumb_keep - scrumbs.len().min(self.crumb_keep);
        let mut rubble_room = self.rubble_keep;
        let mut rubble: Vec<u32> = Vec::new();
        let mut spieces = Vec::new();
        if !extra.is_empty() || !scrumbs.is_empty() || !side.is_empty() {
            let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE).collect();
            let mut idx = vec![NONE; self.b.len()];
            for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
            let load: Vec<f32> = set.iter().map(|&i| structure::mass(&self.b, i) * 9.81).collect();
            // (at the contact: a hit on a 22 m slab loads the cell it lands on)
            let mut point = Vec::new();
            for &(i, f, p) in &extra { let k = idx[i as usize]; if k != NONE { point.push((k, p[0], p[2], f)); } }
            let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
            if !extra.is_empty() {
                let a = structure::analyse_at(&self.b, &set, &load, &point, &sink, v3(0.0, -1.0, 0.0));
                self.max_ratio = self.max_ratio.max(a.max_ratio);
                let work = self.apply_failures(&a, 256);
                self.failed_total += work.len();
            }
            // sideways hits: their loads alone, in their main direction, flow to the ground (shear and tension in
            // the joints on the way). Bricks that failed in the vertical case above are still in the set; joints
            // removed there are gone from the graph already.
            let total = side.iter().fold(v3(0.0, 0.0, 0.0), |a, &(_, l, _)| a + l);
            if total.len() > 0.0 {
                let dirn = total * (1.0 / total.len());
                let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE).collect();
                let mut idx = vec![NONE; self.b.len()];
                for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
                let lload = vec![0.0f32; set.len()];
                let mut point = Vec::new();
                for &(i, l, p) in &side { let k = idx[i as usize]; if k != NONE { point.push((k, p[0], p[2], l.dot(dirn).max(0.0))); } }
                let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
                let a = structure::analyse_at(&self.b, &set, &lload, &point, &sink, dirn);
                self.max_ratio = self.max_ratio.max(a.max_ratio);
                let work = self.apply_failures(&a, 256);
                self.failed_total += work.len();
                self.side_failed += work.len();
            }
            spieces = self.detach();
        }
        self.emit(&scrumbs, &spieces);
        // ---- falling pieces
        let mut ids: Vec<i32> = hits.iter().filter(|h| h.who >= 0).map(|h| h.who).collect();
        ids.sort_unstable(); ids.dedup();
        let mut broken: Vec<(u32, Vec<(i32, bool, Vec<u32>)>)> = Vec::new();
        for pid in ids {
            let pid = pid as usize;
            if pid >= self.pieces.len() || self.pieces[pid].is_empty() { continue; }
            let mine: Vec<&Hit> = hits.iter().filter(|h| h.who == pid as i32).collect();
            let mut set = core::mem::take(&mut self.pieces[pid]);
            let total_mass: f32 = set.iter().map(|&i| structure::mass(&self.b, i)).sum();
            let mut crumbs = Vec::new();
            let mut rs = Vec::new();
            for h in &mine {
                let Some(&nearest) = set.iter().min_by(|&&x, &&y| dist2(&self.b, x, h.p).partial_cmp(&dist2(&self.b, y, h.p)).unwrap()) else { break };
                let r = crush_r(0.5 * h.j * h.v, structure::crush_energy(self.b.mat[nearest as usize]), ce(h.other));
                rs.push(r);
                if r <= 0.12 { continue; }
                let rt = r / TICK;
                let near: Vec<u32> = set.iter().copied().filter(|&i| sphere_hits(&self.b, i, h.p, rt + 1.0)).collect();
                let mut created = Vec::new();
                self.b.refine_bricks(near, h.p, rt + 1.0, 2, &mut created);
                set.extend_from_slice(&created);
                for k in (0..set.len()).rev() {
                    let i = set[k];
                    if dist2(&self.b, i, h.p) <= rt * rt && set.len() > 1 { crumbs.push(i); set.swap_remove(k); }
                }
            }
            for &i in &crumbs { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
            { let keep = crumb_room; if crumbs.len() > keep { let n = crumbs.len(); let kept: Vec<u32> = (0..keep).map(|k| crumbs[k * n / keep.max(1)]).collect(); self.dust += n - keep; crumbs = kept; } crumb_room -= crumbs.len(); }
            // crush-down: the weakest section within a storey of the contacts carries at most its crushing
            // force; when stopping the piece takes more, that section gives way and everything beyond it is cut free
            let mut cracked = 0;
            let mut cap_used = 0.0f32;
            let mut cut_at: Option<(usize, i32, i32)> = None;
            {
                let mut jd = v3(0.0, 0.0, 0.0);
                let mut jabs = v3(0.0, 0.0, 0.0);
                let mut jsum = 0.0;
                for h in &mine { jd = jd + h.d * h.j; jabs = jabs + v3(h.d.x.abs(), h.d.y.abs(), h.d.z.abs()) * h.j; jsum += h.j; }
                // a piece stopped from one side carries the net impulse; one squeezed between two sides (a storey
                // between the floors falling on it and the ground) carries the larger of the two: (sum + net) / 2
                let net = jd.len();
                let f_req = 0.5 * (jsum + net) / dt;
                let squeezed = net < 0.5 * jsum;
                let dn = if squeezed { jabs.norm() } else { jd.norm() };
                let comps = [dn.x.abs(), dn.y.abs(), dn.z.abs()];
                let ax = if comps[0] >= comps[1] && comps[0] >= comps[2] { 0 } else if comps[1] >= comps[2] { 1 } else { 2 };
                let sg = if [dn.x, dn.y, dn.z][ax] >= 0.0 { 1 } else { -1 };
                if f_req > 0.0 {
                    let band = if squeezed { i32::MAX / 4 } else { tower::STOREY + 4 };
                    let from: Vec<f32> = if squeezed { set.iter().map(|&i| { let t = self.b.tbox(i); if sg > 0 { t.min[ax] as f32 } else { t.max[ax] as f32 } }).collect() } else { mine.iter().map(|h| h.p[ax]).collect() };
                    if let Some((at, cap, epm)) = weakest_section(&self.b, &set, ax, sg, &from, band) {
                        if f_req > cap {
                            // a failed section stops resisting once its material is crushed: the energy it takes
                            // is the crush energy of its area over the crushed depth
                            cap_used = epm * self.crush_depth;
                            cut_at = Some((ax, sg, at));
                            let s0 = from.iter().map(|&p| p * sg as f32).fold(f32::MAX, f32::min);
                            self.cut_dbg = [s0, at as f32, cap / 1e6, epm / 1e6, f_req / 1e6, if squeezed { 1.0 } else { 0.0 }];
                            cracked += cut_section(&mut self.b, &mut set, ax, at);
                        }
                    }
                }
            }
            // inertia: every brick decelerates with the piece, its load flows to the contacts
            let near_contact = |b: &Bricks, i: u32| mine.iter().zip(rs.iter()).any(|(h, &r)| { let rt = (r / TICK).max(2.0) + 2.0; dist2(b, i, h.p) <= rt * rt });
            if !set.is_empty() {
                let mut jd = v3(0.0, 0.0, 0.0);
                for h in &mine { jd = jd + h.d * h.j; }
                let jl = jd.len();
                if jl > 0.0 {
                    let accel = jl / (total_mass * dt);
                    let load: Vec<f32> = set.iter().map(|&i| structure::mass(&self.b, i) * accel).collect();
                    let mut sink: Vec<bool> = set.iter().map(|&i| near_contact(&self.b, i)).collect();
                    if !sink.iter().any(|&s| s) {
                        let h = mine[0];
                        let k = (0..set.len()).min_by(|&x, &y| dist2(&self.b, set[x], h.p).partial_cmp(&dist2(&self.b, set[y], h.p)).unwrap()).unwrap();
                        sink[k] = true;
                    }
                    let before = self.b.len();
                    let mut a = structure::analyse(&self.b, &set, &load, &sink, jd * (-1.0 / jl));
                    // a rigid stop in one frame would load every storey with the full deceleration. In reality the
                    // material next to the contacts gives way first, at the force it can carry, and the rest keeps
                    // moving: so only failures within one storey of the contacts count; parts cut free above them
                    // carry on and meet the crushed zone again (a crush-down, storey by storey)
                    let dn = jd * (1.0 / jl);
                    let s0 = mine.iter().map(|h| dn.dot(v3(h.p[0], h.p[1], h.p[2]))).fold(f32::MAX, f32::min);
                    let band = tower::STOREY as f32 + 4.0;
                    let up = |b: &Bricks, i: u32| { let c = centre(b, i); dn.dot(v3(c[0], c[1], c[2])) - s0 };
                    let b = &self.b;
                    a.failed.retain(|&(j, _)| { let (p, q) = (b.ja[j as usize], b.jb[j as usize]); p != NONE && 0.5 * (up(b, p) + up(b, q)) <= band });
                    a.cracks.retain(|c| up(b, c.brick) <= band);
                    if cfg!(test) {
                        self.dbg.clear();
                        for &(j, _) in &a.failed { let (p, q) = (self.b.ja[j as usize], self.b.jb[j as usize]); if p != NONE { self.dbg.push(((self.b.y[p as usize] as i32).max(self.b.y[q as usize] as i32)) as f32); self.dbg.push(self.b.jaxis[j as usize] as f32); } }
                    }
                    cracked = self.apply_failures(&a, usize::MAX).len();
                    self.piece_ratio = a.max_ratio; self.piece_info = [a.failed.len() as f32, a.cracks.len() as f32, sink.iter().filter(|&&s| s).count() as f32, set.len() as f32];
                    for i in before..self.b.len() { set.push(i as u32); }
                }
            }
            if crumbs.is_empty() && cracked == 0 { self.pieces[pid] = set; continue; }
            self.caps.push(cap_used);
            let mut kids: Vec<(i32, bool, Vec<u32>)> = crumbs.iter().map(|&i| (-1, true, vec![i])).collect();
            sever_infill(&mut self.b, &set);
            // rubble budget: a crushed zone falls apart into many fragments; the biggest stay as bodies, the
            // finest go as dust (removed) so a collapse does not run out of bodies
            let mut groups = components(&self.b, &set);
            groups.sort_by_key(|g| core::cmp::Reverse(g.len()));
            groups.retain(|g| {
                if g.len() >= RUBBLE_MIN { return true; }
                if rubble_room > 0 { rubble_room -= 1; return true; }
                for &i in g { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
                self.dust += g.len();
                false
            });
            for g in groups {
                // with a section cut, the parts wholly beyond it were cut free; otherwise the parts at a contact
                let hit = match cut_at {
                    Some((ax, sg, at)) => g.iter().any(|&i| { let t = self.b.tbox(i); (t.min[ax] * sg).min(t.max[ax] * sg) < at }),
                    None => g.iter().any(|&i| near_contact(&self.b, i)),
                };
                // the side of a section that gave way, where the contacts are, is the crushed storey: it does not
                // stay a solid block for the floors above to land on, it falls apart into rubble
                if hit && cut_at.is_some() { rubble.extend_from_slice(&g); continue; }
                let id = self.new_piece(g.clone());
                kids.push((id as i32, hit, g));
            }
            if !rubble.is_empty() {
                // the biggest pieces of rubble stay as bodies (within the crumb budget), the rest goes as dust
                let vol = |b: &Bricks, i: u32| { let t = b.tbox(i); ((t.max[0] - t.min[0]) * (t.max[1] - t.min[1]) * (t.max[2] - t.min[2])) as i64 };
                rubble.sort_by_key(|&i| core::cmp::Reverse(vol(&self.b, i)));
                let keep = rubble.len().min(crumb_room);
                crumb_room -= keep;
                for (k, &i) in rubble.iter().enumerate() {
                    self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC;
                    if k < keep { kids.push((-1, true, vec![i])); } else { self.dust += 1; }
                }
                rubble.clear();
            }
            broken.push((pid as u32, kids));
        }
        self.out.push(broken.len() as f32);
        let mut n = 0;
        let caps = core::mem::take(&mut self.caps);
        for (k, (parent, kids)) in broken.iter().enumerate() {
            self.out.push(*parent as f32); self.out.push(kids.len() as f32); self.out.push(caps[k]);
            for (id, hit, g) in kids {
                self.out.push(*id as f32); self.out.push(if *hit { 1.0 } else { 0.0 }); self.out.push(g.len() as f32);
                for &i in g { Self::push_box(&mut self.out, &self.b, i); }
                n += 1;
            }
        }
        self.out.extend_from_slice(&self.cut_dbg);
        n + scrumbs.len() + spieces.len()
    }

    /// A blast at c (m, tower frame) with radius r (m) on falling piece `piece`: what is inside crumbles, the
    /// rest splits into its connected parts. Output as `impacts` (no static changes, one broken piece).
    pub fn blast_piece(&mut self, piece: u32, c: V3, r: f32) -> usize {
        self.b.log.clear();
        self.emit(&[], &[]);
        let pid = piece as usize;
        if pid >= self.pieces.len() || self.pieces[pid].is_empty() { self.out.push(0.0); return 0; }
        let mut set = core::mem::take(&mut self.pieces[pid]);
        let ct = [c.x / TICK, c.y / TICK, c.z / TICK];
        let rt = r / TICK;
        let near: Vec<u32> = set.iter().copied().filter(|&i| sphere_hits(&self.b, i, ct, rt + 1.0)).collect();
        if near.is_empty() { self.pieces[pid] = set; self.out.push(0.0); return 0; }
        let mut created = Vec::new();
        self.b.refine_bricks(near, ct, rt + 1.0, 2, &mut created);
        set.extend_from_slice(&created);
        let mut crumbs = Vec::new();
        for k in (0..set.len()).rev() { let i = set[k]; if dist2(&self.b, i, ct) <= rt * rt { crumbs.push(i); set.swap_remove(k); } }
        for &i in &crumbs { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
        let keep = self.crumb_keep;
        if crumbs.len() > keep { let n = crumbs.len(); let kept: Vec<u32> = (0..keep).map(|k| crumbs[k * n / keep]).collect(); self.dust += n - keep; crumbs = kept; }
        let mut kids: Vec<(i32, bool, Vec<u32>)> = crumbs.iter().map(|&i| (-1, true, vec![i])).collect();
        if !set.is_empty() {
            sever_infill(&mut self.b, &set);
            let mut groups = components(&self.b, &set);
            groups.sort_by_key(|g| core::cmp::Reverse(g.len()));
            let mut room = self.rubble_keep;
            for g in groups {
                if g.len() < RUBBLE_MIN { if room == 0 { for &i in &g { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; } self.dust += g.len(); continue; } room -= 1; }
                let id = self.new_piece(g.clone());
                kids.push((id as i32, true, g));
            }
        }
        self.out.push(1.0);
        self.out.push(pid as f32); self.out.push(kids.len() as f32); self.out.push(0.0);
        for (id, hit, g) in &kids {
            self.out.push(*id as f32); self.out.push(if *hit { 1.0 } else { 0.0 }); self.out.push(g.len() as f32);
            for &i in g { Self::push_box(&mut self.out, &self.b, i); }
        }
        self.out.extend_from_slice(&[0.0; 6]);
        kids.len()
    }

    /// Ray from o along unit d (m): distance to the first static brick, or a negative value.
    /// The standing brick whose top is the first under the point (x, y, z m), within `max` m below it.
    pub fn brick_below(&self, p: V3, max: f32) -> Option<u32> {
        let (x, y, z) = (p.x / TICK, p.y / TICK, p.z / TICK);
        let (xi, zi) = (floor(x) as i32, floor(z) as i32);
        let mut near = Vec::new();
        self.b.query([xi, floor(y - max / TICK) as i32, zi], [xi + 1, floor(y) as i32 + 1, zi + 1], false, &mut near);
        near.into_iter().filter(|&i| { let t = self.b.tbox(i); (t.max[1] as f32) <= y + 1.0 }).max_by_key(|&i| self.b.tbox(i).max[1])
    }
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
/// The weakest section of `set` across axis `ax` within `band` ticks beyond the contacts (`sg` is the direction
/// the contacts push, the contacts at coordinates `at` along `ax`): (plane, crushing force it can carry in N).
/// A section's capacity is the compressive strength times the area of the structural bricks it cuts, and of the
/// joints lying in it.
fn weakest_section(b: &Bricks, set: &[u32], ax: usize, sg: i32, at: &[f32], band: i32) -> Option<(i32, f32, f32)> {
    use crate::brick::F_INFILL;
    let s0 = at.iter().map(|&p| p * sg as f32).fold(f32::MAX, f32::min);
    // planes right at the contacts would only shave off the crushed layer
    let (lo, hi) = (s0 as i32 + 3, s0 as i32 + band);
    let mut idx = vec![NONE; b.len()];
    for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
    let (a1, a2) = ((ax + 1) % 3, (ax + 2) % 3);
    let orient = |t: &crate::brick::TBox| { let (p, q) = (t.min[ax] * sg, t.max[ax] * sg); (p.min(q), p.max(q)) };
    // candidate planes: brick faces inside the band
    let mut planes: Vec<i32> = Vec::new();
    for &i in set { let (p, q) = orient(&b.tbox(i)); for v in [p, q] { if v > lo && v < hi { planes.push(v); } } }
    planes.sort_unstable(); planes.dedup();
    if planes.len() > 64 { let step = planes.len() / 64 + 1; planes = planes.into_iter().step_by(step).collect(); }
    let mut best: Option<(i32, f32, f32)> = None;
    for &y in &planes {
        let mut cap = 0.0f32;
        let mut epm = 0.0f32;   // crush energy per metre of crushing: area x crush energy density
        for &i in set {
            if b.flags[i as usize] & F_INFILL != 0 { continue; }
            let t = b.tbox(i); let (p, q) = orient(&t);
            let sc = structure::strength(b.mat[i as usize])[0];
            if p < y && y < q {
                let area = ((t.max[a1] - t.min[a1]) * (t.max[a2] - t.min[a2])) as f32 * TICK * TICK;
                cap += area * sc; epm += area * structure::crush_energy(b.mat[i as usize]);
            } else if p == y {
                // joints across the plane: to bricks on the near side
                let mut j = b.jhead[i as usize];
                while j != NONE {
                    let o = b.joint_other(j, i);
                    if b.jaxis[j as usize] as usize == ax && idx[o as usize] != NONE && b.flags[o as usize] & F_INFILL == 0 && orient(&b.tbox(o)).1 == y {
                        let so = structure::strength(b.mat[o as usize])[0];
                        let area = b.jarea[j as usize] as f32 * TICK * TICK;
                        cap += area * sc.min(so); epm += area * structure::crush_energy(b.mat[i as usize]);
                    }
                    j = b.joint_next(j, i);
                }
            }
        }
        if cap > 0.0 && best.map_or(true, |(_, c, _)| cap < c) { best = Some((y, cap, epm)); }
    }
    best
}

/// Cut `set` at oriented plane `at` across `ax`: bricks crossing it split (the pieces join `set`), and every
/// joint lying in the plane goes. Returns the number of cuts.
fn cut_section(b: &mut Bricks, set: &mut Vec<u32>, ax: usize, at_oriented: i32) -> usize {
    let mut n = 0;
    // the plane in real coordinates: oriented = coord * sg, so coord = +-at
    for &plane in &[at_oriented, -at_oriented] {
        let mut hit = false;
        for k in 0..set.len() { let t = b.tbox(set[k]); if t.min[ax] == plane || t.max[ax] == plane || (t.min[ax] < plane && plane < t.max[ax]) { hit = true; break; } }
        if !hit { continue; }
        let mut idx = vec![false; b.len()];
        for &i in set.iter() { idx[i as usize] = true; }
        let len0 = set.len();
        for k in 0..len0 {
            let i = set[k];
            let t = b.tbox(i);
            if t.min[ax] < plane && plane < t.max[ax] {
                let m = b.split(i, ax, plane);
                let mut jj = b.jhead[m as usize];
                while jj != NONE { if b.joint_other(jj, m) == i { b.remove_joint(jj); break; } jj = b.joint_next(jj, m); }
                set.push(m); n += 1;
            }
        }
        let mut idx2 = vec![false; b.len()];
        for &i in set.iter() { idx2[i as usize] = true; }
        let mut cut = Vec::new();
        for &i in set.iter() {
            let t = b.tbox(i);
            if t.max[ax] != plane { continue; }
            let mut j = b.jhead[i as usize];
            while j != NONE { let o = b.joint_other(j, i); if b.jaxis[j as usize] as usize == ax && idx2[o as usize] && b.tbox(o).min[ax] == plane { cut.push(j); } j = b.joint_next(j, i); }
        }
        cut.sort_unstable(); cut.dedup();
        for &j in &cut { b.remove_joint(j); n += 1; }
        let _ = idx;
        break;
    }
    n
}

/// Infill (glass, cladding, partitions) carries no load, so it must not hold structure together either: where
/// the structural bricks of `set` have come apart, the infill joints bridging the parts are cut. Each infill
/// brick stays with the structural part it is joined to lowest down (it hangs on or stands on that).
/// Returns the number of joints cut.
fn sever_infill(b: &mut Bricks, set: &[u32]) -> usize {
    use crate::brick::F_INFILL;
    let mut idx = vec![NONE; b.len()];
    for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
    let infill = |b: &Bricks, i: u32| b.flags[i as usize] & F_INFILL != 0;
    // structural parts
    let mut comp = vec![NONE; set.len()];
    let mut nc = 0u32;
    for s in 0..set.len() {
        if comp[s] != NONE || infill(b, set[s]) { continue; }
        let mut st = vec![s as u32]; comp[s] = nc;
        while let Some(k) = st.pop() {
            let i = set[k as usize];
            let mut j = b.jhead[i as usize];
            while j != NONE { let o = b.joint_other(j, i); let l = idx[o as usize]; if l != NONE && comp[l as usize] == NONE && !infill(b, o) { comp[l as usize] = nc; st.push(l); } j = b.joint_next(j, i); }
        }
        nc += 1;
    }
    if nc <= 1 { return 0; }
    // infill: the lowest structural neighbour, then (for infill joined only to infill) a neighbour's part
    for s in 0..set.len() {
        let i = set[s];
        if !infill(b, i) { continue; }
        let (mut best, mut by) = (NONE, i32::MAX);
        let mut j = b.jhead[i as usize];
        while j != NONE { let o = b.joint_other(j, i); let l = idx[o as usize];
            if l != NONE && !infill(b, o) && (b.y[o as usize] as i32) < by { by = b.y[o as usize] as i32; best = comp[l as usize]; }
            j = b.joint_next(j, i); }
        comp[s] = best;
    }
    for _ in 0..8 {
        let mut changed = false;
        for s in 0..set.len() {
            if comp[s] != NONE { continue; }
            let i = set[s];
            let mut j = b.jhead[i as usize];
            while j != NONE { let o = b.joint_other(j, i); let l = idx[o as usize]; if l != NONE && comp[l as usize] != NONE { comp[s] = comp[l as usize]; changed = true; break; } j = b.joint_next(j, i); }
        }
        if !changed { break; }
    }
    let mut cut = Vec::new();
    for s in 0..set.len() {
        let i = set[s];
        if !infill(b, i) { continue; }
        let mut j = b.jhead[i as usize];
        while j != NONE { let o = b.joint_other(j, i); let l = idx[o as usize];
            if l != NONE && comp[s] != NONE && comp[l as usize] != NONE && comp[s] != comp[l as usize] { cut.push(j); }
            j = b.joint_next(j, i); }
    }
    cut.sort_unstable(); cut.dedup();
    for &j in &cut { b.remove_joint(j); }
    cut.len()
}

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
    #[test] fn a_tower_left_on_one_column_comes_down() {
        // the ground storey blasted away except one column: the upper building must not stand on it.
        for (kind, keep) in [(0u8, (-10.7f32, -10.7f32)), (0, (-3.7, -3.7)), (0, (-3.7, -10.7)), (1, (-3.7, -3.7)), (2, (-3.7, -3.7)), (1, (-10.7, -10.7))] {
            let mut g = Game::new_kind(4, kind);
            assert_eq!(g.structure_step(64), 0, "intact building stands");
            let mut x = -11.6f32;
            while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
                if (x - keep.0).abs() > 1.3 || (z - keep.1).abs() > 1.3 { g.blast(v3(x, 1.6, z), 0.7); }
                z += 0.8; } x += 0.8; }
            let mut rounds = 0; let mut biggest = 0;
            while rounds < 200 {
                let n = g.structure_step(64);
                for (id, sz) in bodies_in(&g.out) { if id >= 0 { biggest = biggest.max(sz); } }
                rounds += 1;
                if n == 0 { break; }
            }
            std::println!("kind {kind}, one column at {keep:?}: {} joints failed in {rounds} rounds, biggest falling piece {biggest} bricks", g.failed_total);
            assert!(biggest > 300, "kind {kind}, column at {keep:?}: the floors above come down");
        }
    }
    #[test] fn rubble_lying_on_a_floor_weighs_on_it() {
        // rubble heaped over one bay of the second floor (6.5 m square between columns). Light rubble (3 kPa, a
        // 0.15 m layer) is carried; 60 kPa (about 3 m of loose concrete rubble) is not: the bay's static moment
        // q L^2 / 8 = 66 kPa x 6.4^2 / 8 = 338 kNm/m against 2 x 125 kNm/m (f_flex 12 MPa, 0.25 m) is 1.35 times
        // what it carries. Before resting loads the floor never felt rubble; with them spread over the whole
        // 22 m slab brick, 0.59 of strength; at their points through the cell graph, 0.70.
        for (kpa, expect) in [(3.0f32, false), (60.0, true)] {
            let mut g = Game::new_kind(4, 0);
            assert_eq!(g.structure_step(64), 0);
            let (mut x, mut pts) = (-10.25f32, 0);
            while x <= -4.25 { let mut z = -10.25f32; while z <= -4.25 { g.rest.push([x, 7.9, z, kpa * 1000.0 * 0.25]); pts += 1; z += 0.5; } x += 0.5; }
            let (mut failed, mut fell, mut applied) = (0, 0, 0.0f32);
            for round in 0..40 {
                if round > 0 && failed > 0 {
                    // the rubble stays on what is left of the floor
                    let (mut x, mut k) = (-10.25f32, 0); while x <= -4.25 { let mut z = -10.25f32; while z <= -4.25 { g.rest.push([x, 7.9, z, kpa * 1000.0 * 0.25]); k += 1; z += 0.5; } x += 0.5; } let _ = k;
                }
                let n = g.structure_step(64);
                if round == 0 { applied = g.rest_applied; std::println!("  round 0: worst {:.2}, panel {:.2}", g.max_ratio, g.panel_ratio); }
                failed += n;
                for (id, sz) in bodies_in(&g.out) { if id >= 0 { fell += sz; } }
                if n == 0 { break; }
            }
            std::println!("{kpa} kPa of rubble on a bay ({pts} points, {:.0} kN found a floor): {failed} failures, {fell} bricks fell", applied / 1000.0);
            assert!(applied > 0.9 * kpa * 1000.0 * 0.25 * pts as f32, "the rubble rests on the slab");
            assert_eq!(fell > 0, expect, "{kpa} kPa");
        }
    }
    #[test] fn the_mixed_buildings_timber_floor_gives_way_under_rubble_and_its_brick_cracks() {
        // a timber floor bay of the third storey (joists, 2.4 MPa over 0.25 m: m_p = 25 kNm/m) between the brick
        // wall and the steel beam at x = -7.5 m (3.25 m clear, beams on both sides): 3 kPa is carried; 40 kPa
        // (q L^2 / 8 = 41 x 3.25^2 / 8 = 54 kNm/m against 2 m_p = 50) is not
        for (kpa, expect) in [(3.0f32, false), (40.0, true)] {
            let mut g = Game::new_kind(5, 3);
            assert_eq!(g.structure_step(64), 0, "the mixed building stands");
            let y = 3.0 * 3.75 + 0.3;
            let push = |g: &mut Game| { let mut x = -10.75f32; while x <= -7.75 { let mut z = -10.75f32; while z <= -7.75 { g.rest.push([x, y, z, kpa * 1000.0 * 0.25]); z += 0.5; } x += 0.5; } };
            let mut fell = 0;
            for _ in 0..40 { push(&mut g); let n = g.structure_step(64); for (id, sz) in bodies_in(&g.out) { if id >= 0 { fell += sz; } } if n == 0 { break; } }
            std::println!("mixed, timber floor bay with {kpa} kPa: panel ratio {:.2}, {fell} bricks fell", g.panel_ratio);
            assert_eq!(fell > 0, expect, "{kpa} kPa on the timber floor");
        }
        // a hit on a brick pier of the second storey: the mortar cracks first
        let mut g = Game::new_kind(5, 3);
        g.inp = vec![-1.0, -7.5, 5.5, -11.2, 0.0, 0.0, 1.0, 6000.0, 6.0, 0.0, 0.0];
        g.impacts(1.0 / 60.0);
        std::println!("mixed, sideways hit on a brick pier: {} cracked, {} failed sideways", g.cracked_total, g.side_failed);
        assert!(g.cracked_total > 0, "the brick cracks");
    }
    #[test] fn a_sideways_hit_knocks_bricks_out_of_a_wall() {
        // a 1 t piece swinging into a masonry pier of the second storey at 6 m/s (J = 6 kN s over a frame):
        // 360 kN sideways on a few bricks of mortar (0.3 MPa in tension, 0.5 in shear). Before, only the
        // downward part of a contact force loaded the structure, so a sideways hit only crushed what it touched.
        let mut g = Game::new_kind(4, 1);
        assert_eq!(g.structure_step(64), 0);
        // contact on the south facade (z = -11.25 m) at 5.5 m up, pushing the wall north (into the building)
        // The mortar cracks first and the pier still stands (it bears in compression); a second hit where it is
        // cracked knocks bricks out.
        let mut fell = [0usize; 2]; let mut cracked = [0usize; 2];
        for k in 0..2 {
            g.inp = vec![-1.0, -6.0, 5.5, -11.2, 0.0, 0.0, 1.0, 6000.0, 6.0, 0.0, 0.0];
            g.impacts(1.0 / 60.0);
            for (id, sz) in bodies_in(&g.out) { if id >= 0 { fell[k] += sz; } }
            cracked[k] = g.cracked_total;
            for _ in 0..10 { if g.structure_step(64) == 0 { break; } for (id, sz) in bodies_in(&g.out) { if id >= 0 { fell[k] += sz; } } }
        }
        std::println!("sideways hits on a masonry pier: {} joints failed by sideways loads, cracked after each hit {cracked:?}, bricks loose after each {fell:?}", g.side_failed);
        assert!(g.side_failed > 0, "the sideways load fails joints");
        assert!(cracked[0] > 0, "the first hit cracks the mortar");
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
    #[test] fn a_dropped_tower_crushes_its_stumps_and_its_lowest_storey() {
        // cut the whole ground storey at 1.6 m: the tower above falls 1.4 m and lands at 5.2 m/s on the stumps
        let mut g = Game::new(8);
        let mut x = -11.6f32;
        let mut piece = -1i32;
        while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
            g.blast(v3(x, 1.6, z), 0.7);
            for (id, sz) in bodies_in(&g.out) { if id >= 0 && sz > 100 { piece = id; } }
            z += 0.8; } x += 0.8; }
        assert!(piece >= 0, "the tower came loose");
        let m: f32 = g.pieces[piece as usize].iter().map(|&i| structure::mass(&g.b, i)).sum();
        let low = g.pieces[piece as usize].iter().map(|&i| g.b.y[i as usize] as i32).min().unwrap();
        // the stumps: static bricks reaching up to the cut
        let tops: std::vec::Vec<u32> = (0..g.b.len() as u32).filter(|&i| g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && {
            let t = g.b.tbox(i); t.max[1] >= 5 && t.max[1] <= low && g.b.flags[i as usize] & crate::brick::F_INFILL == 0 }).collect();
        assert!(!tops.is_empty());
        let v = 5.2; let j = m * v / tops.len() as f32;
        // the stumps fell short of the piece by the crushed gap; they meet at the stump tops after the fall
        for &i in &tops {
            let t = g.b.tbox(i);
            let p = [(t.min[0] + t.max[0]) as f32 * 0.5 * TICK, t.max[1] as f32 * TICK, (t.min[2] + t.max[2]) as f32 * 0.5 * TICK];
            // the piece's contact point is at its underside, shifted down by the fall
            let fall = (low - t.max[1]) as f32 * TICK;
            g.inp.extend_from_slice(&[-1.0, p[0], p[1], p[2], 0.0, -1.0, 0.0, j, v, 0.0, 0.0]);
            g.inp.extend_from_slice(&[piece as f32, p[0], p[1] + fall, p[2], 0.0, 1.0, 0.0, j, v, 0.0, 0.0]);
        }
        let t = std::time::Instant::now();
        let n = g.impacts(1.0 / 60.0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // read the output
        let o = &g.out; let nu = o[0] as usize; let mut k = 1 + nu * 9; let nb = o[k] as usize; k += 1;
        let mut static_crumbs = 0; for _ in 0..nb { let id = o[k] as i32; let c = o[k + 1] as usize; if id < 0 { static_crumbs += 1; } k += 2 + c * 7; }
        let nbr = o[k] as usize; k += 1;
        let (mut free, mut free_bricks, mut held, mut crumbs) = (0, 0, 0, 0);
        let mut cap = 0.0; for _ in 0..nbr { k += 1; let nk = o[k] as usize; cap = o[k + 1]; k += 2; for _ in 0..nk {
            let id = o[k] as i32; let hit = o[k + 1] > 0.5; let c = o[k + 2] as usize; k += 3 + c * 7;
            let (mut y0, mut y1) = (f32::MAX, f32::MIN); for q in 0..c { let bx = &o[k - c * 7 + q * 7..]; y0 = y0.min(bx[1]); y1 = y1.max(bx[1] + bx[4]); }
            if id >= 0 { std::println!("  part {id}: hit {hit}, {c} bricks, y {y0}..{y1} ticks"); }
            if id < 0 { crumbs += 1 } else if hit { held += 1 } else { free += 1; free_bricks += c; } } }
        std::println!("tower of {m:.0} kg lands on {} stumps: {n} new bodies in {ms:.0} ms; static: {} changes, {static_crumbs} crumbs, {} joints failed; piece: {crumbs} crumbs, {held} parts at the contacts, {free} parts cut free ({free_bricks} bricks)", tops.len(), nu, g.failed_total);
        std::println!("crushing the section that gave way took {:.1} MJ (the falling tower carries {:.0} MJ)", cap / 1e6, 0.5 * m * v * v / 1e6);
        std::println!("piece analysis: worst {:.2}, info {:?}, piece starts at y {low}", g.piece_ratio, g.piece_info);
        let mut h = std::collections::BTreeMap::new(); for c in g.dbg.chunks(2) { *h.entry((c[0] as i32 / 10 * 10, c[1] as i32)).or_insert(0) += 1; }
        std::println!("failed joints by (y tick /10, axis): {:?}", h);
        assert!(static_crumbs > 0, "the stumps are crushed");
        assert!(free > 0 && free_bricks > 100, "the floors above are cut free of the crushed storey and keep falling");
        // the next frames: the upper tower meets the storey it left behind, which sits on the crushed stumps
        let mut total_free_cuts = 0;
        for round in 0..4 {
            let live: std::vec::Vec<usize> = (0..g.pieces.len()).filter(|&p| g.pieces[p].len() > 50).collect();
            let top = *live.iter().max_by_key(|&&p| g.pieces[p].iter().map(|&i| g.b.y[i as usize] as i32 + g.b.sy[i as usize] as i32).max().unwrap()).unwrap();
            let tlow = g.pieces[top].iter().map(|&i| g.b.y[i as usize] as i32).min().unwrap();
            let below: std::vec::Vec<usize> = live.iter().copied().filter(|&p| p != top).collect();
            let mt: f32 = g.pieces[top].iter().map(|&i| structure::mass(&g.b, i)).sum();
            let mut recs: std::vec::Vec<(usize, [f32; 3])> = std::vec::Vec::new();
            for &p in &below { for &i in &g.pieces[p] { let t = g.b.tbox(i); if t.max[1] <= tlow && t.max[1] >= tlow - 8 && g.b.flags[i as usize] & crate::brick::F_INFILL == 0 {
                recs.push((p, [(t.min[0] + t.max[0]) as f32 * 0.5 * TICK, t.max[1] as f32 * TICK, (t.min[2] + t.max[2]) as f32 * 0.5 * TICK])); } } }
            // nothing left of the crushed storey under it: it falls onto what stands below (the stumps)
            let mut on_static = false;
            if recs.is_empty() {
                let stat: std::vec::Vec<u32> = (0..g.b.len() as u32).filter(|&i| g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && g.b.flags[i as usize] & crate::brick::F_INFILL == 0 && g.b.tbox(i).max[1] <= tlow).collect();
                let topy = stat.iter().map(|&i| g.b.tbox(i).max[1]).filter(|&y| y > 0).max().unwrap_or(0);
                for &i in &stat { let t = g.b.tbox(i); if t.max[1] >= topy - 2 && topy > 0 { recs.push((usize::MAX, [(t.min[0] + t.max[0]) as f32 * 0.5 * TICK, t.max[1] as f32 * TICK, (t.min[2] + t.max[2]) as f32 * 0.5 * TICK])); } }
                on_static = true;
                std::println!("round {round}: the top piece falls {:.1} m onto {} stumps", (tlow - topy) as f32 * TICK, recs.len());
            }
            if recs.is_empty() { break; }
            let j = mt * 5.0 / recs.len() as f32;
            for (p, c) in &recs {
                // the top piece's own contact point is at its underside
                g.inp.extend_from_slice(&[top as f32, c[0], tlow as f32 * TICK, c[2], 0.0, 1.0, 0.0, j, 5.0, 0.0, 0.0]);
                if on_static { g.inp.extend_from_slice(&[-1.0, c[0], c[1], c[2], 0.0, -1.0, 0.0, j, 5.0, 0.0, 0.0]); continue; }
                g.inp.extend_from_slice(&[*p as f32, c[0], c[1], c[2], 0.0, -1.0, 0.0, j, 5.0, 0.0, 0.0]);
                // the storey below passes the load on to the ground
                g.inp.extend_from_slice(&[*p as f32, c[0], 2.0, c[2], 0.0, 1.0, 0.0, j, 0.5, 0.0, 0.0]);
            }
            g.impacts(1.0 / 60.0);
            let o = &g.out; let nu = o[0] as usize; let mut k = 1 + nu * 9; let nb = o[k] as usize; k += 1;
            for _ in 0..nb { let c = o[k + 1] as usize; k += 2 + c * 7; }
            let nbr = o[k] as usize; k += 1;
            let mut cut_free = std::vec::Vec::new();
            for _ in 0..nbr { let parent = o[k] as usize; let nk = o[k + 1] as usize; if parent == top { std::println!("    top broke: crush force {:.0} MN, {nk} parts", o[k + 2] / 1e6); } k += 3; for _ in 0..nk {
                let id = o[k] as i32; let hit = o[k + 1] > 0.5; let c = o[k + 2] as usize; k += 3 + c * 7;
                if id >= 0 && c > 20 && parent == top { let (mut y0, mut y1) = (f32::MAX, f32::MIN); for q in 0..c { let bx = &o[k - c * 7 + q * 7..]; y0 = y0.min(bx[1]); y1 = y1.max(bx[1] + bx[4]); } std::println!("    part {id} hit {hit} {c} bricks y {y0}..{y1}"); }
                if id >= 0 && !hit && parent == top && c > 20 { cut_free.push(c); } } }
            std::println!("round {round}: top piece ({} kg, bottom at {tlow}) on {} contacts -> parts cut free above: {:?}", mt as i64, recs.len(), cut_free);
            if cut_free.is_empty() { break; }
            total_free_cuts += 1;
        }
        assert!(total_free_cuts >= 2, "the crush-down goes on storey by storey");
    }
    #[test] fn a_blast_on_a_falling_piece_crumbles_it_and_splits_the_rest() {
        let mut g = Game::new(3);
        // cut the ground storey all round: the upper floors come loose as a piece
        let mut x = -11.6f32; let mut piece = -1i32;
        while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 { g.blast(v3(x, 1.6, z), 0.7); for (id, sz) in bodies_in(&g.out) { if id >= 0 && sz > 100 { piece = id; } } z += 0.8; } x += 0.8; }
        assert!(piece >= 0);
        let before = g.pieces[piece as usize].len();
        let n = g.blast_piece(piece as u32, v3(0.0, 4.0, -11.0), 1.2);
        std::println!("blast on a piece of {before} bricks: {n} bodies");
        assert!(n > 5, "crumbs and the rest");
    }
    #[test] fn each_kind_of_building_stands_at_its_height() {
        for (kind, storeys, name) in [(0u8, 12, "concrete frame"), (1, 5, "masonry"), (2, 6, "timber"), (3, 5, "mixed")] {
            let mut g = Game::new_kind(storeys, kind);
            {
                let set: std::vec::Vec<u32> = (0..g.b.len() as u32).filter(|&i| g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE).collect();
                let load: std::vec::Vec<f32> = set.iter().map(|&i| structure::mass(&g.b, i) * 9.81).collect();
                let sink: std::vec::Vec<bool> = set.iter().map(|&i| g.b.y[i as usize] <= 0).collect();
                let a = structure::analyse(&g.b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
                let mut w: std::vec::Vec<_> = a.failed.iter().map(|&(j, r)| { let (x, y) = (g.b.ja[j as usize] as usize, g.b.jb[j as usize] as usize); (r, g.b.paint[x], g.b.mat[x], g.b.paint[y], g.b.mat[y], g.b.jaxis[j as usize], g.b.y[x]) }).collect();
                w.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap());
                for f in w.iter().take(6) { std::println!("  joint {:.2}: paint/mat {}/{} - {}/{}, axis {}, y {}", f.0, f.1, f.2, f.3, f.4, f.5, f.6); }
                for &(j, r) in a.failed.iter().take(12) { let (x, y) = (g.b.ja[j as usize], g.b.jb[j as usize]); std::println!("    {:.1}: {:?}-{:?} / {:?}-{:?}", r, g.b.tbox(x).min, g.b.tbox(x).max, g.b.tbox(y).min, g.b.tbox(y).max); }
                for c in a.cracks.iter().take(6) { std::println!("  crack {:.2}: paint/mat {}/{}, axis {} at {} in {:?}-{:?}", c.ratio, g.b.paint[c.brick as usize], g.b.mat[c.brick as usize], c.axis, c.at, g.b.tbox(c.brick).min, g.b.tbox(c.brick).max); }
            }
            let n = g.structure_step(64);
            std::println!("{name}, {storeys} storeys: worst joint at {:.0}% of strength, {n} failures", g.max_ratio * 100.0);
            assert_eq!(n, 0, "{name} stands");
        }
    }
}
