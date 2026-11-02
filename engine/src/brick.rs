//! Adaptive brick store: real-size axis-aligned boxes on the 0.125 m tick grid, joints as shared faces with
//! exact areas, and on-demand splitting that keeps the joint graph exact. Layout follows docs/SPEC.md.
use alloc::vec::Vec;
use hashbrown::HashMap;

pub const NONE: u32 = u32::MAX;
pub const TICK: f32 = 0.125;

pub mod mat {
    pub const CONCRETE: u8 = 0;
    pub const MASONRY: u8 = 1;
    pub const STEEL: u8 = 2;
    pub const GLASS: u8 = 3;
    pub const TIMBER: u8 = 4;
    /// course grid in ticks that splits snap to before going finer (0 = none)
    pub fn course(m: u8) -> i32 { match m { CONCRETE | MASONRY => 2, _ => 0 } }
    /// density kg/m^3
    pub fn density(m: u8) -> f32 { match m { CONCRETE => 2400.0, MASONRY => 1900.0, STEEL => 7850.0, GLASS => 2500.0, _ => 500.0 } }
}
pub const F_ALIVE: u8 = 1;
pub const F_INFILL: u8 = 2;
pub const F_DYNAMIC: u8 = 4;

/// Grid cell for the spatial index, in ticks (8 ticks = 1 m).
const CELL: i32 = 8;

/// A box in ticks, [min, max).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TBox { pub min: [i32; 3], pub max: [i32; 3] }

/// Face contact between two boxes: (normal axis, area in tick^2, true if the first box is on the min side).
pub fn face_contact(p: &TBox, q: &TBox) -> Option<(u8, u32, bool)> {
    for ax in 0..3 {
        let p_low = p.max[ax] == q.min[ax];
        let q_low = q.max[ax] == p.min[ax];
        if !p_low && !q_low { continue; }
        let (o1, o2) = ((ax + 1) % 3, (ax + 2) % 3);
        let a = p.max[o1].min(q.max[o1]) - p.min[o1].max(q.min[o1]);
        let b = p.max[o2].min(q.max[o2]) - p.min[o2].max(q.min[o2]);
        if a > 0 && b > 0 { return Some((ax as u8, (a * b) as u32, p_low)); }
        return None;
    }
    None
}

#[derive(Clone, Default)]
pub struct Bricks {
    // bricks, SoA (24 B each)
    pub x: Vec<i16>, pub y: Vec<i16>, pub z: Vec<i16>,
    pub sx: Vec<u8>, pub sy: Vec<u8>, pub sz: Vec<u8>,
    pub mat: Vec<u8>, pub paint: Vec<u8>, pub flags: Vec<u8>,
    pub structure: Vec<u16>, pub body: Vec<u32>, pub jhead: Vec<u32>, pub damage: Vec<u8>,
    // joints, SoA (20 B each); `ja == NONE` marks a free slot
    pub ja: Vec<u32>, pub jb: Vec<u32>, pub jna: Vec<u32>, pub jnb: Vec<u32>,
    pub jarea: Vec<u16>, pub jaxis: Vec<u8>, pub jdmg: Vec<u8>,
    jfree: Vec<u32>,
    pub live_joints: usize,
    // spatial index (an accelerator, not part of the brick format): 1 m cells -> bricks touching them.
    // Entries can go stale after splits; queries test real overlap.
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    stamp: Vec<u32>, stamp_id: u32,
}

impl Bricks {
    pub fn new() -> Self { Self::default() }
    #[inline] pub fn len(&self) -> usize { self.x.len() }
    #[inline] pub fn tbox(&self, i: u32) -> TBox {
        let i = i as usize;
        let min = [self.x[i] as i32, self.y[i] as i32, self.z[i] as i32];
        TBox { min, max: [min[0] + self.sx[i] as i32, min[1] + self.sy[i] as i32, min[2] + self.sz[i] as i32] }
    }
    #[inline] pub fn size(&self, i: u32) -> [i32; 3] { let i = i as usize; [self.sx[i] as i32, self.sy[i] as i32, self.sz[i] as i32] }

    /// Add a brick: min corner and size in ticks. Sizes 1..=255.
    pub fn add(&mut self, min: [i32; 3], size: [i32; 3], material: u8, paint: u8, structure: u16) -> u32 {
        debug_assert!(size.iter().all(|&s| (1..=255).contains(&s)));
        let i = self.x.len() as u32;
        self.x.push(min[0] as i16); self.y.push(min[1] as i16); self.z.push(min[2] as i16);
        self.sx.push(size[0] as u8); self.sy.push(size[1] as u8); self.sz.push(size[2] as u8);
        self.mat.push(material); self.paint.push(paint); self.flags.push(F_ALIVE);
        self.structure.push(structure); self.body.push(NONE); self.jhead.push(NONE); self.damage.push(0);
        self.stamp.push(0);
        self.index(i);
        i
    }

    fn index(&mut self, i: u32) {
        let b = self.tbox(i);
        let c0 = [b.min[0].div_euclid(CELL), b.min[1].div_euclid(CELL), b.min[2].div_euclid(CELL)];
        let c1 = [(b.max[0] - 1).div_euclid(CELL), (b.max[1] - 1).div_euclid(CELL), (b.max[2] - 1).div_euclid(CELL)];
        for cx in c0[0]..=c1[0] { for cy in c0[1]..=c1[1] { for cz in c0[2]..=c1[2] {
            self.grid.entry((cx, cy, cz)).or_default().push(i);
        } } }
    }

    /// Bricks whose box overlaps [min, max) in ticks (inclusive of touching when `touch`), each once.
    pub fn query(&mut self, min: [i32; 3], max: [i32; 3], touch: bool, out: &mut Vec<u32>) {
        self.stamp_id = self.stamp_id.wrapping_add(1);
        if self.stamp_id == 0 { for s in self.stamp.iter_mut() { *s = 0; } self.stamp_id = 1; }
        let t = if touch { 1 } else { 0 };
        let c0 = [(min[0] - t).div_euclid(CELL), (min[1] - t).div_euclid(CELL), (min[2] - t).div_euclid(CELL)];
        let c1 = [(max[0] - 1 + t).div_euclid(CELL), (max[1] - 1 + t).div_euclid(CELL), (max[2] - 1 + t).div_euclid(CELL)];
        for cx in c0[0]..=c1[0] { for cy in c0[1]..=c1[1] { for cz in c0[2]..=c1[2] {
            if let Some(v) = self.grid.get(&(cx, cy, cz)) {
                for &i in v {
                    if self.stamp[i as usize] == self.stamp_id || self.flags[i as usize] & F_ALIVE == 0 { continue; }
                    self.stamp[i as usize] = self.stamp_id;
                    let b = self.tbox(i);
                    let ok = (0..3).all(|k| if touch { b.min[k] <= max[k] && b.max[k] >= min[k] } else { b.min[k] < max[k] && b.max[k] > min[k] });
                    if ok { out.push(i); }
                }
            }
        } } }
    }

    // ---- joints ----
    #[inline] pub fn joint_next(&self, j: u32, k: u32) -> u32 { if self.ja[j as usize] == k { self.jna[j as usize] } else { self.jnb[j as usize] } }
    #[inline] fn set_next(&mut self, j: u32, k: u32, n: u32) { if self.ja[j as usize] == k { self.jna[j as usize] = n } else { self.jnb[j as usize] = n } }
    #[inline] pub fn joint_other(&self, j: u32, k: u32) -> u32 { if self.ja[j as usize] == k { self.jb[j as usize] } else { self.ja[j as usize] } }

    /// Joint between a (min side along `axis`) and b.
    pub fn add_joint(&mut self, a: u32, b: u32, axis: u8, area: u32, dmg: u8) -> u32 {
        let j = match self.jfree.pop() {
            Some(j) => j,
            None => {
                self.ja.push(0); self.jb.push(0); self.jna.push(0); self.jnb.push(0);
                self.jarea.push(0); self.jaxis.push(0); self.jdmg.push(0);
                (self.ja.len() - 1) as u32
            }
        };
        let ju = j as usize;
        self.ja[ju] = a; self.jb[ju] = b; self.jarea[ju] = area as u16; self.jaxis[ju] = axis; self.jdmg[ju] = dmg;
        self.jna[ju] = self.jhead[a as usize]; self.jhead[a as usize] = j;
        self.jnb[ju] = self.jhead[b as usize]; self.jhead[b as usize] = j;
        self.live_joints += 1;
        j
    }

    fn unlink(&mut self, j: u32, k: u32) {
        let mut prev = NONE;
        let mut cur = self.jhead[k as usize];
        while cur != j { debug_assert!(cur != NONE); prev = cur; cur = self.joint_next(cur, k); }
        let nxt = self.joint_next(j, k);
        if prev == NONE { self.jhead[k as usize] = nxt } else { self.set_next(prev, k, nxt) }
    }

    pub fn remove_joint(&mut self, j: u32) {
        let (a, b) = (self.ja[j as usize], self.jb[j as usize]);
        self.unlink(j, a); self.unlink(j, b);
        self.ja[j as usize] = NONE; self.jb[j as usize] = NONE;
        self.jfree.push(j);
        self.live_joints -= 1;
    }

    fn join_if_touching(&mut self, p: u32, q: u32, dmg: u8) {
        if let Some((ax, area, p_low)) = face_contact(&self.tbox(p), &self.tbox(q)) {
            if p_low { self.add_joint(p, q, ax, area, dmg); } else { self.add_joint(q, p, ax, area, dmg); }
        }
    }

    /// Rebuild every joint from geometry.
    pub fn build_joints(&mut self) {
        self.ja.clear(); self.jb.clear(); self.jna.clear(); self.jnb.clear();
        self.jarea.clear(); self.jaxis.clear(); self.jdmg.clear(); self.jfree.clear();
        self.live_joints = 0;
        for h in self.jhead.iter_mut() { *h = NONE; }
        let mut cand = Vec::new();
        for i in 0..self.len() as u32 {
            if self.flags[i as usize] & F_ALIVE == 0 { continue; }
            let b = self.tbox(i);
            cand.clear();
            self.query(b.min, b.max, true, &mut cand);
            for k in 0..cand.len() { let o = cand[k]; if o > i { self.join_if_touching(i, o, 0); } }
        }
    }

    /// Split brick `i` along `axis` at absolute tick `at` (strictly inside). The lower part keeps `i`;
    /// returns the index of the new upper part. Joints are redistributed exactly.
    pub fn split(&mut self, i: u32, axis: usize, at: i32) -> u32 {
        let b = self.tbox(i);
        debug_assert!(at > b.min[axis] && at < b.max[axis]);
        // detach i's joints, remembering neighbour and damage
        let mut nb: [(u32, u8); 64] = [(NONE, 0); 64];
        let mut big: Vec<(u32, u8)> = Vec::new();
        let mut cnt = 0usize;
        let mut j = self.jhead[i as usize];
        while j != NONE {
            let nxt = self.joint_next(j, i);
            let o = self.joint_other(j, i);
            let d = self.jdmg[j as usize];
            if cnt < 64 { nb[cnt] = (o, d) } else { big.push((o, d)) }
            cnt += 1;
            self.unlink(j, o);
            self.ja[j as usize] = NONE; self.jb[j as usize] = NONE;
            self.jfree.push(j); self.live_joints -= 1;
            j = nxt;
        }
        self.jhead[i as usize] = NONE;
        // new upper brick
        let iu = i as usize;
        let mut min = b.min; min[axis] = at;
        let mut size = [b.max[0] - min[0], b.max[1] - min[1], b.max[2] - min[2]];
        let n = self.add(min, size, self.mat[iu], self.paint[iu], self.structure[iu]);
        let nu = n as usize;
        self.flags[nu] = self.flags[iu]; self.body[nu] = self.body[iu]; self.damage[nu] = self.damage[iu];
        // shrink the lower part
        size = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
        size[axis] = at - b.min[axis];
        self.sx[iu] = size[0] as u8; self.sy[iu] = size[1] as u8; self.sz[iu] = size[2] as u8;
        // redistribute
        for k in 0..cnt {
            let (o, d) = if k < 64 { nb[k] } else { big[k - 64] };
            self.join_if_touching(i, o, d);
            self.join_if_touching(n, o, d);
        }
        // the cut itself, at full strength
        let (o1, o2) = ((axis + 1) % 3, (axis + 2) % 3);
        self.add_joint(i, n, axis as u8, ((b.max[o1] - b.min[o1]) * (b.max[o2] - b.min[o2])) as u32, 0);
        n
    }

    /// Where to split brick i along `axis`: the midpoint, snapped to the material's course grid while the
    /// brick is larger than a course.
    pub fn split_point(&self, i: u32, axis: usize) -> i32 {
        let b = self.tbox(i);
        let (lo, hi) = (b.min[axis], b.max[axis]);
        let mid = lo + (hi - lo) / 2;
        let c = mat::course(self.mat[i as usize]);
        if c > 1 && hi - lo > c {
            let snapped = (mid + c / 2).div_euclid(c) * c;
            if snapped > lo && snapped < hi { return snapped; }
        }
        mid
    }

    /// Split every brick touching the sphere (centre and radius in ticks) until none inside is larger than
    /// `target` ticks on any axis. Returns the number of splits.
    pub fn refine_sphere(&mut self, c: [f32; 3], r: f32, target: i32) -> usize {
        let target = target.max(1);
        let mut work = Vec::new();
        let min = [floor_i(c[0] - r), floor_i(c[1] - r), floor_i(c[2] - r)];
        let max = [floor_i(c[0] + r) + 1, floor_i(c[1] + r) + 1, floor_i(c[2] + r) + 1];
        self.query(min, max, false, &mut work);
        let mut splits = 0;
        while let Some(b) = work.pop() {
            let tb = self.tbox(b);
            if !sphere_box(c, r, &tb) { continue; }
            let s = self.size(b);
            let mut ax = 3; let mut best = target;
            for k in 0..3 { if s[k] > best { best = s[k]; ax = k; } }
            if ax == 3 { continue; }
            let at = self.split_point(b, ax);
            let n = self.split(b, ax, at);
            splits += 1;
            work.push(b); work.push(n);
        }
        splits
    }

    pub fn alive_count(&self) -> usize { self.flags.iter().filter(|&&f| f & F_ALIVE != 0).count() }
    /// Bytes of brick + joint records per live brick (the format in SPEC.md; excludes the spatial index).
    pub fn bytes_per_brick(&self) -> f32 {
        let n = self.alive_count().max(1);
        (n * 24 + self.live_joints * 20) as f32 / n as f32
    }
    /// Approximate bytes held by the spatial index.
    pub fn index_bytes(&self) -> usize { self.grid.values().map(|v| v.capacity() * 4 + 40).sum() }

    /// Sorted list of (a, b, axis, area) for every live joint, for tests.
    pub fn joint_set(&self) -> Vec<(u32, u32, u8, u16)> {
        let mut v: Vec<_> = (0..self.ja.len()).filter(|&j| self.ja[j] != NONE)
            .map(|j| (self.ja[j], self.jb[j], self.jaxis[j], self.jarea[j])).collect();
        v.sort_unstable();
        v
    }
}

/// The split-gate scenario: a 7.5 x 3.75 x 0.25 m masonry panel as one brick (index 0), between two
/// concrete piers and two slabs so it has outside joints.
pub fn panel() -> (Bricks, u32) {
    let mut b = Bricks::new();
    let w = b.add([0, 0, 0], [60, 30, 2], mat::MASONRY, 0, 0);
    b.add([-4, 0, 0], [4, 30, 2], mat::CONCRETE, 0, 0);
    b.add([60, 0, 0], [4, 30, 2], mat::CONCRETE, 0, 0);
    b.add([-4, -2, -4], [68, 2, 10], mat::CONCRETE, 0, 0);
    b.add([-4, 30, -4], [68, 2, 10], mat::CONCRETE, 0, 0);
    b.build_joints();
    (b, w)
}

#[inline] fn floor_i(x: f32) -> i32 { crate::math::floor(x) as i32 }
fn sphere_box(c: [f32; 3], r: f32, b: &TBox) -> bool {
    let mut d2 = 0.0;
    for k in 0..3 {
        let v = c[k].max(b.min[k] as f32).min(b.max[k] as f32) - c[k];
        d2 += v * v;
    }
    d2 < r * r
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::time::Instant;

    use super::panel;

    #[test] fn contact_cases() {
        let a = TBox { min: [0, 0, 0], max: [4, 2, 2] };
        assert_eq!(face_contact(&a, &TBox { min: [4, 1, 0], max: [8, 3, 2] }), Some((0, 2, true)));
        assert_eq!(face_contact(&TBox { min: [4, 1, 0], max: [8, 3, 2] }, &a), Some((0, 2, false)));
        assert_eq!(face_contact(&a, &TBox { min: [4, 2, 0], max: [8, 3, 2] }), None); // edge only
        assert_eq!(face_contact(&a, &TBox { min: [5, 0, 0], max: [8, 2, 2] }), None); // gap
    }

    #[test] fn split_keeps_joints_exact() {
        let (mut b, w) = panel();
        let before = b.joint_set();
        assert_eq!(before.len(), 8); // wall to 2 piers and 2 slabs, each pier to both slabs
        let n = b.split(w, 0, 20);
        b.split(n, 1, 10);
        b.refine_sphere([30.0, 15.0, 2.0], 4.0, 1);
        b.refine_sphere([5.0, 3.0, 0.0], 6.0, 2);
        // incremental result must equal a rebuild from geometry
        let inc = b.joint_set();
        let mut r = b.clone(); r.build_joints();
        assert_eq!(inc, r.joint_set());
        // area to the outside is conserved: wall-to-slab contact stays 60 x 2 ticks on each slab
        let wall: Vec<u32> = (0..b.len() as u32).filter(|&i| b.mat[i as usize] == mat::MASONRY).collect();
        let vol: i32 = wall.iter().map(|&i| { let s = b.size(i); s[0] * s[1] * s[2] }).sum();
        assert_eq!(vol, 60 * 30 * 2);
        // (the slab under the impact was split too, so count by material rather than by index)
        let is_wall = |i: u32| wall.contains(&i);
        let bearing: u32 = inc.iter().filter(|j| j.2 == 1 && is_wall(j.0) != is_wall(j.1)).map(|j| j.3 as u32).sum();
        let side: u32 = inc.iter().filter(|j| j.2 == 0 && is_wall(j.0) != is_wall(j.1)).map(|j| j.3 as u32).sum();
        assert_eq!(bearing, 2 * 60 * 2);
        assert_eq!(side, 2 * 30 * 2);
    }

    #[test] fn course_grid_snaps() {
        let mut b = Bricks::new();
        let w = b.add([0, 0, 0], [60, 30, 2], mat::MASONRY, 0, 0);
        assert_eq!(b.split_point(w, 1), 16); // 15 snapped to the 2-tick course
        let g = b.add([0, 0, 10], [60, 30, 1], mat::GLASS, 0, 0);
        assert_eq!(b.split_point(g, 1), 15);
    }

    #[test] fn gate_split_timing_and_memory() {
        // SPEC gate: panel refined to 0.125 m within 0.5 m (4 ticks) of an impact point in under 1 ms.
        let runs = 400;
        let mut times = std::vec::Vec::new(); let mut bricks = 0; let mut splits = 0;
        for k in 0..runs {
            let (mut b, _) = panel();
            let cx = 8.0 + (k % 45) as f32;
            let t = Instant::now();
            splits = b.refine_sphere([cx, 15.0, 2.0], 4.0, 1);
            times.push(t.elapsed().as_secs_f64() * 1e3);
            bricks = b.alive_count();
            if k == 0 {
                let mut r = b.clone(); r.build_joints();
                assert_eq!(b.joint_set(), r.joint_set());
            }
        }
        // memory on a realistic wall: 0.5 x 0.25 x 0.25 m masonry blocks, running bond, 7.5 x 3.75 m
        let mut w = Bricks::new();
        for row in 0..15 { let off = if row % 2 == 1 { 2 } else { 0 };
            let mut x = 0; if off > 0 { w.add([0, row * 2, 0], [2, 2, 2], mat::MASONRY, 0, 0); x = 2; }
            while x < 60 { let len = (60 - x).min(4); w.add([x, row * 2, 0], [len, 2, 2], mat::MASONRY, 0, 0); x += len; } }
        w.build_joints();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (med, p99, worst) = (times[runs / 2], times[runs * 99 / 100], times[runs - 1]);
        std::println!("split gate: {splits} splits -> {bricks} bricks, median {med:.4} ms, p99 {p99:.4} ms, worst {worst:.4} ms (gate 1 ms)");
        std::println!("memory: masonry wall {} bricks, {} joints, {:.1} B/brick (gate 96); refined panel {:.1} B/brick",
            w.alive_count(), w.live_joints, w.bytes_per_brick(), { let (mut b, _) = panel(); b.refine_sphere([30.0, 15.0, 2.0], 4.0, 1); b.bytes_per_brick() });
        assert!(p99 < 1.0, "split gate");
        assert!(w.bytes_per_brick() <= 96.0);
    }
}
