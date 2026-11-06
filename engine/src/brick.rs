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
/// cast monolithically with its neighbours (stair flights): its joints are not sections for bending
pub const F_MONO: u8 = 8;
/// reinforced concrete: a failure first cracks it (the steel holds, half the strength is left), a second one separates
pub const F_REINF: u8 = 16;
/// on a fracture surface: a joint of it broke, or a neighbour was crushed away. It waits in `frac_pending` to be
/// split into the convex cells of a fracture pattern (`shatter`), so a break is jagged, not a step of boxes.
pub const F_FRAC: u8 = 32;
/// The cell pattern of fractured brick `i` (0..16, see the hull library) and the mask of its cells that spall off
/// (fall away as crumbs): one or two of them, never all. Fixed by the brick's index.
pub fn frac_cells(i: u32, ncells: u32) -> (u32, u32) {
    let h = i.wrapping_mul(2654435761).rotate_right(7) ^ 0x9e37_79b9;
    let pattern = h % 16;
    if ncells < 2 { return (pattern, 0); }
    let a = (h >> 8) % ncells;
    let mut mask = 1u32 << a;
    if ncells > 3 && (h >> 16) & 1 == 1 { mask |= 1 << ((a + 1 + (h >> 20) % (ncells - 1)) % ncells); }
    (pattern, mask)
}

/// Spatial index: each brick sits in exactly one cell, the cell holding its min corner, at the finest of four
/// levels (1, 4, 16, 64 m) whose cell is at least as large as the brick. A query visits, per level, the cells
/// that can hold the min corner of an overlapping brick.
const LEVELS: usize = 4;
const fn lcell(l: usize) -> i32 { 8 << (2 * l) }

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
    // bricks, SoA (25 B each)
    pub x: Vec<i16>, pub y: Vec<i16>, pub z: Vec<i16>,
    pub sx: Vec<u8>, pub sy: Vec<u8>, pub sz: Vec<u8>,
    pub mat: Vec<u8>, pub paint: Vec<u8>, pub flags: Vec<u8>,
    pub structure: Vec<u16>, pub body: Vec<u32>, pub jhead: Vec<u32>, pub damage: Vec<u8>,
    /// the brick's convex cell: 0 = its whole box; else a cell of the hull library (`hull::lib`), mapped into the box
    /// (the box is the brick it was split from: the cells of one fractured brick share it)
    pub hull: Vec<u8>,
    // joints, SoA (20 B each); `ja == NONE` marks a free slot
    pub ja: Vec<u32>, pub jb: Vec<u32>, pub jna: Vec<u32>, pub jnb: Vec<u32>,
    pub jarea: Vec<u16>, pub jaxis: Vec<u8>, pub jdmg: Vec<u8>,
    jfree: Vec<u32>,
    pub live_joints: usize,
    // spatial index (an accelerator, not part of the brick format): cells -> bricks touching them, at two
    // levels so a 30 m slab is a few entries, not hundreds. Entries can go stale after splits and removals;
    // queries test real overlap.
    cells: HashMap<u64, u32>,   // cell key -> first brick in it
    cnext: Vec<u32>,            // next brick in the same cell
    /// bricks added, resized or removed since the log was last drained (for meshes and physics shapes)
    pub log: Vec<u32>,
    /// bricks marked on a fracture surface (F_FRAC) and not yet split into cells
    pub frac_pending: Vec<u32>,
}
#[inline] fn ckey(l: usize, c: [i32; 3]) -> u64 {
    ((l as u64) << 60) | (((c[0] + (1 << 19)) as u64 & 0xfffff) << 40) | (((c[1] + (1 << 19)) as u64 & 0xfffff) << 20) | ((c[2] + (1 << 19)) as u64 & 0xfffff)
}

/// Raw save and load of a brick store (to hand it between a worker and the page): every array as its length
/// and bytes, little endian.
trait Pod: Copy + Default {}
impl Pod for u8 {} impl Pod for i16 {} impl Pod for u16 {} impl Pod for u32 {} impl Pod for u64 {}
fn put<T: Pod>(out: &mut Vec<u8>, v: &[T]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    let bytes = unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * core::mem::size_of::<T>()) };
    out.extend_from_slice(bytes);
    while out.len() % 8 != 0 { out.push(0); }
}
fn get<T: Pod>(src: &[u8], at: &mut usize) -> Vec<T> {
    let n = u32::from_le_bytes([src[*at], src[*at + 1], src[*at + 2], src[*at + 3]]) as usize; *at += 4;
    let sz = n * core::mem::size_of::<T>();
    let mut v = alloc::vec![T::default(); n];
    unsafe { core::ptr::copy_nonoverlapping(src[*at..*at + sz].as_ptr(), v.as_mut_ptr() as *mut u8, sz); }
    *at += sz;
    while *at % 8 != 0 { *at += 1; }
    v
}
impl Bricks {
    pub fn save(&self, out: &mut Vec<u8>) {
        out.clear();
        put(out, &self.x); put(out, &self.y); put(out, &self.z);
        put(out, &self.sx); put(out, &self.sy); put(out, &self.sz);
        put(out, &self.mat); put(out, &self.paint); put(out, &self.flags);
        put(out, &self.structure); put(out, &self.body); put(out, &self.jhead); put(out, &self.damage); put(out, &self.hull);
        put(out, &self.ja); put(out, &self.jb); put(out, &self.jna); put(out, &self.jnb);
        put(out, &self.jarea); put(out, &self.jaxis); put(out, &self.jdmg);
        put(out, &self.jfree); put(out, &[self.live_joints as u32]);
        let keys: Vec<u64> = self.cells.keys().copied().collect();
        let vals: Vec<u32> = keys.iter().map(|k| self.cells[k]).collect();
        put(out, &keys); put(out, &vals); put(out, &self.cnext);
    }
    pub fn load(src: &[u8]) -> Bricks {
        let at = &mut 0usize;
        let mut b = Bricks::default();
        b.x = get(src, at); b.y = get(src, at); b.z = get(src, at);
        b.sx = get(src, at); b.sy = get(src, at); b.sz = get(src, at);
        b.mat = get(src, at); b.paint = get(src, at); b.flags = get(src, at);
        b.structure = get(src, at); b.body = get(src, at); b.jhead = get(src, at); b.damage = get(src, at); b.hull = get(src, at);
        b.ja = get(src, at); b.jb = get(src, at); b.jna = get(src, at); b.jnb = get(src, at);
        b.jarea = get(src, at); b.jaxis = get(src, at); b.jdmg = get(src, at);
        b.jfree = get(src, at); b.live_joints = get::<u32>(src, at)[0] as usize;
        let keys: Vec<u64> = get(src, at); let vals: Vec<u32> = get(src, at);
        b.cells = keys.into_iter().zip(vals).collect();
        b.cnext = get(src, at);
        b
    }
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
        self.structure.push(structure); self.body.push(NONE); self.jhead.push(NONE); self.damage.push(0); self.hull.push(0);
        self.cnext.push(NONE);
        self.index(i);
        self.log.push(i);
        i
    }

    fn index(&mut self, i: u32) {
        let b = self.tbox(i);
        let m = (0..3).map(|k| b.max[k] - b.min[k]).max().unwrap();
        let l = (0..LEVELS).find(|&l| lcell(l) >= m).unwrap_or(LEVELS - 1);
        let c = lcell(l);
        let key = ckey(l, [b.min[0].div_euclid(c), b.min[1].div_euclid(c), b.min[2].div_euclid(c)]);
        let head = self.cells.entry(key).or_insert(NONE);
        self.cnext[i as usize] = *head;
        *head = i;
    }

    /// Remove a brick: its joints go, the record stays (dead) so indices remain stable.
    pub fn remove(&mut self, i: u32) {
        // what it touched is now a fracture surface
        let mut j = self.jhead[i as usize];
        while j != NONE { let o = self.joint_other(j, i); self.mark_frac(o); j = self.joint_next(j, i); }
        while self.jhead[i as usize] != NONE { let j = self.jhead[i as usize]; self.remove_joint(j); }
        self.flags[i as usize] &= !F_ALIVE;
        self.log.push(i);
    }
    /// Remove a brick without marking what it touched (a cell spalling off a fracture surface).
    pub fn remove_quiet(&mut self, i: u32) {
        while self.jhead[i as usize] != NONE { let j = self.jhead[i as usize]; self.remove_joint(j); }
        self.flags[i as usize] &= !F_ALIVE;
        self.log.push(i);
    }
    #[inline] pub fn alive(&self, i: u32) -> bool { self.flags[i as usize] & F_ALIVE != 0 }

    /// Bricks not connected to the ground through joints, grouped into connected pieces. Grounded bricks are
    /// those resting on y = 0.
    pub fn detached(&self) -> Vec<Vec<u32>> {
        let n = self.len();
        let mut mark = alloc::vec![0u8; n];   // 0 unvisited, 1 grounded, 2 assigned to a piece
        let mut stack: Vec<u32> = Vec::new();
        for i in 0..n as u32 {
            if !self.alive(i) || self.flags[i as usize] & F_DYNAMIC != 0 { mark[i as usize] = 1; continue; }
            if self.y[i as usize] <= 0 { mark[i as usize] = 1; stack.push(i); }
        }
        while let Some(i) = stack.pop() {
            let mut j = self.jhead[i as usize];
            while j != NONE { let o = self.joint_other(j, i); if mark[o as usize] == 0 { mark[o as usize] = 1; stack.push(o); } j = self.joint_next(j, i); }
        }
        let mut pieces = Vec::new();
        for s in 0..n as u32 {
            if !self.alive(s) || mark[s as usize] != 0 { continue; }
            let mut piece = Vec::new();
            mark[s as usize] = 2; stack.push(s);
            while let Some(i) = stack.pop() {
                piece.push(i);
                let mut j = self.jhead[i as usize];
                while j != NONE { let o = self.joint_other(j, i); if mark[o as usize] == 0 { mark[o as usize] = 2; stack.push(o); } j = self.joint_next(j, i); }
            }
            pieces.push(piece);
        }
        pieces
    }

    /// Nearest brick hit by a ray (origin and direction in ticks), within `max_t` ticks: (brick, t, face axis).
    pub fn raycast(&mut self, o: [f32; 3], d: [f32; 3], max_t: f32) -> Option<(u32, f32, usize)> {
        let mut best: Option<(u32, f32, usize)> = None;
        let mut cand = Vec::new();
        let step = 8.0f32;
        let mut t = 0.0f32;
        while t < max_t {
            let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
            let r = step;
            cand.clear();
            self.query([floor_i(p[0] - r), floor_i(p[1] - r), floor_i(p[2] - r)], [floor_i(p[0] + r) + 1, floor_i(p[1] + r) + 1, floor_i(p[2] + r) + 1], false, &mut cand);
            for &i in &cand {
                let b = self.tbox(i);
                let (mut t0, mut t1, mut ax) = (0.0f32, max_t, 0usize);
                let mut ok = true;
                for k in 0..3 {
                    if crate::math::fabs(d[k]) < 1e-9 { if o[k] < b.min[k] as f32 || o[k] > b.max[k] as f32 { ok = false; break; } continue; }
                    let (mut a, mut c) = ((b.min[k] as f32 - o[k]) / d[k], (b.max[k] as f32 - o[k]) / d[k]);
                    if a > c { core::mem::swap(&mut a, &mut c); }
                    if a > t0 { t0 = a; ax = k; }
                    if c < t1 { t1 = c; }
                    if t0 > t1 { ok = false; break; }
                }
                if ok && best.map_or(true, |(_, bt, _)| t0 < bt) { best = Some((i, t0, ax)); }
            }
            if let Some((_, bt, _)) = best { if bt < t + step { return best; } }
            t += step;
        }
        best
    }

    /// Bricks whose box overlaps [min, max) in ticks (inclusive of touching when `touch`), each once.
    pub fn query(&self, min: [i32; 3], max: [i32; 3], touch: bool, out: &mut Vec<u32>) {
        let t = if touch { 1 } else { 0 };
        for l in 0..LEVELS {
            let c = lcell(l);
            // an overlapping brick's min corner lies in [min - c - t, max + t)
            let c0 = [(min[0] - c - t).div_euclid(c), (min[1] - c - t).div_euclid(c), (min[2] - c - t).div_euclid(c)];
            let c1 = [(max[0] - 1 + t).div_euclid(c), (max[1] - 1 + t).div_euclid(c), (max[2] - 1 + t).div_euclid(c)];
            for cx in c0[0]..=c1[0] { for cy in c0[1]..=c1[1] { for cz in c0[2]..=c1[2] {
                let mut i = match self.cells.get(&ckey(l, [cx, cy, cz])) { Some(&h) => h, None => continue };
                while i != NONE {
                    let iu = i as usize;
                    // the static world only: bricks of falling pieces (dynamic) live in the physics
                    if self.flags[iu] & (F_ALIVE | F_DYNAMIC) == F_ALIVE {
                        let mn = [self.x[iu] as i32, self.y[iu] as i32, self.z[iu] as i32];
                        let mx = [mn[0] + self.sx[iu] as i32, mn[1] + self.sy[iu] as i32, mn[2] + self.sz[iu] as i32];
                        let ok = (0..3).all(|k| if touch { mn[k] <= max[k] && mx[k] >= min[k] } else { mn[k] < max[k] && mx[k] > min[k] });
                        if ok { out.push(i); }
                    }
                    i = self.cnext[iu];
                }
            } } }
        }
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

    /// Mark brick `i` as on a fracture surface (if it is a size that reads as rubble: 0.25 to 2 m a side; bigger
    /// ones are refined near a break, smaller ones are crumbs already).
    pub fn mark_frac(&mut self, i: u32) {
        let iu = i as usize;
        if self.flags[iu] & F_ALIVE == 0 || self.flags[iu] & F_FRAC != 0 || self.hull[iu] != 0 || self.mat[iu] == mat::GLASS { return; }
        let s = self.size(i);
        if s.iter().any(|&v| v < 2) || s.iter().any(|&v| v > 16) { return; }
        self.flags[iu] |= F_FRAC;
        self.frac_pending.push(i);
    }
    /// A joint that breaks: both bricks are now on a fracture surface.
    pub fn fracture_joint(&mut self, j: u32) {
        let (a, b) = (self.ja[j as usize], self.jb[j as usize]);
        self.mark_frac(a); self.mark_frac(b);
        self.remove_joint(j);
    }
    pub fn remove_joint(&mut self, j: u32) {
        let (a, b) = (self.ja[j as usize], self.jb[j as usize]);
        self.unlink(j, a); self.unlink(j, b);
        self.ja[j as usize] = NONE; self.jb[j as usize] = NONE;
        self.jfree.push(j);
        self.live_joints -= 1;
    }

    fn join_if_touching(&mut self, p: u32, q: u32, dmg: u8) {
        if let Some((ax, area, p_low)) = self.contact(p, q) {
            if p_low { self.add_joint(p, q, ax, area, dmg); } else { self.add_joint(q, p, ax, area, dmg); }
        }
    }

    // ---- convex cells ----
    /// the brick's convex cell, if it is one
    #[inline] pub fn cell(&self, i: u32) -> Option<&'static crate::hull::Cell> { let h = self.hull[i as usize]; if h == 0 { None } else { Some(&crate::hull::lib().cells[h as usize]) } }
    /// centre and half size of the brick's box (m)
    pub fn box_m(&self, i: u32) -> ([f64; 3], [f64; 3]) {
        let t = self.tbox(i); let k = TICK as f64;
        ([0.5 * (t.min[0] + t.max[0]) as f64 * k, 0.5 * (t.min[1] + t.max[1]) as f64 * k, 0.5 * (t.min[2] + t.max[2]) as f64 * k],
         [0.5 * (t.max[0] - t.min[0]) as f64 * k, 0.5 * (t.max[1] - t.min[1]) as f64 * k, 0.5 * (t.max[2] - t.min[2]) as f64 * k])
    }
    /// the brick's share of its box's volume (1 for a whole box)
    #[inline] pub fn vol_share(&self, i: u32) -> f64 { self.cell(i).map_or(1.0, |c| c.vol) }
    /// the brick's centroid (m)
    pub fn centroid(&self, i: u32) -> [f64; 3] {
        let (c, h) = self.box_m(i);
        match self.cell(i) { None => c, Some(cl) => [c[0] + h[0] * cl.cen[0], c[1] + h[1] * cl.cen[1], c[2] + h[2] * cl.cen[2]] }
    }
    /// The brick's polygon on side `side` of its box (0..6: +x, -x, +y, -y, +z, -z), in m, counter-clockwise seen
    /// from outside; none if it is a cell that does not reach that side.
    pub fn side_poly(&self, i: u32, side: u8) -> Option<Vec<[f64; 3]>> {
        let (c, h) = self.box_m(i);
        if let Some(cl) = self.cell(i) { return cl.side(side).map(|f| cl.face_in(f, c, h)); }
        let ax = (side / 2) as usize; let sg = if side % 2 == 0 { 1.0 } else { -1.0 };
        let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
        let corner = |a: f64, b: f64| { let mut p = c; p[ax] += sg * h[ax]; p[u] += a * h[u]; p[w] += b * h[w]; p };
        let mut v = alloc::vec![corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
        if sg < 0.0 { v.reverse(); }
        Some(v)
    }
    /// Contact between bricks p and q: (axis, area in tick^2, true if p is on the min side). Two cells of one
    /// fractured brick touch on their shared face (axis 3, p the lower cell of the pattern); otherwise the boxes
    /// touch on a face and the contact is where the bricks' polygons on it overlap.
    pub fn contact(&self, p: u32, q: u32) -> Option<(u8, u32, bool)> {
        let (cp, cq) = (self.cell(p), self.cell(q));
        if let (Some(a), Some(b)) = (cp, cq) {
            if a.pattern == b.pattern && self.tbox(p) == self.tbox(q) {
                let f = a.shared(b.slot)?;
                let (c, h) = self.box_m(p);
                let area = crate::hull::polygon(&a.face_in(f, c, h)).0;
                return Some((3, area_ticks(area), a.slot < b.slot));
            }
        }
        let (ax, area, p_low) = face_contact(&self.tbox(p), &self.tbox(q))?;
        if cp.is_none() && cq.is_none() { return Some((ax, area, p_low)); }
        let (lo, hi) = if p_low { (p, q) } else { (q, p) };
        let poly = self.contact_poly(lo, hi, ax as usize);
        let a = crate::hull::polygon(&poly).0;
        if a < 1e-6 { return None; }
        Some((ax, area_ticks(a), p_low))
    }
    /// Where `lo` (on the min side along `ax`) and `hi` meet: the overlap of their polygons on the shared plane, in m,
    /// counter-clockwise about +ax.
    pub fn contact_poly(&self, lo: u32, hi: u32, ax: usize) -> Vec<[f64; 3]> {
        let (pa, pb) = match (self.side_poly(lo, 2 * ax as u8), self.side_poly(hi, 2 * ax as u8 + 1)) { (Some(a), Some(b)) => (a, b), _ => return Vec::new() };
        let mut n = [0.0; 3]; n[ax] = 1.0;
        crate::hull::intersect(&pa, &pb, n)
    }
    /// The face of joint j (m), counter-clockwise about the normal from ja to jb.
    pub fn joint_poly(&self, j: u32) -> Vec<[f64; 3]> {
        let ju = j as usize;
        let (a, b) = (self.ja[ju], self.jb[ju]);
        let ax = self.jaxis[ju] as usize;
        if ax == 3 {
            let (ca, cb) = (self.cell(a).unwrap(), self.cell(b).unwrap());
            let (c, h) = self.box_m(a);
            return ca.shared(cb.slot).map(|f| ca.face_in(f, c, h)).unwrap_or_default();
        }
        self.contact_poly(a, b, ax)
    }
    /// Split fractured brick `i` into the cells of its pattern (`frac_cells`). The first cell keeps `i`; its joints
    /// go to the cells that reach them, and the cells are joined on their shared faces. Returns the cells and,
    /// of them, the one or two that spall off (the caller removes them).
    pub fn shatter(&mut self, i: u32) -> (Vec<u32>, Vec<u32>) {
        let iu = i as usize;
        self.flags[iu] &= !F_FRAC;
        if self.hull[iu] != 0 || self.flags[iu] & F_ALIVE == 0 { return (Vec::new(), Vec::new()); }
        let lib = crate::hull::lib();
        let (pattern, _) = frac_cells(i, 1);
        let ids = &lib.patterns[pattern as usize];
        let (_, mask) = frac_cells(i, ids.len() as u32);
        // its joints, with neighbour and damage
        let mut nb: Vec<(u32, u8)> = Vec::new();
        while self.jhead[iu] != NONE { let j = self.jhead[iu]; nb.push((self.joint_other(j, i), self.jdmg[j as usize])); self.remove_joint(j); }
        let t = self.tbox(i);
        let size = [t.max[0] - t.min[0], t.max[1] - t.min[1], t.max[2] - t.min[2]];
        let mut cells = Vec::with_capacity(ids.len());
        for (k, &id) in ids.iter().enumerate() {
            let c = if k == 0 { i } else {
                let n = self.add(t.min, size, self.mat[iu], self.paint[iu], self.structure[iu]);
                let nu = n as usize;
                self.flags[nu] = self.flags[iu]; self.body[nu] = self.body[iu]; self.damage[nu] = self.damage[iu];
                n
            };
            self.hull[c as usize] = id as u8;
            cells.push(c);
        }
        self.log.push(i);
        for &c in &cells { for &(o, d) in &nb { self.join_if_touching(c, o, d); } }
        for s in 0..cells.len() { for t in s + 1..cells.len() { self.join_if_touching(cells[s], cells[t], 0); } }
        let spall = (0..cells.len()).filter(|&k| mask & (1 << k) != 0).map(|k| cells[k]).collect();
        (cells, spall)
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
        debug_assert!(self.hull[i as usize] == 0, "a convex cell is not split");
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
        self.log.push(i);
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
        let mut work = Vec::new();
        let min = [floor_i(c[0] - r), floor_i(c[1] - r), floor_i(c[2] - r)];
        let max = [floor_i(c[0] + r) + 1, floor_i(c[1] + r) + 1, floor_i(c[2] + r) + 1];
        self.query(min, max, false, &mut work);
        self.refine_bricks(work, c, r, target, &mut Vec::new())
    }

    /// Refine the given bricks (and their split-off parts) inside the sphere; new bricks go to `created`.
    pub fn refine_bricks(&mut self, mut work: Vec<u32>, c: [f32; 3], r: f32, target: i32, created: &mut Vec<u32>) -> usize {
        let target = target.max(1);
        let mut splits = 0;
        while let Some(b) = work.pop() {
            let tb = self.tbox(b);
            // (a convex cell is as small as fracture makes bricks: it is not split)
            if !sphere_box(c, r, &tb) || self.hull[b as usize] != 0 { continue; }
            let s = self.size(b);
            let mut ax = 3; let mut best = target;
            for k in 0..3 { if s[k] > best { best = s[k]; ax = k; } }
            if ax == 3 { continue; }
            let at = self.split_point(b, ax);
            let n = self.split(b, ax, at);
            splits += 1;
            created.push(n);
            work.push(b); work.push(n);
        }
        splits
    }

    pub fn alive_count(&self) -> usize { self.flags.iter().filter(|&&f| f & F_ALIVE != 0).count() }
    /// Bytes of brick + joint records per live brick (the format in SPEC.md; excludes the spatial index).
    pub fn bytes_per_brick(&self) -> f32 {
        let n = self.alive_count().max(1);
        (n * 25 + self.live_joints * 20) as f32 / n as f32
    }
    /// Approximate bytes held by the spatial index.
    pub fn index_bytes(&self) -> usize {
        // per brick a 4 B list link; per cell a hash entry (8 B key + 4 B head + 1 control byte, ~7/8 load)
        self.cnext.len() * 4 + self.cells.len() * 15
    }
    /// Everything per live brick: records, joints and the spatial index.
    pub fn total_bytes_per_brick(&self) -> f32 { self.bytes_per_brick() + self.index_bytes() as f32 / self.alive_count().max(1) as f32 }

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

/// an area in m^2 as tick^2 for a joint record (at least 1)
fn area_ticks(a: f64) -> u32 { ((a / (TICK as f64 * TICK as f64)) + 0.5).clamp(1.0, 65535.0) as u32 }
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

    #[test] fn a_fractured_brick_splits_into_its_convex_cells() {
        // a 1 x 1 x 0.5 m masonry brick on a ground block, with a brick on each side and one on top: split into the
        // cells of its pattern, the cells fill its volume, its old joints are shared by the cells that reach each
        // face (the cells' faces tile it), and the cells are joined to each other on their shared faces
        for idx in 0..16u32 {
            let mut b = Bricks::new();
            for _ in 0..idx { b.add([100, 0, 100], [2, 2, 2], mat::CONCRETE, 0, 0); }   // (to vary the pattern)
            b.add([0, 0, 0], [8, 8, 4], mat::CONCRETE, 0, 0);
            let i = b.add([0, 8, 0], [8, 8, 4], mat::MASONRY, 0, 0);
            b.add([-8, 8, 0], [8, 8, 4], mat::MASONRY, 0, 0);
            b.add([8, 8, 0], [8, 8, 4], mat::MASONRY, 0, 0);
            b.add([0, 16, 0], [8, 8, 4], mat::MASONRY, 0, 0);
            b.build_joints();
            let face = |b: &Bricks, k: u32, o: u32| -> f64 { let mut a = 0.0; let mut j = b.jhead[k as usize]; while j != NONE { if b.joint_other(j, k) == o { a += crate::hull::polygon(&b.joint_poly(j)).0; } j = b.joint_next(j, k); } a };
            let others: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&o| o != i && b.contact(i, o).is_some()).collect();
            assert_eq!(others.len(), 4);
            let (cells, spall) = b.shatter(i);
            let pat = &crate::hull::lib().patterns[frac_cells(i, 1).0 as usize];
            assert_eq!(cells.len(), pat.len());
            assert!(!spall.is_empty() && spall.len() < cells.len());
            let vol: f64 = cells.iter().map(|&c| b.vol_share(c)).sum();
            assert!((vol - 1.0).abs() < 1e-9, "the cells fill the brick: {vol}");
            for &o in &others {
                let a: f64 = cells.iter().map(|&c| face(&b, c, o)).sum();
                let want = 0.5;   // (every face of a 1 x 1 x 0.5 m brick that touches a neighbour here is 0.5 m^2)
                assert!((a - want).abs() < 1e-3, "pattern {}: the cells share the face to {o}: {a} m^2 against {want}", frac_cells(i, 1).0);
            }
            // cells joined where they share a face; each cell joined to something
            let shared: usize = (0..cells.len()).map(|s| (s + 1..cells.len()).filter(|&t| { let (x, y) = (b.cell(cells[s]).unwrap(), b.cell(cells[t]).unwrap()); x.shared(y.slot).is_some() }).count()).sum();
            let joined: usize = (0..b.ja.len()).filter(|&j| b.ja[j] != NONE && b.jaxis[j] == 3).count();
            assert_eq!(joined, shared);
            for &c in &cells { assert!(b.jhead[c as usize] != NONE); }
        }
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
#[cfg(test)]
mod snap_tests {
    #[test] fn save_and_load_round_trip() {
        let mut g = crate::game::Game::new(3);
        g.blast(crate::math::v3(-2.0, 3.875, -2.0), 0.6);
        let mut buf = alloc::vec::Vec::new();
        g.b.save(&mut buf);
        let c = super::Bricks::load(&buf);
        assert_eq!(c.len(), g.b.len());
        assert_eq!(c.joint_set(), g.b.joint_set());
        let mut q1 = alloc::vec::Vec::new(); let mut q2 = alloc::vec::Vec::new();
        g.b.query([-40, 0, -40], [40, 40, 40], true, &mut q1); c.query([-40, 0, -40], [40, 40, 40], true, &mut q2);
        q1.sort(); q2.sort(); assert_eq!(q1, q2);
        extern crate std; std::println!("store of {} bricks: {} KB", c.len(), buf.len() / 1024);
    }
}
