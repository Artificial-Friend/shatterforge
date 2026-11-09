//! Adaptive brick store: real-size axis-aligned boxes on the 0.125 m tick grid, joints as shared faces with
//! exact areas, and on-demand splitting that keeps the joint graph exact. Layout follows docs/SPEC.md.
use alloc::vec::Vec;
use crate::Map as HashMap;

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
/// loose: room contents and the like, built in place, joined only to the other parts of the same object (the same
/// `structure`) and founded on nothing: each object is a detached piece at the first structure step, a body resting
/// on the floor it stands on.
pub const F_LOOSE: u8 = 64;
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

/// `hull` of a brick that is a free convex piece (its shape is in `pieces`, not the hull library)
pub const HULL_FREE: u8 = 255;
/// `jaxis` of a joint with a free piece: its face is where two faces meet flat (`Bricks::free_face`), any angle
pub const JFREE: u8 = 4;
/// Two faces meet flat (a joint) when they face each other within 1 degree (cos 1 deg) and their planes are within
/// 5 mm: what a built joint is, mortar bed or a plate bolted on, and far tighter than anything that only rests.
pub const JOIN_COS: f64 = 0.999_847_695_156_391_2;
pub const JOIN_GAP: f64 = 0.005;
/// the smallest joint face (m^2): 1 cm^2 (a sliver where two faces only graze is no joint)
pub const JOIN_MIN_AREA: f64 = 1e-4;
/// Grid of a free piece's centre (m): 1 mm.
pub const PIECE_GRID: f64 = 0.001;

/// A free piece's shape in its own frame, about its centre (m): corners and faces (corner loops, counter-clockwise
/// seen from outside), with its volume and centroid. Pieces of the same shape share one (`Bricks::shapes`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PieceShape { pub v: Vec<[f32; 3]>, pub faces: Vec<Vec<u8>>, pub vol: f64, pub cen: [f64; 3] }
/// A free convex piece, at any angle: its centre on the 1 mm grid (`c`, in mm), its rotation (`q`, x y z w), and its
/// shape (`s`, in `Bricks::shapes`). Its brick's box is the tick box that holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Piece { pub c: [i32; 3], pub q: [f32; 4], pub s: u32 }
/// A piece with its shape (`Bricks::piece`).
#[derive(Clone, Copy, Debug)]
pub struct PieceRef<'a> { pub c: [i32; 3], pub q: [f32; 4], pub v: &'a [[f32; 3]], pub faces: &'a [Vec<u8>], pub vol: f64, pub cen: [f64; 3] }
fn qrot(q: [f32; 4], v: [f64; 3]) -> [f64; 3] {
    let (x, y, z, w) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    // v + 2 w (q x v) + 2 q x (q x v)
    let t = [2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])];
    [v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0])]
}
impl PieceRef<'_> {
    /// centre (m)
    pub fn centre(&self) -> [f64; 3] { [self.c[0] as f64 * PIECE_GRID, self.c[1] as f64 * PIECE_GRID, self.c[2] as f64 * PIECE_GRID] }
    /// corners in the world (m)
    pub fn world(&self) -> Vec<[f64; 3]> {
        let c = self.centre();
        self.v.iter().map(|p| { let r = qrot(self.q, [p[0] as f64, p[1] as f64, p[2] as f64]); [c[0] + r[0], c[1] + r[1], c[2] + r[2]] }).collect()
    }
    /// the faces in the world: (outward normal, polygon in m, counter-clockwise seen from outside)
    pub fn world_faces(&self) -> Vec<([f64; 3], Vec<[f64; 3]>)> {
        let w = self.world();
        self.faces.iter().map(|f| { let poly: Vec<[f64; 3]> = f.iter().map(|&k| w[k as usize]).collect(); (crate::hull::polygon(&poly).2, poly) }).collect()
    }
}
/// a shape's key for sharing: its corners to 10 um and its faces (pieces cut alike from identical members share it)
fn shape_key(s: &PieceShape) -> (u64, Vec<i32>) {
    let q: Vec<i32> = s.v.iter().flat_map(|p| p.iter().map(|&x| crate::math::floor64(x as f64 * 1e5 + 0.5) as i32)).collect();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| { h ^= x; h = h.wrapping_mul(0x0000_0100_0000_01b3); };
    for &x in &q { mix(x as u32 as u64); }
    for f in &s.faces { mix(0xffff_0000 | f.len() as u64); for &k in f { mix(k as u64); } }
    (h, q)
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
    /// free convex pieces (any angle; `hull == HULL_FREE`), in a side table so box bricks keep their 25 B: the
    /// piece of each such brick, and the pieces
    pub piece_of: HashMap<u32, u32>,
    pub pieces: Vec<Piece>,
    /// the pieces' shapes, each kept once, and their index by `shape_key`
    pub shapes: Vec<PieceShape>,
    pub shape_ix: HashMap<u64, Vec<u32>>,
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
impl Pod for u8 {} impl Pod for i16 {} impl Pod for i32 {} impl Pod for f32 {} impl Pod for f64 {} impl Pod for u16 {} impl Pod for u32 {} impl Pod for u64 {}
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
        // free pieces: their bricks, centres, rotations and shapes; the shapes: corner and face counts, corners, face
        // loops, volume and centroid
        let ids: Vec<u32> = { let mut v: Vec<(u32, u32)> = self.piece_of.iter().map(|(&b, &p)| (p, b)).collect(); v.sort(); v.into_iter().map(|x| x.1).collect() };
        let (mut c, mut q, mut sh) = (Vec::new(), Vec::new(), Vec::new());
        for p in &self.pieces { c.extend_from_slice(&p.c); q.extend_from_slice(&p.q); sh.push(p.s); }
        let (mut nv, mut nf, mut v, mut f, mut m) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for p in &self.shapes {
            nv.push(p.v.len() as u32); nf.push(p.faces.len() as u32);
            for x in &p.v { v.extend_from_slice(x); }
            for fl in &p.faces { f.push(fl.len() as u8); f.extend_from_slice(fl); }
            m.push(p.vol); m.extend_from_slice(&p.cen);
        }
        put(out, &ids); put(out, &c); put(out, &q); put(out, &sh); put(out, &nv); put(out, &nf); put(out, &v); put(out, &f); put(out, &m);
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
        if *at < src.len() {
            let ids: Vec<u32> = get(src, at); let c: Vec<i32> = get(src, at); let q: Vec<f32> = get(src, at); let sh: Vec<u32> = get(src, at);
            let nv: Vec<u32> = get(src, at); let nf: Vec<u32> = get(src, at); let v: Vec<f32> = get(src, at); let f: Vec<u8> = get(src, at); let m: Vec<f64> = get(src, at);
            for k in 0..ids.len() {
                b.pieces.push(Piece { c: [c[3 * k], c[3 * k + 1], c[3 * k + 2]], q: [q[4 * k], q[4 * k + 1], q[4 * k + 2], q[4 * k + 3]], s: sh[k] });
                b.piece_of.insert(ids[k], k as u32);
            }
            let (mut av, mut af) = (0usize, 0usize);
            for k in 0..nv.len() {
                let vs = (0..nv[k] as usize).map(|t| [v[3 * (av + t)], v[3 * (av + t) + 1], v[3 * (av + t) + 2]]).collect();
                av += nv[k] as usize;
                let mut faces = Vec::new();
                for _ in 0..nf[k] { let n = f[af] as usize; faces.push(f[af + 1..af + 1 + n].to_vec()); af += 1 + n; }
                let shape = PieceShape { v: vs, faces, vol: m[4 * k], cen: [m[4 * k + 1], m[4 * k + 2], m[4 * k + 3]] };
                let (h, _) = shape_key(&shape);
                b.shape_ix.entry(h).or_default().push(k as u32);
                b.shapes.push(shape);
            }
        }
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

    /// Add a free convex piece: the hull of `pts` (m, in the piece's own frame about `centre`), turned by `q` (x y z
    /// w) and placed at `centre` (m, snapped to the 1 mm grid). None if the points have no volume, the hull is beyond
    /// the GPU's limits on a hull, or it is too big for a brick's box (32 m a side).
    pub fn add_piece(&mut self, centre: [f64; 3], q: [f32; 4], pts: &[[f64; 3]], material: u8, paint: u8, structure: u16) -> Option<u32> {
        let ql = { let l = crate::math::sqrt64(q.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()) as f32; if l > 0.0 { [q[0] / l, q[1] / l, q[2] / l, q[3] / l] } else { [0.0, 0.0, 0.0, 1.0] } };
        let snap = |x: f64| crate::math::floor64(x / PIECE_GRID + 0.5) as i32;
        let c = [snap(centre[0]), snap(centre[1]), snap(centre[2])];
        let (shape, t) = Self::piece_of_points(c, ql, pts)?;
        let (lo, size) = (t.min, [t.max[0] - t.min[0], t.max[1] - t.min[1], t.max[2] - t.min[2]]);
        let i = self.add(lo, size, material, paint, structure);
        self.hull[i as usize] = HULL_FREE;
        let s = self.intern(shape);
        self.piece_of.insert(i, self.pieces.len() as u32);
        self.pieces.push(Piece { c, q: ql, s });
        Some(i)
    }
    /// keep a shape once: the index of the same shape if there is one, else of this one, added
    fn intern(&mut self, shape: PieceShape) -> u32 {
        let (h, q) = shape_key(&shape);
        if let Some(c) = self.shape_ix.get(&h) { for &k in c { let o = &self.shapes[k as usize]; if o.faces == shape.faces && shape_key(o).1 == q { return k; } } }
        let k = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_ix.entry(h).or_default().push(k);
        k
    }
    /// Cut free piece `i` by the plane n.x = d (m; n a unit vector) into two convex pieces: `i` keeps the side
    /// n.x <= d, the new piece (returned) the other. Each keeps the material, its joints go to the part(s) that still
    /// meet the neighbour, and the two parts are joined on the cut. None (and nothing changed) if the plane misses the
    /// piece or leaves a part too thin (under 1 cm) or beyond the hull limits.
    pub fn cut_piece(&mut self, i: u32, n: [f64; 3], d: f64) -> Option<u32> {
        let (w, faces, pq) = { let p = self.piece(i)?; (p.world(), p.faces.to_vec(), p.q) };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (lo, hi) = w.iter().fold((f64::MAX, f64::MIN), |a, v| (a.0.min(dot(n, *v) - d), a.1.max(dot(n, *v) - d)));
        if lo > -0.01 || hi < 0.01 { return None; }
        // each side's points: its corners, and where the edges cross the plane
        let (mut below, mut above): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (Vec::new(), Vec::new());
        for v in &w { let e = dot(n, *v) - d; if e <= 0.0 { below.push(*v); } if e >= 0.0 { above.push(*v); } }
        for f in &faces { for k in 0..f.len() {
            let (a, b) = (w[f[k] as usize], w[f[(k + 1) % f.len()] as usize]);
            let (ea, eb) = (dot(n, a) - d, dot(n, b) - d);
            if (ea < 0.0 && eb > 0.0) || (ea > 0.0 && eb < 0.0) {
                let t = ea / (ea - eb);
                let x = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
                below.push(x); above.push(x);
            }
        } }
        let make = |pts: &[[f64; 3]]| -> Option<([i32; 3], PieceShape, TBox)> {
            let m = pts.len() as f64;
            let c = pts.iter().fold([0.0; 3], |a, x| [a[0] + x[0] / m, a[1] + x[1] / m, a[2] + x[2] / m]);
            let snap = |x: f64| crate::math::floor64(x / PIECE_GRID + 0.5) as i32;
            let ci = [snap(c[0]), snap(c[1]), snap(c[2])];
            let cm = [ci[0] as f64 * PIECE_GRID, ci[1] as f64 * PIECE_GRID, ci[2] as f64 * PIECE_GRID];
            // in the piece's own frame: the world points turned back by its rotation
            let qi = [-pq[0], -pq[1], -pq[2], pq[3]];
            let local: Vec<[f64; 3]> = pts.iter().map(|x| qrot(qi, [x[0] - cm[0], x[1] - cm[1], x[2] - cm[2]])).collect();
            Self::piece_of_points(ci, pq, &local).map(|(s, t)| (ci, s, t))
        };
        let (ca, sa, ta) = make(&below)?; let (cb, sb, tb) = make(&above)?;
        let pa = Piece { c: ca, q: pq, s: self.intern(sa) };
        let pb = Piece { c: cb, q: pq, s: self.intern(sb) };
        // the joints of i, with neighbour and damage
        let mut nb: Vec<(u32, u8)> = Vec::new();
        while self.jhead[i as usize] != NONE { let j = self.jhead[i as usize]; nb.push((self.joint_other(j, i), self.jdmg[j as usize])); self.remove_joint(j); }
        let iu = i as usize;
        let k = *self.piece_of.get(&i).unwrap() as usize;
        self.pieces[k] = pa;
        self.set_box(i, ta);
        let s = [tb.max[0] - tb.min[0], tb.max[1] - tb.min[1], tb.max[2] - tb.min[2]];
        let o = self.add(tb.min, s, self.mat[iu], self.paint[iu], self.structure[iu]);
        let ou = o as usize;
        self.flags[ou] = self.flags[iu]; self.body[ou] = self.body[iu]; self.damage[ou] = self.damage[iu];
        self.hull[ou] = HULL_FREE;
        self.piece_of.insert(o, self.pieces.len() as u32);
        self.pieces.push(pb);
        for &(q, dm) in &nb { self.join_if_touching(i, q, dm); self.join_if_touching(o, q, dm); }
        self.join_if_touching(i, o, 0);
        Some(o)
    }
    /// A piece from points in its own frame (centre in mm, rotation), and its tick box; none as in `add_piece`.
    fn piece_of_points(c: [i32; 3], q: [f32; 4], pts: &[[f64; 3]]) -> Option<(PieceShape, TBox)> {
        let sh = crate::hull::convex(pts)?;
        let faces: Vec<Vec<u8>> = sh.faces.iter().map(|f| f.2.iter().map(|&k| k as u8).collect()).collect();
        let v: Vec<[f32; 3]> = sh.v.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        // mass properties from the stored (f32) corners, so they are those of the shape kept
        let vd: Vec<[f64; 3]> = v.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
        let fi: Vec<Vec<usize>> = faces.iter().map(|f| f.iter().map(|&k| k as usize).collect()).collect();
        let (vol, cen, _) = crate::hull::moments(&vd, &fi);
        let shape = PieceShape { v, faces, vol, cen };
        // its box: the ticks that hold its corners
        let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for p in &(PieceRef { c, q, v: &shape.v, faces: &shape.faces, vol, cen }).world() { for k in 0..3 { let t = p[k] / TICK as f64; lo[k] = lo[k].min(crate::math::floor64(t + 1e-9) as i32); hi[k] = hi[k].max(crate::math::ceil64(t - 1e-9) as i32); } }
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        if size.iter().any(|&s| s < 1 || s > 255) || lo.iter().chain(hi.iter()).any(|&t| t < i16::MIN as i32 || t > i16::MAX as i32) { return None; }
        Some((shape, TBox { min: lo, max: hi }))
    }
    /// Move brick `i` to tick box `t`, and in the spatial index to the cell of its new box. (Re-indexing without
    /// taking it out of its old cell's list first linked the lists into a loop: a query never ended.)
    fn set_box(&mut self, i: u32, t: TBox) {
        let iu = i as usize;
        self.unindex(i);
        self.x[iu] = t.min[0] as i16; self.y[iu] = t.min[1] as i16; self.z[iu] = t.min[2] as i16;
        self.sx[iu] = (t.max[0] - t.min[0]) as u8; self.sy[iu] = (t.max[1] - t.min[1]) as u8; self.sz[iu] = (t.max[2] - t.min[2]) as u8;
        self.index(i);
        self.log.push(i);
    }
    /// Cut a long free piece across its length into segments of at most `max` m (a brace bends as boxes' cells do:
    /// the structure solver has one node per piece). Its length is its longest extent along one of its edges'
    /// directions. Returns the segments, `i` first, in order along the length.
    pub fn segment(&mut self, i: u32, max: f64) -> Vec<u32> {
        let mut out = alloc::vec![i];
        let Some(p) = self.piece(i) else { return out };
        let w = p.world();
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (mut best, mut dir, mut lo) = (0.0, [1.0, 0.0, 0.0], 0.0);
        for f in p.faces { for k in 0..f.len() {
            let (a, b) = (w[f[k] as usize], w[f[(k + 1) % f.len()] as usize]);
            let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; let l = crate::math::sqrt64(dot(e, e));
            if l < 1e-9 { continue; }
            let e = [e[0] / l, e[1] / l, e[2] / l];
            let (mn, mx) = w.iter().fold((f64::MAX, f64::MIN), |a, v| (a.0.min(dot(e, *v)), a.1.max(dot(e, *v))));
            if mx - mn > best + 1e-9 { best = mx - mn; dir = e; lo = mn; }
        } }
        // (corners are f32: an 8 m piece measures 8.0000001 m, and must make 4 segments of 2 m, not 5)
        let n = crate::math::ceil64(best / max - 1e-5) as usize;
        if n < 2 { return out; }
        // from the far end back: the piece left below each plane keeps `i`
        let mut segs: Vec<u32> = Vec::new();
        for k in (1..n).rev() {
            match self.cut_piece(i, dir, lo + best * k as f64 / n as f64) { Some(o) => segs.push(o), None => {} }
        }
        segs.reverse();
        out.extend(segs);
        out
    }
    /// the brick's free piece, if it is one
    #[inline] pub fn piece(&self, i: u32) -> Option<PieceRef<'_>> { if self.hull[i as usize] != HULL_FREE { return None; } self.piece_of.get(&i).and_then(|&k| self.piece_k(k)) }
    /// piece `k` of `pieces`, with its shape
    pub fn piece_k(&self, k: u32) -> Option<PieceRef<'_>> {
        let p = self.pieces.get(k as usize)?; let s = &self.shapes[p.s as usize];
        Some(PieceRef { c: p.c, q: p.q, v: &s.v, faces: &s.faces, vol: s.vol, cen: s.cen })
    }

    fn key_of(&self, i: u32) -> u64 {
        let b = self.tbox(i);
        let m = (0..3).map(|k| b.max[k] - b.min[k]).max().unwrap();
        let l = (0..LEVELS).find(|&l| lcell(l) >= m).unwrap_or(LEVELS - 1);
        let c = lcell(l);
        ckey(l, [b.min[0].div_euclid(c), b.min[1].div_euclid(c), b.min[2].div_euclid(c)])
    }
    /// take brick `i` out of its cell's list in the spatial index
    fn unindex(&mut self, i: u32) {
        let key = self.key_of(i);
        let Some(&head) = self.cells.get(&key) else { return };
        if head == i { self.cells.insert(key, self.cnext[i as usize]); }
        else {
            let mut k = head;
            while k != NONE && self.cnext[k as usize] != i { k = self.cnext[k as usize]; }
            if k != NONE { self.cnext[k as usize] = self.cnext[i as usize]; }
        }
        self.cnext[i as usize] = NONE;
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
            // (loose things on the ground are not founded on it: they come off as bodies too)
            if self.y[i as usize] <= 0 && self.flags[i as usize] & F_LOOSE == 0 { mark[i as usize] = 1; stack.push(i); }
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
        if self.flags[iu] & F_ALIVE == 0 || self.flags[iu] & F_FRAC != 0 || (self.hull[iu] != 0 && self.hull[iu] != HULL_FREE) || self.mat[iu] == mat::GLASS { return; }
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
    #[inline] pub fn cell(&self, i: u32) -> Option<&'static crate::hull::Cell> { let h = self.hull[i as usize]; if h == 0 || h == HULL_FREE { None } else { Some(&crate::hull::lib().cells[h as usize]) } }
    /// centre and half size of the brick's box (m)
    pub fn box_m(&self, i: u32) -> ([f64; 3], [f64; 3]) {
        let t = self.tbox(i); let k = TICK as f64;
        ([0.5 * (t.min[0] + t.max[0]) as f64 * k, 0.5 * (t.min[1] + t.max[1]) as f64 * k, 0.5 * (t.min[2] + t.max[2]) as f64 * k],
         [0.5 * (t.max[0] - t.min[0]) as f64 * k, 0.5 * (t.max[1] - t.min[1]) as f64 * k, 0.5 * (t.max[2] - t.min[2]) as f64 * k])
    }
    /// the brick's share of its box's volume (1 for a whole box)
    #[inline] pub fn vol_share(&self, i: u32) -> f64 {
        if let Some(p) = self.piece(i) { let s = self.size(i); let k = TICK as f64; return p.vol / (s[0] as f64 * s[1] as f64 * s[2] as f64 * k * k * k); }
        self.cell(i).map_or(1.0, |c| c.vol)
    }
    /// the brick's volume (m^3)
    pub fn volume(&self, i: u32) -> f64 { if let Some(p) = self.piece(i) { return p.vol; } let s = self.size(i); let k = TICK as f64; s[0] as f64 * s[1] as f64 * s[2] as f64 * k * k * k * self.vol_share(i) }
    /// The brick's corners and faces in the world (m): a box's 8 corners, a cell mapped into its box, a free piece
    /// turned and placed.
    pub fn shape(&self, i: u32) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
        if let Some(p) = self.piece(i) { return (p.world(), p.faces.iter().map(|f| f.iter().map(|&k| k as usize).collect()).collect()); }
        let (c, h) = self.box_m(i);
        let cl = self.cell(i).unwrap_or(&crate::hull::lib().cells[0]);
        (cl.v.iter().map(|p| [c[0] + h[0] * p[0], c[1] + h[1] * p[1], c[2] + h[2] * p[2]]).collect(), cl.faces.iter().map(|f| f.v.iter().map(|&k| k as usize).collect()).collect())
    }
    /// The brick's inertia tensor about its centroid in the world, per unit density (m^5): times the density, kg m^2.
    pub fn inertia(&self, i: u32) -> [[f64; 3]; 3] { let (v, f) = self.shape(i); crate::hull::moments(&v, &f).2 }
    /// the brick's centroid (m)
    pub fn centroid(&self, i: u32) -> [f64; 3] {
        if let Some(p) = self.piece(i) { let c = p.centre(); let r = qrot(p.q, p.cen); return [c[0] + r[0], c[1] + r[1], c[2] + r[2]]; }
        let (c, h) = self.box_m(i);
        match self.cell(i) { None => c, Some(cl) => [c[0] + h[0] * cl.cen[0], c[1] + h[1] * cl.cen[1], c[2] + h[2] * cl.cen[2]] }
    }
    /// The brick's polygon on side `side` of its box (0..6: +x, -x, +y, -y, +z, -z), in m, counter-clockwise seen
    /// from outside; none if it is a cell that does not reach that side.
    pub fn side_poly(&self, i: u32, side: u8) -> Option<Vec<[f64; 3]>> {
        if self.hull[i as usize] == HULL_FREE { return None; }
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
        // loose things (furniture) join only the parts of the same object (the same `structure`), and nothing else
        let (fp, fq) = (self.flags[p as usize] & F_LOOSE, self.flags[q as usize] & F_LOOSE);
        if fp | fq != 0 && !(fp != 0 && fq != 0 && self.structure[p as usize] == self.structure[q as usize]) { return None; }
        // a free piece is not its box: it joins where its faces meet the other's flat (p is then the joint's first
        // brick, the face's normal from p to q)
        if self.hull[p as usize] == HULL_FREE || self.hull[q as usize] == HULL_FREE {
            let poly = self.free_face(p, q)?;
            return Some((JFREE, area_ticks(crate::hull::polygon(&poly).0), true));
        }
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
    /// Where p and q meet flat: of the faces of p and q that face each other within 1 degree with planes within 5 mm,
    /// the largest overlap (q's face projected onto p's plane, cut to p's face), at least 1 cm^2. In m, on p's face,
    /// counter-clockwise about p's outward normal (from p to q). Any two shapes (boxes, cells, free pieces).
    pub fn free_face(&self, p: u32, q: u32) -> Option<Vec<[f64; 3]>> {
        let (vp, fp) = self.shape(p); let (vq, fq) = self.shape(q);
        let polys = |v: &[[f64; 3]], f: &[Vec<usize>]| -> Vec<(Vec<[f64; 3]>, [f64; 3], [f64; 3])> {
            f.iter().map(|l| { let pl: Vec<[f64; 3]> = l.iter().map(|&k| v[k]).collect(); let (_, c, n) = crate::hull::polygon(&pl); (pl, c, n) }).collect()
        };
        let (pp, qq) = (polys(&vp, &fp), polys(&vq, &fq));
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let mut best: Option<(f64, Vec<[f64; 3]>)> = None;
        for (pa, ca, na) in &pp {
            for (pb, cb, nb) in &qq {
                if dot(*na, *nb) > -JOIN_COS { continue; }
                let gap = dot(*na, [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]]);
                if gap.abs() > JOIN_GAP { continue; }
                // q's face onto p's plane
                let proj: Vec<[f64; 3]> = pb.iter().map(|x| { let e = dot(*na, [x[0] - ca[0], x[1] - ca[1], x[2] - ca[2]]); [x[0] - e * na[0], x[1] - e * na[1], x[2] - e * na[2]] }).collect();
                let ov = crate::hull::intersect(pa, &proj, *na);
                let a = crate::hull::polygon(&ov).0;
                if a >= JOIN_MIN_AREA && best.as_ref().map_or(true, |b| a > b.0) { best = Some((a, ov)); }
            }
        }
        best.map(|b| b.1)
    }
    /// The face of joint j (m), counter-clockwise about the normal from ja to jb.
    pub fn joint_poly(&self, j: u32) -> Vec<[f64; 3]> {
        let ju = j as usize;
        let (a, b) = (self.ja[ju], self.jb[ju]);
        let ax = self.jaxis[ju] as usize;
        if ax == JFREE as usize { return self.free_face(a, b).unwrap_or_default(); }
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
        if self.hull[iu] == HULL_FREE && self.flags[iu] & F_ALIVE != 0 { return self.shatter_piece(i); }
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

    /// Shatter a free piece on a fracture surface: two cuts through (near) its centroid, the first across its
    /// longest extent, the second across the first, at angles and offsets from a hash of the brick (as the boxes'
    /// patterns), into up to four convex parts; joints as `cut_piece`. Returns the parts and, of them, the one or
    /// two that spall off (`frac_cells`' mask).
    pub fn shatter_piece(&mut self, i: u32) -> (Vec<u32>, Vec<u32>) {
        let h = i.wrapping_mul(2654435761).rotate_right(7) ^ 0x9e37_79b9;
        let u = |k: u32| ((h.rotate_left(k * 5) & 0xffff) as f64 / 65535.0) * 2.0 - 1.0;   // in -1..1
        let w = match self.piece(i) { Some(p) => p.world(), None => return (Vec::new(), Vec::new()) };
        let c = self.centroid(i);
        // the longest extent, among the corners' pairs
        let mut ax = [1.0, 0.0, 0.0]; let mut best = 0.0;
        for a in 0..w.len() { for b in a + 1..w.len() {
            let d = [w[b][0] - w[a][0], w[b][1] - w[a][1], w[b][2] - w[a][2]]; let l = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if l > best { best = l; ax = d; }
        } }
        let nrm = |v: [f64; 3]| { let l = crate::math::sqrt64(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).max(1e-12); [v[0] / l, v[1] / l, v[2] / l] };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let ext = |n: [f64; 3]| { let (lo, hi) = w.iter().fold((f64::MAX, f64::MIN), |a, v| (a.0.min(dot(n, *v)), a.1.max(dot(n, *v)))); hi - lo };
        // across the longest extent, tilted up to about 25 degrees
        let n1 = nrm([nrm(ax)[0] + 0.45 * u(1), nrm(ax)[1] + 0.45 * u(2), nrm(ax)[2] + 0.45 * u(3)]);
        // the second across the first: a direction at right angles to n1
        let r = [u(4), u(5), u(6)];
        let pr = dot(r, n1);
        let mut n2 = [r[0] - pr * n1[0], r[1] - pr * n1[1], r[2] - pr * n1[2]];
        if dot(n2, n2) < 1e-6 { n2 = if n1[0].abs() < 0.9 { [0.0, -n1[2], n1[1]] } else { [-n1[2], 0.0, n1[0]] }; }
        let n2 = nrm(n2);
        let mut parts = alloc::vec![i];
        if let Some(o) = self.cut_piece(i, n1, dot(n1, c) + 0.15 * u(7) * ext(n1)) { parts.push(o); }
        let first = parts.clone();
        for &p in &first { if let Some(o) = self.cut_piece(p, n2, dot(n2, c) + 0.15 * u(8) * ext(n2)) { parts.push(o); } }
        let (_, mask) = frac_cells(i, parts.len() as u32);
        let spall = (0..parts.len()).filter(|&k| mask & (1 << k) != 0).map(|k| parts[k]).collect();
        (parts, spall)
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

    /// Cut brick `i` at tick plane `at` across `axis`: a box (or cell) is `split`, a free piece is cut by the plane
    /// itself (`cut_piece`). `i` keeps the low side; the new brick is returned, or none if the plane leaves a free
    /// piece whole (it misses it, or a part would be under 1 cm).
    pub fn cut(&mut self, i: u32, axis: usize, at: i32) -> Option<u32> {
        if self.hull[i as usize] == HULL_FREE { let mut n = [0.0; 3]; n[axis] = 1.0; return self.cut_piece(i, n, at as f64 * TICK as f64); }
        Some(self.split(i, axis, at))
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
            // (a convex cell is as small as fracture makes bricks: it is not split. A free piece is cut by the plane,
            // as boxes are: left whole, a blast on a brace took a few crumbs off the boxes round it and none off the
            // brace, its 1.5 m segments' centres outside the 1.5 m blast)
            let h = self.hull[b as usize];
            if !sphere_box(c, r, &tb) || (h != 0 && h != HULL_FREE) { continue; }
            let s = self.size(b);
            let mut ax = 3; let mut best = target;
            for k in 0..3 { if s[k] > best { best = s[k]; ax = k; } }
            if ax == 3 { continue; }
            let at = self.split_point(b, ax);
            let Some(n) = self.cut(b, ax, at) else { continue };
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
    /// Approximate bytes held by the free pieces' side table: per piece its centre, rotation and shape (32 B) and its
    /// entry in `piece_of` (~10 B); per shape (kept once) its corners (12 B each), face loops (a byte a corner, 24 B a
    /// list), the lists' headers (48 B), volume and centroid (32 B) and its entry in the index (~36 B).
    pub fn piece_bytes(&self) -> usize {
        self.pieces.len() * (12 + 16 + 4 + 10)
            + self.shapes.iter().map(|p| 12 * p.v.len() + p.faces.iter().map(|f| 24 + f.len()).sum::<usize>() + 48 + 32 + 36).sum::<usize>()
    }
    /// Everything per live brick: records, joints, the spatial index and the free pieces' side table.
    pub fn total_bytes_per_brick(&self) -> f32 { self.bytes_per_brick() + (self.index_bytes() + self.piece_bytes()) as f32 / self.alive_count().max(1) as f32 }

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
/// free pieces (docs/PHASE2.md, step 1)
#[cfg(test)]
mod piece_tests {
    use super::*;
    extern crate std;
    fn rmat(q: [f32; 4]) -> [[f64; 3]; 3] {
        let c = |e: [f64; 3]| qrot(q, e);
        let (x, y, z) = (c([1.0, 0.0, 0.0]), c([0.0, 1.0, 0.0]), c([0.0, 0.0, 1.0]));
        [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    }
    /// R diag(d) R^T
    fn turned(r: &[[f64; 3]; 3], d: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 { for j in 0..3 { for k in 0..3 { for l in 0..3 { o[i][j] += r[i][k] * d[k][l] * r[j][l]; } } } }
        o
    }
    fn close(a: [[f64; 3]; 3], b: [[f64; 3]; 3], tol: f64) -> bool { (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() <= tol)) }
    fn quat(axis: [f64; 3], ang: f64) -> [f32; 4] {
        let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt(); let (s, c) = ((ang / 2.0).sin(), (ang / 2.0).cos());
        [(axis[0] / l * s) as f32, (axis[1] / l * s) as f32, (axis[2] / l * s) as f32, c as f32]
    }
    fn box_pts(h: [f64; 3]) -> std::vec::Vec<[f64; 3]> {
        let mut v = std::vec::Vec::new();
        for &a in &[-1.0, 1.0] { for &b in &[-1.0, 1.0] { for &c in &[-1.0, 1.0] { v.push([a * h[0], b * h[1], c * h[2]]); } } }
        v
    }
    #[test] fn a_rotated_box_piece_has_the_mass_properties_of_its_box() {
        // a 2 x 1 x 0.5 m box turned 0.7 rad about (1, 2, 3): volume 1 m^3, centroid at its centre (snapped to 1 mm),
        // inertia R diag((b^2+c^2)/12, (a^2+c^2)/12, (a^2+b^2)/12) R^T per unit density
        let mut b = Bricks::new();
        let q = quat([1.0, 2.0, 3.0], 0.7);
        let i = b.add_piece([3.2, 4.0011, -1.5], q, &box_pts([1.0, 0.5, 0.25]), mat::CONCRETE, 0, 0).unwrap();
        let p = b.piece(i).unwrap();
        assert_eq!(p.c, [3200, 4001, -1500], "centre on the 1 mm grid");
        assert_eq!((p.v.len(), p.faces.len()), (8, 6));
        assert!((b.volume(i) - 1.0).abs() < 1e-6, "volume {}", b.volume(i));
        let c = b.centroid(i);
        assert!((c[0] - 3.2).abs() < 1e-6 && (c[1] - 4.001).abs() < 1e-6 && (c[2] + 1.5).abs() < 1e-6, "centroid {c:?}");
        let want = turned(&rmat(q), [[1.25 / 12.0, 0.0, 0.0], [0.0, 4.25 / 12.0, 0.0], [0.0, 0.0, 5.0 / 12.0]]);
        let got = b.inertia(i);
        assert!(close(got, want, 1e-6), "inertia {got:?} want {want:?}");
        // its tick box holds every corner and is the tight box rounded out to ticks
        let t = b.tbox(i);
        for w in b.piece(i).unwrap().world() { for k in 0..3 {
            assert!(w[k] >= t.min[k] as f64 * TICK as f64 - 1e-9 && w[k] <= t.max[k] as f64 * TICK as f64 + 1e-9);
        } }
        for k in 0..3 {
            let ws = b.piece(i).unwrap().world();
            let (lo, hi) = ws.iter().fold((f64::MAX, f64::MIN), |a, w| (a.0.min(w[k]), a.1.max(w[k])));
            assert!(lo - t.min[k] as f64 * 0.125 < 0.125 && t.max[k] as f64 * 0.125 - hi < 0.125);
        }
        // a box brick's own inertia by the same path: a 1 x 0.5 x 2 m box
        let j = b.add([0, 0, 0], [8, 4, 16], mat::CONCRETE, 0, 0);
        assert!(close(b.inertia(j), [[(0.25 + 4.0) / 12.0, 0.0, 0.0], [0.0, 5.0 / 12.0, 0.0], [0.0, 0.0, 1.25 / 12.0]], 1e-9));
    }
    #[test] fn a_wedge_piece_matches_the_closed_form() {
        // a right triangular prism: legs a = 1.2 m along x and b = 0.6 m along y, length L = 2 m along z.
        // V = abL/2; centroid (a/3, b/3, L/2); about it, with A = ab/2: Ixx = L ab^3/36 + A L^3/12,
        // Iyy = L ba^3/36 + A L^3/12, Izz = L (ab^3 + ba^3)/36, Ixy = -L * (integral of xy dA = -a^2 b^2/72)
        let (a, bb, l) = (1.2f64, 0.6f64, 2.0f64);
        let pts = [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [0.0, bb, 0.0], [0.0, 0.0, l], [a, 0.0, l], [0.0, bb, l]];
        let mut b = Bricks::new();
        let i = b.add_piece([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], &pts, mat::MASONRY, 0, 0).unwrap();
        assert_eq!((b.piece(i).unwrap().v.len(), b.piece(i).unwrap().faces.len()), (6, 5));
        let ar = a * bb / 2.0;
        assert!((b.volume(i) - ar * l).abs() < 1e-6);
        let c = b.centroid(i);
        assert!((c[0] - a / 3.0).abs() < 1e-6 && (c[1] - bb / 3.0).abs() < 1e-6 && (c[2] - l / 2.0).abs() < 1e-6, "centroid {c:?}");
        let want = [[l * a * bb.powi(3) / 36.0 + ar * l.powi(3) / 12.0, l * a * a * bb * bb / 72.0, 0.0],
                    [l * a * a * bb * bb / 72.0, l * bb * a.powi(3) / 36.0 + ar * l.powi(3) / 12.0, 0.0],
                    [0.0, 0.0, l * (a * bb.powi(3) + bb * a.powi(3)) / 36.0]];
        assert!(close(b.inertia(i), want, 1e-6), "inertia {:?} want {want:?}", b.inertia(i));
        // the same wedge turned 90 degrees about x (its length now along -y): the tensor turns with it
        let q = quat([1.0, 0.0, 0.0], core::f64::consts::FRAC_PI_2);
        let k = b.add_piece([5.0, 2.0, 0.0], q, &pts, mat::MASONRY, 0, 0).unwrap();
        assert!(close(b.inertia(k), turned(&rmat(q), want), 1e-6));
        assert!((b.volume(k) - ar * l).abs() < 1e-6);
    }
    #[test] fn a_hexagonal_prism_piece_matches_the_closed_form() {
        // a regular hexagon of side s = 0.4 m, height H = 1.5 m along y: V = 3 sqrt(3)/2 s^2 H, Iyy = V 5 s^2 / 12,
        // Ixx = Izz = V (5 s^2 / 24 + H^2 / 12)
        let (sd, h) = (0.4f64, 1.5f64);
        let mut pts = std::vec::Vec::new();
        for k in 0..6 { let t = k as f64 * core::f64::consts::PI / 3.0; for &y in &[-h / 2.0, h / 2.0] { pts.push([sd * t.cos(), y, sd * t.sin()]); } }
        let mut b = Bricks::new();
        let i = b.add_piece([0.0, 0.75, 0.0], [0.0, 0.0, 0.0, 1.0], &pts, mat::CONCRETE, 0, 0).unwrap();
        let p = b.piece(i).unwrap();
        assert_eq!((p.v.len(), p.faces.len()), (12, 8));
        let v = 1.5 * 3f64.sqrt() * sd * sd * h;
        assert!((b.volume(i) - v).abs() < 1e-6);
        let side = v * (5.0 * sd * sd / 24.0 + h * h / 12.0);
        assert!(close(b.inertia(i), [[side, 0.0, 0.0], [0.0, v * 5.0 * sd * sd / 12.0, 0.0], [0.0, 0.0, side]], 1e-6), "{:?}", b.inertia(i));
        // its mass is its volume times the density, not its box's
        assert!((crate::structure::mass(&b, i) as f64 - v * 2400.0).abs() < 1e-2 * v * 2400.0 * 1e-2, "mass {}", crate::structure::mass(&b, i));
    }
    #[test] fn pieces_beyond_the_hull_limits_or_flat_are_refused() {
        let mut b = Bricks::new();
        // a 20-sided prism: 40 corners (the GPU's hull holds 16)
        let mut pts = std::vec::Vec::new();
        for k in 0..20 { let t = k as f64 * core::f64::consts::PI / 10.0; for &y in &[0.0, 1.0] { pts.push([t.cos(), y, t.sin()]); } }
        assert!(b.add_piece([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], &pts, mat::CONCRETE, 0, 0).is_none());
        // flat: four points in one plane
        assert!(b.add_piece([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], mat::CONCRETE, 0, 0).is_none());
        // too long for a brick's box (40 m)
        assert!(b.add_piece([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], &box_pts([20.0, 0.5, 0.5]), mat::CONCRETE, 0, 0).is_none());
        assert_eq!(b.len(), 0);
    }
    #[test] fn a_free_piece_joins_by_its_faces_not_its_box() {
        // a 1 m cube on the ground and, on it, a 1 m cube turned 45 degrees about y: their tick boxes share a face
        // (y = 1 m) of 1 m^2, but the joint is where the faces overlap: the square and the same square turned 45
        // degrees about its centre share an octagon of 2 (sqrt 2 - 1) = 0.8284 m^2. Before step 2 a free piece
        // joined nothing (0 joints); a box's joint would have been 1 m^2. It is drawn as its own six faces, its mesh
        // encloses its volume, and it survives a save and load.
        let mut b = Bricks::new();
        let base = b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
        let q = quat([0.0, 1.0, 0.0], core::f64::consts::FRAC_PI_4);
        let top = b.add_piece([0.5, 1.5, 0.5], q, &box_pts([0.5, 0.5, 0.5]), mat::CONCRETE, 3, 0).unwrap();
        assert_eq!(b.tbox(top).min[1], 8);
        b.build_joints();
        assert_eq!(b.live_joints, 1);
        let j = b.jhead[base as usize];
        assert_eq!(b.jaxis[j as usize], JFREE);
        let (a, _, n) = crate::hull::polygon(&b.joint_poly(j));
        assert!((a - 2.0 * (2f64.sqrt() - 1.0)).abs() < 1e-6, "joint area {a}");
        // its normal runs from ja to jb
        let up = if b.ja[j as usize] == base { 1.0 } else { -1.0 };
        assert!((n[1] - up).abs() < 1e-9, "normal {n:?}");
        assert!(b.detached().is_empty(), "the turned cube stands on the base");
        let m = crate::mesh::build(&b);
        // the base: 5 faces (its bottom is on the ground); the piece: 6
        assert_eq!(m.quads, 11);
        // the piece's triangles (the last 12) enclose its volume: sum of p0 . (p1 x p2) / 6
        let tri = &m.i[m.i.len() - 36..];
        let pt = |k: u32| { let o = k as usize * 9; [m.v[o] as f64, m.v[o + 1] as f64, m.v[o + 2] as f64] };
        let mut vol = 0.0;
        for t in tri.chunks(3) { let (a, c, d) = (pt(t[0]), pt(t[1]), pt(t[2])); vol += (a[0] * (c[1] * d[2] - c[2] * d[1]) - a[1] * (c[0] * d[2] - c[2] * d[0]) + a[2] * (c[0] * d[1] - c[1] * d[0])) / 6.0; }
        assert!((vol - 1.0).abs() < 1e-5, "mesh volume {vol}: faces wound outward");
        let mut buf = std::vec::Vec::new(); b.save(&mut buf);
        let c = Bricks::load(&buf);
        assert_eq!(c.pieces, b.pieces);
        assert_eq!(c.shapes, b.shapes);
        assert_eq!(c.piece_of.get(&top), Some(&0));
        assert_eq!(c.centroid(top), b.centroid(top));
        assert_eq!(c.hull[top as usize], HULL_FREE);
        assert_eq!(c.joint_set(), b.joint_set());
    }

    // ---- joints of free pieces (docs/PHASE2.md, step 2) ----
    fn area_of(b: &Bricks, j: u32) -> f64 { crate::hull::polygon(&b.joint_poly(j)).0 }
    /// a piece from world points: centre at their mean, not turned
    fn piece_at(b: &mut Bricks, pts: &[[f64; 3]], m: u8) -> u32 {
        let n = pts.len() as f64;
        let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
        let cs = [(c[0] * 1000.0).round() / 1000.0, (c[1] * 1000.0).round() / 1000.0, (c[2] * 1000.0).round() / 1000.0];
        let local: std::vec::Vec<[f64; 3]> = pts.iter().map(|p| [p[0] - cs[0], p[1] - cs[1], p[2] - cs[2]]).collect();
        b.add_piece(cs, [0.0, 0.0, 0.0, 1.0], &local, m, 0, 0).unwrap()
    }
    #[test] fn a_turned_block_on_a_slab_joins_on_its_whole_face() {
        // a 1 m cube turned 30 degrees about y, wholly on a 2 x 1 x 2 m block: the joint is the cube's 1 m^2 bottom
        let mut b = Bricks::new();
        b.add([0, 0, 0], [16, 8, 16], mat::CONCRETE, 0, 0);
        b.add_piece([1.0, 1.5, 1.0], quat([0.0, 1.0, 0.0], 0.5236), &box_pts([0.5, 0.5, 0.5]), mat::CONCRETE, 0, 0).unwrap();
        b.build_joints();
        assert_eq!(b.live_joints, 1);
        assert!((area_of(&b, 0) - 1.0).abs() < 1e-6, "{}", area_of(&b, 0));
    }
    #[test] fn two_wedges_join_on_the_diagonal() {
        // a 1 m cube cut on its diagonal plane x = y into two wedges, both on the ground: one joint, the diagonal
        // face, sqrt 2 x 1 = 1.4142 m^2, its normal along (1, -1, 0) / sqrt 2
        let mut b = Bricks::new();
        let lo = piece_at(&mut b, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], mat::MASONRY);
        let _hi = piece_at(&mut b, &[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], mat::MASONRY);
        b.build_joints();
        assert_eq!(b.live_joints, 1);
        assert!((area_of(&b, 0) - 2f64.sqrt()).abs() < 1e-6, "{}", area_of(&b, 0));
        let n = crate::hull::polygon(&b.joint_poly(0)).2;
        let s = if b.ja[0] == lo { 1.0 } else { -1.0 };
        let r = 0.5f64.sqrt();
        assert!((n[0] + s * r).abs() < 1e-9 && (n[1] - s * r).abs() < 1e-9, "normal {n:?}");
        assert_eq!(b.detached().len(), 0);
    }
    #[test] fn a_brace_end_joins_a_column_face() {
        // a 0.5 m square column, 3 m tall; a 0.2 x 0.2 m brace at 45 degrees, its end cut vertical against the
        // column's face x = 0.5 m: the joint is the end, a 0.2 / cos 45 x 0.2 parallelogram, 0.05657 m^2, normal +x
        let mut b = Bricks::new();
        let col = b.add([0, 0, 0], [4, 24, 4], mat::STEEL, 0, 0);
        let (c45, l) = (0.5f64.sqrt(), 2.0f64);
        let mut pts = std::vec::Vec::new();
        for &sd in &[-0.1, 0.1] { for &z in &[0.15, 0.35] {
            let p = [0.5, 1.5 + sd / c45, z];
            pts.push(p); pts.push([p[0] + l * c45, p[1] + l * c45, z]);
        } }
        let brace = piece_at(&mut b, &pts, mat::STEEL);
        b.build_joints();
        assert_eq!(b.live_joints, 1);
        let j = b.jhead[brace as usize];
        assert!((area_of(&b, j) - 0.04 / c45).abs() < 1e-6, "{}", area_of(&b, j));
        let n = crate::hull::polygon(&b.joint_poly(j)).2;
        let s = if b.ja[j as usize] == col { 1.0 } else { -1.0 };
        assert!((n[0] - s).abs() < 1e-9, "normal {n:?}");
    }
    #[test] fn faces_join_within_one_degree_and_five_millimetres() {
        // a 1 m cube on a 2 m block: tilted 0.5 degrees it joins, 1.5 degrees not; lifted 4 mm it joins, 6 mm not
        let joins = |tilt_deg: f64, lift: f64| {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [16, 8, 16], mat::CONCRETE, 0, 0);
            let t = tilt_deg.to_radians();
            b.add_piece([1.0, 1.0 + 0.5 * t.cos() + lift, 1.0], quat([1.0, 0.0, 0.0], t), &box_pts([0.5, 0.5, 0.5]), mat::CONCRETE, 0, 0).unwrap();
            b.build_joints();
            b.live_joints
        };
        assert_eq!(joins(0.5, 0.0), 1);
        assert_eq!(joins(1.5, 0.0), 0);
        assert_eq!(joins(0.0, 0.004), 1);
        assert_eq!(joins(0.0, 0.006), 0);
    }
    #[test] fn a_turned_block_bears_on_a_slab_through_its_joint() {
        // a 1 m cube turned 30 degrees about y with 3 MN on it, centred on the line between two of a 4 m slab's
        // cells (the 4 x 0.5 x 2 m slab, on the ground, is cut into 2 m cells): the joint is split between the two cells, and summed
        // back about its centre it is pure compression, (weight + 3 MN) / 1 m^2 against concrete's 30 MPa
        let mut b = Bricks::new();
        b.add([0, 0, 0], [32, 4, 16], mat::CONCRETE, 0, 0);
        let top = b.add_piece([2.0, 1.0, 1.0], quat([0.0, 1.0, 0.0], 0.5236), &box_pts([0.5, 0.5, 0.5]), mat::CONCRETE, 0, 0).unwrap();
        b.build_joints();
        assert_eq!(b.live_joints, 1);
        let set: std::vec::Vec<u32> = (0..b.len() as u32).collect();
        let w = crate::structure::mass(&b, top) as f64 * 9.81;
        let load: std::vec::Vec<f32> = set.iter().map(|&i| if i == top { (w + 3.0e6) as f32 } else { 0.0 }).collect();
        let sink: std::vec::Vec<bool> = set.iter().map(|&i| i != top).collect();
        let a = crate::structure::analyse(&b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
        assert_eq!(a.nodes, 3, "two slab cells and the cube");
        assert_eq!(a.ratios.len(), 1);
        let want = (w + 3.0e6) / 1.0 / 30e6;
        let got = a.ratios.iter().find(|r| r.0 == 0).unwrap().1 as f64;
        std::println!("turned block on a slab: ratio {got:.5}, pure compression {want:.5}");
        assert!((got - want).abs() < 0.01 * want, "ratio {got} want {want}");
    }

    // ---- the structure over free pieces (docs/PHASE2.md, step 3) ----
    #[test] fn a_cut_piece_keeps_its_volume_and_is_joined_on_the_cut() {
        // a 3 m long 0.3 m square bar turned 20 degrees about z, cut by a plane across it through its centre: two
        // pieces of half the volume each, joined on the cut (the bar's section, 0.09 m^2)
        let mut b = Bricks::new();
        let q = quat([0.0, 0.0, 1.0], 0.349);
        let i = b.add_piece([0.0, 5.0, 0.0], q, &box_pts([1.5, 0.15, 0.15]), mat::TIMBER, 0, 0).unwrap();
        let ax = qrot(q, [1.0, 0.0, 0.0]);
        let d = ax[0] * 0.0 + ax[1] * 5.0;
        let o = b.cut_piece(i, ax, d).unwrap();
        assert!((b.volume(i) - 0.135).abs() < 1e-6 && (b.volume(o) - 0.135).abs() < 1e-6, "{} {}", b.volume(i), b.volume(o));
        assert_eq!(b.live_joints, 1);
        let j = b.jhead[i as usize];
        assert!((area_of(&b, j) - 0.09).abs() < 1e-6, "{}", area_of(&b, j));
        // segmenting a 7 m bar: four pieces of 1.75 m, joined in a row
        let mut c = Bricks::new();
        let k = c.add_piece([0.0, 5.0, 0.0], q, &box_pts([3.5, 0.15, 0.15]), mat::TIMBER, 0, 0).unwrap();
        let segs = c.segment(k, 2.0);
        assert_eq!(segs.len(), 4);
        for &s in &segs { assert!((c.volume(s) - 0.63 / 4.0).abs() < 1e-6, "{}", c.volume(s)); }
        assert_eq!(c.live_joints, 3);
        let cs: std::vec::Vec<f64> = segs.iter().map(|&s| { let p = c.centroid(s); p[0] * ax[0] + p[1] * ax[1] }).collect();
        // in order along the bar (whichever way its length was measured)
        for w in cs.windows(2) { assert!(((w[1] - w[0]).abs() - 1.75).abs() < 1e-6 && (w[1] - w[0]) * (cs[1] - cs[0]) > 0.0, "{cs:?}"); }
    }
    /// An A-frame: two 0.2 m square steel struts at `theta` from the ground, 4 m long, their feet cut level on
    /// ground blocks, their heads cut vertical against a 0.5 m block between them that carries `p` (no self weight).
    /// Returns the store, the struts' segments (left, right) and the block.
    fn a_frame(theta: f64) -> (Bricks, std::vec::Vec<u32>, std::vec::Vec<u32>, u32) { let (b, l, r, t, _) = a_frame_tied(theta, None, false); (b, l, r, t) }
    /// The A-frame below, with a 0.1 m deep tie between its struts at `tie` m above their feet (ends cut to the
    /// struts' inner faces), its feet on 3 m concrete piers (0.375 x 0.25 m, 12 bricks, on footings below the
    /// ground line) if `piers`. Also returns the tie.
    fn a_frame_tied(theta: f64, tie: Option<f64>, piers: bool) -> (Bricks, std::vec::Vec<u32>, std::vec::Vec<u32>, u32, u32) {
        let mut b = Bricks::new();
        let (c, s) = (theta.cos(), theta.sin());
        let la = 4.0;
        let (xt, yf) = (0.25, if piers { 3.0 } else { 0.5 });
        let yt = yf + la * s;
        let top = b.add([-2, crate::math::floor64((yt - 0.5) / 0.125) as i32, 0], [4, 8, 2], mat::STEEL, 0, 0);
        let mut sides = std::vec::Vec::new();
        for &side in &[-1.0f64, 1.0] {
            // the strut's axis from its foot (x = side (xt + la c), y = yf) up and in to (side xt, yt)
            let d = [-side * c, s, 0.0];
            let nrm = [s, side * c, 0.0];
            let foot = [side * (xt + la * c), yf, 0.0];
            let mut pts = std::vec::Vec::new();
            for &o in &[-0.1, 0.1] { for &z in &[0.025, 0.225] {
                let e = [foot[0] + o * nrm[0], foot[1] + o * nrm[1]];
                // along the edge e + t d: level at y = yf, vertical at x = side xt
                let t0 = (yf - e[1]) / d[1]; let t1 = (side * xt - e[0]) / d[0];
                pts.push([e[0] + t0 * d[0], yf, z]); pts.push([side * xt, e[1] + t1 * d[1], z]);
            } }
            let k = piece_at(&mut b, &pts, mat::STEEL);
            sides.push(b.segment(k, 2.0));
            let fx = crate::math::floor64(foot[0] / 0.125) as i32;
            if piers {
                b.add([fx - 4, -4, -2], [11, 4, 6], mat::CONCRETE, 0, 0);
                for k in 0..12 { b.add([fx - 1, 2 * k, 0], [3, 2, 2], mat::CONCRETE, 0, 0); }
            } else { b.add([fx - 8, 0, 0], [16, 4, 2], mat::STEEL, 0, 0); }
        }
        let mut tid = NONE;
        if let Some(e) = tie {
            // the struts' inner faces: (p - foot) . (s, -c) = 0.1 on the left, mirrored on the right
            let fx = xt + la * c;
            let xin = |y: f64| -fx + (0.1 + (y - yf) * c) / s;
            let mut pts = std::vec::Vec::new();
            for &y in &[yf + e - 0.05, yf + e + 0.05] { for &z in &[0.025, 0.225] { pts.push([xin(y), y, z]); pts.push([-xin(y), y, z]); } }
            tid = piece_at(&mut b, &pts, mat::STEEL);
        }
        b.build_joints();
        let r = sides.pop().unwrap(); let l = sides.pop().unwrap();
        (b, l, r, top, tid)
    }
    #[test] fn an_a_frame_carries_its_load_as_struts() {
        // A load P on the head of an A-frame of struts at theta to the ground goes down the struts: each carries
        // N = P / (2 sin theta) along its length, and at its foot pushes down P / 2 and out H = P / (2 tan theta). The
        // joints resist bending too, but struts 20 times longer than deep carry almost all of it axially.
        for &deg in &[45.0f64, 60.0] {
            let th = deg.to_radians();
            let (b, l, r, top) = a_frame(th);
            // (a strut's longest edge is 4 m plus its cut ends: 3 segments of at most 2 m)
            assert_eq!((l.len(), r.len()), (3, 3));
            assert!(b.detached().is_empty());
            let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
            let p = 100e3;
            let load: std::vec::Vec<f32> = set.iter().map(|&i| if i == top { p as f32 } else { 0.0 }).collect();
            let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
            let a = crate::structure::analyse(&b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
            let force = |x: u32, y: u32| -> [f64; 3] {
                let j = { let mut j = b.jhead[x as usize]; while j != NONE && b.joint_other(j, x) != y { j = b.joint_next(j, x); } j };
                assert!(j != NONE, "joint {x}-{y}");
                a.forces.iter().find(|f| f.0 == j).unwrap().1
            };
            let n_want = p / (2.0 * th.sin());
            for seg in [&l, &r] {
                // the joints between its segments: across the strut, pure compression N
                for w in seg.windows(2) {
                    let f = force(w[0], w[1]);
                    let sh = (f[1] * f[1] + f[2] * f[2]).sqrt();
                    std::println!("A-frame at {deg} degrees: strut force {:.2} kN (P / 2 sin theta = {:.2}), shear across it {:.3} kN", -f[0] / 1e3, n_want / 1e3, sh / 1e3);
                    assert!((-f[0] - n_want).abs() < 0.02 * n_want, "strut force {} want {n_want}", -f[0]);
                    assert!(sh < 0.02 * n_want);
                }
                // the foot: on the ground block below its lower segment
                let foot = { let mut j = b.jhead[seg[0] as usize]; let mut g = NONE; while j != NONE { let o = b.joint_other(j, seg[0]); if b.y[o as usize] == 0 { g = o; } j = b.joint_next(j, seg[0]); } g };
                let f = force(seg[0], foot);
                let (v, h) = (f[0].abs(), (f[1] * f[1] + f[2] * f[2]).sqrt());
                std::println!("  foot: down {:.1} kN (P / 2 = {:.1}), out {:.1} kN (P / 2 tan theta = {:.1})", v / 1e3, p / 2e3, h / 1e3, p / (2e3 * th.tan()));
                assert!((v - p / 2.0).abs() < 0.02 * p / 2.0);
                assert!((h - p / (2.0 * th.tan())).abs() < 0.02 * p / (2.0 * th.tan()));
            }
        }
    }
    #[test] fn a_parabolic_arch_of_wedge_stones_thrusts_w_l2_over_8f() {
        // The thrust of a rigid funicular arch is H = w L^2 / 8 f. An elastic arch without hinges thrusts less: its
        // stones shorten under the compression, and the shortfall grows as I / (A f^2), the thickness squared (a fixed
        // parabolic arch: H = H0 / (1 + 45/4 I / (A f^2)), 0.94 H0 at 0.4 m; two-hinged 15/8 instead of 45/4, 0.99 H0).
        // Measured (12 stones): 143.8, 148.4 and 149.6 kN at 0.4, 0.2 and 0.1 m thick, against 150.0: shortfalls 6.2,
        // 1.6, 0.4 kN, a quarter at each halving; extrapolated to no thickness, (4 H(t/2) - H(t)) / 3 = 149.9 kN.
        let h0 = 50e3 * 36.0 / (8.0 * 1.5);
        let hs: std::vec::Vec<f64> = [0.4f64, 0.2, 0.1].iter().map(|&t| arch_thrust(t)).collect();
        for k in 0..2 {
            let r = (h0 - hs[k]) / (h0 - hs[k + 1]);
            assert!(r > 3.5 && r < 4.5, "shortfall ratio {r} at halving {k} (thrusts {hs:?})");
            let ex = (4.0 * hs[k + 1] - hs[k]) / 3.0;
            assert!((ex - h0).abs() < 0.005 * h0, "extrapolated thrust {ex}, w L^2 / 8 f = {h0}");
        }
        assert!((hs[2] - h0).abs() < 0.005 * h0, "0.1 m thick: {}", hs[2]);
    }
    /// The thrust at the crown of the arch below, `t` m thick (and checks its crown joint carries no shear).
    fn arch_thrust(t: f64) -> f64 { arch_run(t, None).0 }
    /// The arch (12 stones, L = 6, f = 1.5, `t` thick) under 50 kN/m, plus `extra` N on one stone if given:
    /// (crown thrust, per joint along the arch from the left foot: normal force, moment about the arch's axis,
    /// ratio).
    fn arch_run(t: f64, extra: Option<(usize, f64)>) -> (f64, std::vec::Vec<(f64, f64, f32)>) {
        // An arch of 12 wedge stones (voussoirs) on the parabola y = 4 f x (L - x) / L^2, span L = 6 m, rise f = 1.5 m,
        // t thick and 0.5 m deep, its joints cut square to the arch, springing from two sloped abutments on the
        // ground, under w = 50 kN per metre of span (each stone carries w times its share of the span; no self weight).
        // A parabola is the funicular of a uniform load, so the stones carry it in compression with little bending.
        // The joint at the crown is vertical: it carries the thrust as its normal force and nothing in shear.
        let (l, f, nst, w) = (6.0f64, 1.5f64, 12usize, 50e3f64);
        let y0 = 1.0;   // the springing, above the ground
        let at = |x: f64| -> ([f64; 3], [f64; 3]) {
            let y = y0 + 4.0 * f * x * (l - x) / (l * l);
            let sl = 4.0 * f * (l - 2.0 * x) / (l * l);
            let n = 1.0 / (1.0 + sl * sl).sqrt();
            ([x, y, 0.0], [-sl * n, n, 0.0])
        };
        let mut b = Bricks::new();
        let mut stones = std::vec::Vec::new();
        for k in 0..nst {
            let (x0, x1) = (l * k as f64 / nst as f64, l * (k + 1) as f64 / nst as f64);
            let mut pts = std::vec::Vec::new();
            for &x in &[x0, x1] { let (p, n) = at(x); for &o in &[-t / 2.0, t / 2.0] { for &z in &[0.0, 0.5] { pts.push([p[0] + o * n[0], p[1] + o * n[1], z]); } } }
            stones.push((piece_at(&mut b, &pts, mat::MASONRY), x1 - x0));
        }
        // the abutments: the springing face, swept back along the arch's tangent down to the ground
        for &x in &[0.0, l] {
            let (p, n) = at(x);
            let tg = if x == 0.0 { [n[1], -n[0]] } else { [-n[1], n[0]] };   // pointing away from the arch, downward
            let mut pts = std::vec::Vec::new();
            for &o in &[-t / 2.0, t / 2.0] { for &z in &[0.0, 0.5] {
                let c = [p[0] + o * n[0], p[1] + o * n[1]];
                let s = -c[1] / tg[1];
                pts.push([c[0], c[1], z]); pts.push([c[0] + s * tg[0], 0.0, z]);
            } }
            piece_at(&mut b, &pts, mat::MASONRY);
        }
        b.build_joints();
        assert_eq!(b.live_joints, nst + 1, "stone to stone and to the abutments");
        assert!(b.detached().is_empty());
        let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let load: std::vec::Vec<f32> = set.iter().map(|&i| stones.iter().position(|s| s.0 == i).map_or(0.0, |k| (w * stones[k].1 + extra.map_or(0.0, |e| if e.0 == k { e.1 } else { 0.0 })) as f32)).collect();
        let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
        let a = crate::structure::analyse(&b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
        let jb = |sa: u32, sb: u32| { let mut j = b.jhead[sa as usize]; while j != NONE && b.joint_other(j, sa) != sb { j = b.joint_next(j, sa); } j };
        let (sa, sb) = (stones[nst / 2 - 1].0, stones[nst / 2].0);
        let j = jb(sa, sb);
        let fc = a.forces.iter().find(|x| x.0 == j).unwrap();
        // every joint, from the left abutment to the right: the abutments are the last two pieces
        let (abl, abr) = ((nst) as u32, (nst + 1) as u32);
        let mut chain = alloc::vec![abl]; chain.extend(stones.iter().map(|s| s.0)); chain.push(abr);
        let per: std::vec::Vec<(f64, f64, f32)> = chain.windows(2).map(|p| {
            let j = jb(p[0], p[1]);
            let f = a.forces.iter().find(|x| x.0 == j).unwrap();
            let r = a.ratios.iter().find(|x| x.0 == j).unwrap().1;
            // the face's normal runs from ja to jb: orient the moment so it is the same sense along the arch
            let sg = if b.ja[j as usize] == p[0] { 1.0 } else { -1.0 };
            // the bending moment about z (the arch's axis), from the face frame's moments: the frame's u and w axes
            let fr = crate::structure::face_frame(&b.joint_poly(j));
            let mz = f.2[1] * fr[1][2] + f.2[2] * fr[2][2];
            (-f.1[0], sg * mz, r)
        }).collect();
        if extra.is_some() { return (-fc.1[0], per); }
        let h = w * l * l / (8.0 * f);
        let shear = (fc.1[1] * fc.1[1] + fc.1[2] * fc.1[2]).sqrt();
        assert!(shear < 0.001 * h, "shear at the crown {shear}");
        std::println!("parabolic arch t={t}: thrust at the crown {:.1} kN (w L^2 / 8 f = {:.1}), shear {:.2} kN, moment {:.2} kN m; worst ratio {:.3}", -fc.1[0] / 1e3, h / 1e3, shear / 1e3, (fc.2[1] * fc.2[1] + fc.2[2] * fc.2[2]).sqrt() / 1e3, a.max_ratio);
        (-fc.1[0], per)
    }
    /// A portal frame: two 0.25 m square concrete columns 3 m tall (footings below the ground line), 4 m apart, a
    /// 0.5 m deep beam on them; braced or not by a 0.15 m concrete brace from the left footing up to the right
    /// column's face. Returns the store, the beam and the brace's segments.
    fn portal(braced: bool, col_cells: i32) -> (Bricks, u32, std::vec::Vec<u32>) {
        let mut b = Bricks::new();
        b.add([-4, -4, -4], [16, 4, 10], mat::CONCRETE, 0, 0);
        b.add([28, -4, -4], [12, 4, 10], mat::CONCRETE, 0, 0);
        let hh = 24 / col_cells;
        for k in 0..col_cells { b.add([0, k * hh, 0], [2, hh, 2], mat::CONCRETE, 0, 0); b.add([32, k * hh, 0], [2, hh, 2], mat::CONCRETE, 0, 0); }
        let beam = b.add([0, 24, 0], [34, 4, 2], mat::CONCRETE, 0, 0);
        let mut segs = std::vec::Vec::new();
        if braced {
            // from (0.6, 0) on the left footing, by the column, to (4.0, 2.85) on the right column's face, under the
            // beam, 0.15 m square
            let (f, h) = ([0.6f64, 0.0], [4.0f64, 2.85]);
            let d = { let e = [h[0] - f[0], h[1] - f[1]]; let l = (e[0] * e[0] + e[1] * e[1]).sqrt(); [e[0] / l, e[1] / l] };
            let nrm = [-d[1], d[0]];
            let mut pts = std::vec::Vec::new();
            for &o in &[-0.075, 0.075] { for &z in &[0.05, 0.2] {
                let e = [f[0] + o * nrm[0], f[1] + o * nrm[1]];
                let t0 = (0.0 - e[1]) / d[1]; let t1 = (4.0 - e[0]) / d[0];
                pts.push([e[0] + t0 * d[0], 0.0, z]); pts.push([4.0, e[1] + t1 * d[1], z]);
            } }
            let k = piece_at(&mut b, &pts, mat::CONCRETE);
            segs = b.segment(k, 2.0);
        }
        b.build_joints();
        (b, beam, segs)
    }
    /// The sideways sway (m) of a portal under 10 kN on its beam, and the brace's horizontal share of the load.
    fn sway(braced: bool, cells: i32) -> (f64, f64) {
        let (b, beam, segs) = portal(braced, cells);
        let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let p = 10e3;
        let load: std::vec::Vec<f32> = set.iter().map(|&i| if i == beam { p as f32 } else { 0.0 }).collect();
        let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] < 0).collect();
        let mut w = crate::structure::Warm::default();
        let a = crate::structure::analyse_warm(&b, &set, &load, &[], &sink, crate::math::v3(1.0, 0.0, 0.0), Some(&mut w));
        let ux = w.get(&(beam, 0, 0)).unwrap()[0];
        if !braced { return (ux, 0.0); }
        let j = { let mut j = b.jhead[segs[0] as usize]; while j != NONE && b.joint_other(j, segs[0]) != segs[1] { j = b.joint_next(j, segs[0]); } j };
        let th = (2.85f64 / 3.4).atan();
        (ux, a.forces.iter().find(|x| x.0 == j).unwrap().1[0] * th.cos() / p)
    }
    #[test] fn a_braced_portal_frame_is_stiffer_by_its_brace() {
        // A portal: 0.25 m square concrete columns, 3 m tall on footings, 4 m apart, a 0.5 m deep beam; 10 kN sideways
        // on the beam. Closed form (fixed feet): k = 24 E Ic / h^3 (1 + 6 rho) / (4 + 6 rho), rho = Ib h / (Ic L), with
        // h from the footings' centres to the beam's (3.5 m: the model's springs reach the nodes' centres):
        // 5.11e6 N/m. A column of one brick is one rigid node, and bends only at its two ends: the frame sways
        // 4.21 mm, 2.2 times the 1.92 mm it converges to with the columns cut finer (2.17, 1.97, 1.92 mm at 3, 6
        // and 12 bricks; docs/PHASE2.md, found on the way).
        // Braced by a 0.15 m concrete brace (a free piece, 3 segments) from the left footing to the right column under
        // the beam, at 40 degrees: k_b = E A cos^2 / L = 8.94e7 N/m. The brace takes the share k_b / (k_b + k_f) of
        // the load: 0.945, against 0.945. (The braced frame is 12.5 times as stiff, not (k_b + k_f) / k_f = 18: its
        // corners, where the brace's line misses the nodes, give a little; the brace itself stretches P l / EA per
        // segment to 1%.)
        let e = 30e9f64; let (ic, ib, h, l) = (0.25f64.powi(4) / 12.0, 0.25 * 0.5f64.powi(3) / 12.0, 3.5, 4.0);
        let rho = ib * h / (ic * l);
        let kf = 24.0 * e * ic / h.powi(3) * (1.0 + 6.0 * rho) / (4.0 + 6.0 * rho);
        let sw: std::vec::Vec<f64> = [1, 3, 6, 12].iter().map(|&c| sway(false, c).0).collect();
        std::println!("unbraced portal sway (mm) at 1, 3, 6, 12 bricks a column: {:?}; closed form {:.3}", sw.iter().map(|x| x * 1e3).collect::<std::vec::Vec<_>>(), 10e3 / kf * 1e3);
        for w in sw.windows(2) { assert!(w[1] < w[0], "finer columns are stiffer: {sw:?}"); }
        let k12 = 10e3 / sw[3];
        assert!((k12 - kf).abs() < 0.03 * kf, "converged frame {k12:.3e}, closed form {kf:.3e}");
        let (ub, share) = sway(true, 12);
        let th = (2.85f64 / 3.4).atan();
        let kb = e * 0.0225 / (3.4f64 * 3.4 + 2.85 * 2.85).sqrt() * th.cos().powi(2);
        let want = kb / (kb + k12);
        std::println!("braced: sway {:.4} mm ({:.1} times stiffer), brace share {share:.4} against k_b / (k_b + k_f) = {want:.4}", ub * 1e3, sw[3] / ub);
        assert!((share - want).abs() < 0.01 * want);
        assert!(sw[3] / ub > 10.0);
    }
    /// The rafter roof on piers under 100 kN at its ridge: (pier-top push out, pier-top load down, tie tension, the
    /// struts' axial force, worst ratio), N.
    fn roof(tie: Option<f64>) -> (f64, f64, f64, f64, f32) {
        let th = 45f64.to_radians();
        let (b, l, _, top, tid) = a_frame_tied(th, tie, true);
        let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let load: std::vec::Vec<f32> = set.iter().map(|&i| if i == top { 100e3 } else { 0.0 }).collect();
        let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] < 0).collect();
        let a = crate::structure::analyse(&b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
        let (mut push, mut down, mut t, mut n) = (0.0, 0.0, 0.0, 0.0);
        let mut j = b.jhead[l[0] as usize];
        while j != NONE {
            let o = b.joint_other(j, l[0]);
            let f = a.forces.iter().find(|x| x.0 == j).unwrap().1;
            let sh = (f[1] * f[1] + f[2] * f[2]).sqrt();
            // the tie's end is on the strut's 45-degree face: its pull is normal and shear in equal parts
            if o == tid { t = (f[0] * f[0] + sh * sh).sqrt(); }
            else if o == l[1] { n = -f[0]; }
            else { push = sh; down = f[0].abs(); }
            j = b.joint_next(j, l[0]);
        }
        (push, down, t, n, a.max_ratio)
    }
    #[test] fn a_rafter_roof_pushes_its_walls_out_unless_tied() {
        // Two 45-degree steel rafters (the A-frame above) on 3 m concrete piers, 100 kN at the ridge: the ridge thrust
        // is H = P / (2 tan 45) = 50 kN. Untied, the rafters push the pier tops out: 29.0 kN (the piers lean, the
        // rigid-jointed frame spreads and its rafters bend, 14.8 kN of shear in them), and the piers are overloaded
        // in bending (worst ratio 1.34). A tie across the rafters takes the thrust: 0.5, 0.3, 0.15 m above the feet
        // the push falls to 8.9, 5.6, 3.8 kN; at 0.15 m tie and push add up to H (46.0 + 3.8 kN), the rafters carry
        // P / 2 sin 45 = 70.71 kN axially again (70.6) and the worst ratio is 0.19. The piers carry P / 2 down always.
        let h = 50e3;
        let (p0, d0, _, _, r0) = roof(None);
        std::println!("untied: push {:.1} kN, down {:.1} kN, worst {r0:.3}", p0 / 1e3, d0 / 1e3);
        assert!(p0 > 0.5 * h && r0 > 1.0);
        assert!((d0 - h).abs() < 0.01 * h);
        let mut last = p0;
        for &e in &[0.5, 0.3, 0.15] {
            let (pu, dn, t, n, r) = roof(Some(e));
            std::println!("tied at {e} m: push {:.1} kN, tie {:.1} kN, rafter {:.2} kN, worst {r:.3}", pu / 1e3, t / 1e3, n / 1e3);
            assert!(pu < last, "a lower tie pushes the piers less");
            assert!((dn - h).abs() < 0.01 * h);
            last = pu;
            if e == 0.15 {
                assert!(pu < 0.1 * h);
                assert!((pu + t - h).abs() < 0.02 * h, "tie {t} and push {pu} against the thrust {h}");
                assert!((n - 100e3 / (2.0 * 45f64.to_radians().sin())).abs() < 0.01 * 70.71e3);
                assert!(r < 0.25);
            }
        }
    }
    // ---- breaking free pieces (docs/PHASE2.md, step 4) ----
    #[test] fn an_arch_loaded_off_centre_forms_four_hinges() {
        // The 0.4 m arch above with 150 kN more on its fourth stone (a quarter of the span): the line of thrust
        // swings out of the stones in the pattern of the four-hinge mechanism, its eccentricity e = M / N at its
        // extremes alternating in sign: the left springing (-0.081 m), under the load (+0.215 m, past the 0.2 m
        // half-thickness), the far haunch (-0.097 m), the right springing (+0.051 m). The joints that give way first
        // (ratio over 1, the masonry's tension) are under the load (joints 3, 4) and at the far haunch (8, 9).
        // With 50 kN the pattern is the same and nothing gives way.
        let (_, per) = arch_run(0.4, Some((3, 150e3)));
        let e: std::vec::Vec<f64> = per.iter().map(|x| x.1 / x.0).collect();
        std::println!("eccentricities (m): {:?}", e.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<std::vec::Vec<_>>());
        let (imax, emax) = (2..7).map(|k| (k, e[k])).fold((0, f64::MIN), |a, x| if x.1 > a.1 { x } else { a });
        let (imin, emin) = (6..11).map(|k| (k, e[k])).fold((0, f64::MAX), |a, x| if x.1 < a.1 { x } else { a });
        assert!(e[0] < 0.0 && e[0] < e[1], "left springing: {}", e[0]);
        assert!(emax > 0.2 && (imax == 3 || imax == 4), "under the load: {emax} at joint {imax}");
        assert!(emin < -0.05 && (imin == 8 || imin == 9), "far haunch: {emin} at joint {imin}");
        assert!(e[12] > 0.0 && e[12] > e[11], "right springing: {}", e[12]);
        let gone: std::vec::Vec<usize> = per.iter().enumerate().filter(|x| x.1 .2 > 1.0).map(|x| x.0).collect();
        assert_eq!(gone, [3, 4, 8, 9]);
        let (_, light) = arch_run(0.4, Some((3, 50e3)));
        assert!(light.iter().all(|x| x.2 < 1.0));
    }
    #[test] fn a_beam_of_free_pieces_breaks_where_its_moment_peaks() {
        // An 8 m concrete beam, 0.2 x 0.3 m, turned 30 degrees in plan (a free piece, 4 segments of 2 m), its ends
        // on two 0.5 m blocks, 60 kN on each of its two middle segments: along the beam the moment peaks at mid-span,
        // at the joint between them, and falls toward the supports; of the beam's own joints that one, and only it,
        // is over strength (ratios 0.32, 1.36, 0.32 at 2, 4, 6 m). (Its ends are clamped on the blocks, not resting on pins: the joints there carry the
        // hogging moment and the reactions on a small bearing, and are checked as such.)
        let mut b = Bricks::new();
        let q = quat([0.0, 1.0, 0.0], 0.5236);
        let k = b.add_piece([0.0, 1.15, 0.0], q, &box_pts([4.0, 0.15, 0.1]), mat::CONCRETE, 0, 0).unwrap();
        let segs = b.segment(k, 2.0);
        assert_eq!(segs.len(), 4);
        let ends: std::vec::Vec<[f64; 3]> = [-3.875f64, 3.875].iter().map(|&x| { let r = qrot(q, [x, 0.0, 0.0]); [r[0], 1.0, r[2]] }).collect();
        for e in &ends { b.add([crate::math::floor64(e[0] / 0.125 + 0.5) as i32 - 2, 0, crate::math::floor64(e[2] / 0.125 + 0.5) as i32 - 2], [4, 8, 4], mat::CONCRETE, 0, 0); }
        b.build_joints();
        assert!(b.detached().is_empty());
        let set: std::vec::Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        // the segments in order along the beam
        let mut order = segs.clone();
        order.sort_by(|x, y| { let (a, c) = (b.centroid(*x), b.centroid(*y)); let ax = qrot(q, [1.0, 0.0, 0.0]); (a[0] * ax[0] + a[2] * ax[2]).partial_cmp(&(c[0] * ax[0] + c[2] * ax[2])).unwrap() });
        let load: std::vec::Vec<f32> = set.iter().map(|&i| if i == order[1] || i == order[2] { 60e3 } else { 0.0 }).collect();
        let sink: std::vec::Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
        let a = crate::structure::analyse(&b, &set, &load, &sink, crate::math::v3(0.0, -1.0, 0.0));
        let jb = |sa: u32, sb: u32| { let mut j = b.jhead[sa as usize]; while j != NONE && b.joint_other(j, sa) != sb { j = b.joint_next(j, sa); } j };
        let r: std::vec::Vec<f32> = order.windows(2).map(|p| a.ratios.iter().find(|x| x.0 == jb(p[0], p[1])).unwrap().1).collect();
        std::println!("beam of free pieces: ratios at 2, 4, 6 m: {r:?}; failed {:?}", a.failed);
        assert!(r[1] > r[0] && r[1] > r[2]);
        assert!(r[1] > 1.0 && r[0] < 1.0 && r[2] < 1.0);
        let own: std::vec::Vec<u32> = a.failed.iter().map(|f| f.0).filter(|&j| segs.contains(&b.ja[j as usize]) && segs.contains(&b.jb[j as usize])).collect();
        assert_eq!(own, [jb(order[1], order[2])]);
    }
    #[test] fn a_free_piece_is_cut_by_an_axis_plane_and_shatters_into_convex_parts() {
        // Bricks::cut on a free piece cuts it by the plane itself (a box is split as before); shatter_piece cuts a
        // piece in two planes into up to four convex parts that keep its volume, are joined to each other and to
        // what the piece was joined to, and of which one or two spall off.
        let mut b = Bricks::new();
        let base = b.add([-16, 0, -16], [32, 8, 32], mat::CONCRETE, 0, 0);
        let q = quat([0.0, 1.0, 0.0], 0.4);
        let i = b.add_piece([0.0, 1.5, 0.0], q, &box_pts([0.9, 0.5, 0.4]), mat::CONCRETE, 0, 0).unwrap();
        b.build_joints();
        let v0 = b.volume(i);
        assert!(b.cut(i, 0, 40).is_none(), "a plane past the piece leaves it whole");
        let o = b.cut(i, 1, 12).unwrap();
        assert!(b.centroid(i)[1] < 1.5 && b.centroid(o)[1] > 1.5);
        assert!((b.volume(i) + b.volume(o) - v0).abs() < 1e-6);
        assert!((b.volume(i) - v0 / 2.0).abs() < 1e-6, "the plane y = 1.5 m halves it");
        // the lower half still stands on the base, the upper on the lower
        assert!(b.detached().is_empty());
        let (parts, spall) = b.shatter_piece(o);
        assert!(parts.len() >= 3 && parts.len() <= 4, "{} parts", parts.len());
        let vs: f64 = parts.iter().map(|&p| b.volume(p)).sum();
        assert!((vs - v0 / 2.0).abs() < 1e-6, "volume {vs}");
        assert!(!spall.is_empty() && spall.len() <= 2 && spall.iter().all(|s| parts.contains(s)));
        assert!(b.detached().is_empty(), "the parts are joined to each other and to the lower half");
        for &p in &parts { assert!(b.piece(p).is_some()); }
        let _ = base;
    }
}
