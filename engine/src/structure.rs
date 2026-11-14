//! Load paths through the joint graph and joint failure (first cut of the structural solver).
//!
//! Every brick carries a load (its weight, or its inertia in an impact). The load flows through joints to the
//! supports like current through a resistor network: a potential per brick, conductance = joint area (infill
//! such as glass and stud partitions conducts very little, so loads go round it), supports at potential 0.
//! The flow through a joint is a force. Its direction against the load direction says how the joint works:
//! along the load = compression, against it = tension, across it = shear. A joint fails when the stress
//! (force / area) exceeds the weaker brick's strength in that mode.
//!
//! Not modelled yet: bending moments (a scalar flow has none), yielding and damage accumulation.
use crate::brick::{mat, Bricks, F_INFILL, NONE, TICK};
use crate::math::*;
use alloc::vec;
use alloc::vec::Vec;

/// Strength in Pa: (compression, tension, shear). Reinforced concrete joints get rebar continuity in tension.
/// a steel member's section as shares of the solid box drawn: area, shear area (web), plastic modulus, second
/// moment (254 x 146 x 37 UB in a 0.25 m box: 47 cm2, 15 cm2 of web, 483 cm3, 5,540 cm4 against 625 cm2, 3,906 cm3
/// and 32,552 cm4)
pub const STEEL_AREA: f32 = 0.16;
pub const STEEL_SHEAR: f32 = 0.10;
pub const STEEL_FLEX: f32 = 0.12;
pub const STEEL_STIFF: f64 = 0.17;
/// What a floor carries besides itself (N/m^2): finishes, partitions, the rooms' contents; 1,500 to 3,000 in a real
/// building. 2,000 since 14 Nov (0 before, the buildings having been sized for their own weight alone: on 13 Nov
/// at 1,000 the masonry building stood at 0.65, the mixed one at 0.67 and the 3 x 3 city at 0.89). What made room
/// for it: a concrete floor bears on a brick wall as a hinge (`hinge`) and the masonry building's piers are
/// 1.125 m (tower.rs). Standing with it, worst joint over strength: frame 0.59 to 0.64 (3 to 12 storeys), masonry
/// 0.41 to 0.54 (2 to 5), mixed 0.15 to 0.19, steel 0.48 to 0.53, complex 0.57
/// (dyn_tests::the_buildings_standing_margins, docs/LAB.md 14 November). It is weight and mass both (`weight`,
/// `Node::mass`). Test hook: SF_IMPOSED.
pub const IMPOSED: f32 = 2000.0;
/// The same on a timber floor: lighter finishes, a house's rooms.
pub const IMPOSED_TIMBER: f32 = 1000.0;
/// The imposed load on brick `i` (N): `IMPOSED` on its plan if it is a floor, a concrete or timber plate at most
/// 0.4 m thick and at least 1 m each way, off the ground.
pub fn imposed(b: &Bricks, i: u32) -> f32 {
    let iu = i as usize;
    if !(b.mat[iu] == mat::CONCRETE || b.mat[iu] == mat::TIMBER) || b.cell(i).is_some() || b.piece(i).is_some() { return 0.0; }
    let s = b.size(i);
    if s[1] > 3 || s[0] < 8 || s[2] < 8 || b.y[iu] <= 0 { return 0.0; }
    #[cfg(test)] { if let Ok(v) = std::env::var("SF_IMPOSED") { return if b.mat[iu] == mat::TIMBER { 0.0 } else { v.parse::<f32>().unwrap() * (s[0] * s[2]) as f32 * TICK * TICK }; } }
    (if b.mat[iu] == mat::TIMBER { IMPOSED_TIMBER } else { IMPOSED }) * (s[0] * s[2]) as f32 * TICK * TICK
}
/// The weight on brick `i` (N): its own and what it carries as a floor.
pub fn weight(b: &Bricks, i: u32) -> f32 { mass(b, i) * 9.81 + imposed(b, i) }
pub fn strength(m: u8) -> [f32; 3] {
    match m {
        // (shear 3 MPa since 14 Nov, for 4: a flat slab fails round a column in punching at about 0.33 sqrt(fc),
        // 1.8 MPa for this concrete, and with shear reinforcement up to about twice that; at 4 a slab hung on from
        // columns that took a lost neighbour's share. Not 2: the frame's slabs here are 0.25 m with no drops or
        // column heads, and at 2 with the floor loads the frame stands at 0.89 (3 storeys) to 0.97 (12), at 3 at 0.59 to 0.64.)
        #[cfg(test)] mat::CONCRETE if std::env::var("SF_SHEAR").is_ok() => [30e6, 4e6, std::env::var("SF_SHEAR").unwrap().parse::<f32>().unwrap() * 1e6],
        mat::CONCRETE => [30e6, 4e6, 3e6],
        mat::MASONRY => [10e6, 0.3e6, 0.5e6],
        // a steel member is a rolled section, not the solid box it is drawn as: a 254 x 146 universal beam has 16% of
        // that box's area (axial strength), its web 10% (shear). Solid, the mixed building stood on its core with
        // the whole ground storey gone, on steel 6 times too strong (STEEL_AREA, STEEL_SHEAR; bending: flexural)
        mat::STEEL => [250e6 * STEEL_AREA, 250e6 * STEEL_AREA, 150e6 * STEEL_SHEAR],
        mat::GLASS => [50e6, 7e6, 7e6],
        // structural timber (C24 / GL24h: compression 21, tension 14, shear 4 MPa along the grain). It was 5, 5
        // and 1, a stud partition's: with bending checked at the compressed fibre too (the spring network), a
        // 5 m timber cantilever came out at 4 times its strength
        _ => [21e6, 14e6, 4e6],
    }
}
/// Energy (J/m^3) to crush the material to rubble.
/// How strong a joint is relative to the weaker of its two bricks, in [compression, tension, shear]: a joint
/// is a connection between two elements, and connections are usually weaker than what they join.
/// - a seam inside one element (two bricks of one slab, one column) is the material itself
/// - concrete slab to column or core: a cast connection, sound in compression; in shear it is the slab's
///   punching strength (about 0.33 sqrt(f_c) = 1.8 MPa against 4 MPa) and in tension what the connection
///   steel carries
/// - timber elements joined to each other: nailed or bolted, a fraction of the wood
/// - anything else between different elements: somewhat weaker than the material
pub fn connection(b: &Bricks, j: u32) -> [f32; 3] {
    let (a, c) = (b.ja[j as usize] as usize, b.jb[j as usize] as usize);
    let (ma, mc) = (b.mat[a], b.mat[c]);
    let (pa, pc) = (b.paint[a], b.paint[c]);
    if pa == pc && ma == mc { return [1.0, 1.0, 1.0]; }
    if ma == mat::TIMBER && mc == mat::TIMBER { return [1.0, 0.1, 0.25]; }
    if ma == mat::CONCRETE && mc == mat::CONCRETE {
        let slab = |p: u8| p == 0;
        let support = |p: u8| p == 1 || p == 3;
        if (slab(pa) && support(pc)) || (slab(pc) && support(pa)) { return [1.0, 0.5, 0.45]; }
    }
    [1.0, 0.7, 0.7]
}

/// Stones on a mortar bed: masonry where a free piece (a cut stone: an arch's voussoir, a stone beam, a wall panel)
/// meets another member (a different `structure`, the member id; the segments of one member share it, and their
/// seams are the stone itself), however the joint is turned. Checked as masonry arches are (Heyman): the thrust inside
/// the face, no tension (`bed_ratio`). (As the stone itself, 0.3 MPa in tension, 0.5 in bending, a half arch stood as a
/// cantilever off its pier with its keystone gone, at 0.78 of strength. Limited to joints within 70 degrees of level,
/// the joints by a keystone, near upright, were left as stone and failed in tension once the arcade carried a floor.)
pub fn mortar_bed(b: &Bricks, j: u32, _normal: [f64; 3]) -> bool {
    let (a, c) = (b.ja[j as usize] as usize, b.jb[j as usize] as usize);
    let free = |i: usize| b.hull[i] == crate::brick::HULL_FREE;
    b.mat[a] == mat::MASONRY && b.mat[c] == mat::MASONRY && (free(a) || free(c)) && b.structure[a] != b.structure[c]
}
#[inline] fn fabs64s(x: f64) -> f64 { if x < 0.0 { -x } else { x } }
/// A bed joint's ratio, with no tension (Heyman): force `f` and moment `m` on the max side in the face's frame
/// (normal, u, w). The thrust N presses (with none, see below), and must pass inside the face: its
/// eccentricities over the extreme fibres' distances, e_w / z_u + e_u / z_w, is 1 where it reaches the edge (a hinge
/// opens through: the joint gives way). Its compression is checked on what still bears, A (1 - that), and its
/// shear against cohesion on that area plus friction (0.4) on N. The worst of the three.
fn bed_ratio(s: &Sect, f: [f64; 3], m: [f64; 3], cap: [f32; 3]) -> f32 {
    let n = -f[0];
    // a bed that carries no thrust (a wall's upright corner, a stone beam's butt joint) holds by what mortar does
    // without it: a little bond in tension (a tenth of the stone's) and cohesion in shear
    if n <= 1e-9 {
        let bond = 0.1 * strength(mat::MASONRY)[1] as f64;
        let v = sqrt64(f[1] * f[1] + f[2] * f[2]);
        return ((-n) / s.area / bond).max(v / (cap[2] as f64 * s.area)) as f32;
    }
    let ecc = fabs64s(m[1]) / n / s.zu.max(1e-6) + fabs64s(m[2]) / n / s.zw.max(1e-6);
    let bearing = (1.0 - ecc).max(1e-3) * s.area;
    let comp = n / bearing / cap[0] as f64;
    let v = sqrt64(f[1] * f[1] + f[2] * f[2]) + fabs64s(m[0]) * s.rmax / s.j * s.area;
    let shear = v / (cap[2] as f64 * bearing + 0.4 * n);
    ecc.max(comp).max(shear) as f32
}

pub fn crush_energy(m: u8) -> f32 {
    match m { mat::CONCRETE => 1.0e6, mat::MASONRY => 0.4e6, mat::STEEL => 20e6, mat::GLASS => 0.05e6, _ => 0.2e6 }
}
pub fn mass(b: &Bricks, i: u32) -> f32 {
    let s = b.size(i);
    // (a steel member weighs its section, not the solid box: STEEL_AREA)
    let share = if b.mat[i as usize] == mat::STEEL { STEEL_AREA } else { 1.0 };
    (s[0] * s[1] * s[2]) as f32 * TICK * TICK * TICK * mat::density(b.mat[i as usize]) * b.vol_share(i) as f32 * share
}

/// Flexural strength (Pa) for bending of a brick: reinforced concrete gets its rebar.
/// Bending strength of brick `i` (Pa over its own depth). A timber floor (paint 8) is boards on joists
/// (50 x 225 mm at 400 mm, 24 MPa), not solid wood: per metre of width it carries 2.5 x 0.05 x 0.225^2 / 6 x
/// 24 MPa = 25 kNm, which over the plate's 0.25 m depth is 2.4 MPa. As solid timber (20 MPa) a floor bay
/// carried 300 kPa of rubble.
pub fn flexural_of(b: &Bricks, i: usize) -> f32 { if b.mat[i] == mat::TIMBER && b.paint[i] == 8 { 2.4e6 } else { flexural(b.mat[i]) } }
/// A joint at which a timber floor meets another material (the core's concrete, a brick wall): the joists bear on a
/// ledger or in pockets, a hinge. It carries shear but no bending (checked as one plate, it failed under the
/// floor's own weight).
/// Also steel bearing on brick (a beam's end or a lintel on a pier): it sits on a padstone, a hinge (bonded, its
/// end rotation bent the brick pier under it: 1.31 of strength in the mixed building).
fn hinge(b: &Bricks, x: usize, y: usize) -> bool {
    let tf = |i: usize| b.mat[i] == mat::TIMBER && b.paint[i] == 8;
    let (mx, my) = (b.mat[x], b.mat[y]);
    #[cfg(test)] let cm = std::env::var("SF_NOCM").is_err();
    #[cfg(not(test))] let cm = true;
    // (14 Nov) and a concrete floor on brickwork or brickwork on a concrete floor: a mortar bed, which turns before
    // it bends the brick. Bonded, the floor's turning at its bearing put the brick in tension wherever little weight
    // stood on it: 0.89 of strength at the foot of the mixed building's first-floor walls when it has 3 storeys (0.56
    // with 5), 0.83 on the masonry building's top piers when it has 5. Checked as a bed with no tension at all
    // (`mortar_bed`) the masonry building's wall heads, 0.25 m wide under a floor's edge, failed standing.
    ((tf(x) || tf(y)) && mx != my) || (mx == mat::STEEL && my == mat::MASONRY) || (mx == mat::MASONRY && my == mat::STEEL)
        // (brickwork, not cut stone: a free piece's joints are its beds, `mortar_bed`, and the complex building's
        // arcade under its floor went from 0.42 to 0.77 as a hinge)
        || (cm && b.hull[x] != crate::brick::HULL_FREE && b.hull[y] != crate::brick::HULL_FREE && ((mx == mat::CONCRETE && my == mat::MASONRY) || (mx == mat::MASONRY && my == mat::CONCRETE)))
}
pub fn flexural(m: u8) -> f32 {
    match m { mat::CONCRETE => 10e6, mat::MASONRY => 0.5e6, mat::STEEL => 250e6 * STEEL_FLEX, mat::GLASS => 40e6, _ => 20e6 }
}

/// A crack through a brick from bending: split brick `brick` at tick `at` across `axis`.
#[derive(Clone, Copy, Debug)]
pub struct Crack { pub brick: u32, pub axis: usize, pub at: i32, pub ratio: f32 }

// ======================================================================================================
// Structural analysis: a rigid-body-spring network over cells.
//
// Big bricks are single records in the store, but a 22 m slab must be able to bend and crack inside. For the
// analysis each brick is cut into cells of at most CELL ticks (2 m) across x and z, and at the edges of what
// bears on it. Each cell is a rigid body with 6 unknowns (3 displacements, 3 rotations); cells of one brick are
// joined through the brick's own section, and joints between bricks become springs between the cells that
// share the face (normal, shear, bending and torsion stiffness from the materials' moduli, two halves in
// series). One solve gives every face's force and moment; then every face is checked:
//   - joints between bricks (sub-faces summed about the joint's centre): tension, compression, Mohr-Coulomb
//     shear and torsion on the face as a section; over strength the joint fails (or, brick, cracks)
//   - faces inside a brick: the brick's own section; over strength it cracks there
// ======================================================================================================

pub const CELL: i32 = 16;
#[cfg(test)] mod dyn_spike;

pub struct Analysis {
    pub failed: Vec<(u32, f32)>,  // joints over strength, with their worst ratio
    /// joints that are yielding (`Plastic`): at their strength, turning, not failed
    pub yielded: Vec<u32>,
    pub cracks: Vec<Crack>,       // bricks cracking inside
    pub max_ratio: f32,
    pub iterations: usize,
    pub nodes: usize,
    /// the worst link inside a brick: (ratio, brick, axis, flow N, moment N m, section width m)
    pub worst_in: (f32, u32, usize, f32, f32, f32),
    /// every joint checked, with its ratio
    pub ratios: Vec<(u32, f32)>,
    /// every joint checked: its force and moment about its face's centre on jb from ja, in the face's frame
    /// (normal from ja to jb, u, w; N and N m; a positive normal force is tension)
    pub forces: Vec<(u32, [f64; 3], [f64; 3])>,
    /// faces inside bricks loaded past 0.7 of strength but under it: (brick, axis, plane, ratio) (they wear)
    pub inner: Vec<(u32, usize, i32, f32)>,
    /// the joints checked as stones' beds (`mortar_bed`): their ratio is where the thrust passes, which does not
    /// wear (an arch whose thrust runs at 0.7 of the way to the edge stands for centuries)
    pub beds: Vec<u32>,
    /// with `Dyn::want_cells`: every face inside a brick, with its ratio (brick, load / strength)
    pub inner_all: Vec<(u32, f32)>,
    /// the free cells (`Dyn::want_cells`), see `Dyn::cells`
    pub cells: Vec<([i32; 5], [f64; 3], [f64; 6])>,
    /// a dynamic step: joints and faces cut because one side has left the structure (their ratio is `DYN_CUT`)
    pub left_cut: usize,
}
impl Analysis {
    pub fn empty() -> Analysis { Analysis { yielded: Vec::new(), failed: Vec::new(), cracks: Vec::new(), max_ratio: 0.0, iterations: 0, nodes: 0, worst_in: (0.0, 0, 0, 0.0, 0.0, 0.0), ratios: Vec::new(), forces: Vec::new(), inner: Vec::new(), beds: Vec::new(), cells: Vec::new(), inner_all: Vec::new(), left_cut: 0 } }
    /// add another part's analysis (a city's lots, each analysed on its own)
    pub fn merge(&mut self, o: Analysis) {
        self.failed.extend(o.failed); self.cracks.extend(o.cracks); self.ratios.extend(o.ratios); self.forces.extend(o.forces);
        self.inner.extend(o.inner); self.beds.extend(o.beds); self.cells.extend(o.cells); self.inner_all.extend(o.inner_all); self.left_cut += o.left_cut;
        self.max_ratio = self.max_ratio.max(o.max_ratio); self.iterations += o.iterations; self.nodes += o.nodes;
        if o.worst_in.0 > self.worst_in.0 { self.worst_in = o.worst_in; }
    }
}

struct Node { brick: u32, lo: [i32; 3], hi: [i32; 3], load: f32, mass: f32, sink: bool, c: [f64; 3] }
/// A link: between two cells of one brick (`joint` NONE), or across a joint. `face_lo..face_hi` is its face on the
/// tick grid; a face where a convex cell meets something is a polygon, and `geo` holds it.
struct Link { a: u32, b: u32, joint: u32, axis: usize, face_lo: [i32; 3], face_hi: [i32; 3], geo: Option<Geo> }
/// A face as a section: centre (m), frame (rows: normal, u, w), area, second moments about u and w, the extreme
/// fibres' distances for each, the polar moment and the farthest point (m, m^2, m^4).
#[derive(Clone, Copy)]
struct Geo { c: [f64; 3], fr: [[f64; 3]; 3], s: Sect }
#[derive(Clone, Copy)]
struct Sect { area: f64, iu: f64, iw: f64, zu: f64, zw: f64, j: f64, rmax: f64 }
impl Sect {
    /// a rectangle eu x ew (along u and w)
    fn rect(eu: f64, ew: f64, area: f64) -> Sect {
        let (eu, ew) = (eu.max(1e-3), ew.max(1e-3));
        Sect { area: area.max(1e-6), iu: eu * ew * ew * ew / 12.0, iw: ew * eu * eu * eu / 12.0, zu: ew / 2.0, zw: eu / 2.0, j: eu * ew * (eu * eu + ew * ew) / 12.0, rmax: 0.5 * sqrt64(eu * eu + ew * ew) }
    }
}
/// a polygon face (m) as a section; `ax` < 3 for one on an axis plane (its frame is then the axes')
fn geo_of(poly: &[[f64; 3]], ax: usize) -> Option<Geo> {
    let (area, c, n) = crate::hull::polygon(poly);
    if area < 1e-6 { return None; }
    let fr = if ax < 3 {
        let e = |d: usize| { let mut v = [0.0; 3]; v[d] = 1.0; v };
        [e(ax), e((ax + 1) % 3), e((ax + 2) % 3)]
    } else {
        // u: the axis least along the normal, made perpendicular
        let k = (0..3).min_by(|&a, &b| n[a].abs().partial_cmp(&n[b].abs()).unwrap()).unwrap();
        let mut e = [0.0; 3]; e[k] = 1.0;
        let d = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
        let u = [e[0] - d * n[0], e[1] - d * n[1], e[2] - d * n[2]];
        let l = sqrt64(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let u = [u[0] / l, u[1] / l, u[2] / l];
        let w = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        [n, u, w]
    };
    let ([iu, iw], [zu, zw]) = crate::hull::polygon_section(poly, c, fr[1], fr[2]);
    let mut rmax = 0.0f64;
    for p in poly { let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; rmax = rmax.max(sqrt64(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])); }
    Some(Geo { c, fr, s: Sect { area, iu: iu.max(1e-12), iw: iw.max(1e-12), zu: zu.max(1e-4), zw: zw.max(1e-4), j: (iu + iw).max(1e-12), rmax } })
}
/// The frame (rows: normal, u, w) a polygonal face is checked in: that of `Analysis::forces`.
pub fn face_frame(poly: &[[f64; 3]]) -> [[f64; 3]; 3] { geo_of(poly, 3).map_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], |g| g.fr) }
fn rot(fr: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] { [fr[0][0] * v[0] + fr[0][1] * v[1] + fr[0][2] * v[2], fr[1][0] * v[0] + fr[1][1] * v[1] + fr[1][2] * v[2], fr[2][0] * v[0] + fr[2][1] * v[1] + fr[2][2] * v[2]] }
fn rot_t(fr: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] { [fr[0][0] * v[0] + fr[1][0] * v[1] + fr[2][0] * v[2], fr[0][1] * v[0] + fr[1][1] * v[1] + fr[2][1] * v[2], fr[0][2] * v[0] + fr[1][2] * v[1] + fr[2][2] * v[2]] }

fn cells_of(t: &crate::brick::TBox) -> (Vec<i32>, Vec<i32>) {
    let cut = |lo: i32, hi: i32| { let n = ((hi - lo + CELL - 1) / CELL).max(1); let mut v = vec![lo]; for k in 1..n { v.push(lo + (hi - lo) * k / n); } v.push(hi); v };
    (cut(t.min[0], t.max[0]), cut(t.min[2], t.max[2]))
}

/// Analyse `set` under per-brick loads `load` (N) acting along `dir`, supported at `sink` bricks.
/// Moments are evaluated for vertical loads (gravity).
pub fn analyse(b: &Bricks, set: &[u32], load: &[f32], sink: &[bool], dir: V3) -> Analysis { analyse_at(b, set, load, &[], sink, dir) }
/// As `analyse`, plus loads at points: (index into `set`, x, z in ticks, N). A point load goes to the cell of
/// that brick under the point (a heap of rubble on one bay of a 22 m slab loads that bay, not the whole slab).
pub fn analyse_at(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3) -> Analysis { analyse_warm(b, set, load, point, sink, dir, None) }
/// The last solution of a load case, per cell (brick, cell's corner x, z ticks): the next solve of the standing
/// building starts from it. Between two checks most of the building moves the same.
#[derive(Default)]
pub struct Warm(crate::Map<(u32, i32, i32), [f64; 6]>);
impl Warm {
    /// the largest displacement of any cell (m)
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn max_disp(&self) -> f64 { self.0.values().map(|x| sqrt64(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])).fold(0.0, f64::max) }
    /// a cell's displacement and rotation (brick, cell corner x, z ticks)
    pub fn get(&self, k: &(u32, i32, i32)) -> Option<&[f64; 6]> { self.0.get(k) }
}
/// As `analyse_at`, starting from (and keeping) the last solution in `warm`.
pub fn analyse_warm(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3, warm: Option<&mut Warm>) -> Analysis { analyse_dyn(b, set, load, point, sink, dir, warm, None) }

/// The standing structure as a dynamic model (docs/DYNAMIC.md): its cells' velocities between steps, and the step.
/// A step of `h` seconds is implicit Euler on the spring network,
///     (K + M / h^2) x' = f + (M / h^2) (x + h v),   v' = (x' - x) / h,
/// and M / h^2 is one more spring a cell, from its centre to the ground (translation m / h^2, rotation I / h^2): the
/// dynamic step is the static network with those springs, solved by the same multigrid from x + h v, in at most
/// `max_it` iterations. What follows:
///   - a change (a column gone, a hit) is answered in the step it happens, at a bounded cost, and the forces rise
///     over the frames the building takes to move (`structure::dyn_spike`: 4 iterations a step are within 5-7% of
///     the step solved exactly from the first frame, on the 12-storey tower);
///   - a mechanism is no longer a singular solve (a building on one column took 340 iterations): its part starts to
///     move at the speed its weight gives it, and its joints fail as they stretch;
///   - at rest nothing is solved: the step from a resting solution needs no iteration, and the cells stay where
///     they are.
#[derive(Default)]
pub struct Dyn {
    /// the step (s); 0 solves statically, as `analyse_warm`
    pub h: f64,
    /// iterations a step at most (0: `DYN_IT`)
    pub max_it: usize,
    /// each cell's velocity and spin (brick, cell corner x, z ticks), m/s and rad/s
    pub vel: crate::Map<(u32, i32, i32), [f64; 6]>,
    /// and its acceleration (the step is second order)
    /// settling steps taken in a row (0: not settling), what was out of balance at the last calm step over the
    /// largest load, and the times the structure was rested after `DYN_SETTLE_MAX` settling steps still out of balance
    pub settle: usize,
    pub imbalance: f64,
    pub forced: usize,
    /// a settling step was refused (it moved a part out of the linear range) since the structure last rested
    pub no_settle: bool,
    /// the most the last rest took off a cell's loads as left out of balance (`bias`), as a share of the cell's own
    /// load (over the cells heavy enough for `DYN_BIAS` and not `DYN_BIAS_FLOOR` to be their limit)
    pub bias_share: f64,
    /// the cap on a step's iterations is `DYN_IT` times 2 to this (`DYN_BOOST`)
    pub boost: usize,
    /// s until a structure that could not settle tries again
    pub wait: f64,
    /// without the material's damping (`DYN_DAMP`, `DYN_DAMP_K`): for tests of the step against theory
    pub undamped: bool,
    /// with `undamped`, after a step: the work of the loads over the cells' displacements, and kinetic plus strain energy less that work (J)
    pub work: f64,
    pub energy: f64,
    /// what each cell's loads and springs left unbalanced when the structure last rested (N, N m): the solver's
    /// tolerance, taken off the loads of the steps that follow
    pub bias: crate::Map<(u32, i32, i32), [f64; 6]>,
    pub acc: crate::Map<(u32, i32, i32), [f64; 6]>,
    /// after a step: the fastest cell (m/s), the iterations it took, and whether it stopped at `max_it`
    pub speed: f64,
    pub its: usize,
    pub capped: bool,
    /// after a step: the cells' kinetic energy (J), its part without spin, and the speed that part gives the
    /// structure's whole mass (m/s)
    pub kinetic: f64,
    pub kinetic_lin: f64,
    pub rms: f64,
    /// after a step: the cells that left the structure (`DYN_TURN`, `DYN_MOVE`) and the joints cut for it
    pub left: usize,
    pub cut: usize,
    /// asked for (`want_cells`): the free cells after the last step, for the page's view of the structure:
    /// (brick, corner x, z; far corner x, z (ticks); centre (m); displacement (m) and rotation (rad))
    pub want_cells: bool,
    /// the view of this lot has been written since it was last asked for (`Sim::view_out`): a lot at rest is
    /// not written again
    pub viewed: bool,
    pub cells: Vec<([i32; 5], [f64; 3], [f64; 6])>,
    /// where each cell last rested (displacement and rotation): `DYN_TURN` and `DYN_MOVE` are measured from there
    /// the cells beyond that range and watched: (how long for (s), how far out when they crossed (ranges))
    pub beyond: crate::Map<(u32, i32, i32), (f64, f64)>,
    pub rest: crate::Map<(u32, i32, i32), [f64; 6]>,
    /// the stones' beds open (1) or hinged (2) after the last step: (joint, its two cells)
    pub beds: crate::Map<(u32, (u32, i32, i32), (u32, i32, i32)), u8>,
    /// the links that have yielded (`Plastic`), by their two cells
    pub plastic: crate::Map<((u32, i32, i32), (u32, i32, i32)), Plastic>,
    /// joints yielding after the last step, and the most any has turned plastically (rad)
    pub yielding: usize,
    pub plastic_turn: f64,
    /// no joint yields (for tests of the elastic step against theory)
    pub no_yield: bool,
    /// cells with no way to a support are stepped too, held by their mass alone (a piece in flight: docs/DEFORM.md).
    /// Only a dynamic step can: at rest nothing holds them. Not used by the game yet.
    pub free: bool,
    /// the lot has sagged enough to be drawn where it is (`sim::VIEW_BEND`)
    pub bent: bool,
    /// after a solve at rest: the most any cell's loads and springs leave out of balance, over the structure's
    /// largest load on a cell (what the solve's tolerance leaves; `bias` is this, limited)
    pub residual: f64,
    /// hits: where (m), the impulse (N s) and how much of `KICK_TIME` is left of it (s). Each is a force of
    /// impulse / `KICK_TIME` on the cell nearest it for that long (docs/DYNPLAN.md stage 5).
    pub kicks: Vec<([f64; 3], [f64; 3], f64)>,
    /// with `want_cells`: the velocities of the cells in `Analysis::cells`, in their order
    pub cell_vel: Vec<[f32; 6]>,
}
pub const DYN_IT: usize = 16;
/// A step that stops at its cap is not solved: the step after it may take twice the iterations, up to `DYN_IT` times
/// 2^`DYN_BOOST` (128), and one that converges gives half back. Left at 16, a structure whose steps all stopped
/// there moved slower than it should (the solve finds the stiff motions first and the slow sag last), was taken
/// for calm, and was rested where it hung.
pub const DYN_BOOST: usize = 3;
/// **Plastic joints** (the dynamic structure only). A joint of a ductile material that bending takes over its
/// strength does not break: it yields. From there it turns against `PL_H` of its elastic stiffness (the moment it
/// carried at yield, and a little more as it turns: strain hardening), and keeps the turn when the load goes. It
/// breaks when it has turned by its material's `ductility`, or if its ratio passes `PL_OVER` all the same (the
/// part of its moment that its links' own turning does not carry cannot yield here).
/// What takes it over strength without bending (shear, tension, crushing) breaks it as before.
pub const PL_H: f64 = 0.02;
/// A hit's impulse reaches the structure over this long (s): a heavy thing stopping takes about that (at 12 m/s,
/// 0.6 m of crushing). In one step of 1/60 s the force was six times as large and gone before a joint that it
/// took over its strength could turn: the joint's state changes after a step, not in it.
pub const KICK_TIME: f64 = 0.1;
pub const PL_OVER: f32 = 1.5;
/// How far a joint of this material turns plastically before it breaks (rad): steel 0.12, reinforced concrete 0.06,
/// timber 0.015; masonry, stone and glass do not yield. Between two materials the lesser holds.
/// With `PL_H` = 0.02 a concrete joint that yields at a turn of 0.0016 rad carries 1.75 times its yield moment
/// when it breaks. A hinge alone does not: a reinforced section has 1.1 to 1.25 of its yield moment at 0.02 to
/// 0.05 rad. The rest stands for what a linear model has not, the tension a sagging slab or beam develops along
/// itself (membrane and catenary action), which carries more than bending does once it hangs by a tenth of its
/// span. It is a game's stand-in, not a section's property.
pub fn ductility(m: u8) -> f64 { match m { mat::STEEL => 0.12, mat::CONCRETE => 0.06, mat::MASONRY | mat::GLASS => 0.0, _ => 0.015 } }
/// A link's plastic state: yielding now, the rotation taken off its strain (world), the rotation at which it
/// yielded and the last step's.
#[derive(Clone, Copy, Default)]
pub struct Plastic { pub on: bool, pub t0: [f64; 3], pub ry: [f64; 3], pub r: [f64; 3],
    /// the ratio it had hardened to when it last stopped yielding (it yields again past that, not past 1), and the
    /// turn it had made by then
    pub lev: f32, pub turned: f64 }
/// The most of a cell's own load that a rest may write off as left out of balance by the solve (`Dyn::bias`); or, if
/// that is more, `DYN_BIAS_FLOOR` of the structure's largest load: what the solve's tolerance leaves on a light cell
/// beside a heavy one (the standing 12-storey tower: up to 449 N on a cell, 0.39% of its largest load of 116 kN;
/// with less allowed it was never quiet).
pub const DYN_BIAS_FLOOR: f64 = 0.005;
pub const DYN_BIAS: f64 = 0.03;
/// How much of a motion too fast for the step survives it (the generalised-alpha method's spectral radius at
/// infinity; 1 keeps all of it, 0 none). A building's joints ring at kilohertz, and a step is a 60th of a second:
/// that ringing is noise here and is damped, while motion the step resolves (a floor's bounce, a tower's sway) is
/// kept to second order. Implicit Euler, the first step tried, damped both: a column lost suddenly peaked at
/// 1.04 of the new static force where a resolved motion gives 2 (docs/DYNAMIC.md).
pub const DYN_RHO: f64 = 0.5;
/// A structure with no cell moving and no cell out of balance by more than this share of its largest load is at
/// rest, and its step is not solved; see `analyse_dyn`.
pub const DYN_QUIET: f64 = 1.0e-3;
/// A calm structure with no more than this share of its largest load out of balance on any cell is at rest (the
/// solve at rest leaves 1.5% on the 12-storey tower); with more it settles: steps of `DYN_SETTLE_H` s from rest,
/// `DYN_SETTLE_MAX` at most in a row.
pub const DYN_SETTLED: f64 = 0.03;
pub const DYN_SETTLE_H: f64 = 1.0;
pub const DYN_SETTLE_MAX: usize = 20;
/// Damping in proportion to stiffness (s): a mode of frequency f is damped at pi f DYN_DAMP_K of critical (10 Hz,
/// a floor's bounce: 5%; 1 Hz: 0.5%; the joints' ringing far above the step's rate: all of it). It does not touch
/// a part that moves as one piece, so a falling part falls. With mass damping alone the 4-storey frame, a corner
/// column gone, rang at 25-80 mm/s for the 4 s tried and never rested.
pub const DYN_DAMP_K: f64 = 1.6e-3;
/// Below this speed (m/s; the cells' speeds weighted by mass) a structure whose step converged is at rest, if no
/// cell is faster than `DYN_REST_CELL`: its cells' velocities are dropped. A part let go from rest moves 163 mm/s
/// after its first 60th of a second (g h).
/// - Without it the complex building, a corner column shot out, crept at 0.1-0.3 mm/s for the 8 s tried.
/// - By the fastest cell alone, the 4-storey frame with a corner column gone never rested: the stubs the blast
///   left, 9-19 kg on one joint each, kept 7-11 mm/s for the 4 s tried. That is the solver's tolerance, which is
///   set against the building's loads: a hundredth of a per cent of 1 MN is 11 m/s^2 on 9 kg.
pub const DYN_REST: f64 = 0.5e-3;
pub const DYN_REST_CELL: f64 = 0.05;
/// Damping in proportion to mass (1/s): a mode of frequency f is damped at DYN_DAMP / (4 pi f) of critical (1 Hz,
/// a tall building's sway: 8%; real buildings 2-5%, cracked and opening joints more), and a part in free fall
/// loses DYN_DAMP h of its speed a step (2% at 60 Hz) for the tenth of a second before it leaves the structure.
pub const DYN_DAMP: f64 = 1.0;
/// A cell that has turned or moved this far (rad, m) from where it last rested has left the standing structure: the network is linear (small
/// displacements), and a part turned by t stretches by t^2 / 2 of its length for nothing (0.005 rad: 1.25e-5,
/// an eighth of the strain concrete cracks at; at 0.01 rad the 4-storey frame left on one column broke into
/// pieces of 1,242 bricks at most, 1,171 joints failed, against 1,845 and 843: dyn_tests). What moves that far is cut loose where it meets what does not, and
/// falls as a body; joints between cells beyond the range are not checked (their strains are the turn's, not
/// load). A cell beyond the range has not left yet: it is watched (`DYN_WATCH`).
/// (13 Nov: 0.03 rad and 0.25 m, for 0.005 and 0.05. A part hanging on yielded joints (`Plastic`) sags by centimetres
/// and by hundredths of a radian and is still held; at 0.03 rad a part is drawn 0.05% too long. The old range is
/// kept for what a settling step may move, `DYN_SETTLE_TURN`.)
pub const DYN_TURN: f64 = 0.03;
pub const DYN_MOVE: f64 = 0.25;
pub const DYN_SETTLE_TURN: f64 = 0.005;
pub const DYN_SETTLE_MOVE: f64 = 0.05;
/// A cell beyond that range is watched for this long (s) before it is taken to have left; see `analyse_dyn`.
/// Taken at once, the lab's ball (two steel balls into a facade) cut 14 joints and sent 3 pieces off where the
/// structure solved at rest fails 2 joints and sends none: cracked pieces by the holes, hanging by their softened
/// joints, sag past the range and stay.
pub const DYN_WATCH: f64 = 0.15;
/// the ratio given to a joint cut because one side has left the structure (so that it goes first)
pub const DYN_CUT: f32 = 1.0e3;
impl Dyn {
    /// is the structure moving (so that another step is due)?
    pub fn moving(&self) -> bool { self.h > 0.0 && (!self.vel.is_empty() || self.settle > 0 || !self.kicks.is_empty()) }
    /// a cell's velocity and spin
    pub fn get(&self, k: &(u32, i32, i32)) -> Option<&[f64; 6]> { self.vel.get(k) }
}
/// As `analyse_warm`; with `dy` (and `warm`, which holds the cells' displacements between steps), a dynamic step.
pub fn analyse_dyn(b: &Bricks, set: &[u32], load: &[f32], point: &[(u32, f32, f32, f32)], sink: &[bool], dir: V3, warm: Option<&mut Warm>, mut dy: Option<&mut Dyn>) -> Analysis {
    let gravity = fabs(dir.y) > 0.99;
    let mut nodes: Vec<Node> = Vec::new();
    let mut first = vec![NONE; b.len()];     // brick -> first node
    let mut grid: Vec<(usize, usize)> = vec![(0, 0); set.len()];
    let mut links: Vec<Link> = Vec::new();
    let mut cuts: Vec<(Vec<i32>, Vec<i32>)> = Vec::with_capacity(set.len());   // each brick's cell boundaries
    let mut slot = vec![0u32; b.len()];   // brick -> its place in `set`
    for (k, &i) in set.iter().enumerate() { slot[i as usize] = k as u32; }
    for (k, &i) in set.iter().enumerate() {
        let t = b.tbox(i);
        let infill = b.flags[i as usize] & F_INFILL != 0;
        // (a convex cell is one node: it is no bigger than fracture makes bricks)
        let (xs, zs) = if infill || b.hull[i as usize] != 0 { (vec![t.min[0], t.max[0]], vec![t.min[2], t.max[2]]) } else {
            // the edges of the supports it rests on are where the moment peaks: cells are cut there too
            let (mut xs, mut zs) = cells_of(&t);
            let mut j = b.jhead[i as usize];
            let (mut ex, mut ez) = (Vec::new(), Vec::new());
            while j != NONE {
                let oi = b.joint_other(j, i);
                let o = b.tbox(oi);
                // (infill carries nothing: glass, cladding and partitions under a slab are no supports)
                if b.jaxis[j as usize] == 1 && o.max[1] == t.min[1] && b.flags[oi as usize] & F_INFILL == 0 {
                    for v in [o.min[0], o.max[0]] { if v > t.min[0] && v < t.max[0] { ex.push(v); } }
                    for v in [o.min[2], o.max[2]] { if v > t.min[2] && v < t.max[2] { ez.push(v); } }
                }
                j = b.joint_next(j, i);
            }
            let merge = |base: &mut Vec<i32>, extra: Vec<i32>| {
                for v in extra { base.push(v); }
                base.sort_unstable(); base.dedup();
            };
            merge(&mut xs, ex); merge(&mut zs, ez);
            (xs, zs)
        };
        first[i as usize] = nodes.len() as u32;
        grid[k] = (xs.len() - 1, zs.len() - 1);
        cuts.push((xs.clone(), zs.clone()));
        let vol = ((t.max[0] - t.min[0]) * (t.max[2] - t.min[2])) as f32;
        for cz in 0..zs.len() - 1 { for cx in 0..xs.len() - 1 {
            let (lo, hi) = ([xs[cx], t.min[1], zs[cz]], [xs[cx + 1], t.max[1], zs[cz + 1]]);
            let share = ((hi[0] - lo[0]) * (hi[2] - lo[2])) as f32 / vol;
            let tk = TICK as f64;
            let c = if b.hull[i as usize] != 0 { b.centroid(i) } else { [0.5 * (lo[0] + hi[0]) as f64 * tk, 0.5 * (lo[1] + hi[1]) as f64 * tk, 0.5 * (lo[2] + hi[2]) as f64 * tk] };
            nodes.push(Node { brick: i, lo, hi, load: load[k] * share, mass: (mass(b, i) + imposed(b, i) / 9.81) * share, sink: sink[k], c });
        } }
        // links inside the brick
        let (nx, nz) = grid[k];
        let base = first[i as usize];
        for cz in 0..nz { for cx in 0..nx {
            let id = base + (cz * nx + cx) as u32;
            let n = &nodes[id as usize];
            if cx + 1 < nx {
                let (lo, hi) = ([n.hi[0], n.lo[1], n.lo[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                links.push(Link { a: id, b: id + 1, joint: NONE, axis: 0, face_lo: lo, face_hi: hi, geo: None });
            }
            if cz + 1 < nz {
                let (lo, hi) = ([n.lo[0], n.lo[1], n.hi[2]], [n.hi[0], n.hi[1], n.hi[2]]);
                links.push(Link { a: id, b: id + nx as u32, joint: NONE, axis: 2, face_lo: lo, face_hi: hi, geo: None });
            }
        } }
    }
    // links across joints, cell to cell where the face overlaps
    for (k, &i) in set.iter().enumerate() {
        let mut j = b.jhead[i as usize];
        while j != NONE {
            let o = b.joint_other(j, i);
            if first[o as usize] != NONE && b.ja[j as usize] == i && b.jaxis[j as usize] == crate::brick::JFREE {
                // a free piece meets something flat, at any angle: the joint's polygon, cut to each cell of a box on
                // either side (a free piece and a convex cell are one node each)
                let poly = b.joint_poly(j);
                let (ka, kb) = (k, slot[o as usize] as usize);
                let cells = |kk: usize, br: u32| -> Vec<u32> { let (nx, nz) = grid[kk]; (0..(nx * nz) as u32).map(|c| first[br as usize] + c).collect() };
                let (ca, cb) = (cells(ka, i), cells(kb, o));
                let tk = TICK as f64;
                let cut = |pl: &[[f64; 3]], nd: &Node| -> Vec<[f64; 3]> {
                    let mut q = pl.to_vec();
                    for (d, lo, hi) in [(0usize, nd.lo[0], nd.hi[0]), (2usize, nd.lo[2], nd.hi[2])] {
                        let mut e = [0.0; 3]; e[d] = 1.0; q = crate::hull::clip(&q, e, hi as f64 * tk + 1e-9);
                        let mut e = [0.0; 3]; e[d] = -1.0; q = crate::hull::clip(&q, e, -(lo as f64 * tk) + 1e-9);
                        if q.len() < 3 { return Vec::new(); }
                    }
                    q
                };
                for &na in &ca { for &nb in &cb {
                    let mut pl = poly.clone();
                    if ca.len() > 1 { pl = cut(&pl, &nodes[na as usize]); }
                    if cb.len() > 1 && pl.len() >= 3 { pl = cut(&pl, &nodes[nb as usize]); }
                    if pl.len() < 3 { continue; }
                    if let Some(g) = geo_of(&pl, 3) {
                        let q = |v: f64| floor64(v / tk + 0.5) as i32;
                        let lo = [q(g.c[0]), q(g.c[1]), q(g.c[2])];
                        links.push(Link { a: na, b: nb, joint: j, axis: 3, face_lo: lo, face_hi: lo, geo: Some(g) });
                    }
                } }
            } else if first[o as usize] != NONE && b.ja[j as usize] == i && b.jaxis[j as usize] == 3 {
                // two cells of one fractured brick, on their shared face
                let (ia, ib) = (first[i as usize], first[o as usize]);
                if let Some(g) = geo_of(&b.joint_poly(j), 3) {
                    let tk = TICK as f64;
                    let q = |v: f64| floor64(v / tk + 0.5) as i32;
                    let lo = [q(g.c[0]), q(g.c[1]), q(g.c[2])];
                    links.push(Link { a: ia, b: ib, joint: j, axis: 3, face_lo: lo, face_hi: lo, geo: Some(g) });
                }
            } else if first[o as usize] != NONE && b.ja[j as usize] == i {
                let ax = b.jaxis[j as usize] as usize;
                // a face of a convex cell: the joint's polygon, cut to each pair of cells below
                let poly = if b.hull[i as usize] != 0 || b.hull[o as usize] != 0 { Some(b.joint_poly(j)) } else { None };
                let ko = slot[o as usize] as usize;
                let (na, nb) = (grid[k], grid[ko]);
                let plane = b.tbox(i).max[ax];
                // the cells of o that overlap a cell of i: found by the cell boundaries (both are grids)
                let (bx, bz) = (&cuts[ko].0, &cuts[ko].1);
                let range = |c: &Vec<i32>, lo: i32, hi: i32| -> (usize, usize) {
                    let n = c.len() - 1;
                    let a = c.partition_point(|&v| v <= lo).saturating_sub(1).min(n);
                    let z = c.partition_point(|&v| v < hi).min(n);
                    (a, z.max(a))
                };
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for ca in 0..na.0 * na.1 {
                    let p = &nodes[(first[i as usize] + ca as u32) as usize];
                    if ax != 1 && p.hi[ax] != plane { continue; }
                    let (x0, x1) = if ax == 0 { (0, nb.0) } else { range(bx, p.lo[0], p.hi[0]) };
                    let (z0, z1) = if ax == 2 { (0, nb.1) } else { range(bz, p.lo[2], p.hi[2]) };
                    for cz in z0..z1 { for cx in x0..x1 { pairs.push((ca, cz * nb.0 + cx)); } }
                }
                for (ca, cb) in pairs {
                    let (ia, ib) = (first[i as usize] + ca as u32, first[o as usize] + cb as u32);
                    let (p, q) = (&nodes[ia as usize], &nodes[ib as usize]);
                    let mut lo = [0; 3]; let mut hi = [0; 3]; let mut ok = true;
                    for d in 0..3 {
                        if d == ax { lo[d] = plane; hi[d] = plane; continue; }
                        lo[d] = p.lo[d].max(q.lo[d]); hi[d] = p.hi[d].min(q.hi[d]);
                        if hi[d] <= lo[d] { ok = false; }
                    }
                    if !ok { continue; }
                    let geo = match &poly {
                        None => None,
                        Some(pl) => {
                            let tk = TICK as f64;
                            let (mut rl, mut rh) = ([0.0; 3], [0.0; 3]);
                            for d in 0..3 { rl[d] = lo[d] as f64 * tk; rh[d] = hi[d] as f64 * tk; }
                            match geo_of(&crate::hull::clip_rect(pl, ax, rl, rh), ax) { Some(g) => Some(g), None => continue }
                        }
                    };
                    links.push(Link { a: ia, b: ib, joint: j, axis: ax, face_lo: lo, face_hi: hi, geo });
                }
            }
            j = b.joint_next(j, i);
        }
    }
    for &(k, x, z, f) in point {
        let i = set[k as usize];
        let (fst, (nx, nz)) = (first[i as usize], grid[k as usize]);
        if fst == NONE { continue; }
        let (xs, zs) = (&cuts[k as usize].0, &cuts[k as usize].1);
        let cx = xs.partition_point(|&v| (v as f32) <= x).saturating_sub(1).min(nx - 1);
        let cz = zs.partition_point(|&v| (v as f32) <= z).saturating_sub(1).min(nz - 1);
        nodes[(fst + (cz * nx + cx) as u32) as usize].load += f;
    }
    let n = nodes.len();
    // nodes connected to a support
    let mut head = vec![NONE; n]; let mut next = vec![NONE; 2 * links.len()];
    for (l, k) in links.iter().enumerate() {
        next[2 * l] = head[k.a as usize]; head[k.a as usize] = (2 * l) as u32;
        next[2 * l + 1] = head[k.b as usize]; head[k.b as usize] = (2 * l + 1) as u32;
    }
    let other = |e: u32| -> u32 { let l = (e / 2) as usize; if e % 2 == 0 { links[l].b } else { links[l].a } };
    let mut live = vec![dy.as_ref().map_or(false, |d| d.free && d.h > 0.0); n];
    let mut st: Vec<u32> = (0..n as u32).filter(|&k| nodes[k as usize].sink).collect();
    for &k in &st { live[k as usize] = true; }
    while let Some(k) = st.pop() { let mut e = head[k as usize]; while e != NONE { let o = other(e); if !live[o as usize] { live[o as usize] = true; st.push(o); } e = next[e as usize]; } }
    // ---- the spring network: each cell a rigid body (3 displacements, 3 rotations), each link a spring at the
    // centre of its face, solved for the displacements under the loads with the ground cells fixed
    let mut springs: Vec<Spring> = links.iter().map(|l| spring(b, &nodes, l)).collect();
    let fixed: Vec<bool> = (0..n).map(|k| nodes[k].sink || !live[k]).collect();
    let mut f = vec![[0.0f64; 6]; n];
    for k in 0..n { let w = nodes[k].load as f64; f[k][0] = w * dir.x as f64; f[k][1] = w * dir.y as f64; f[k][2] = w * dir.z as f64; }
    let mut ends: Vec<(u32, u32)> = links.iter().map(|l| (l.a, l.b)).collect();
    let tk = TICK as f64;
    let mut pos: Vec<[f64; 3]> = nodes.iter().map(|nd| nd.c).collect();
    // A cell is known between steps by where it is (its corners), not by its brick: a brick cut by a blast becomes
    // new bricks, and the cells of it that the blast did not touch keep their displacements and velocities. (By
    // brick, every cell of a cut brick started again from no displacement beside neighbours 6 mm down.)
    // (a convex cell or a free piece is known by its brick: the boxes round two fragments of one brick can be the
    // same box, and the two then shared one velocity)
    let shaped: Vec<bool> = nodes.iter().map(|nd| b.cell(nd.brick).is_some() || b.piece(nd.brick).is_some()).collect();
    let key = |k: usize| { let nd = &nodes[k]; if shaped[k] { (nd.brick | 0x8000_0000, nd.lo[0], nd.lo[2]) } else { (((nd.lo[1] as u32).wrapping_mul(73856093) ^ (nd.hi[0] as u32).wrapping_mul(19349663) ^ (nd.hi[1] as u32).wrapping_mul(83492791) ^ (nd.hi[2] as u32).wrapping_mul(2654435761)) & 0x7FFF_FFFF, nd.lo[0], nd.lo[2]) } };
    let known: Vec<bool> = warm.as_ref().map(|w| (0..n).map(|k| fixed[k] || w.0.contains_key(&key(k))).collect()).unwrap_or_default();
    let mut x0: Option<Vec<[f64; 6]>> = warm.as_ref().map(|w| (0..n).map(|k| if fixed[k] { [0.0; 6] } else { w.0.get(&key(k)).copied().unwrap_or([0.0; 6]) }).collect());
    // a dynamic step: each cell's inertia as a spring to an anchor (one more cell, fixed), and M / h^2 (x + h v)
    // on the right-hand side. `xn` keeps where the cells were.
    let dynamic = dy.as_ref().map_or(false, |d| d.h > 0.0) && warm.is_some() && gravity;
    let want_cells = dy.as_ref().map_or(false, |d| d.want_cells);
    // (a settling step, see the end of the step. It is solved to the solver's tolerance on its own move, not on the
    // loads: it is there to bring what is out of balance down, and against the loads a step can pass with 3% of
    // them left on a cell.)
    let settling = dynamic && dy.as_ref().unwrap().settle > 0;
    let mut fixed = fixed;
    let mut xn: Vec<[f64; 6]> = Vec::new();
    let mut cap = MAX_IT;
    if dynamic { let d = dy.as_ref().unwrap(); cap = if d.max_it > 0 { d.max_it } else { DYN_IT << d.boost.min(DYN_BOOST) }; xn = x0.take().unwrap(); }
    let bed_of: Vec<bool> = links.iter().map(|l| l.joint != NONE && l.geo.is_some() && mortar_bed(b, l.joint, [0.0; 3])).collect();
    let mut bed_state = vec![0u8; links.len()];    // 1 open, 2 hinged
    let mut bed_changes = 0usize;
    // a dynamic step starts from the beds as the last step left them (open, hinged): started closed each time, a
    // building with open beds never came to rest (lab complexcol: 70 steps in 280 frames, each re-opening them)
    let bed_key = |l: usize| (links[l].joint, key(links[l].a as usize), key(links[l].b as usize));
    // (a settling step starts them closed, as the solve at rest does: where the structure comes to rest does not
    // then depend on which beds opened on the way. Kept through, the complex building with a corner column shot out
    // rested with its worst joint at 0.435 of its strength; the solve at rest, and now this, has 0.855.)
    if dynamic && !settling {
        let d = dy.as_ref().unwrap();
        for l in 0..links.len() {
            if !bed_of[l] { continue; }
            match d.beds.get(&bed_key(l)) {
                Some(1) => { bed_state[l] = 1; let sp = &mut springs[l]; for k in 0..3 { sp.k[k] *= 1e-6; sp.kr[k] *= 1e-6; } }
                Some(2) => { bed_state[l] = 2; let sp = &mut springs[l]; for k in 0..3 { sp.kr[k] *= 1e-6; } }
                _ => {}
            }
        }
    }
    // The dynamic step (generalised alpha, Chung and Hulbert 1993; `DYN_RHO`). With a' = (x' - xp) / (beta h^2) and
    // v' = vp + gamma h a', where xp = x + h v + h^2 (1/2 - beta) a and vp = v + h (1 - gamma) a are what the cells
    // do left alone, the balance M a + C v + K x = f taken between the step's ends (at 1 - am for a, 1 - af for the
    // rest; C = DYN_DAMP M) is
    //     (K + cm M) x' = (f - af K x - am M a - DYN_DAMP af M v) / (1 - af) + cm M xp - DYN_DAMP M vp,
    //     cm = (1 - am) / ((1 - af) beta h^2) + DYN_DAMP gamma / (beta h)
    // so each cell's inertia is a spring cm M to an anchor (one more cell, fixed), as before. A cell with no
    // velocity kept starts at rest.
    let nl = links.len();
    // (the first step from rest is implicit Euler, am = af = 0, gamma = beta = 1: what a change lets go lands on
    // cells that their joints hold stiffly, a motion far too fast for the step, and Euler damps all of it. Started
    // with the acceleration the imbalance gives, a = (f - K x) / M, the cell over a lost column, 234 kg let go by
    // 966 kN, left its first step at 4.4 m/s and 56 joints failed round it; it moves 0.2 mm.)
    // (not when the first step is a hit on a structure at rest: nothing is let go, the cells are in balance with no
    // acceleration, which is the state the step needs, and Euler would take a part of the hit's energy with it.
    // With Euler there too, and the first step's acceleration kept, a hit on a structure at rest was 37/36 of its
    // impulse and a hit in one step 7/6: the step after counted the first one's acceleration again by 1 - gamma.
    // dyn_closed: a_free_strip_hit_in_the_middle_leaves_with_the_impulse, a_mass_arriving_at_speed_peaks_as_a_mass_dropped.)
    let first = dynamic && dy.as_ref().unwrap().vel.is_empty() && dy.as_ref().unwrap().kicks.is_empty();
    // (the damping: the material's, or none for a test of the step itself)
    let (damp_m, damp_k) = if dy.as_ref().map_or(false, |d| d.undamped) { (0.0, 0.0) } else { (DYN_DAMP, DYN_DAMP_K) };
    // (the step's length: a settling step is `DYN_SETTLE_H`)
    let hs = if dynamic { let d = dy.as_ref().unwrap(); if d.settle > 0 { DYN_SETTLE_H } else { d.h } } else { 0.0 };
    let (am, af) = if first { (0.0, 0.0) } else { ((2.0 * DYN_RHO - 1.0) / (DYN_RHO + 1.0), DYN_RHO / (DYN_RHO + 1.0)) };
    let (gam, bet) = if first { (1.0, 1.0) } else { (0.5 - am + af, (1.0 - am + af) * (1.0 - am + af) / 4.0) };
    let (mut m6, mut vn, mut an, mut xp, mut vp): (Vec<[f64; 6]>, Vec<[f64; 6]>, Vec<[f64; 6]>, Vec<[f64; 6]>, Vec<[f64; 6]>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // The loads of the step are the cells' loads less what the last rest left unbalanced (`Dyn::bias`): the solve
    // at rest stops at its tolerance, which leaves up to 1.5% of the largest load on a cell of the 12-storey tower,
    // 2.4 kN on one that weighs 2.8 kN. Taken for real, the standing tower never rested (9,600 iterations in 600
    // steps). Rest is then exact, and what a change lets go is the whole of what is out of balance.
    // hits: an impulse J at a point is the force J / h on the cell nearest it for this step, with its moment
    // about the cell's centre
    if dynamic { let d = dy.as_mut().unwrap(); let mut kicks = core::mem::take(&mut d.kicks); for (p, j, left) in kicks.iter_mut() {
        let mut best = (f64::MAX, usize::MAX);
        for k in 0..n { if fixed[k] { continue; } let c = nodes[k].c; let q = (c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) + (c[2] - p[2]) * (c[2] - p[2]); if q < best.0 { best = (q, k); } }
        // (the share of the impulse that falls in this step, as a force over the step)
        let share = left.min(hs) / KICK_TIME / hs;
        *left -= hs;
        if best.1 == usize::MAX || best.0 > 9.0 { continue; }
        let (k, c) = (best.1, nodes[best.1].c);
        let (fo, r) = ([j[0] * share, j[1] * share, j[2] * share], [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
        for i in 0..3 { f[k][i] += fo[i]; }
        f[k][3] += r[1] * fo[2] - r[2] * fo[1]; f[k][4] += r[2] * fo[0] - r[0] * fo[2]; f[k][5] += r[0] * fo[1] - r[1] * fo[0];
    } kicks.retain(|k| k.2 > 1e-9); d.kicks = kicks; }
    // plastic joints: a yielding link turns against `PL_H` of its stiffness, and every link that has yielded has
    // its plastic rotation taken off its strain, which is a pair of moments on its two cells
    let lkey = |l: usize| (key(links[l].a as usize), key(links[l].b as usize));
    let mut pl_on = vec![false; nl];
    let mut pl_off = vec![[0.0f64; 3]; nl];
    let kr_el: Vec<[f64; 3]> = springs.iter().map(|sp| sp.kr).collect();
    if let Some(d) = dy.as_ref() { if !d.plastic.is_empty() {
        for l in 0..nl { if let Some(p) = d.plastic.get(&lkey(l)) {
            pl_off[l] = p.t0;
            if p.on { pl_on[l] = true; for k in 0..3 { springs[l].kr[k] *= PL_H; } }
            let mo = springs[l].force([0.0; 3], p.t0).1;
            let (a, c) = (links[l].a as usize, links[l].b as usize);
            for i in 0..3 { f[a][3 + i] -= mo[i]; f[c][3 + i] += mo[i]; }
        } }
    } }
    let mut pl_update = false;
    // (a solve that the structure will rest on, a lot's first and a settling step, is taken `TIGHT` further: what
    // it leaves out of balance is what a rest writes off, `Dyn::bias`, and that has a limit)
    let tight = dy.is_some() && (!dynamic || settling);
    let raw: Vec<[f64; 6]> = if dy.is_some() { f.clone() } else { Vec::new() };
    let loads: Vec<[f64; 6]> = if dynamic { let d = dy.as_ref().unwrap(); (0..n).map(|k| { let mut l = f[k]; if let Some(b) = d.bias.get(&key(k)) { for i in 0..6 { l[i] -= b[i]; } } l }).collect() } else { Vec::new() };
    let mut cm = 1.0;
    // K x of the cells, from the links' springs
    let kx_of = |springs: &[Spring], x: &[[f64; 6]]| -> Vec<[f64; 6]> {
        let mut y = vec![[0.0f64; 6]; n];
        for l in 0..nl {
            let (a, c) = (links[l].a as usize, links[l].b as usize);
            let sp = &springs[l];
            let (d, r) = sp.strain(&x[a], &x[c]);
            let (fo, mo) = sp.force(d, r);
            let (ra, rb) = (&sp.ra, &sp.rb);
            let ma = [mo[0] + ra[1] * fo[2] - ra[2] * fo[1], mo[1] + ra[2] * fo[0] - ra[0] * fo[2], mo[2] + ra[0] * fo[1] - ra[1] * fo[0]];
            let mb = [mo[0] + rb[1] * fo[2] - rb[2] * fo[1], mo[1] + rb[2] * fo[0] - rb[0] * fo[2], mo[2] + rb[0] * fo[1] - rb[1] * fo[0]];
            for i in 0..3 { y[a][i] -= fo[i]; y[a][3 + i] -= ma[i]; y[c][i] += fo[i]; y[c][3 + i] += mb[i]; }
        }
        y
    };
    // The step is solved for the cells' move from xp, d = x' - xp:
    //     (K + cm M) d = (f - af K x - am M a - DYN_DAMP af M v) / (1 - af) - DYN_DAMP M vp - K xp
    // so that the solver's tolerance is on the move and not on where the cells are (6 mm of standing displacement
    // against a move of micrometres; solved for x', the cells' accelerations, (x' - xp) / (beta h^2), were the
    // solver's error times 8,100 and the step blew up: the 4-storey frame's cells at 10^8 m/s in 6 steps).
    // With damping in proportion to stiffness too, C = DYN_DAMP M + DYN_DAMP_K K, the springs' part of C v joins K:
    // with s = 1 + DYN_DAMP_K gamma / (beta h), and all of it divided by s,
    //     (K + (cm / s) M) d = ((f - af K x - am M a - DYN_DAMP af M v) / (1 - af) - DYN_DAMP M vp - K xp
    //                           - DYN_DAMP_K K vp - DYN_DAMP_K af / (1 - af) K v) / s
    let rhs_of = |springs: &[Spring], m6: &[[f64; 6]], vn: &[[f64; 6]], an: &[[f64; 6]], xn: &[[f64; 6]], xp: &[[f64; 6]], vp: &[[f64; 6]], sd: f64| -> Vec<[f64; 6]> {
        let (kx, kxp, kvp, kvn) = (kx_of(springs, xn), kx_of(springs, xp), kx_of(springs, vp), kx_of(springs, vn));
        let mut r = vec![[0.0f64; 6]; n + 1];
        for k in 0..n { if m6[k][0] == 0.0 { continue; } for i in 0..6 {   // (a fixed cell has no mass here)
            r[k][i] = ((loads[k][i] - af * kx[k][i] - am * m6[k][i] * an[k][i] - damp_m * af * m6[k][i] * vn[k][i]) / (1.0 - af) - damp_m * m6[k][i] * vp[k][i] - kxp[k][i]
                - damp_k * kvp[k][i] - damp_k * af / (1.0 - af) * kvn[k][i]) / sd;
        } }
        r
    };
    let mut sd = 1.0;
    let scale: Vec<[f64; 6]> = if dynamic { let mut v = loads.clone(); v.push([0.0; 6]); v } else { Vec::new() };
    let mut quiet = false;
    if dynamic {
        let d = dy.as_ref().unwrap();
        let h = hs;
        sd = 1.0 + damp_k * gam / (bet * h);
        cm = ((1.0 - am) / ((1.0 - af) * bet * h * h) + damp_m * gam / (bet * h)) / sd;
        m6 = vec![[0.0; 6]; n]; vn = vec![[0.0; 6]; n]; an = vec![[0.0; 6]; n];
        // New cells (cut from a brick since the last step) are put where the cells round them hold them: a solve
        // at rest of the new cells alone, the others where they are. They start in balance, and from rest.
        // (not with `Dyn::free`: a piece held by nothing has no rest to be put at, and starts where it is)
        if !d.free && known.iter().any(|k| !*k) {
            let kx = kx_of(&springs, &xn);
            let r: Vec<[f64; 6]> = (0..n).map(|k| if known[k] { [0.0; 6] } else { let mut v = loads[k]; for i in 0..6 { v[i] -= kx[k][i]; } v }).collect();
            let (xu, _) = solve_network(&springs, &ends, &known, &pos, &r, None, MAX_IT, None, false);
            for k in 0..n { if !known[k] { xn[k] = xu[k]; } }
        }
        for k in 0..n {
            if fixed[k] { continue; }
            let nd = &nodes[k];
            // (the cell's own mass. It was its load over g: a load that is not a weight, a hit or a test's pull,
            // then moved as mass; rubble lying on a cell loads it and is not counted in its mass)
            let m = (nd.mass as f64).max(1e-6);
            let (ex, ey, ez) = ((nd.hi[0] - nd.lo[0]) as f64 * tk, (nd.hi[1] - nd.lo[1]) as f64 * tk, (nd.hi[2] - nd.lo[2]) as f64 * tk);
            m6[k] = [m, m, m, (m * (ey * ey + ez * ez) / 12.0).max(1e-9), (m * (ex * ex + ez * ez) / 12.0).max(1e-9), (m * (ex * ex + ey * ey) / 12.0).max(1e-9)];
        }
        let noise = DYN_QUIET * loads.iter().take(n).map(|l| sqrt64(l[0] * l[0] + l[1] * l[1] + l[2] * l[2])).fold(0.0, f64::max);
        for k in 0..n {
            if fixed[k] { continue; }
            match (d.vel.get(&key(k)), d.acc.get(&key(k))) {
                (Some(v), Some(a)) => { vn[k] = *v; an[k] = *a; }
                _ => {}
            }
        }
        xp = xn.clone(); vp = vn.clone();
        for k in 0..n { for i in 0..6 { xp[k][i] += h * vn[k][i] + h * h * (0.5 - bet) * an[k][i]; vp[k][i] += h * (1.0 - gam) * an[k][i]; } }
        for k in 0..n {
            if fixed[k] { continue; }
            springs.push(Spring { c: nodes[k].c, ra: [0.0; 3], rb: [0.0; 3], k: [cm * m6[k][0]; 3], kr: [cm * m6[k][3], cm * m6[k][4], cm * m6[k][5]], fr: None, frr: None });
            ends.push((n as u32, k as u32));
        }
        f = rhs_of(&springs, &m6, &vn, &an, &xn, &xp, &vp, sd);
        // nothing moving and nothing out of balance: the structure is at rest, and the step is not solved
        quiet = d.vel.is_empty() && f.iter().take(n).all(|r| r[0] * r[0] + r[1] * r[1] + r[2] * r[2] <= noise * noise);
        fixed.push(true); pos.push([0.0, -100.0, 0.0]); xn.push([0.0; 6]); xp.push([0.0; 6]);
        x0 = None;
    }
    // (x' from the move d)
    let moved = |xp: &[[f64; 6]], d: Vec<[f64; 6]>| -> Vec<[f64; 6]> { let mut x = d; for k in 0..x.len() { for i in 0..6 { x[k][i] += xp[k][i]; } } x };
    // (the dynamic-model spike reads the network a test analyses: structure/dyn_spike.rs)
    #[cfg(test)] dyn_spike::capture(&springs, &ends, &fixed, &pos, &f, &nodes.iter().map(|nd| { let tk = TICK as f64; [(nd.hi[0] - nd.lo[0]) as f64 * tk, (nd.hi[1] - nd.lo[1]) as f64 * tk, (nd.hi[2] - nd.lo[2]) as f64 * tk, nd.load as f64] }).collect::<Vec<_>>());
    let (mut x, mut it) = if !dynamic { solve_network(&springs, &ends, &fixed, &pos, &f, x0, cap, None, tight) } else if quiet { (xn.clone(), 0) } else { let (d, i) = solve_network(&springs, &ends, &fixed, &pos, &f, None, cap, Some(&scale), tight); (moved(&xp, d), i) };
    let mut capped = dynamic && it >= cap;
    // Stones' beds carry no tension, and where the thrust leaves a bed's face a hinge opens through it (Heyman): a bed
    // the solution pulls apart opens (its springs all but go), one whose thrust passes outside its face hinges (it
    // carries force, no moment), and the network is solved again, until nothing more opens or hinges (8 rounds at
    // most). A hinge is not a failure: a mechanism is, and then the solution runs away (`mech`, below). Checked on the
    // elastic solution alone, the keystones of an arcade carrying a stiff floor hung from it, their beds at 10 times
    // what mortar holds, and a stone beam's butt joint over a keystone failed in hogging (complex::tests).
    // the beds' joints, each with its links, face centre and frame: a joint hinges as a whole, when the thrust summed
    // over its links leaves its face. (Tested link by link, a stone's bed under a stone beam had each link's thrust
    // inside its own part of the face and the joint's outside it, 1.04: checked as one section it failed, where a
    // hinge opens.)
    let mut bed_joints: Vec<(Vec<usize>, Geo)> = Vec::new();
    {
        let mut slot = crate::Map::default();
        for (l, k) in links.iter().enumerate() {
            if !bed_of[l] { continue; }
            let s = match slot.get(&k.joint) {
                Some(&s) => s,
                None => {
                    let Some(g) = geo_of(&b.joint_poly(k.joint), b.jaxis[k.joint as usize] as usize) else { continue };
                    bed_joints.push((Vec::new(), g)); slot.insert(k.joint, bed_joints.len() - 1); bed_joints.len() - 1
                }
            };
            bed_joints[s].0.push(l);
        }
    }
    // a joint's force and moment about its face's centre, in its face's frame
    let joint_load = |springs: &[Spring], x: &[[f64; 6]], ls: &[usize], g: &Geo| -> ([f64; 3], [f64; 3]) {
        let (mut ft, mut mt) = ([0.0f64; 3], [0.0f64; 3]);
        for &l in ls {
            let (d, r) = springs[l].strain(&x[links[l].a as usize], &x[links[l].b as usize]);
            let (fo, mo) = springs[l].force(d, r);
            let p = springs[l].c;
            let rr = [p[0] - g.c[0], p[1] - g.c[1], p[2] - g.c[2]];
            for i in 0..3 { ft[i] += fo[i]; mt[i] += mo[i]; }
            let rx = [rr[1] * fo[2] - rr[2] * fo[1], rr[2] * fo[0] - rr[0] * fo[2], rr[0] * fo[1] - rr[1] * fo[0]];
            for i in 0..3 { mt[i] += rx[i]; }
        }
        (rot(&g.fr, ft), rot(&g.fr, mt))
    };
    if !bed_joints.is_empty() {
        for _ in 0..8 {
            let mut changed = false;
            // (a dynamic step: an open bed pressed together again closes)
            if dynamic {
                for (l, lk) in links.iter().enumerate() {
                    if !bed_of[l] || bed_state[l] != 1 { continue; }
                    let g = lk.geo.as_ref().unwrap();
                    let (d, _) = springs[l].strain(&x[lk.a as usize], &x[lk.b as usize]);
                    if rot(&g.fr, d)[0] < -1e-6 { bed_state[l] = 0; springs[l] = spring(b, &nodes, lk); changed = true; bed_changes += 1; }
                }
            }
            for (l, lk) in links.iter().enumerate() {
                if !bed_of[l] || bed_state[l] == 1 { continue; }
                let g = lk.geo.as_ref().unwrap();
                let (d, r) = springs[l].strain(&x[lk.a as usize], &x[lk.b as usize]);
                let (fo, _) = springs[l].force(d, r);
                if rot(&g.fr, fo)[0] > 1.0 { bed_state[l] = 1; let sp = &mut springs[l]; for k in 0..3 { sp.k[k] *= 1e-6; sp.kr[k] *= 1e-6; } changed = true; bed_changes += 1; }
            }
            for (ls, g) in bed_joints.iter() {
                if ls.iter().any(|&l| bed_state[l] != 0) { continue; }
                let (fl, ml) = joint_load(&springs, &x, ls, g);
                let nn = -fl[0];
                if nn > 0.0 && (fabs64s(ml[1]) / nn / g.s.zu.max(1e-6) + fabs64s(ml[2]) / nn / g.s.zw.max(1e-6)) > 1.0 {
                    for &l in ls { bed_state[l] = 2; let sp = &mut springs[l]; for k in 0..3 { sp.kr[k] *= 1e-6; } }
                    changed = true; bed_changes += 1;
                }
            }
            if !changed { break; }
            // (a dynamic step: K has changed with the beds, and with it K x on the right-hand side)
            let (x2, it2) = if dynamic {
                f = rhs_of(&springs, &m6, &vn, &an, &xn, &xp, &vp, sd);
                let d0: Vec<[f64; 6]> = (0..x.len()).map(|k| { let mut d = x[k]; for i in 0..6 { d[i] -= xp[k][i]; } d }).collect();
                let (d, i) = solve_network(&springs, &ends, &fixed, &pos, &f, Some(d0), cap, Some(&scale), tight);
                (moved(&xp, d), i)
            } else { solve_network(&springs, &ends, &fixed, &pos, &f, Some(x.clone()), cap, None, tight) };
            x = x2; it += it2; if dynamic && it2 >= cap { capped = true; }
        }
    }
    // a mechanism: the network with its open and hinged beds no longer holds, and moves without bound (stone moves
    // fractions of a millimetre): a node by an open or hinged bed displaced over 5 cm
    let disp = |k: usize| sqrt64(x[k][0] * x[k][0] + x[k][1] * x[k][1] + x[k][2] * x[k][2]);
    let mech_link: Vec<bool> = links.iter().enumerate().map(|(l, lk)| bed_state[l] != 0 && (disp(lk.a as usize) > 0.05 || disp(lk.b as usize) > 0.05)).collect();
    // a dynamic step: each cell in the linear range (0), beyond it and watched (1), or gone (2); see `DYN_TURN`
    let mut state: Vec<u8> = Vec::new();
    let mut left = 0usize;
    if dynamic {
        let d = dy.as_mut().unwrap();
        let h = hs;
        if quiet { x = xn.clone(); }
        // (a settling step that moves a cell out of the linear range has found a part with no resting place, a
        // mechanism: a second of its fall in one step. It is not taken: the part falls in steps of real time.)
        let mut refused = false;
        if settling {
            for k in 0..n {
                if fixed[k] { continue; }
                let (t, u) = ([x[k][3] - xn[k][3], x[k][4] - xn[k][4], x[k][5] - xn[k][5]], [x[k][0] - xn[k][0], x[k][1] - xn[k][1], x[k][2] - xn[k][2]]);
                if t[0] * t[0] + t[1] * t[1] + t[2] * t[2] > DYN_SETTLE_TURN * DYN_SETTLE_TURN || u[0] * u[0] + u[1] * u[1] + u[2] * u[2] > DYN_SETTLE_MOVE * DYN_SETTLE_MOVE { refused = true; break; }
            }
            if refused { x = xn.clone(); }
        }
        d.vel.clear(); d.acc.clear(); d.speed = 0.0; d.kinetic = 0.0; d.kinetic_lin = 0.0; d.rms = 0.0; d.its = it; d.capped = capped; d.left = 0; d.cut = bed_changes;
        d.beds.clear(); for l in 0..links.len() { if bed_state[l] != 0 { d.beds.insert(bed_key(l), bed_state[l]); } }
        // the cells' new velocities and accelerations
        let (mut vel, mut acc): (crate::Map<(u32, i32, i32), [f64; 6]>, crate::Map<(u32, i32, i32), [f64; 6]>) = (Default::default(), Default::default());
        if !quiet && !refused { for k in 0..n {
            if fixed[k] { continue; }
            let (mut v, mut a) = ([0.0; 6], [0.0; 6]);
            for i in 0..6 { a[i] = (x[k][i] - xp[k][i]) / (bet * h * h); v[i] = vp[k][i] + gam * h * a[i]; }
            let s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if s2 > d.speed * d.speed { d.speed = sqrt64(s2); }
            d.kinetic_lin += 0.5 * m6[k][0] * s2;
            d.kinetic += 0.5 * m6[k][0] * s2 + 0.5 * (m6[k][3] * v[3] * v[3] + m6[k][4] * v[4] * v[4] + m6[k][5] * v[5] * v[5]);
            // (after a first step that was implicit Euler no acceleration is kept: Euler has given the step's force
            // whole, and the next step, taking this one's acceleration as the old one, would give 1 - gamma = 1/6
            // of it again)
            vel.insert(key(k), v); acc.insert(key(k), if first { [0.0; 6] } else { a });
        } }
        let mass: f64 = (0..n).map(|k| m6[k][0]).sum();
        d.rms = if mass > 0.0 { sqrt64(d.kinetic_lin * 2.0 / mass) } else { 0.0 };
        if d.undamped {
            // (for tests of the step: the work the loads have done, and kinetic plus strain energy less that work,
            // which a step without damping keeps)
            let kx = kx_of(&springs, &x);
            let (mut u, mut w) = (0.0, 0.0);
            for k in 0..n { if !fixed[k] { for i in 0..6 { u += 0.5 * x[k][i] * kx[k][i]; w += loads[k][i] * x[k][i]; } } }
            d.work = w; d.energy = d.kinetic + u - w;
        }
        // Which cells have left. How far a cell is from where it rested, in ranges: the larger of its turn over
        // DYN_TURN and its move over DYN_MOVE. Beyond 1 it is watched: after DYN_WATCH s, a cell a range further
        // out than when it crossed has left (it is falling: from rest a part falls 5 cm in 0.1 s and 31 cm by
        // 0.25 s); one that after twice as long has not gone a quarter of a range further has found a new resting
        // place (a cracked piece hanging by its softened joints), and rests there. (A settling step is no time.)
        // (Nor has one left that after twice as long is slower than `DYN_REST_CELL`, a speed a falling part passes
        // in 5 ms: it creeps on softened joints. Without this the 4-storey masonry building left on one pier
        // watched 3 or 4 such cells at 2-8 mm/s for 600 steps and never rested: dyn_tests.)
        let slow = |k: usize| vel.get(&key(k)).map_or(true, |v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2] <= DYN_REST_CELL * DYN_REST_CELL && v[3] * v[3] + v[4] * v[4] + v[5] * v[5] <= DYN_REST_CELL * DYN_REST_CELL);
        state = vec![0u8; n];
        let mut beyond: crate::Map<(u32, i32, i32), (f64, f64)> = Default::default();
        for k in 0..n {
            if fixed[k] { continue; }
            let Some(r) = d.rest.get(&key(k)).copied() else { d.rest.insert(key(k), x[k]); continue };
            let (t, u) = ([x[k][3] - r[3], x[k][4] - r[4], x[k][5] - r[5]], [x[k][0] - r[0], x[k][1] - r[1], x[k][2] - r[2]]);
            let m = (sqrt64(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]) / DYN_TURN).max(sqrt64(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / DYN_MOVE);
            if m <= 1.0 { continue; }
            let (t0, m0) = d.beyond.get(&key(k)).copied().unwrap_or((0.0, m));
            let t1 = t0 + if d.settle > 0 { 0.0 } else { d.h };
            if t1 >= DYN_WATCH && m >= m0 + 1.0 { state[k] = 2; left += 1; beyond.insert(key(k), (t1, m0)); }
            else if t1 >= 2.0 * DYN_WATCH && (m < m0 + 0.25 || slow(k)) { d.rest.insert(key(k), x[k]); }
            else { state[k] = 1; beyond.insert(key(k), (t1, m0)); }
        }
        d.beyond = beyond;
        d.left = left;
        // Rest. The step converged, the structure as a whole is slower than `DYN_REST`, no cell is faster than
        // `DYN_REST_CELL`, and none is watched. If what is then out of balance is within the solver's tolerance
        // (`DYN_SETTLED`), the structure rests there, and that remainder is kept off the loads from now on
        // (`Dyn::bias`). If more is out of balance, it is creeping to a resting place it has not reached (stones on
        // opened beds a millionth as stiff move for minutes): the next step is a settling step, `DYN_SETTLE_H`
        // long and from rest, which is nearly the solve at rest, and so on until it is in balance. Rested while it
        // crept, the complex building with a corner column shot out kept its worst joint at 0.435 of its strength,
        // where the solve at rest has 0.855: half of what had come onto that joint was written off as tolerance.
        // Calm asks that the step converged: one that stopped at its cap has not found the slow motions, and a
        // structure moving slower than it should is not calm. (Not asked for, with a cap of 16: the complex building
        // shot through from the west had 401 of 412 moving steps at the cap and 8 rests forced on it.)
        d.boost = if capped { (d.boost + 1).min(DYN_BOOST) } else { d.boost.saturating_sub(1) };
        pl_update = !quiet && !refused && !d.no_yield;
        let calm = !capped && d.rms <= DYN_REST && d.speed <= DYN_REST_CELL && d.beyond.is_empty() && d.kicks.is_empty();
        // What a rest writes off (`bias`) is what the solve's tolerance left, and never more than `DYN_BIAS` of the
        // cell's own load. It was all that the cell's loads and springs left out of balance, whatever its size: a
        // rest forced on a structure not in balance then held its cells where they were, against up to 4.65 times
        // their own weight (dyn_tests::a_rest_writes_off_no_more_than_the_tolerance), a part hanging in the air.
        let big_raw = raw.iter().take(n).map(|l| sqrt64(l[0] * l[0] + l[1] * l[1] + l[2] * l[2])).fold(0.0, f64::max);
        let rest_now = |d: &mut Dyn, x: &[[f64; 6]]| {
            let kx = kx_of(&springs, x);
            d.settle = 0; d.no_settle = false; d.wait = 0.0;
            d.rest.clear(); d.bias.clear(); d.beyond.clear();
            d.bias_share = 0.0;
            for k in 0..n { if !fixed[k] { d.rest.insert(key(k), x[k]); let mut b = raw[k]; for i in 0..6 { b[i] -= kx[k][i]; }
                let (bn, rn) = (sqrt64(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]), sqrt64(raw[k][0] * raw[k][0] + raw[k][1] * raw[k][1] + raw[k][2] * raw[k][2]));
                let lim = (DYN_BIAS * rn).max(DYN_BIAS_FLOOR * big_raw);
                if bn > lim { let f = lim / bn; for i in 0..6 { b[i] *= f; } }
                if rn > 0.0 && DYN_BIAS * rn >= DYN_BIAS_FLOOR * big_raw { d.bias_share = d.bias_share.max(bn.min(lim) / rn); }
                d.bias.insert(key(k), b); } }
        };
        if quiet { d.settle = 0; }
        else if refused { d.settle = 0; d.no_settle = true; d.wait = DYN_SETTLE_H; }
        else if settling {
            let kx = kx_of(&springs, &x);
            let big = loads.iter().take(n).map(|l| sqrt64(l[0] * l[0] + l[1] * l[1] + l[2] * l[2])).fold(0.0, f64::max);
            let out = (0..n).filter(|&k| !fixed[k]).map(|k| { let r = [loads[k][0] - kx[k][0], loads[k][1] - kx[k][1], loads[k][2] - kx[k][2]]; sqrt64(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) }).fold(0.0, f64::max);
            d.imbalance = if big > 0.0 { out / big } else { 0.0 };
            // (it rests after a settling step that converged in balance: that step set the beds as they are at
            // rest. One that stopped at its cap, or left too much out of balance, is followed by another; after
            // `DYN_SETTLE_MAX` of them it is not at rest, and goes on in steps of real time. No rest is forced.)
            if !capped && d.imbalance <= DYN_SETTLED { rest_now(d, &x); }
            else if d.settle >= DYN_SETTLE_MAX { d.forced += 1; d.settle = 0; d.no_settle = true; d.wait = DYN_SETTLE_H; }
            else { d.settle += 1; }
        }
        else if calm {
            // (calm after it could not settle: it tries again a second later)
            if d.no_settle { d.wait -= d.h; if d.wait <= 0.0 { d.no_settle = false; } d.vel = vel; d.acc = acc; } else { d.settle = 1; }
        }
        else { d.vel = vel; d.acc = acc; d.settle = 0; }
    } else if let Some(d) = dy.as_mut() {
        // (solved at rest: the beds as this solve left them are where the next dynamic step starts)
        d.vel.clear(); d.acc.clear(); d.speed = 0.0; d.kinetic = 0.0; d.its = it; d.capped = false; d.left = 0; d.cut = 0; d.settle = 0; d.no_settle = false;
        d.rest.clear(); d.beyond.clear(); for k in 0..n { if !fixed[k] { d.rest.insert(key(k), x[k]); } }
        let big_raw = raw.iter().take(n).map(|l| sqrt64(l[0] * l[0] + l[1] * l[1] + l[2] * l[2])).fold(0.0, f64::max);
        let kx = kx_of(&springs, &x); d.bias.clear(); d.bias_share = 0.0; d.boost = 0; d.wait = 0.0; d.residual = 0.0;
        for k in 0..n { if !fixed[k] { let mut b = raw[k]; for i in 0..6 { b[i] -= kx[k][i]; }
            let (bn, rn) = (sqrt64(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]), sqrt64(raw[k][0] * raw[k][0] + raw[k][1] * raw[k][1] + raw[k][2] * raw[k][2]));
            if big_raw > 0.0 { d.residual = d.residual.max(bn / big_raw); }
            let lim = (DYN_BIAS * rn).max(DYN_BIAS_FLOOR * big_raw);
            if bn > lim { let f = lim / bn; for i in 0..6 { b[i] *= f; } }
            if rn > 0.0 && DYN_BIAS * rn >= DYN_BIAS_FLOOR * big_raw { d.bias_share = d.bias_share.max(bn.min(lim) / rn); }
            d.bias.insert(key(k), b); } }
        d.beds.clear(); for l in 0..links.len() { if bed_state[l] != 0 { d.beds.insert((links[l].joint, key(links[l].a as usize), key(links[l].b as usize)), bed_state[l]); } }
    }
    if let Some(w) = warm { w.0.clear(); for k in 0..n { if !fixed[k] { w.0.insert(key(k), x[k]); } } }
    let cells_out: Vec<([i32; 5], [f64; 3], [f64; 6])> = if want_cells { (0..n).filter(|&k| !fixed[k]).map(|k| ([nodes[k].brick as i32, nodes[k].lo[0], nodes[k].lo[2], nodes[k].hi[0], nodes[k].hi[2]], nodes[k].c, x[k])).collect() } else { Vec::new() };
    let mut out = Analysis { yielded: Vec::new(), failed: Vec::new(), cracks: Vec::new(), max_ratio: 0.0, iterations: it, nodes: n, worst_in: (0.0, 0, 0, 0.0, 0.0, 0.0), ratios: Vec::new(), forces: Vec::new(), inner: Vec::new(), beds: Vec::new(), cells: Vec::new(), inner_all: Vec::new(), left_cut: 0 };
    // forces on each face (on the max side's cell, from the min side's): force and moment about the face centre
    let face_load = |l: usize| -> ([f64; 3], [f64; 3]) {
        let sp = &springs[l];
        let (a, c) = (links[l].a as usize, links[l].b as usize);
        let (d, r) = sp.strain(&x[a], &x[c]);
        let o = pl_off[l];
        sp.force(d, [r[0] - o[0], r[1] - o[1], r[2] - o[2]])
    };
    // a link's or joint's force and moment in its face's frame (normal, u, w)
    let local = |ax: usize, geo: &Option<Geo>, v: [f64; 3]| -> [f64; 3] {
        match geo { Some(g) => rot(&g.fr, v), None => [v[ax], v[(ax + 1) % 3], v[(ax + 2) % 3]] }
    };
    // joints: their sub-links summed about the joint's face centre, then checked as one section
    let mut jsum: Vec<(u32, [f64; 3], [f64; 3], [f64; 3], f64)> = Vec::new();   // joint, force, moment, force-weighted.., area
    let mut jslot = crate::Map::default();
    let mut parts: Vec<Vec<usize>> = Vec::new();
    for (l, k) in links.iter().enumerate() {
        if k.joint == NONE || !live[k.a as usize] || !live[k.b as usize] { continue; }
        let s = *jslot.entry(k.joint).or_insert_with(|| { jsum.push((k.joint, [0.0; 3], [0.0; 3], [0.0; 3], 0.0)); parts.push(Vec::new()); jsum.len() - 1 });
        parts[s].push(l);
    }
    let mut pl_now = dy.as_mut().map(|d| core::mem::take(&mut d.plastic)).unwrap_or_default();
    let (mut pl_count, mut pl_turn) = (0usize, 0.0f64);
    for (s, ls) in parts.iter().enumerate() {
        let j = jsum[s].0; let ju = j as usize;
        let ax = b.jaxis[ju] as usize;
        // the joint's face: the union of its sub-faces (or the polygon where a convex cell meets it); its centre,
        // area and section
        let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for &l in ls { for d in 0..3 { lo[d] = lo[d].min(links[l].face_lo[d]); hi[d] = hi[d].max(links[l].face_hi[d]); } }
        let tk = TICK as f64;
        let jgeo = if links[ls[0]].geo.is_some() { geo_of(&b.joint_poly(j), ax) } else { None };
        let c = match &jgeo { Some(g) => g.c, None => [0.5 * (lo[0] + hi[0]) as f64 * tk, 0.5 * (lo[1] + hi[1]) as f64 * tk, 0.5 * (lo[2] + hi[2]) as f64 * tk] };
        let (mut ft, mut mt) = ([0.0f64; 3], [0.0f64; 3]);
        for &l in ls {
            let (fo, mo) = face_load(l);
            let p = springs[l].c;
            let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            for i in 0..3 { ft[i] += fo[i]; mt[i] += mo[i]; }
            let rx = [r[1] * fo[2] - r[2] * fo[1], r[2] * fo[0] - r[0] * fo[2], r[0] * fo[1] - r[1] * fo[0]];
            for i in 0..3 { mt[i] += rx[i]; }
        }
        let (ia, ib) = (b.ja[ju] as usize, b.jb[ju] as usize);
        let (ma, mb) = (b.mat[ia], b.mat[ib]);
        let sect = match &jgeo { Some(g) => g.s, None => { let (u, w) = ((ax + 1) % 3, (ax + 2) % 3); Sect::rect((hi[u] - lo[u]) as f64 * tk, (hi[w] - lo[w]) as f64 * tk, b.jarea[ju] as f64 * tk * tk) } };
        let conn = connection(b, j);
        let cracked_masonry = ma == mat::MASONRY || mb == mat::MASONRY;
        let residual: [f32; 3] = if b.jdmg[ju] & 1 == 0 { [1.0; 3] } else if cracked_masonry { [1.0, 0.05, 0.4] } else { [0.5; 3] };
        let sa = strength(ma); let sb = strength(mb);
        let cap = [sa[0].min(sb[0]) * conn[0] * residual[0], sa[1].min(sb[1]) * conn[1] * residual[1], sa[2].min(sb[2]) * conn[2] * residual[2]];
        let flex = flexural_of(b, ia).min(flexural_of(b, ib)) * conn[1] * residual[1];
        let hinged = hinge(b, ia, ib);
        let (fl, mut ml) = (local(ax, &jgeo, ft), local(ax, &jgeo, mt));
        out.forces.push((j, fl, ml));
        if hinged { ml = [0.0; 3]; }
        // (a stone's bed: no tension, the thrust inside the face)
        let normal = match &jgeo { Some(g) => g.fr[0], None => { let mut e = [0.0; 3]; e[ax.min(2)] = 1.0; e } };
        let bed = mortar_bed(b, j, normal);
        if bed { out.beds.push(j); }
        // (a bed: open, it carries nothing; hinged, only sliding is checked; part of a mechanism, it gives way)
        let (mut opened, mut hinged_bed, mut mech) = (true, false, false);
        for &l in ls.iter() { if bed_state[l] != 1 { opened = false; } if bed_state[l] == 2 { hinged_bed = true; } if mech_link[l] { mech = true; } }
        let ratio = if !bed { section_ratio(&sect, fl, ml, cap, flex) }
            else if mech { 2.0 }
            else if opened && !ls.is_empty() { 0.0 }
            else if hinged_bed { let n = (-fl[0]).max(0.0); let v = sqrt64(fl[1] * fl[1] + fl[2] * fl[2]); (v / (0.2 * sa[2].min(sb[2]) as f64 * sect.area + 0.4 * n)) as f32 }
            else { bed_ratio(&sect, fl, ml, [sa[0].min(sb[0]), 0.0, 0.2 * sa[2].min(sb[2])]) };
        // (a dynamic step: a joint between a cell that has left and one that has not is cut; one between two that
        // have left goes with them unchecked)
        // (watched cells are not yet cut from what stands, but a joint between two cells beyond the range, watched
        // or gone, is not checked: what it shows is their turning)
        let (mut g_all, mut g_cross) = (!state.is_empty(), false);
        if !state.is_empty() { for &l in ls.iter() { let (sa, sb) = (state[links[l].a as usize], state[links[l].b as usize]); if sa == 0 || sb == 0 { g_all = false; } if (sa == 2 && sb == 0) || (sb == 2 && sa == 0) { g_cross = true; } } }
        // a ductile joint that bending takes over its strength yields (`Plastic`)
        let mut ratio = ratio;
        let theta_u = ductility(ma).min(ductility(mb));
        if pl_update && theta_u > 0.0 && !bed && !hinged && !g_cross && !g_all {
            let rot_of = |l: usize| springs[l].strain(&x[links[l].a as usize], &x[links[l].b as usize]).1;
            if ls.iter().any(|&l| pl_on[l]) {
                let (mut turned, mut work) = (0.0f64, 0.0f64);
                for &l in ls.iter() {
                    let r = rot_of(l);
                    let p = pl_now.entry(lkey(l)).or_insert_with(Plastic::default);
                    let t = [r[0] - p.ry[0], r[1] - p.ry[1], r[2] - p.ry[2]];
                    turned = turned.max(p.turned + sqrt64(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]));
                    let m = springs[l].force([0.0; 3], [r[0] - p.t0[0], r[1] - p.t0[1], r[2] - p.t0[2]]).1;
                    for i in 0..3 { work += m[i] * (r[i] - p.r[i]); }
                }
                pl_turn = pl_turn.max(turned);
                if turned > theta_u || ratio > PL_OVER { ratio = ratio.max(1.01); for &l in ls.iter() { pl_now.remove(&lkey(l)); } }
                else if work < 0.0 {
                    // (turning back: elastic again, about the turn it keeps)
                    for &l in ls.iter() {
                        let (r, sp) = (rot_of(l), &springs[l]);
                        let p = pl_now.get_mut(&lkey(l)).unwrap();
                        let (rl, tl) = (sp.rloc(r), sp.rloc(p.t0));
                        let mut nt = [0.0; 3];
                        for i in 0..3 { nt[i] = if kr_el[l][i] > 0.0 { rl[i] - sp.kr[i] * (rl[i] - tl[i]) / kr_el[l][i] } else { rl[i] }; }
                        p.on = false; p.t0 = sp.rwor(nt); p.r = r; p.lev = ratio.max(1.0); p.turned = turned;
                    }
                    ratio = ratio.min(0.999);
                }
                else { for &l in ls.iter() { let r = rot_of(l); pl_now.get_mut(&lkey(l)).unwrap().r = r; } ratio = ratio.min(0.999); out.yielded.push(j); pl_count += 1; }
            } else if ratio > ls.iter().map(|&l| pl_now.get(&lkey(l)).map_or(1.0, |p| p.lev.max(1.0))).fold(1.0f32, f32::max) && section_ratio(&sect, fl, [0.0; 3], cap, flex) < 1.0 {
                let lev = ls.iter().map(|&l| pl_now.get(&lkey(l)).map_or(1.0, |p| p.lev.max(1.0))).fold(1.0f32, f32::max);
                for &l in ls.iter() {
                    let (r, sp) = (rot_of(l), &springs[l]);
                    let p = pl_now.entry(lkey(l)).or_insert_with(Plastic::default);
                    let (rl, tl) = (sp.rloc(r), sp.rloc(p.t0));
                    let mut nt = [0.0; 3];
                    // (the moment it carries now, brought down to its strength, is the moment it yields at)
                    for i in 0..3 { nt[i] = if kr_el[l][i] > 0.0 { rl[i] - (rl[i] - tl[i]) * lev as f64 / (ratio as f64 * PL_H) } else { rl[i] }; }
                    p.on = true; p.t0 = sp.rwor(nt); p.ry = r; p.r = r;
                }
                ratio = 0.999; out.yielded.push(j); pl_count += 1;
            }
        }
        // (a joint that has yielded and hardened carries up to what it hardened to)
        if ratio > 1.0 && ratio < DYN_CUT && !ls.is_empty() { let lev = ls.iter().map(|&l| pl_now.get(&lkey(l)).map_or(1.0, |p| if p.on { 1.0 } else { p.lev })).fold(1.0f32, f32::max); if ratio <= lev { ratio = 0.999; } }
        let ratio = if g_cross { DYN_CUT } else if g_all { 0.0 } else { ratio };
        out.ratios.push((j, ratio));
        if ratio > out.max_ratio && ratio < DYN_CUT { out.max_ratio = ratio; }
        if ratio > 1.0 { out.failed.push((j, ratio)); }
        if ratio >= DYN_CUT { out.left_cut += 1; }
    }
    if let Some(d) = dy.as_mut() { d.plastic = pl_now; d.yielding = pl_count; d.plastic_turn = d.plastic_turn.max(pl_turn); }
    // faces inside bricks: the brick's own section; over strength it cracks there
    for (l, k) in links.iter().enumerate() {
        if k.joint != NONE || !live[k.a as usize] { continue; }
        let i = nodes[k.a as usize].brick;
        let m = b.mat[i as usize];
        let (u, w) = ((k.axis + 1) % 3, (k.axis + 2) % 3);
        let tk = TICK as f64;
        let (eu, ew) = ((k.face_hi[u] - k.face_lo[u]) as f64 * tk, (k.face_hi[w] - k.face_lo[w]) as f64 * tk);
        let (fo, mo) = face_load(l);
        let ratio = section_ratio(&Sect::rect(eu, ew, eu * ew), local(k.axis, &None, fo), local(k.axis, &None, mo), strength(m), flexural_of(b, i as usize));
        let ratio = if state.is_empty() { ratio } else { let (sa, sb) = (state[k.a as usize], state[k.b as usize]); if (sa == 2 && sb == 0) || (sb == 2 && sa == 0) { out.left_cut += 1; DYN_CUT } else if sa != 0 && sb != 0 { 0.0 } else { ratio } };
        if want_cells { out.inner_all.push((i, ratio.min(1.0))); }
        if ratio > out.worst_in.0 { out.worst_in = (ratio, i, k.axis, fo[k.axis] as f32, (mo[u] * mo[u] + mo[w] * mo[w]) as f32, 0.0); }
        if ratio > out.max_ratio && ratio < DYN_CUT { out.max_ratio = ratio; }
        if ratio > 1.0 { out.cracks.push(Crack { brick: i, axis: k.axis, at: k.face_lo[k.axis], ratio }); }
        else if ratio > 0.7 { out.inner.push((i, k.axis, k.face_lo[k.axis], ratio)); }
    }
    let _ = gravity;
    if want_cells { if let Some(d) = dy.as_mut() { d.cell_vel = (0..n).filter(|&k| !fixed[k]).map(|k| d.vel.get(&key(k)).map_or([0.0; 6], |v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32, v[4] as f32, v[5] as f32])).collect(); } }
    out.cells = cells_out;
    out
}

/// A spring at a face between two cells: stiffness per axis in world axes (the faces are axis-aligned) for the
/// relative displacement at the face centre (`k`: normal along the face's axis, shear along the others) and the
/// relative rotation (`kr`: torsion about the axis, bending about the others), and the arms from each cell's
/// centre to the face centre (m).
/// A face that is not on an axis plane (between two cells of a fractured brick) has its frame in `fr`, and `k`, `kr`
/// are then along its normal, u and w.
#[derive(Clone)]
/// A face's springs: translation stiffness `k` along the rows of frame `fr` (world axes if none), rotation
/// stiffness `kr` along the rows of `frr` (`fr` if none: a free piece's link has its own frames for the two).
struct Spring { c: [f64; 3], ra: [f64; 3], rb: [f64; 3], k: [f64; 3], kr: [f64; 3], fr: Option<[[f64; 3]; 3]>, frr: Option<[[f64; 3]; 3]> }
impl Spring {
    /// force and moment (world) for relative displacement d and rotation r
    #[inline] fn force(&self, d: [f64; 3], r: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        match &self.fr {
            None => ([self.k[0] * d[0], self.k[1] * d[1], self.k[2] * d[2]], [self.kr[0] * r[0], self.kr[1] * r[1], self.kr[2] * r[2]]),
            Some(fr) => {
                let frr = self.frr.as_ref().unwrap_or(fr);
                let (dl, rl) = (rot(fr, d), rot(frr, r));
                (rot_t(fr, [self.k[0] * dl[0], self.k[1] * dl[1], self.k[2] * dl[2]]), rot_t(frr, [self.kr[0] * rl[0], self.kr[1] * rl[1], self.kr[2] * rl[2]]))
            }
        }
    }
    /// a rotation in the frame of the face's rotational stiffness, and back
    fn rloc(&self, r: [f64; 3]) -> [f64; 3] { match &self.fr { None => r, Some(fr) => rot(self.frr.as_ref().unwrap_or(fr), r) } }
    fn rwor(&self, r: [f64; 3]) -> [f64; 3] { match &self.fr { None => r, Some(fr) => rot_t(self.frr.as_ref().unwrap_or(fr), r) } }
    /// the 6 x 6 stiffness of the face (force and moment against displacement and rotation), world axes
    fn d6(&self) -> M6 {
        let mut d = [[0.0; 6]; 6];
        match &self.fr {
            None => { for i in 0..3 { d[i][i] = self.k[i]; d[3 + i][3 + i] = self.kr[i]; } }
            Some(fr) => { let frr = self.frr.as_ref().unwrap_or(fr); for i in 0..3 { for j in 0..3 { for a in 0..3 { d[i][j] += fr[a][i] * self.k[a] * fr[a][j]; d[3 + i][3 + j] += frr[a][i] * self.kr[a] * frr[a][j]; } } } }
        }
        d
    }
    /// relative displacement at the face centre and relative rotation (cell b against cell a)
    fn strain(&self, xa: &[f64; 6], xb: &[f64; 6]) -> ([f64; 3], [f64; 3]) {
        let pt = |x: &[f64; 6], r: &[f64; 3]| [x[0] + x[4] * r[2] - x[5] * r[1], x[1] + x[5] * r[0] - x[3] * r[2], x[2] + x[3] * r[1] - x[4] * r[0]];
        let (pa, pb) = (pt(xa, &self.ra), pt(xb, &self.rb));
        ([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]], [xb[3] - xa[3], xb[4] - xa[4], xb[5] - xa[5]])
    }
}
#[inline] fn fabs64(x: f64) -> f64 { if x < 0.0 { -x } else { x } }
/// Eigenvalues and eigenvectors (rows) of a symmetric 3 x 3 matrix (Jacobi rotations).
fn eig3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];   // columns are the eigenvectors
    for _ in 0..50 {
        let (mut p, mut q, mut big) = (0, 1, 0.0);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] { if fabs64(a[i][j]) > big { big = fabs64(a[i][j]); p = i; q = j; } }
        if big < 1e-30 + 1e-14 * (fabs64(a[0][0]) + fabs64(a[1][1]) + fabs64(a[2][2])) { break; }
        // the rotation that clears a[p][q], without trigonometry
        let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = (if th >= 0.0 { 1.0 } else { -1.0 }) / (fabs64(th) + sqrt64(th * th + 1.0));
        let c = 1.0 / sqrt64(t * t + 1.0); let s = t * c;
        for k in 0..3 { let (akp, akq) = (a[k][p], a[k][q]); a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for k in 0..3 { let (apk, aqk) = (a[p][k], a[q][k]); a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for k in 0..3 { let (vkp, vkq) = (v[k][p], v[k][q]); v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
    ([a[0][0], a[1][1], a[2][2]], [[v[0][0], v[1][0], v[2][0]], [v[0][1], v[1][1], v[2][1]], [v[0][2], v[1][2], v[2][2]]])
}
/// Young's modulus of brick `i` (Pa). A timber floor is joists (50 x 225 mm at 400 mm): its plate has the
/// bending stiffness of the joists, about a tenth of solid timber's.
pub fn modulus_of(b: &Bricks, i: usize) -> f64 {
    match b.mat[i] { mat::CONCRETE => 30e9, mat::MASONRY => 5e9, mat::STEEL => 200e9 * STEEL_STIFF, mat::GLASS => 70e9, _ => if b.paint[i] == 8 { 1e9 } else { 11e9 } }
}
fn spring(b: &Bricks, nodes: &[Node], l: &Link) -> Spring {
    let (na, nb) = (&nodes[l.a as usize], &nodes[l.b as usize]);
    let t = TICK as f64;
    let (pa, pb) = (na.c, nb.c);
    let ax = l.axis;
    // the face as a section, in its frame (normal, u, w): a rectangle on the grid, or a convex cell's polygon
    let (c, fr, sec) = match &l.geo {
        Some(g) => (g.c, g.fr, g.s),
        None => {
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            let (eu, ew) = ((l.face_hi[u] - l.face_lo[u]) as f64 * t, (l.face_hi[w] - l.face_lo[w]) as f64 * t);
            let e = |d: usize| { let mut v = [0.0; 3]; v[d] = 1.0; v };
            ([0.5 * (l.face_lo[0] + l.face_hi[0]) as f64 * t, 0.5 * (l.face_lo[1] + l.face_hi[1]) as f64 * t, 0.5 * (l.face_lo[2] + l.face_hi[2]) as f64 * t], [e(ax), e(u), e(w)], Sect::rect(eu, ew, eu * ew))
        }
    };
    let n = fr[0];
    let along = |p: [f64; 3]| ((c[0] - p[0]) * n[0] + (c[1] - p[1]) * n[1] + (c[2] - p[2]) * n[2]).abs().max(0.5 * t);
    let (la, lb) = (along(pa), along(pb));
    let (ea, eb) = (modulus_of(b, na.brick as usize), modulus_of(b, nb.brick as usize));
    // the two halves in series (compliance per unit E: length / E)
    let comp = la / ea + lb / eb;
    let gcomp = comp / 0.4;
    // in the face's frame: normal, two shears; torsion, bending about u (stresses vary along w), about w
    let kl = [sec.area / comp, sec.area / gcomp, sec.area / gcomp];
    let krl = [sec.j / gcomp, sec.iu / comp, sec.iw / comp];
    let (mut k, mut kr, mut gen_fr) = if ax < 3 {
        let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
        let (mut k, mut kr) = ([0.0; 3], [0.0; 3]);
        k[ax] = kl[0]; k[u] = kl[1]; k[w] = kl[2]; kr[ax] = krl[0]; kr[u] = krl[1]; kr[w] = krl[2];
        (k, kr, None)
    } else { (kl, krl, Some(fr)) };
    // the links' own factors: infill conducts little, a stair flight is flexible, steel in a brick pocket bears only
    // vertically, a timber floor's seam with another material is a hinge
    let (ia, ib) = (na.brick as usize, nb.brick as usize);
    let infill = (b.flags[ia] | b.flags[ib]) & F_INFILL != 0;
    let mono = (b.flags[ia] | b.flags[ib]) & crate::brick::F_MONO != 0;
    let (mi, mo) = (b.mat[ia], b.mat[ib]);
    let pocket = l.joint != NONE && ax != 1 && ax < 3 && ((mi == mat::STEEL && mo == mat::MASONRY) || (mi == mat::MASONRY && mo == mat::STEEL));
    let mut f = if infill || pocket { 0.02 } else if mono { 0.05 } else { 1.0 };
    // a cracked joint is about half as stiff (a cracked concrete section's I is 0.3-0.5 of the uncracked one),
    // and wear softens it further, to a fifth when worn through (sim.rs, wear_joints): its load moves on
    if l.joint != NONE { let d = b.jdmg[l.joint as usize]; f *= (if d & 1 != 0 { 0.5 } else { 1.0 }) * (1.0 - 0.6 * (d >> 1) as f64 / 127.0); }
    for i in 0..3 { k[i] *= f; kr[i] *= f; }
    if l.joint != NONE && hinge(b, ia, ib) { if ax < 3 { kr[(ax + 1) % 3] = 0.0; kr[(ax + 2) % 3] = 0.0; } else { kr[1] = 0.0; kr[2] = 0.0; } }
    // a joist floor spans one way, along its joists (x: from the walls to the beams): across them only the
    // boards bend, and open joists hardly twist, each about a twentieth as stiff as a solid plate. As a plate
    // bending both ways, a 3.25 m bay under 40 kPa came out at a third of its strength; with the boards soft but
    // the deck still twisting as a solid plate, 0.68 (torsion about the joists carried a third of the load to the
    // walls along the bay's sides)
    // A free piece's side of a face at an angle to it (a brace's end cut level on a footing) is a bar from its
    // centroid to the face: along that line it is stretched, across it sheared, over its true section A |n . r|
    // (taken along the face's normal, a brace force crossing a level cut at 40 degrees sheared the brace's whole
    // length to the cut: a braced portal frame came out half as stiff as frame plus brace). The two halves'
    // compliances add as tensors; the stiffness is that sum's inverse, in its own frame. Rotation keeps the face's.
    let free = |i: usize| b.hull[i] == crate::brick::HULL_FREE;
    let mut frr = None;
    if l.geo.is_some() && (free(ia) || free(ib)) {
        let mut cm = [[0.0f64; 3]; 3];
        for (p, e, side_free) in [(pa, ea, free(ia)), (pb, eb, free(ib))] {
            let r = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
            let rl = sqrt64(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            let (dir, len, area) = if side_free && rl > 1e-9 {
                let d = [r[0] / rl, r[1] / rl, r[2] / rl];
                let cs = fabs64(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).max(0.1);
                (d, rl.max(0.5 * t), sec.area * cs)
            } else { (n, along(p), sec.area) };
            for i in 0..3 { for j in 0..3 {
                let dd = dir[i] * dir[j];
                let id = if i == j { 1.0 } else { 0.0 };
                cm[i][j] += len / (e * area) * dd + len / (0.4 * e * area) * (id - dd);
            } }
        }
        let (vals, vecs) = eig3(cm);
        k = [f / vals[0], f / vals[1], f / vals[2]];
        frr = Some(fr);
        gen_fr = Some(vecs);
    }
    let tf = |i: usize| b.mat[i] == mat::TIMBER && b.paint[i] == 8;
    if tf(ia) && tf(ib) && ax == 2 { kr[0] *= 0.05; kr[2] *= 0.05; }
    if tf(ia) && tf(ib) && ax == 0 { kr[0] *= 0.05; }
    Spring { c, ra: [c[0] - pa[0], c[1] - pa[1], c[2] - pa[2]], rb: [c[0] - pb[0], c[1] - pb[1], c[2] - pb[2]], k, kr, fr: gen_fr, frr }
}
/// Stress over strength of a face (section `s`) under force `f` and moment `m` on the max side, in the face's frame
/// (normal, u, w; N and N m): the worst of
/// - tension at the extreme fibre, N / A + M / S, against the tensile strength, or the flexural strength where
///   bending makes most of it
/// - compression at the other extreme fibre, against the compressive strength
/// - shear (V / A, plus torsion) against cohesion plus friction on the compression: 0.4 sigma_n (Mohr-Coulomb;
///   a brick wall under load is stronger in shear than mortar alone)
fn section_ratio(s: &Sect, f: [f64; 3], m: [f64; 3], cap: [f32; 3], flex: f32) -> f32 {
    // normal force: positive = the max side pulled away (tension: the spring is stretched)
    let sn = f[0] / s.area;
    let bend = m[1].abs() * s.zu / s.iu + m[2].abs() * s.zw / s.iw;
    let top = sn + bend;          // the most tensile fibre
    let bottom = sn - bend;       // the most compressed fibre
    let mut r = 0.0f64;
    if top > 0.0 {
        let tens = sn.max(0.0);
        let fe = (tens * cap[1] as f64 + bend * flex as f64) / (tens + bend).max(1e-9);
        r = r.max(top / fe.max(1.0));
    }
    if bottom < 0.0 { r = r.max(-bottom / cap[0] as f64); }
    let v = sqrt64(f[1] * f[1] + f[2] * f[2]) / s.area;
    let tor = m[0].abs() * s.rmax / s.j;
    let cohesion = cap[2] as f64 + 0.4 * (-sn).max(0.0);
    r = r.max((v + tor) / cohesion);
    r as f32
}
/// Conjugate gradients for K x = f on the spring network (6 unknowns per cell, the fixed ones held at 0),
/// preconditioned by a multigrid V-cycle. The levels: cells grouped into blocks of 4, 8, 16... m, each block
/// rigid. A spring between two blocks becomes the same spring with its arms measured from the blocks'
/// centres (the Galerkin coarse problem of a rigid grouping is again a spring network); springs inside a block
/// drop out. Damped block-Jacobi smoothing (each cell's 6 x 6 block inverted) on every level.
/// With block Jacobi alone the 12-storey tower took 1,552 iterations and 1.2 s (the long, slow bending modes of a
/// tall building converge slowly).
fn solve_network(sp: &[Spring], ends: &[(u32, u32)], fixed: &[bool], pos: &[[f64; 3]], f: &[[f64; 6]], x0: Option<Vec<[f64; 6]>>, cap: usize, scale: Option<&[[f64; 6]]>, tight: bool) -> (Vec<[f64; 6]>, usize) {
    let n = fixed.len();
    #[cfg(test)] let t0 = { extern crate std; std::time::Instant::now() };
    let mut levels = vec![Level::from_springs(sp, ends, fixed.to_vec(), pos.to_vec())];
    let mut bin = 3.0f64;
    while levels.last().unwrap().n > 64 && levels.len() < 8 {
        let c = levels.last().unwrap().coarsen(bin);
        if c.n as f64 > 0.7 * levels.last().unwrap().n as f64 { bin *= 2.0; if bin > 256.0 { break; } continue; }
        levels.push(c); bin *= 2.0;
    }
    levels.last_mut().unwrap().factor();
    // (the finest level's blocks were needed only to build the next one)
    levels[0].off = Vec::new();
    #[cfg(test)] { extern crate std; if std::env::var("SF_MG").is_ok() { std::println!("  levels built in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); for (i, l) in levels.iter().enumerate() { std::println!("  level {i}: {} cells, {} couplings", l.n, l.off.len()); } } }
    let dot = |a: &[[f64; 6]], b: &[[f64; 6]]| -> f64 { let mut s = 0.0; for k in 0..a.len() { for i in 0..6 { s += a[k][i] * b[k][i]; } } s };
    let mut r: Vec<[f64; 6]> = (0..n).map(|k| if fixed[k] { [0.0; 6] } else { f[k] }).collect();
    let mut ap = vec![[0.0f64; 6]; n];
    // the tolerance is measured against the residual from zero, so that a warm start stops at the same accuracy
    // (a dynamic step is solved for the cells' move, whose right-hand side is next to nothing near rest: its
    // tolerance is measured against the loads, `scale`, as the solve at rest's is)
    let r0 = match scale {
        Some(sc) => { let q: Vec<[f64; 6]> = (0..n).map(|k| if fixed[k] { [0.0; 6] } else { sc[k] }).collect(); let z0 = vcycle(&levels, 0, &q); dot(&q, &z0).max(1e-30) }
        None => { let z0 = vcycle(&levels, 0, &r); dot(&r, &z0).max(1e-30) }
    };
    let mut x = match x0 {
        Some(x0) => { levels[0].apply(&x0, &mut ap); for k in 0..n { for i in 0..6 { r[k][i] -= ap[k][i]; } } x0 }
        None => vec![[0.0f64; 6]; n],
    };
    // (tests: `TOL_TEST` or SF_TOL solve to another tolerance, without the cap on iterations)
    #[cfg(test)] let tol = { extern crate std; let t = TOL_TEST.with(|t| t.get()); if t > 0.0 { t } else { std::env::var("SF_TOL").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(TOL) } };
    #[cfg(test)] let cap = { extern crate std; if TOL_TEST.with(|t| t.get()) > 0.0 || std::env::var("SF_CAP").is_ok() { 20000 } else { cap } };
    #[cfg(not(test))] let tol = TOL;
    let tol = if tight { tol * TIGHT } else { tol };
    let mut z = vcycle(&levels, 0, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    if rz < tol * r0 { return (x, 0); }
    let mut it = 0;
    while it < cap {
        levels[0].apply(&p, &mut ap);
        let pap = dot(&p, &ap); if pap <= 0.0 { break; }
        let alpha = rz / pap;
        for k in 0..n { for i in 0..6 { x[k][i] += alpha * p[k][i]; r[k][i] -= alpha * ap[k][i]; } }
        it += 1;
        z = vcycle(&levels, 0, &r);
        let rz2 = dot(&r, &z);
        if rz2 < tol * r0 { break; }
        let beta = rz2 / rz; rz = rz2;
        for k in 0..n { for i in 0..6 { p[k][i] = z[k][i] + beta * p[k][i]; } }
    }
    #[cfg(test)] { extern crate std; if std::env::var("SF_MG").is_ok() { std::println!("  solved in {:.0} ms, {it} iterations", t0.elapsed().as_secs_f64() * 1e3); } }
    (x, it)
}
#[cfg(test)] extern crate std;
#[cfg(test)] std::thread_local! { pub static TOL_TEST: core::cell::Cell<f64> = core::cell::Cell::new(0.0); }
const MAX_IT: usize = 2000;
// r z down by 1e-7 from the start. It was 1e-5, which read well on frames and towers and badly where parts stand on
// soft joints: the complex building's worst joint, a corner column shot out, 0.889 at 1e-5, 0.824 at 1e-6, 0.819 at
// 1e-7, 0.817 at 1e-8 and at 1e-12 (5 solves: 93, 139, 197, 260 and 1,045 iterations), and its arch under the loggia
// without its keystone lost a joint at 1e-5 (worst 0.76) that it keeps from 1e-6 on (0.51): dyn_tests, complex::tests.
const TOL: f64 = 1e-7;
const SMOOTH: usize = 1;
/// a solve the dynamic structure rests on is taken this much further than `TOL`
const TIGHT: f64 = 1e-3;
/// two cells go into one coarse block if the joints between them hold at least this share of what holds the weaker
/// of the two (`Level::coarsen`)
const STRONG: f64 = 1.0e-3;
/// a cell whose joints hold it in one direction less than this share of another is held by `Level::finish`
const REG_FREE: f64 = 1.0e-3;
/// the coarsest level is factored if it has no more free cells than this (6 unknowns each: 384, 9 M operations)
const DIRECT: usize = 64;
const OMEGA: f64 = 0.8;

type M6 = [[f64; 6]; 6];
fn mmul(a: &M6, b: &M6) -> M6 { let mut c = [[0.0; 6]; 6]; for i in 0..6 { for k in 0..6 { let v = a[i][k]; if v != 0.0 { for j in 0..6 { c[i][j] += v * b[k][j]; } } } } c }
fn tr(a: &M6) -> M6 { let mut c = [[0.0; 6]; 6]; for i in 0..6 { for j in 0..6 { c[j][i] = a[i][j]; } } c }
/// the rigid map from a body's 6 unknowns to a point at offset d: [I, -[d]x; 0, I]
fn rigid(d: &[f64; 3]) -> M6 {
    let mut t = [[0.0; 6]; 6];
    for i in 0..6 { t[i][i] = 1.0; }
    // u_point = u + theta x d = u - [d]x theta
    t[0][4] = d[2]; t[0][5] = -d[1]; t[1][3] = -d[2]; t[1][5] = d[0]; t[2][3] = d[1]; t[2][4] = -d[0];
    t
}
/// A level of the solve as a block matrix: each cell's own 6 x 6 block and one block per pair of coupled cells.
/// The finest level also keeps its springs (`sp`, `ends`) and applies K from them directly: 15 numbers a spring
/// against a 6 x 6 block, and the solve is bound by memory (the 12-storey tower: 4.3 ms an iteration with blocks).
struct Level { n: usize, diag: Vec<M6>, off: Vec<(u32, u32, M6)>, fixed: Vec<bool>, pos: Vec<[f64; 3]>, inv: Vec<M6>, up: Vec<u32>, sp: Vec<Spring>, ends: Vec<(u32, u32)>, reg: Vec<[f64; 2]>,
    /// the coarsest level, factored (`factor`): the free cells' unknowns in order, and L of K = L L^T, row by row
    dof: Vec<u32>, chol: Vec<f64> }
impl Level {
    /// the finest level, from the springs: strain e = B_b x_b - B_a x_a (B the rigid map to the face centre),
    /// energy e^T D e / 2, so K_aa = B_a^T D B_a, K_bb = B_b^T D B_b, K_ab = -B_a^T D B_b
    fn from_springs(sp: &[Spring], ends: &[(u32, u32)], fixed: Vec<bool>, pos: Vec<[f64; 3]>) -> Level {
        let n = fixed.len();
        let mut diag = vec![[[0.0; 6]; 6]; n];
        let mut off = Vec::with_capacity(sp.len());
        for (s, &(a, c)) in sp.iter().zip(ends) {
            let d = s.d6();
            let (ba, bb) = (rigid(&s.ra), rigid(&s.rb));
            let (dba, dbb) = (mmul(&d, &ba), mmul(&d, &bb));
            let (au, cu) = (a as usize, c as usize);
            if !fixed[au] { let m = mmul(&tr(&ba), &dba); for x in 0..6 { for y in 0..6 { diag[au][x][y] += m[x][y]; } } }
            if !fixed[cu] { let m = mmul(&tr(&bb), &dbb); for x in 0..6 { for y in 0..6 { diag[cu][x][y] += m[x][y]; } } }
            if !fixed[au] && !fixed[cu] {
                let mut m = mmul(&tr(&ba), &dbb); for x in 0..6 { for y in 0..6 { m[x][y] = -m[x][y]; } }
                off.push((a, c, m));
            }
        }
        let mut l = Level::finish(n, diag, off, fixed, pos);
        l.sp = sp.to_vec(); l.ends = ends.to_vec();
        l
    }
    fn finish(n: usize, mut diag: Vec<M6>, off: Vec<(u32, u32, M6)>, fixed: Vec<bool>, pos: Vec<[f64; 3]>) -> Level {
        // (a little added to the diagonal: a cell held only by hinges or infill would be free to turn)
        let mut inv = vec![[[0.0; 6]; 6]; n];
        let mut reg = vec![[0.0; 2]; n];
        for k in 0..n {
            if fixed[k] { continue; }
            let big = (0..6).map(|i| diag[k][i][i].abs()).fold(0.0, f64::max).max(1.0);
            // (on the turns; on the displacements only of a cell that its joints do not hold in some direction.
            // On every cell's displacements it held the structure to the ground by springs of 1e-7 of its
            // stiffest: a 7.6 m cantilever strip of 31 cells sagged 53.1 mm under its weight where beam theory gives
            // 62.3, and its worst joint read 1.152 for 1.313; now 61.5 mm: dyn_closed)
            let (lo, hi) = ((0..3).map(|i| diag[k][i][i]).fold(f64::MAX, f64::min), (0..3).map(|i| diag[k][i][i]).fold(0.0, f64::max));
            #[cfg(test)] let lo = { extern crate std; if std::env::var("SF_OLDREG").is_ok() { -1.0 } else { lo } };
            reg[k] = [if lo < REG_FREE * hi { 1e-7 * big } else { 0.0 }, 1e-7 * big];
            for i in 0..3 { diag[k][i][i] += reg[k][0]; diag[k][3 + i][3 + i] += reg[k][1]; }
            inv[k] = invert6(diag[k]);
        }
        Level { n, diag, off, fixed, pos, inv, up: Vec::new(), sp: Vec::new(), ends: Vec::new(), reg, dof: Vec::new(), chol: Vec::new() }
    }
    /// Factor the level (the coarsest: at most `DIRECT` free cells), so that the V-cycle solves it exactly; it was
    /// 30 sweeps of block Jacobi. A structure of up to `DIRECT` cells is then solved in one iteration.
    fn factor(&mut self) {
        let free: Vec<u32> = (0..self.n as u32).filter(|&k| !self.fixed[k as usize]).collect();
        let m = free.len() * 6;
        #[cfg(test)] { extern crate std; if std::env::var("SF_OLDREG").is_ok() { return; } }
        if free.is_empty() || free.len() > DIRECT { return; }
        let mut at = vec![NONE; self.n];
        for (i, &k) in free.iter().enumerate() { at[k as usize] = i as u32; }
        let mut a = vec![0.0f64; m * m];
        for (i, &k) in free.iter().enumerate() { let d = &self.diag[k as usize]; for x in 0..6 { for y in 0..6 { a[(i * 6 + x) * m + i * 6 + y] = d[x][y]; } } }
        for &(p, q, ref b) in &self.off {
            let (i, j) = (at[p as usize], at[q as usize]); if i == NONE || j == NONE { continue; }
            let (i, j) = (i as usize, j as usize);
            for x in 0..6 { for y in 0..6 { a[(i * 6 + x) * m + j * 6 + y] += b[x][y]; a[(j * 6 + y) * m + i * 6 + x] += b[x][y]; } }
        }
        // Cholesky, the lower triangle in place
        for j in 0..m {
            let mut d = a[j * m + j]; for k in 0..j { d -= a[j * m + k] * a[j * m + k]; }
            if !(d > 0.0) { return; }   // (not positive definite to rounding: the sweeps stay)
            let d = sqrt64(d); a[j * m + j] = d;
            for i in j + 1..m {
                let mut v = a[i * m + j];
                let (ri, rj) = (&a[i * m..i * m + j], &a[j * m..j * m + j]);
                for k in 0..j { v -= ri[k] * rj[k]; }
                a[i * m + j] = v / d;
            }
        }
        self.dof = free; self.chol = a;
    }
    /// z = K^-1 r with the factor
    fn direct(&self, r: &[[f64; 6]], z: &mut [[f64; 6]]) {
        let m = self.dof.len() * 6;
        let l = &self.chol;
        let mut y = vec![0.0f64; m];
        for (i, &k) in self.dof.iter().enumerate() { for x in 0..6 { y[i * 6 + x] = r[k as usize][x]; } }
        for i in 0..m { let mut v = y[i]; let row = &l[i * m..i * m + i]; for k in 0..i { v -= row[k] * y[k]; } y[i] = v / l[i * m + i]; }
        for i in (0..m).rev() { let mut v = y[i]; for k in i + 1..m { v -= l[k * m + i] * y[k]; } y[i] = v / l[i * m + i]; }
        for (i, &k) in self.dof.iter().enumerate() { for x in 0..6 { z[k as usize][x] = y[i * 6 + x]; } }
    }
    /// y = K x (fixed cells: y = 0)
    fn apply(&self, x: &[[f64; 6]], y: &mut [[f64; 6]]) {
        if !self.sp.is_empty() {
            for k in 0..self.n { let r = self.reg[k]; for i in 0..3 { y[k][i] = r[0] * x[k][i]; y[k][3 + i] = r[1] * x[k][3 + i]; } }
            for (s, &(a, c)) in self.sp.iter().zip(&self.ends) {
                let (a, c) = (a as usize, c as usize);
                let (d, r) = s.strain(&x[a], &x[c]);
                let (f, m) = s.force(d, r);
                // on c: f, and m plus the arm's moment rb x f; on a the opposite with ra
                let (ra, rb) = (&s.ra, &s.rb);
                let mb = [m[0] + rb[1] * f[2] - rb[2] * f[1], m[1] + rb[2] * f[0] - rb[0] * f[2], m[2] + rb[0] * f[1] - rb[1] * f[0]];
                let ma = [m[0] + ra[1] * f[2] - ra[2] * f[1], m[1] + ra[2] * f[0] - ra[0] * f[2], m[2] + ra[0] * f[1] - ra[1] * f[0]];
                for i in 0..3 { y[c][i] += f[i]; y[c][3 + i] += mb[i]; y[a][i] -= f[i]; y[a][3 + i] -= ma[i]; }
            }
            for k in 0..self.n { if self.fixed[k] { y[k] = [0.0; 6]; } }
            return;
        }
        for k in 0..self.n {
            if self.fixed[k] { y[k] = [0.0; 6]; continue; }
            let m = &self.diag[k]; let v = &x[k];
            for i in 0..6 { let mut s = 0.0; for j in 0..6 { s += m[i][j] * v[j]; } y[k][i] = s; }
        }
        for &(a, c, ref m) in &self.off {
            let (a, c) = (a as usize, c as usize);
            let (xa, xc) = (x[a], x[c]);
            for i in 0..6 { let mut s = 0.0; let mut t = 0.0; for j in 0..6 { s += m[i][j] * xc[j]; t += m[j][i] * xa[j]; } y[a][i] += s; y[c][i] += t; }
        }
    }
    fn jacobi(&self, r: &[[f64; 6]], z: &mut [[f64; 6]], w: f64) {
        for k in 0..self.n { if self.fixed[k] { continue; } for i in 0..6 { let mut s = 0.0; for j in 0..6 { s += self.inv[k][i][j] * r[k][j]; } z[k][i] += w * s; } }
    }
    /// the next level: free cells grouped into blocks of `bin` m, each block rigid (K_c = T^T K T, T the rigid
    /// map from the block's centre to each cell); fixed cells stay out (they are not unknowns)
    fn coarsen(&self, bin: f64) -> Level {
        let mut map = crate::Map::default();
        let mut up = vec![NONE; self.n];
        let mut pos: Vec<[f64; 3]> = Vec::new(); let mut cnt: Vec<f64> = Vec::new();
        // The cells of a bin make one block only as far as they hold each other: joined through joints at least
        // `STRONG` of what holds the weaker of the two. A part on open beds or softened joints (a millionth as
        // stiff as sound ones) moves against its neighbours, and a block rigid across such a joint cannot show
        // that motion: it was left to the smoothing, which does not find it, and the solve stopped at its
        // tolerance far from the answer (the complex building's worst joint read 0.917 standing; solved to 1e-12
        // it is 0.416: structure tests).
        let mut bin_of = vec![NONE; self.n];
        let mut nb = 0u32;
        for k in 0..self.n {
            if self.fixed[k] { continue; }
            let p = self.pos[k];
            let key = (floor64(p[0] / bin) as i64, floor64(p[1] / bin) as i64, floor64(p[2] / bin) as i64);
            bin_of[k] = *map.entry(key).or_insert_with(|| { nb += 1; nb - 1 });
        }
        let mut par: Vec<u32> = (0..self.n as u32).collect();
        fn find(par: &mut [u32], mut a: u32) -> u32 { while par[a as usize] != a { let g = par[par[a as usize] as usize]; par[a as usize] = g; a = g; } a }
        let hold = |m: &M6| fabs64(m[0][0]) + fabs64(m[1][1]) + fabs64(m[2][2]);
        for &(a, c, ref m) in &self.off {
            let (au, cu) = (a as usize, c as usize);
            if bin_of[au] == NONE || bin_of[au] != bin_of[cu] { continue; }
            #[cfg(test)] { extern crate std; if std::env::var("SF_NOSTRONG").is_ok() { let (ra, rc) = (find(&mut par, a), find(&mut par, c)); if ra != rc { par[ra.max(rc) as usize] = ra.min(rc); } continue; } }
            if hold(m) < STRONG * hold(&self.diag[au]).min(hold(&self.diag[cu])) { continue; }
            let (ra, rc) = (find(&mut par, a), find(&mut par, c));
            if ra != rc { par[ra.max(rc) as usize] = ra.min(rc); }
        }
        #[cfg(test)] let merge_all = { extern crate std; std::env::var("SF_NOSTRONG").is_ok() };
        #[cfg(not(test))] let merge_all = false;
        let mut block: crate::Map<(u32, u32), u32> = crate::Map::default();
        for k in 0..self.n {
            if self.fixed[k] { continue; }
            let root = if merge_all { 0 } else { find(&mut par, k as u32) };
            let c = *block.entry((bin_of[k], root)).or_insert_with(|| { pos.push([0.0; 3]); cnt.push(0.0); pos.len() as u32 - 1 }) as usize;
            let p = self.pos[k];
            up[k] = c as u32; for d in 0..3 { pos[c][d] += p[d]; } cnt[c] += 1.0;
        }
        for c in 0..pos.len() { for d in 0..3 { pos[c][d] /= cnt[c]; } }
        let nc = pos.len();
        let t: Vec<M6> = (0..self.n).map(|k| if up[k] == NONE { [[0.0; 6]; 6] } else { let c = up[k] as usize; rigid(&[self.pos[k][0] - pos[c][0], self.pos[k][1] - pos[c][1], self.pos[k][2] - pos[c][2]]) }).collect();
        let mut diag = vec![[[0.0; 6]; 6]; nc];
        let add = |m: &mut M6, a: &M6| { for x in 0..6 { for y in 0..6 { m[x][y] += a[x][y]; } } };
        for k in 0..self.n { if up[k] == NONE { continue; } let m = mmul(&tr(&t[k]), &mmul(&self.diag[k], &t[k])); add(&mut diag[up[k] as usize], &m); }
        let mut pairs: crate::Map<(u32, u32), M6> = crate::Map::default();
        for &(a, c, ref m) in &self.off {
            let (ua, uc) = (up[a as usize], up[c as usize]);
            if ua == NONE || uc == NONE { continue; }
            let q = mmul(&tr(&t[a as usize]), &mmul(m, &t[c as usize]));
            if ua == uc { let qt = tr(&q); add(&mut diag[ua as usize], &q); add(&mut diag[ua as usize], &qt); }
            else if ua < uc { add(pairs.entry((ua, uc)).or_insert([[0.0; 6]; 6]), &q); }
            else { add(pairs.entry((uc, ua)).or_insert([[0.0; 6]; 6]), &tr(&q)); }
        }
        let off: Vec<(u32, u32, M6)> = pairs.into_iter().map(|((a, c), m)| (a, c, m)).collect();
        // (the diagonal already holds the small additions of the level above, carried down)
        let fixed = vec![false; nc];
        let mut l = Level::finish(nc, diag, off, fixed, pos);
        l.up = up;
        l
    }
}
/// One V-cycle on level `l` for residual `r`: smooth, correct on the next level, smooth.
fn vcycle(levels: &[Level], l: usize, r: &[[f64; 6]]) -> Vec<[f64; 6]> {
    let lv = &levels[l];
    let n = lv.n;
    let mut z = vec![[0.0f64; 6]; n];
    let mut kz = vec![[0.0f64; 6]; n];
    let coarsest = l + 1 >= levels.len();
    if coarsest && !lv.chol.is_empty() { lv.direct(r, &mut z); return z; }
    let sweeps = if coarsest { 30 } else { SMOOTH };
    let mut rr = vec![[0.0f64; 6]; n];
    let mut smooth = |z: &mut Vec<[f64; 6]>, kz: &mut Vec<[f64; 6]>, first: bool| {
        for s in 0..sweeps {
            // (from z = 0 the first residual is r itself)
            if first && s == 0 { lv.jacobi(r, z, OMEGA); continue; }
            lv.apply(z, kz);
            for k in 0..n { for i in 0..6 { rr[k][i] = r[k][i] - kz[k][i]; } }
            lv.jacobi(&rr, z, OMEGA);
        }
    };
    smooth(&mut z, &mut kz, true);
    if coarsest { return z; }
    let nx = &levels[l + 1];
    // restrict the residual: forces summed, their moments about the block's centre added
    lv.apply(&z, &mut kz);
    let mut rc = vec![[0.0f64; 6]; nx.n];
    for k in 0..n {
        let c = nx.up[k]; if c == NONE { continue; }
        let c = c as usize;
        let mut v = [0.0; 6]; for i in 0..6 { v[i] = r[k][i] - kz[k][i]; }
        let d = [lv.pos[k][0] - nx.pos[c][0], lv.pos[k][1] - nx.pos[c][1], lv.pos[k][2] - nx.pos[c][2]];
        for i in 0..3 { rc[c][i] += v[i]; }
        rc[c][3] += v[3] + d[1] * v[2] - d[2] * v[1];
        rc[c][4] += v[4] + d[2] * v[0] - d[0] * v[2];
        rc[c][5] += v[5] + d[0] * v[1] - d[1] * v[0];
    }
    let zc = vcycle(levels, l + 1, &rc);
    // prolong: each cell moves with its block
    for k in 0..n {
        let c = nx.up[k]; if c == NONE { continue; }
        let c = c as usize;
        let d = [lv.pos[k][0] - nx.pos[c][0], lv.pos[k][1] - nx.pos[c][1], lv.pos[k][2] - nx.pos[c][2]];
        let t = [zc[c][3], zc[c][4], zc[c][5]];
        z[k][0] += zc[c][0] + t[1] * d[2] - t[2] * d[1];
        z[k][1] += zc[c][1] + t[2] * d[0] - t[0] * d[2];
        z[k][2] += zc[c][2] + t[0] * d[1] - t[1] * d[0];
        for i in 3..6 { z[k][i] += zc[c][i]; }
    }
    smooth(&mut z, &mut kz, false);
    z
}
fn invert6(mut m: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut inv = [[0.0f64; 6]; 6];
    for i in 0..6 { inv[i][i] = 1.0; }
    for c in 0..6 {
        let mut p = c; for r in c + 1..6 { if m[r][c].abs() > m[p][c].abs() { p = r; } }
        if m[p][c].abs() < 1e-300 { continue; }
        m.swap(c, p); inv.swap(c, p);
        let d = m[c][c];
        for j in 0..6 { m[c][j] /= d; inv[c][j] /= d; }
        for r in 0..6 { if r != c { let q = m[r][c]; if q != 0.0 { for j in 0..6 { m[r][j] -= q * m[c][j]; inv[r][j] -= q * inv[c][j]; } } } }
    }
    inv
}

#[cfg(test)]
mod analysis_tests {
    use super::*;
    use crate::tower;
    extern crate std;
    fn gravity(b: &Bricks) -> Analysis {
        let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
        let load: Vec<f32> = set.iter().map(|&i| mass(b, i) * 9.81).collect();
        let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
        analyse(b, &set, &load, &sink, v3(0.0, -1.0, 0.0))
    }
    #[test] fn a_pier_carries_its_load_through_a_fractured_brick() {
        // a 1 m square concrete pier of five 1 m blocks with 2,000 kN on top (2 MPa): with its middle block split
        // into the convex cells of each of the 16 patterns, the load goes down through the cells (their faces on the
        // blocks above and below tile the block's) and stands. The faces between cells are inclined, and on a plane
        // at angle t to the load the shear is sigma sin t cos t, at most sigma / 2: that, against the shear
        // strength, bounds the ratio.
        let build = || { let mut b = Bricks::new(); for k in 0..5 { b.add([0, k * 8, 0], [8, 8, 8], mat::CONCRETE, 0, 1); } b.build_joints(); b };
        let run = |b: &Bricks| {
            let set: Vec<u32> = (0..b.len() as u32).filter(|&i| b.alive(i)).collect();
            let top = set.iter().copied().max_by_key(|&i| b.y[i as usize]).unwrap();
            let load: Vec<f32> = set.iter().map(|&i| mass(b, i) * 9.81 + if i == top { 2.0e6 } else { 0.0 }).collect();
            let sink: Vec<bool> = set.iter().map(|&i| b.y[i as usize] <= 0).collect();
            analyse(b, &set, &load, &sink, v3(0.0, -1.0, 0.0))
        };
        let whole = run(&build());
        let mut worst = 0.0f32; let mut its = 0;
        for p in 0..16u32 {
            let mut b = build();
            // (the pattern comes from the brick's index: pad with dead bricks to reach each)
            let mut k = 0; while crate::brick::frac_cells(b.len() as u32, 1).0 != p { b.add([100 + 2 * k, 0, 100], [2, 2, 2], mat::CONCRETE, 0, 0); k += 1; }
            let m = b.add([0, 16, 0], [8, 8, 8], mat::CONCRETE, 0, 1);
            b.remove(2); for i in 5..m { b.remove(i); }
            b.build_joints();
            let (cells, _) = b.shatter(m);
            assert_eq!(crate::brick::frac_cells(m, 1).0, p);
            let a = run(&b);
            assert!(a.failed.is_empty() && a.cracks.is_empty(), "pattern {p}: {} cells, worst {}", cells.len(), a.max_ratio);
            worst = worst.max(a.max_ratio); its = its.max(a.iterations);
        }
        std::println!("pier under 2,000 kN: whole {:.3}, through each pattern's cells at most {worst:.3} ({its} iterations at most)", whole.max_ratio);
        let bound = 0.5 * 2.1e6 / strength(mat::CONCRETE)[2];
        assert!(worst <= 1.05 * bound.max(whole.max_ratio), "at most sigma / 2 in shear: {bound:.3}");
    }
    #[test] fn cantilever_as_one_brick() {
        for (len, expect) in [(60, true), (16, false)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([0, 8, 0], [8 + len, 2, 8], mat::CONCRETE, 0, 0);
            b.build_joints();
            let a = gravity(&b);
            std::println!("cantilever {} m: worst {:.2}, cracks {:?}, failed joints {}", len as f32 * TICK, a.max_ratio, a.cracks.iter().map(|c| (c.at, c.ratio)).collect::<std::vec::Vec<_>>(), a.failed.len());
            assert_eq!(!a.cracks.is_empty() || !a.failed.is_empty(), expect);
        }
    }
    #[test] fn simply_supported_span() {
        // a slab strip 1 m wide bonded to two 1 m blocks (between clamped and simply supported): 7.5 m holds
        // (w L^2 / 8 = 41 kNm per m, 3.9 MPa even simply supported); 22.5 m does not (clamped: 5.9 x 23.5^2 / 12 =
        // 271 kNm, 26 MPa). (15 m, which the load-flow model failed at 15.9 MPa simply supported, holds bonded:
        // 0.81 of strength.)
        for (span, expect) in [(60, false), (180, true)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([8 + span, 0, 0], [8, 8, 8], mat::CONCRETE, 0, 0);
            b.add([0, 8, 0], [16 + span, 2, 8], mat::CONCRETE, 0, 0);
            b.build_joints();
            let a = gravity(&b);
            std::println!("span {} m: worst {:.2}", span as f32 * TICK, a.max_ratio);
            assert_eq!(a.max_ratio > 1.0, expect);
        }
    }
    #[test] fn a_slab_on_one_off_centre_column_tips_and_a_centred_one_holds() {
        // a 6 m x 6 m concrete slab (0.25 m) on one 0.5 m column 3 m high. Centred: it stands (the resultant is
        // over the column). At a corner: the slab's weight (212 kN) acts 3.9 m off the column, which can bend at
        // most f b d^2 / 6 = 12 MPa x 0.5^3 / 6 = 250 kNm, e = 1.2 m: it tips. Before the overturning check both
        // stood (the scalar flow carried the slab down the corner column).
        for (off, expect) in [(20, false), (0, true)] {
            let mut b = Bricks::new();
            b.add([0, 0, 0], [80, 2, 80], mat::CONCRETE, 0, 0);          // ground slab (the support)
            b.add([off, 2, off], [4, 24, 4], mat::CONCRETE, 1, 0);         // column
            b.add([0, 26, 0], [48, 2, 48], mat::CONCRETE, 0, 0);          // slab above
            b.build_joints();
            let a = gravity(&b);
            std::println!("column at {} m: worst {:.2}, failed {:?}", off as f32 * TICK, a.max_ratio, a.failed.iter().map(|x| x.1).collect::<std::vec::Vec<_>>());
            assert_eq!(!a.failed.is_empty(), expect, "column at {off}");
        }
    }
    #[test] fn intact_tower_stands() {
        let b = tower::build(12, 1);
        let t = std::time::Instant::now();
        let a = gravity(&b);
        std::println!("tower: {} nodes, worst {:.2}, {} joints and {} cracks over, {} CG iterations, {:.0} ms", a.nodes, a.max_ratio, a.failed.len(), a.cracks.len(), a.iterations, t.elapsed().as_secs_f64() * 1e3);
        assert!(a.failed.is_empty() && a.cracks.is_empty());
    }
}

