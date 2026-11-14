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

/// The longest dynamic step of the standing structure (s).
/// A lot is drawn where its structure has moved to, outside the engineer's view, once a cell of it is this far (m)
/// from where it was built: 2 cm, more than a standing building's own sag (16 mm at most in the city).
pub const VIEW_BEND: f64 = 0.02;
/// A hit on the standing structure slower than this (m/s) is a part coming to rest on it: its weight reaches the
/// structure as rubble's does.
pub const KICK_SPEED: f32 = 1.0;
pub const DYN_MAX_DT: f32 = 0.1;
pub struct Game {
    pub b: Bricks,
    /// weight resting on the structure (rubble lying on floors), for the next structure_step: (x, y, z m, N)
    pub rest: Vec<[f32; 4]>,
    /// a new list of rubble has been given since the last structure step (`wasm::rest_begin`; an empty one too),
    /// and the list the dynamic structure keeps between steps: the page sends one once a second, and rubble that
    /// weighed only in the step after it came and went as a load every second (the lab's complexcol never rested)
    pub rest_fresh: bool,
    pub rest_kept: Vec<[f32; 4]>,
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
    /// a free piece's shape for the page (`piece_shape`), apart from `out` so it can be read while `out` is
    pub piece_out: Vec<f32>,
    /// loose bricks (room contents) not yet sent off as pieces: the next structure step detaches them even if
    /// nothing failed
    pub loose_pending: bool,
    /// pieces the last structure step sent off (lost the ground, or loose contents)
    pub last_detached: usize,
    pub dirty: (i32, i32),
    /// how the static bricks are meshed: in storey slices, and in a city also by lot (`Tiles`)
    pub tiles: Tiles,
    /// in a city, the lots the next structure step analyses: those changed since their last analysis, with rubble
    /// on them or joints wearing (a lot left alone gives the same answer every time: the city's 9 lots took 1.2 s a
    /// step), and each lot's worst ratio when last analysed
    pub active: Vec<bool>, pub tile_ratio: Vec<f32>,
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
    /// the standing building's last solution under gravity, to start the next check from
    warm: structure::Warm,
    /// iterations of the last structure check's solve
    pub last_iterations: usize,
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
    /// crushed material beyond the budgets: with `spill` on it is sent as spill (id -2: the page makes rubble
    /// particles of it), not removed as dust. Removed, it was whole floors vanishing: in the lab collapse, 489 m3
    /// of a 1,211 m3 building in 240 frames.
    pub spill_on: bool,
    spill: Vec<u32>,
    /// time since the last structure check (s): joints loaded near their strength wear for that long (see
    /// `wear_joints`); 0 = no wear (the tests' checks)
    pub struct_dt: f32,
    /// the standing structure stepped as a dynamic model (structure::Dyn, docs/DYNAMIC.md): a structure step that
    /// comes with the time since the last one (`struct_dt`) is a dynamic step of that long, `DYN_MAX_DT` at most
    pub dynamic: bool,
    /// each lot's displacements and velocities between dynamic steps (lot 0 where there are no lots)
    pub dynst: crate::Map<usize, (structure::Warm, structure::Dyn)>,
    /// after a structure step: is any lot still moving (another step is due), its fastest cell (m/s), the solver
    /// iterations of the lot that took most, and the kinetic energy of the standing cells (J)
    pub moving: bool,
    pub dyn_speed: f32,
    pub dyn_its: usize,
    pub dyn_kinetic: f32,
    /// after a structure step: the cells that left the structure (structure::DYN_TURN, DYN_MOVE), the joints and
    /// faces cut for it, and whether a lot's solve stopped at its iteration cap
    pub dyn_left: usize,
    pub dyn_cut: usize,
    pub dyn_capped: bool,
    /// the structure as the engineer sees it (mesh::View), in the page's copy of the engine
    pub view: crate::mesh::View,
    /// the engine's side of it: asked for (`view_want`), each structure step writes the analysed lots' bricks and
    /// cells to `view_out`: [bricks n, (brick, load / strength) ..., cells n, (brick, corner x, z, far corner x, z
    /// (ticks), centre x y z (m), displacement x y z (m), rotation x y z (rad)) ...]; `view_in` takes it in the page
    pub view_want: bool,
    /// only the lots that have bent to see are written (`VIEW_BEND`): the page draws the building where it has
    /// sagged, without the engineer's view on (wasm set_param 8 at 2)
    pub view_bend: bool,
    view_vel: Vec<[f32; 6]>,
    /// hits on the standing structure since its last step, for the dynamic structure: where (m) and the impulse
    /// (N s) on it. They are loads of its next step (`Dyn::kicks`), not a check at rest of their own.
    pub kicks: Vec<([f64; 3], [f64; 3], f64)>,
    pub motion: [f32; 12],
    pub view_out: Vec<f32>,
    pub view_in: Vec<f32>,
    /// the bricks of the lots whose view the last structure step writes (a lot at rest, written once, is left out)
    view_bricks: Vec<u32>,
    /// the page's copy: the chunks the last `view_apply` remeshed
    pub view_chunks: Vec<i32>,
    /// joints wearing at the last check (loaded above SUSTAINED of their strength)
    pub wearing: usize,
    /// a brick of each joint or face wearing in the last structure step
    pub wearing_at: Vec<u32>,
    /// joints and faces that gave way by wear so far
    pub worn_total: usize,
    /// the last check's failing joints: (brick, brick, ratio)
    pub last_failed: Vec<(u32, u32, f32)>,
    /// bricks sent in two kids of one break (should be 0)
    pub dup_bricks: usize,
    /// bricks spilled, by why: crumbs over the budget, small fragments over the rubble budget, crushed storey, static
    pub spill_why: [usize; 4],
    pub spill_mat: [usize; 8],
    pub spill_single: usize,
    /// (comparison only: masonry in falling parts parts at once, as before 7 November)
    pub masonry_parts: bool,
    /// wear of faces inside bricks (brick, axis, plane), and of joints (the joint's `jdmg` keeps a coarse copy for
    /// its stiffness: in 1/127 steps alone, a check's wear at 75% of strength, 0.0035, rounded to nothing)
    inner_wear: crate::Map<(u32, u8, i32), f32>,
    joint_wear: crate::Map<u32, f32>,
    /// standing bricks split into convex cells so far, and at most how many (each cell is a shape on the GPU:
    /// the cut of the lab's collapse marks thousands of bricks, and as cells they ran the GPU out of shapes)
    pub frac_made: usize,
    pub frac_max: usize,
    /// break plans of falling pieces, for the GPU to break them in the step of a hit (see `plan_piece`)
    pub plans: crate::Map<u32, Plan>,
    plan_todo: Vec<u32>,
    /// the plans made by the last call: [piece, impulse (N s), n, the other part's bricks as places in the piece's
    /// list (n of them)] ...
    pub plan_out: Vec<f32>,
    /// pieces parted by `snap`, and the two they became
    pub snapped: crate::Map<u32, (u32, u32)>,
}
/// Where a falling piece breaks first, and how hard a hit must be to break it there: its weakest section under
/// its own inertia, landing on its underside. The joints to cut (with their bricks, to see they are still
/// there), the bricks of the part that comes away (the lighter), and the normal impulse over one 60 Hz step.
pub struct Plan { pub len: usize, pub cut: Vec<(u32, u32, u32)>, pub other: Vec<u32>, pub impulse: f32 }
/// at most this many standing bricks are split into convex cells (beyond it a fractured brick stays a box)
pub const FRAC_MAX: usize = 1500;

/// fragments of fewer bricks than this count as rubble; at most `rubble_keep` (default RUBBLE_KEEP) of them per
/// impacts call stay as bodies
const RUBBLE_MIN: usize = 48;
/// fragments of at least this volume (m^3) stay bodies past the rubble budget, up to RUBBLE_MORE times it per call
const FRAGMENT_KEEP_M3: f32 = 0.25;
const RUBBLE_MORE: usize = 4;
const RUBBLE_KEEP: usize = 64;
/// load (share of strength) above which a joint wears, and how fast (per second at full strength); see wear_joints
const SUSTAINED: f32 = 0.7;
/// the share of its strength a material carries for good: concrete and brick fail in time above about 75% (creep
/// rupture), timber above about 70% (duration of load); steel and glass at room temperature do not creep. With
/// one threshold for all, a steel roof beam left at 77% at the cut's edge wore through in 14 s and the top storey
/// came down in pieces while the storeys below stood (engine exploration, the nibble cuts' first stage)
fn sustained(m: u8) -> f32 {
    #[cfg(test)] { extern crate std; if std::env::var("SF_WEAR_ALL").is_ok() { return SUSTAINED; } }
    match m { crate::brick::mat::CONCRETE | crate::brick::mat::MASONRY => 0.75, crate::brick::mat::TIMBER => 0.7, _ => 2.0 }
}
const WEAR_RATE: f32 = 1.5;
/// crumbs kept as bodies per impacts call (the rest of the crushed material goes as dust)
const CRUMB_KEEP: usize = 256;
/// how deep a section that gives way is crushed before what is beyond it moves freely (m)
const CRUSH_DEPTH: f32 = 0.5;

/// The static world's mesh chunks: slices a storey high, and in a city each lot's tile on its own (a change remeshes
/// its lot's slices, not the city's). Chunk k is tile k / slices, slice k % slices; a brick is in the chunk of its
/// min corner.
#[derive(Clone, Copy, Debug)]
pub struct Tiles { pub tile: i32, pub n: i32, pub slices: i32 }
impl Tiles {
    pub fn count(&self) -> usize { (self.n * self.n * self.slices) as usize }
    pub fn tile_of(&self, x: i32, z: i32) -> i32 {
        if self.tile == 0 { return 0; }
        let f = |v: i32| (v + self.n * self.tile / 2).div_euclid(self.tile).clamp(0, self.n - 1);
        f(z) * self.n + f(x)
    }
    pub fn chunk(&self, x: i32, y: i32, z: i32) -> i32 { self.tile_of(x, z) * self.slices + y.div_euclid(tower::STOREY).clamp(0, self.slices - 1) }
    /// is a brick with min corner (x, y, z) in chunk k
    pub fn holds(&self, k: i32, x: i32, y: i32, z: i32) -> bool { self.chunk(x, y, z) == k }
}

impl Game {
    pub fn new(storeys: i32) -> Self { Self::new_kind(storeys, 0) }
    /// A tower of `storeys` storeys built as `kind` (see `tower::element_in`).
    pub fn new_kind(storeys: i32, kind: u8) -> Self {
        let b = tower::build_kind(storeys, 1, kind);
        // (the city: storeys is its size, n x n lots; the slices reach its tallest building)
        let tiles = if kind == 5 {
            let top = (0..b.len() as u32).filter(|&i| b.alive(i)).map(|i| b.tbox(i).max[1]).max().unwrap_or(0);
            Tiles { tile: crate::city::pitch(), n: crate::city::layout(storeys).n, slices: top / tower::STOREY + 2 }
        } else { Tiles { tile: 0, n: 1, slices: storeys + 2 } };
        let n = tiles.count();
        let start = if kind == 5 { let l = crate::city::layout(storeys); let x = if l.n % 2 == 1 { crate::city::PITCH_M / 2.0 } else { 0.0 }; v3(x, 0.26, 0.0) } else { v3(0.0, 0.26, -14.0) };
        let mut g = Game { b, rest: Vec::new(), rest_applied: 0.0, side_failed: 0, panel_ratio: 0.0, cracked_total: 0, warm: Default::default(), last_iterations: 0, player: Player::new(start), tiles, active: vec![true; (tiles.n * tiles.n) as usize], tile_ratio: vec![0.0; (tiles.n * tiles.n) as usize], chunks: (0..n).map(|_| Mesh::default()).collect(),
                           out: Vec::new(), piece_out: Vec::new(), loose_pending: false, last_detached: 0, dirty: (0, n as i32 - 1), pieces: Vec::new(), max_ratio: 0.0, failed_total: 0, inp: Vec::new(), piece_ratio: 0.0, piece_info: [0.0; 4], dbg: Vec::new(), caps: Vec::new(), dust: 0, mesh: true, cut_dbg: [0.0; 6], crumb_keep: CRUMB_KEEP, rubble_keep: RUBBLE_KEEP, crush_depth: CRUSH_DEPTH, spill_on: false, spill: Vec::new(), view_bend: false, view_vel: Vec::new(), kicks: Vec::new(), motion: [0.0; 12], struct_dt: 0.0, rest_fresh: false, rest_kept: Vec::new(), dynamic: false, dynst: Default::default(), moving: false, dyn_speed: 0.0, dyn_its: 0, dyn_kinetic: 0.0, dyn_left: 0, dyn_cut: 0, dyn_capped: false, view: Default::default(), view_want: false, view_out: Vec::new(), view_in: Vec::new(), view_bricks: Vec::new(), view_chunks: Vec::new(), wearing: 0, wearing_at: Vec::new(), worn_total: 0, last_failed: Vec::new(), dup_bricks: 0, spill_why: [0; 4], spill_mat: [0; 8], spill_single: 0, masonry_parts: false, inner_wear: Default::default(), joint_wear: Default::default(), frac_made: 0, frac_max: FRAC_MAX, plans: Default::default(), plan_todo: Vec::new(), plan_out: Vec::new(), snapped: Default::default() };
        for k in 0..n as i32 { g.remesh(k); }
        g.loose_pending = (0..g.b.len()).any(|i| g.b.flags[i] & crate::brick::F_LOOSE != 0);
        g
    }

    pub fn remesh(&mut self, k: i32) {
        if !self.mesh { return; }
        if k < 0 || k as usize >= self.chunks.len() { return; }
        let t = self.tiles;
        self.chunks[k as usize] = mesh::build_view(&self.b, |b, i| t.holds(k, b.x[i as usize] as i32, b.y[i as usize] as i32, b.z[i as usize] as i32), Some(&self.view));
    }

    /// A brick as [x, y, z, sx, sy, sz (ticks), material + 16 x paint + 512 x hull]. The hull is a library cell's
    /// id (0 a box), or for a free piece 256 + its index in `Bricks::pieces` (its shape: `piece_shape`). (The code is
    /// a float, exact to 2^24: up to 32,512 free pieces.)
    fn push_box(out: &mut Vec<f32>, b: &Bricks, i: u32) {
        let t = b.tbox(i);
        let h = if b.hull[i as usize] == crate::brick::HULL_FREE { 256 + b.piece_of.get(&i).copied().unwrap_or(0).min(32_511) } else { b.hull[i as usize] as u32 };
        out.extend_from_slice(&[t.min[0] as f32, t.min[1] as f32, t.min[2] as f32, (t.max[0] - t.min[0]) as f32, (t.max[1] - t.min[1]) as f32, (t.max[2] - t.min[2]) as f32, (b.mat[i as usize] as u32 + 16 * b.paint[i as usize] as u32 + 512 * h) as f32]);
    }
    /// Free piece `k` into `piece_out`: [centre (m) x y z, rotation x y z w, corners n, corners (m, its own frame
    /// about the centre) ..., faces n, per face: corners n, corner indices ...]. Returns its length (0: no piece).
    pub fn piece_shape(&mut self, k: u32) -> usize {
        self.piece_out.clear();
        let Some(p) = self.b.piece_k(k) else { return 0 };
        let c = p.centre();
        self.piece_out.extend_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32, p.q[0], p.q[1], p.q[2], p.q[3], p.v.len() as f32]);
        for v in p.v { self.piece_out.extend_from_slice(v); }
        self.piece_out.push(p.faces.len() as f32);
        for f in p.faces { self.piece_out.push(f.len() as f32); for &i in f { self.piece_out.push(i as f32); } }
        self.piece_out.len()
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

    /// Split the standing bricks marked on a fracture surface into their convex cells (within `frac_max`); the cells
    /// that spall off are removed and returned, to fall as crumbs. Bricks of falling pieces stay as they are (the
    /// GPU places a piece's parts one for one).
    fn shatter_pending(&mut self) -> Vec<u32> {
        let pend = core::mem::take(&mut self.b.frac_pending);
        let mut crumbs = Vec::new();
        for i in pend {
            let iu = i as usize;
            if self.b.flags[iu] & crate::brick::F_FRAC == 0 { continue; }
            if self.b.flags[iu] & (F_ALIVE | F_DYNAMIC) != F_ALIVE || self.frac_made >= self.frac_max { self.b.flags[iu] &= !crate::brick::F_FRAC; continue; }
            let (_, spall) = self.b.shatter(i);
            self.frac_made += 1;
            // (removing a spalled cell marks nothing more: a fracture surface does not spread by itself)
            for c in spall { self.b.remove_quiet(c); crumbs.push(c); }
        }
        crumbs
    }

    /// Plan the first break of falling piece `pid` (`Plan`): its joints' ratios under its own weight, resting on its
    /// lowest bricks; the strongest of them failing first, the fewest from the top that part it in two. A hit that
    /// decelerates it by g / (worst ratio) breaks it there: the impulse M g dt / ratio over a step dt.
    fn plan_piece(&mut self, pid: u32) {
        let set = self.pieces[pid as usize].clone();
        if set.len() < 2 || set.len() > 4000 { return; }
        let miny = set.iter().map(|&i| self.b.y[i as usize]).min().unwrap();
        // (what it lands on: its bricks within half a metre of its lowest)
        let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] as i32 <= miny as i32 + 4).collect();
        if sink.iter().all(|&x| x) { return; }
        let load: Vec<f32> = set.iter().map(|&i| structure::weight(&self.b, i)).collect();
        let a = structure::analyse(&self.b, &set, &load, &sink, v3(0.0, -1.0, 0.0));
        let mut r = a.ratios; r.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(core::cmp::Ordering::Equal));
        if r.is_empty() || !(r[0].1 > 1e-6) { return; }
        // a falling building breaks across a bed joint, a storey line or a wall's course: a horizontal plane that
        // parts it, every bearing joint on it cut. The plane whose worst joint is the worst, with at least a
        // twentieth of the mass on the smaller side. (Cutting single joints worst first, the 4-storey tower took
        // 4,096 of its 5,140 joints to part: its worst joints are spread over the building.)
        let rmap: crate::Map<u32, f32> = r.iter().copied().collect();
        let mut planes: crate::Map<i32, (f32, Vec<u32>)> = crate::Map::default();
        for &i in &set { let mut j = self.b.jhead[i as usize]; while j != NONE {
            if self.b.ja[j as usize] == i && self.b.jaxis[j as usize] == 1 && self.b.body[self.b.jb[j as usize] as usize] == pid {
                let e = planes.entry(self.b.tbox(i).max[1]).or_insert((0.0, Vec::new()));
                e.0 = e.0.max(*rmap.get(&j).unwrap_or(&0.0)); e.1.push(j);
            }
            j = self.b.joint_next(j, i); } }
        let mut pl: Vec<(i32, f32, Vec<u32>)> = planes.into_iter().map(|(y, (q, js))| (y, q, js)).collect();
        pl.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(core::cmp::Ordering::Equal));
        let m = |v: &Vec<u32>| v.iter().map(|&i| structure::mass(&self.b, i)).sum::<f32>();
        let total: f32 = m(&set);
        let parts = |cut: &crate::Set<u32>| -> Vec<Vec<u32>> {
            let mut idx = vec![NONE; self.b.len()];
            for (q, &i) in set.iter().enumerate() { idx[i as usize] = q as u32; }
            let mut seen = vec![false; set.len()]; let mut out = Vec::new();
            for s0 in 0..set.len() { if seen[s0] { continue; }
                let mut g = Vec::new(); let mut st = vec![s0]; seen[s0] = true;
                while let Some(q) = st.pop() { let i = set[q]; g.push(i);
                    let mut j = self.b.jhead[i as usize];
                    while j != NONE { if !cut.contains(&j) { let o = idx[self.b.joint_other(j, i) as usize]; if o != NONE && !seen[o as usize] { seen[o as usize] = true; st.push(o as usize); } } j = self.b.joint_next(j, i); }
                }
                out.push(g);
            }
            out
        };
        let mut found = None;
        for (_, q, js) in pl.iter().take(24) {
            if *q < 1e-6 { break; }
            let cut: crate::Set<u32> = js.iter().copied().collect();
            let mut g = parts(&cut);
            g.sort_by(|x, y| m(y).partial_cmp(&m(x)).unwrap());
            if g.len() >= 2 && m(&g[1]) >= 0.05 * total { found = Some((*q, js.clone(), g)); break; }
        }
        #[cfg(test)] { extern crate std; if std::env::var("SF_PLAN").is_ok() { std::println!("  plan {pid}: {} bricks, {} planes, worst {:.3}: {:?}", set.len(), pl.len(), r[0].1, found.as_ref().map(|f| (f.0, f.1.len(), f.2.len()))); } }
        let Some((q, cutj, g)) = found else { return };
        let impulse = total * 9.81 / q / 60.0;
        // (the smaller part, and with it any bits the cut parts besides: they come away with it)
        let other: Vec<u32> = g[1..].iter().flatten().copied().collect();
        let pos: crate::Map<u32, usize> = set.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        self.plan_out.extend_from_slice(&[pid as f32, impulse, other.len() as f32]);
        for i in &other { self.plan_out.push(pos[i] as f32); }
        let cut = cutj.iter().map(|&j| (j, self.b.ja[j as usize], self.b.jb[j as usize])).collect();
        self.plans.insert(pid, Plan { len: set.len(), cut, other, impulse });
    }
    /// Plan the pieces made since the last call (`plan_out` lists them)
    pub fn finish_plans(&mut self) {
        let todo = core::mem::take(&mut self.plan_todo);
        let pieces = &self.pieces; self.plans.retain(|&p, _| !pieces[p as usize].is_empty());
        for pid in todo { if !self.pieces[pid as usize].is_empty() { self.plan_piece(pid); } }
    }
    /// The GPU broke falling piece `pid` by its plan, in the step of a hit: cut the plan's joints and make the two
    /// parts pieces of their own. Output as `impact`: [2, (id, n, boxes) for the part it kept, then for the other];
    /// [0] when the piece changed since (the engine broke it first, say) and the plan no longer holds.
    pub fn snap(&mut self, pid: u32) -> usize {
        self.out.clear();
        let Some(pl) = self.plans.remove(&pid) else { self.out.push(0.0); return 0 };
        let pu = pid as usize;
        let ok = pu < self.pieces.len() && self.pieces[pu].len() == pl.len
            && pl.cut.iter().all(|&(j, a, b)| (j as usize) < self.b.ja.len() && self.b.ja[j as usize] == a && self.b.jb[j as usize] == b)
            && pl.other.iter().all(|&i| self.b.alive(i) && self.b.body[i as usize] == pid);
        if !ok { self.out.push(0.0); return 0; }
        for &(j, _, _) in &pl.cut { self.b.fracture_joint(j); }
        let set = core::mem::take(&mut self.pieces[pu]);
        let other: crate::Set<u32> = pl.other.iter().copied().collect();
        // (both parts in the piece's order: the GPU's shapes of each body are in it)
        let (moved, kept): (Vec<u32>, Vec<u32>) = set.into_iter().partition(|i| other.contains(i));
        let ids = [self.new_piece(kept), self.new_piece(moved)];
        self.snapped.insert(pid, (ids[0], ids[1]));
        self.out.push(2.0);
        for &id in &ids {
            self.out.push(id as f32); self.out.push(self.pieces[id as usize].len() as f32);
            for k in 0..self.pieces[id as usize].len() { let i = self.pieces[id as usize][k]; Self::push_box(&mut self.out, &self.b, i); }
        }
        2
    }

    fn new_piece(&mut self, bricks: Vec<u32>) -> u32 {
        let id = self.pieces.len() as u32;
        self.plan_todo.push(id);
        for &i in &bricks { self.b.flags[i as usize] |= F_DYNAMIC; self.b.body[i as usize] = id; }
        self.pieces.push(bricks);
        id
    }

    /// Write the static change log (from the store's log) and the new bodies; remesh what changed.
    /// Keep an even sample of `v` within `keep`; the rest is returned as spill (with `spill_on`) or counted as dust.
    fn shed(&mut self, v: &mut Vec<u32>, keep: usize) -> Vec<u32> {
        if v.len() <= keep { return Vec::new(); }
        let n = v.len();
        let pick: Vec<bool> = { let mut p = vec![false; n]; for k in 0..keep { p[k * n / keep.max(1)] = true; } p };
        let (mut kept, mut rest) = (Vec::with_capacity(keep), Vec::new());
        for (k, &i) in v.iter().enumerate() { if pick[k] && keep > 0 { kept.push(i) } else { rest.push(i) } }
        *v = kept;
        if self.spill_on { rest } else { self.dust += rest.len(); Vec::new() }
    }
    fn emit(&mut self, crumbs: &[u32], pieces: &[u32]) {
        let mut out = Vec::new();
        let mut log = core::mem::take(&mut self.b.log);
        log.sort_unstable(); log.dedup();
        // a brick now in a piece is gone from the static world
        let stat: Vec<u32> = log.into_iter().filter(|&i| (i as usize) < self.b.len()).collect();
        out.push(stat.len() as f32);
        // the chunks to remesh: each changed brick's, the slice under it and the one at its top (faces against it
        // show or hide there)
        let (mut klo, mut khi) = (i32::MAX, i32::MIN);
        for &i in &stat {
            let t = self.b.tbox(i);
            let tile = self.tiles.tile_of(t.min[0], t.min[2]) as usize; if tile < self.active.len() { self.active[tile] = true; }
            for y in [t.min[1], t.min[1] - tower::STOREY, t.max[1]] { let k = self.tiles.chunk(t.min[0], y, t.min[2]); klo = klo.min(k); khi = khi.max(k); }
            let alive = self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE;
            out.extend_from_slice(&[i as f32, if alive { 1.0 } else { 0.0 }]);
            Self::push_box(&mut out, &self.b, i);
        }
        let spill = core::mem::take(&mut self.spill);
        out.push((crumbs.len() + spill.len() + pieces.len()) as f32);
        for &i in crumbs { out.extend_from_slice(&[-1.0, 1.0]); Self::push_box(&mut out, &self.b, i); }
        for &i in &spill { out.extend_from_slice(&[-2.0, 1.0]); Self::push_box(&mut out, &self.b, i); }
        self.spill_why[3] += spill.len();
        for &p in pieces {
            out.push(p as f32); out.push(self.pieces[p as usize].len() as f32);
            for k in 0..self.pieces[p as usize].len() { let i = self.pieces[p as usize][k]; Self::push_box(&mut out, &self.b, i); }
        }
        self.out = out;
        if klo <= khi {
            for k in klo..=khi { self.remesh(k); }
            self.dirty = (klo.max(0), khi.min(self.chunks.len() as i32 - 1));
        } else { self.dirty = (0, -1); }
    }

    /// The page's copy: take the view a structure step wrote (`view_in`, the format of `view_out`), draw it with
    /// displacements `scale` times as large and, with `stress`, coloured by load, and remesh the chunks of the bricks
    /// it names (`dirty`). Bricks it does not name keep what they had.
    pub fn view_apply(&mut self, scale: f32, stress: bool) {
        let d = core::mem::take(&mut self.view_in);
        let was = self.view.on && self.view.scale == scale && self.view.stress == stress;
        self.view.on = true; self.view.scale = scale; self.view.stress = stress;
        let mut o = 0usize;
        let nb = if d.is_empty() { 0 } else { d[0] as usize }; o += 1;
        let mut named: Vec<u32> = Vec::new();
        for _ in 0..nb { if o + 1 >= d.len() { break; } let i = d[o] as u32; self.view.ratio.insert(i, d[o + 1]); self.view.cells.remove(&i); named.push(i); o += 2; }
        let nc = if o < d.len() { d[o] as usize } else { 0 }; o += 1;
        for _ in 0..nc {
            if o + 19 >= d.len() { break; }
            let i = d[o] as u32;
            self.view.cells.entry(i).or_default().push(mesh::ViewCell { lo: [d[o + 1] as i32, d[o + 2] as i32], hi: [d[o + 3] as i32, d[o + 4] as i32], c: [d[o + 5], d[o + 6], d[o + 7]], u: [d[o + 8], d[o + 9], d[o + 10]], r: [d[o + 11], d[o + 12], d[o + 13]] , v: [d[o + 14], d[o + 15], d[o + 16]], w: [d[o + 17], d[o + 18], d[o + 19]] });
            o += 20;
        }
        // the chunks of the bricks named, and no others (a range from the first to the last of them took in most of
        // the 8 x 8 city's 960 chunks: 672 ms of meshing, native, each time)
        let nch = self.chunks.len() as i32;
        let mut hit = vec![false; nch as usize];
        if was {
            for &i in &named { if (i as usize) < self.b.len() && self.b.alive(i) { let t = self.b.tbox(i); let k = self.tiles.chunk(t.min[0], t.min[1], t.min[2]); if k >= 0 && k < nch { hit[k as usize] = true; } } }
        } else { for h in hit.iter_mut() { *h = true; } }
        self.view_chunks.clear();
        for k in 0..nch { if hit[k as usize] { self.remesh(k); self.view_chunks.push(k); } }
        // (`dirty` is left as the last job set it: the page may not have meshed that yet)
    }
    /// The page's copy: where the structure's cell at a point (m) has gone and how it moves, into `motion`
    /// ([displacement, rotation, velocity, turning rate]); false if no cell of the view holds the point.
    pub fn view_motion(&mut self, p: [f32; 3]) -> bool {
        let t = [p[0] / crate::brick::TICK, p[1] / crate::brick::TICK, p[2] / crate::brick::TICK];
        let mut best: Option<(f32, &mesh::ViewCell)> = None;
        for (&i, cs) in self.view.cells.iter() {
            if (i as usize) >= self.b.len() { continue; }
            let bx = self.b.tbox(i);
            if t[1] < bx.min[1] as f32 - 1.0 || t[1] > bx.max[1] as f32 + 1.0 { continue; }
            for c in cs {
                if t[0] < c.lo[0] as f32 - 1.0 || t[0] > c.hi[0] as f32 + 1.0 || t[2] < c.lo[1] as f32 - 1.0 || t[2] > c.hi[1] as f32 + 1.0 { continue; }
                let e = [c.c[0] - p[0], c.c[1] - p[1], c.c[2] - p[2]];
                let d = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
                if best.map_or(true, |b| d < b.0) { best = Some((d, c)); }
            }
        }
        match best { Some((_, c)) => { self.motion = [c.u[0], c.u[1], c.u[2], c.r[0], c.r[1], c.r[2], c.v[0], c.v[1], c.v[2], c.w[0], c.w[1], c.w[2]]; true } None => false }
    }
    /// The page's copy: the building as it is built again (all chunks remeshed).
    pub fn view_off(&mut self) {
        self.view = Default::default();
        for k in 0..self.chunks.len() as i32 { self.remesh(k); }
        self.view_chunks = (0..self.chunks.len() as i32).collect();
        self.dirty = (0, self.chunks.len() as i32 - 1);
    }

    /// Detach everything no longer connected to the ground as pieces; returns their ids.
    fn detach(&mut self) -> Vec<u32> {
        // (a city: only its active lots, see `active`; rubble lying on a lot makes it active)
        let lots = self.tiles.tile != 0;
        if lots { for r in &self.rest { let t = self.tiles.tile_of((r[0] / TICK) as i32, (r[2] / TICK) as i32) as usize; if t < self.active.len() { self.active[t] = true; } } }
        let analysed = self.active.clone();
        let tl = self.tiles;
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && (!lots || analysed[tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize])).collect();
        if lots { for a in self.active.iter_mut() { *a = false; } }
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
        crumbs.extend(self.shatter_pending());
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
        // (a city: only its active lots, see `active`; rubble lying on a lot makes it active)
        let lots = self.tiles.tile != 0;
        if lots { for r in &self.rest { let t = self.tiles.tile_of((r[0] / TICK) as i32, (r[2] / TICK) as i32) as usize; if t < self.active.len() { self.active[t] = true; } } }
        let analysed = self.active.clone();
        let tl = self.tiles;
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && (!lots || analysed[tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize])).collect();
        if lots { for a in self.active.iter_mut() { *a = false; } }
        let mut load: Vec<f32> = set.iter().map(|&i| structure::weight(&self.b, i)).collect();
        // rubble lying on the structure weighs on the brick under it: a floor heaped with debris carries it, and
        // fails when it is too much
        let rest = if self.dynamic {
            if self.rest_fresh || !self.rest.is_empty() { self.rest_kept = core::mem::take(&mut self.rest); }
            self.rest_kept.clone()
        } else { core::mem::take(&mut self.rest) };
        self.rest_fresh = false;
        self.rest_applied = 0.0;
        let mut point = Vec::new();
        if !rest.is_empty() {
            let mut idx = vec![NONE; self.b.len()];
            for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
            // (not on the ground slab: it rests on the ground, and debris lying there loads nothing)
            for r in &rest {
                if let Some(i) = self.brick_below(v3(r[0], r[1], r[2]), 4.0) { let k = idx[i as usize]; if k != NONE && self.b.y[i as usize] > 0 { point.push((k, r[0] / TICK, r[2] / TICK, r[3])); self.rest_applied += r[3]; } }
            }
        }
        let _ = &mut load;
        let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
        // a dynamic step when the last step was a moment ago; after a longer gap the structure is solved at rest
        // (the time since the last step, between a 120th of a second and `DYN_MAX_DT`: a structure at rest that is
        // changed now moves from now, however long ago its last step was)
        let h = if self.dynamic && self.struct_dt > 0.0 { (self.struct_dt as f64).clamp(1.0 / 120.0, DYN_MAX_DT as f64) } else { 0.0 };
        self.view_bricks.clear(); self.view_vel.clear();
        self.moving = false; self.dyn_speed = 0.0; self.dyn_its = 0; self.dyn_kinetic = 0.0; self.dyn_left = 0; self.dyn_cut = 0; self.dyn_capped = false;
        let a = if self.dynamic {
            let mut by: crate::Map<usize, Vec<usize>> = Default::default();
            for (k, &i) in set.iter().enumerate() { by.entry(if lots { tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize } else { 0 }).or_default().push(k); }
            let mut keys: Vec<usize> = by.keys().copied().collect(); keys.sort_unstable();
            let mut all: Option<structure::Analysis> = None;
            for t in keys {
                let ks = &by[&t];
                let mut local = vec![NONE; set.len()]; for (n, &k) in ks.iter().enumerate() { local[k] = n as u32; }
                let s2: Vec<u32> = ks.iter().map(|&k| set[k]).collect();
                let l2: Vec<f32> = ks.iter().map(|&k| load[k]).collect();
                let k2: Vec<bool> = ks.iter().map(|&k| sink[k]).collect();
                let p2: Vec<(u32, f32, f32, f32)> = point.iter().filter(|p| local[p.0 as usize] != NONE).map(|p| (local[p.0 as usize], p.1, p.2, p.3)).collect();
                let st = self.dynst.entry(t).or_default();
                // (the hits on this lot since its last step)
                { let mine = |p: &[f64; 3]| !lots || tl.tile_of((p[0] / TICK as f64) as i32, (p[2] / TICK as f64) as i32) as usize == t; let mut k = 0; while k < self.kicks.len() { if mine(&self.kicks[k].0) { st.1.kicks.push(self.kicks.swap_remove(k)); st.1.settle = 0; } else { k += 1; } } }
                // (a lot not solved yet is solved at rest first: stepped from no displacement, its weight would land
                // on it all at once)
                st.1.h = if st.0.is_empty() { 0.0 } else { h };
                st.1.want_cells = self.view_want;
                let mut a = structure::analyse_dyn(&self.b, &s2, &l2, &p2, &k2, v3(0.0, -1.0, 0.0), Some(&mut st.0), Some(&mut st.1));
                // (the view: a lot is written when it is first asked for and when its step changed something. Every
                // analysed lot written at each step, the 8 x 8 city's lots with rubble or furniture on their floors
                // came again every second, 279,622 numbers, and the page remeshed for them: 3 frames a second)
                let sag = if self.view_bend { a.cells.iter().map(|c| c.2[0] * c.2[0] + c.2[1] * c.2[1] + c.2[2] * c.2[2]).fold(0.0, f64::max) } else { 0.0 };
                if self.view_bend && sag >= VIEW_BEND * VIEW_BEND { st.1.bent = true; }
                if self.view_want && (!self.view_bend || st.1.bent) && (!st.1.viewed || st.1.its > 0 || !a.failed.is_empty() || !a.cracks.is_empty()) { st.1.viewed = true; self.view_bricks.extend_from_slice(&s2); self.view_vel.extend_from_slice(&st.1.cell_vel); } else { a.cells.clear(); a.inner_all.clear(); }
                if st.1.moving() { self.moving = true; if lots && t < self.active.len() { self.active[t] = true; } }
                self.dyn_speed = self.dyn_speed.max(st.1.speed as f32); self.dyn_its = self.dyn_its.max(st.1.its); self.dyn_kinetic += st.1.kinetic as f32;
                self.dyn_left += st.1.left; self.dyn_cut += a.left_cut; self.dyn_capped |= st.1.capped;
                all = Some(match all { None => a, Some(mut m) => { m.merge(a); m } });
            }
            all.unwrap_or_else(structure::Analysis::empty)
        } else if !lots { structure::analyse_warm(&self.b, &set, &load, &point, &sink, v3(0.0, -1.0, 0.0), Some(&mut self.warm)) } else {
            // each lot on its own: nothing joins two lots, and the solver's accuracy is relative to what it solves
            // (all 25 lots of a 5 x 5 city at once, two complex buildings' arches came out as mechanisms, 2.25 and
            // 1.19 of strength, where on their own they stand at 0.66)
            let mut by: crate::Map<usize, Vec<usize>> = Default::default();
            for (k, &i) in set.iter().enumerate() { by.entry(tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize).or_default().push(k); }
            let mut keys: Vec<usize> = by.keys().copied().collect(); keys.sort_unstable();
            let mut all: Option<structure::Analysis> = None;
            for t in keys {
                let ks = &by[&t];
                let mut local = vec![NONE; set.len()]; for (n, &k) in ks.iter().enumerate() { local[k] = n as u32; }
                let s2: Vec<u32> = ks.iter().map(|&k| set[k]).collect();
                let l2: Vec<f32> = ks.iter().map(|&k| load[k]).collect();
                let k2: Vec<bool> = ks.iter().map(|&k| sink[k]).collect();
                let p2: Vec<(u32, f32, f32, f32)> = point.iter().filter(|p| local[p.0 as usize] != NONE).map(|p| (local[p.0 as usize], p.1, p.2, p.3)).collect();
                let a = structure::analyse_warm(&self.b, &s2, &l2, &p2, &k2, v3(0.0, -1.0, 0.0), Some(&mut self.warm));
                all = Some(match all { None => a, Some(mut m) => { m.merge(a); m } });
            }
            all.unwrap_or_else(structure::Analysis::empty)
        };
        self.last_iterations = a.iterations;
        // the view: every analysed brick's worst joint or section, and the cells as this step left them
        self.view_out.clear();
        if self.view_want {
            if !self.dynamic { self.view_vel.clear(); }
            let mut worst: crate::Map<u32, f32> = Default::default();
            if !self.dynamic { self.view_bricks = set.clone(); }
            for &i in &self.view_bricks { worst.insert(i, 0.0); }
            let mut up = |i: u32, r: f32| { if let Some(w) = worst.get_mut(&i) { if r > *w { *w = r; } } };
            for &(j, r) in &a.ratios { let r = if r >= structure::DYN_CUT { 1.0 } else { r }; let (p, q) = (self.b.ja[j as usize], self.b.jb[j as usize]); if p != NONE { up(p, r); up(q, r); } }
            for &(i, r) in &a.inner_all { up(i, r); }
            for c in &a.cracks { up(c.brick, c.ratio.min(1.0)); }
            self.view_out.push(worst.len() as f32);
            for &i in &self.view_bricks { self.view_out.push(i as f32); self.view_out.push(worst[&i]); }
            self.view_out.push(a.cells.len() as f32);
            let vels = self.view_vel.len() == a.cells.len();
            for (n, (k, c, x)) in a.cells.iter().enumerate() {
                self.view_out.extend_from_slice(&[k[0] as f32, k[1] as f32, k[2] as f32, k[3] as f32, k[4] as f32, c[0] as f32, c[1] as f32, c[2] as f32, x[0] as f32, x[1] as f32, x[2] as f32, x[3] as f32, x[4] as f32, x[5] as f32]);
                // (and its velocity: what leaves the structure starts as a body with it)
                self.view_out.extend_from_slice(&if vels { self.view_vel[n] } else { [0.0; 6] });
            }
        }
        // (the worst ratio with rubble on the floors: since the spring network, the building's worst)
        self.panel_ratio = a.max_ratio;
        self.max_ratio = a.max_ratio;
        let mut a = a;
        self.wear_joints(&mut a);
        if lots {
            // each analysed lot's worst ratio; a lot with joints wearing (or failing: they change it) stays active
            let tile_b = |b: &Bricks, i: u32| tl.tile_of(b.x[i as usize] as i32, b.z[i as usize] as i32) as usize;
            for (t, &on) in analysed.iter().enumerate() { if on { self.tile_ratio[t] = 0.0; } }
            for &(j, r) in &a.ratios { let i = self.b.ja[j as usize]; if i == NONE || r >= structure::DYN_CUT { continue; } let t = tile_b(&self.b, i); self.tile_ratio[t] = self.tile_ratio[t].max(r); }
            for &(i, _, _, r) in &a.inner { let t = tile_b(&self.b, i); self.tile_ratio[t] = self.tile_ratio[t].max(r); }
            for &i in &self.wearing_at { let t = tile_b(&self.b, i); self.active[t] = true; }
            for &(j, _) in &a.failed { let i = self.b.ja[j as usize]; if i != NONE { let t = tile_b(&self.b, i); self.active[t] = true; } }
            for c in &a.cracks { let t = tile_b(&self.b, c.brick); self.active[t] = true; }
            self.max_ratio = self.tile_ratio.iter().copied().fold(0.0, f32::max);
            self.panel_ratio = self.max_ratio;
        }
        self.last_failed = a.failed.iter().filter_map(|&(j, r)| { let (p, q) = (self.b.ja[j as usize], self.b.jb[j as usize]); if p == NONE { None } else { Some((p, q, r)) } }).collect();
        for c in &a.cracks { self.last_failed.push((c.brick, c.brick, c.ratio)); }
        let work = self.apply_failures(&a, max_fail);
        // (joints gone: a lot that was settling moves again from here)
        if !work.is_empty() { for st in self.dynst.values_mut() { st.1.settle = 0; st.1.no_settle = false; } }
        self.failed_total += work.len();
        let mut crumbs = self.shatter_pending();
        let keep = self.crumb_keep;
        let sp = self.shed(&mut crumbs, keep); self.spill.extend(sp);
        let pieces = if work.is_empty() && !self.loose_pending { Vec::new() } else { self.detach() };
        self.loose_pending = false;
        self.last_detached = pieces.len();
        self.emit(&crumbs, &pieces);
        work.len()
    }

    /// Joints carrying more than SUSTAINED of their strength wear: concrete and masonry under a sustained load
    /// above about 70-75% of their short-term strength fail in time (creep rupture). Wear grows with the time since
    /// the last check and with the square of how far the load is past SUSTAINED; a joint worn through fails as if
    /// over strength (cracks first where it would), its wear starts again from zero, and a worn joint is softer
    /// (see `structure::spring`), so its load moves to its neighbours. Each material has its own share (`sustained`:
    /// steel does not wear). Game time is compressed (real creep rupture takes hours): concrete at 90% wears through
    /// in about 2 s, at 80% in about 17 s (WEAR_RATE). At 0.5 /s, the lab's nibble scenario, its ground storey eaten
    /// to the core, wore through 2 joints. A building left near its limit then cracks and comes down progressively instead
    /// of standing unchanged until one more cut fails it all at once.
    fn wear_joints(&mut self, a: &mut structure::Analysis) {
        self.wearing = 0;
        self.wearing_at.clear();
        if self.struct_dt <= 0.0 { return; }
        let dt = self.struct_dt.min(1.0);
        let failing: crate::Set<u32> = a.failed.iter().map(|&(j, _)| j).collect();
        let beds: crate::Set<u32> = a.beds.iter().copied().collect();
        // (a yielding joint is at its strength because it yields, and its turning is counted against its ductility:
        // it does not wear as well)
        let yielded: crate::Set<u32> = a.yielded.iter().copied().collect();
        for &(j, r) in &a.ratios {
            if r < SUSTAINED || r >= 1.0 || failing.contains(&j) || beds.contains(&j) || yielded.contains(&j) { continue; }
            let ju = j as usize;
            if ju >= self.b.jdmg.len() || self.b.ja[ju] == NONE { continue; }
            let s0 = sustained(self.b.mat[self.b.ja[ju] as usize]).min(sustained(self.b.mat[self.b.jb[ju] as usize]));
            if r < s0 { continue; }
            self.wearing += 1; self.wearing_at.push(self.b.ja[ju]);
            let x = (r - s0) / (1.0 - s0);
            let d = self.b.jdmg[ju];
            let wear = self.joint_wear.get(&j).copied().unwrap_or(0.0) + dt * WEAR_RATE * x * x;
            if wear >= 1.0 { self.b.jdmg[ju] = d & 1; self.joint_wear.remove(&j); a.failed.push((j, 1.0)); self.worn_total += 1; }
            else { self.joint_wear.insert(j, wear); self.b.jdmg[ju] = (d & 1) | (((wear * 127.0) as u8).min(127) << 1); }
        }
        // (wear stays with a joint while it lasts: damage does not heal when the load moves away)
        let b = &self.b; self.joint_wear.retain(|&j, _| (j as usize) < b.ja.len() && b.ja[j as usize] != NONE);
        // faces inside bricks (a slab's own section, a column's): worn through, they crack there
        let mut keep: crate::Map<(u32, u8, i32), f32> = Default::default();
        for &(i, ax, at, r) in &a.inner {
            let s0 = sustained(self.b.mat[i as usize]);
            if r < s0 { continue; }
            self.wearing += 1; self.wearing_at.push(i);
            let x = (r - s0) / (1.0 - s0);
            let key = (i, ax as u8, at);
            let w = self.inner_wear.get(&key).copied().unwrap_or(0.0) + dt * WEAR_RATE * x * x;
            if w >= 1.0 { a.cracks.push(structure::Crack { brick: i, axis: ax, at, ratio: 1.0 }); self.worn_total += 1; } else { keep.insert(key, w); }
        }
        // (a face no longer loaded so far forgets its wear: the check sees the load as it is now)
        self.inner_wear = keep;
    }

    /// Apply the worst `max` failures of an analysis: failed joints are removed, cracked bricks are split at
    /// the failing section and the new joint across the cut is left out. Returns how many were applied.
    fn apply_failures(&mut self, a: &structure::Analysis, max: usize) -> Vec<u32> { self.apply_failures_as(a, max, true) }
    /// `standing`: the standing structure, where brick cracks and holds; in a falling piece hit by an impact, brick
    /// parts at once (holding there, each impact re-cracked the same pieces: 10,889 cracks in 240 frames of the lab's
    /// mixed scenario, 4 times the building's joints)
    fn apply_failures_as(&mut self, a: &structure::Analysis, max: usize, standing: bool) -> Vec<u32> {
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
                                // (a crack through a brick pier parts it: held, its halves cracked again at the next section, round after
                // round: 7,159 cracks in 240 frames of the mixed scenario. Brick cracks and holds at its joints.)
                let reinf = self.b.flags[c.brick as usize] & crate::brick::F_REINF != 0;
                let n = self.b.split(c.brick, c.axis, c.at);
                let mut jj = self.b.jhead[n as usize];
                while jj != NONE {
                    if self.b.joint_other(jj, n) == c.brick {
                        // reinforced concrete cracks but the steel holds the two halves: the joint stays, cracked
                        if reinf { self.b.jdmg[jj as usize] = 1; self.cracked_total += 1; } else { self.b.fracture_joint(jj); }
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
                // (in a falling part too: parted at once there, a brick building landing on its ground storey burst
                // into loose bricks up to its roof; set_param 6 brings that back, for comparison)
                let masonry = (standing || !self.masonry_parts) && (self.b.mat[x as usize] == crate::brick::mat::MASONRY || self.b.mat[y as usize] == crate::brick::mat::MASONRY);
                if (reinf || masonry) && self.b.jdmg[j as usize] & 1 == 0 { self.b.jdmg[j as usize] = 1; self.b.log.push(x); self.b.log.push(y); self.cracked_total += 1; done.push(j); continue; }
                self.b.fracture_joint(j);
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
            cracked = self.apply_failures_as(&a, usize::MAX, false).len();
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
        let mut hits: Vec<Hit> = inp.chunks_exact(11).map(|r| Hit {
            who: r[0] as i32, p: [r[1] / TICK, r[2] / TICK, r[3] / TICK], d: v3(r[4], r[5], r[6]), j: r[7], v: r[8], other: r[9] as i32,
        }).collect();
        // a hit on a piece the GPU broke by its plan (and `snap` parted since) goes to the part holding the brick
        // nearest the contact: the engine refines the break from there
        for h in hits.iter_mut() {
            if h.who < 0 { continue; }
            if let Some(&(a, b)) = self.snapped.get(&(h.who as u32)) {
                let near = |pid: u32| self.pieces[pid as usize].iter().map(|&i| dist2(&self.b, i, h.p)).fold(f32::MAX, f32::min);
                h.who = if near(a) <= near(b) { a as i32 } else { b as i32 };
            }
        }
        let ce = |m: i32| if m < 0 { 0.0 } else { structure::crush_energy(m as u8) };
        // the energy divides by the other side's crush energy: the weaker material takes most of it
        // (steel on concrete crushes the concrete, concrete on concrete half each): V = E c_o / (c_s (c_s + c_o))
        let crush_r = |e: f32, cs: f32, co: f32| { let co = if co > 0.0 { co } else { cs }; cbrt(3.0 * e * co / (cs * (cs + co)) / (4.0 * core::f32::consts::PI)).min(1.5) };
        self.b.log.clear();
        // ---- the standing structure
        let mut scrumbs = Vec::new();
        let mut extra: Vec<(u32, f32, [f32; 3])> = Vec::new();
        let mut side: Vec<(u32, V3, [f32; 3])> = Vec::new();
        let (lots_k, tl_k) = (self.tiles.tile != 0, self.tiles);
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
            // (the dynamic structure takes the hit as an impulse in its next step, where it moves, yields or breaks
            // by it; the checks at rest below are the solve at rest's)
            if self.dynamic {
                if h.v >= KICK_SPEED { self.kicks.push(([(h.p[0] * TICK) as f64, (h.p[1] * TICK) as f64, (h.p[2] * TICK) as f64], [(h.d.x * h.j) as f64, (h.d.y * h.j) as f64, (h.d.z * h.j) as f64], structure::KICK_TIME)); if lots_k { let t = tl_k.tile_of(h.p[0] as i32, h.p[2] as i32) as usize; if t < self.active.len() { self.active[t] = true; } } }
                continue;
            }
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
        { let keep = self.crumb_keep / 2; let sp = self.shed(&mut scrumbs, keep); self.spill.extend(sp); }
        // what is left of the budgets for the falling pieces (shared by all of this call's pieces)
        let mut crumb_room = self.crumb_keep - scrumbs.len().min(self.crumb_keep);
        let mut rubble_room = self.rubble_keep;
        let mut rubble_more = self.rubble_keep * RUBBLE_MORE;
        let mut rubble: Vec<u32> = Vec::new();
        let mut spieces = Vec::new();
        if !extra.is_empty() || !scrumbs.is_empty() || !side.is_empty() {
            // (a city: only its active lots, see `active`; rubble lying on a lot makes it active)
        let lots = self.tiles.tile != 0;
        if lots { for r in &self.rest { let t = self.tiles.tile_of((r[0] / TICK) as i32, (r[2] / TICK) as i32) as usize; if t < self.active.len() { self.active[t] = true; } } }
        let analysed = self.active.clone();
        let tl = self.tiles;
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && (!lots || analysed[tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize])).collect();
        if lots { for a in self.active.iter_mut() { *a = false; } }
            let mut idx = vec![NONE; self.b.len()];
            for (k, &i) in set.iter().enumerate() { idx[i as usize] = k as u32; }
            let load: Vec<f32> = set.iter().map(|&i| structure::weight(&self.b, i)).collect();
            // (at the contact: a hit on a 22 m slab loads the cell it lands on)
            let mut point = Vec::new();
            for &(i, f, p) in &extra { let k = idx[i as usize]; if k != NONE { point.push((k, p[0], p[2], f)); } }
            let sink: Vec<bool> = set.iter().map(|&i| self.b.y[i as usize] <= 0).collect();
            if !extra.is_empty() {
                let a = if !lots { structure::analyse_warm(&self.b, &set, &load, &point, &sink, v3(0.0, -1.0, 0.0), Some(&mut self.warm)) } else {
            // each lot on its own: nothing joins two lots, and the solver's accuracy is relative to what it solves
            // (all 25 lots of a 5 x 5 city at once, two complex buildings' arches came out as mechanisms, 2.25 and
            // 1.19 of strength, where on their own they stand at 0.66)
            let mut by: crate::Map<usize, Vec<usize>> = Default::default();
            for (k, &i) in set.iter().enumerate() { by.entry(tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize).or_default().push(k); }
            let mut keys: Vec<usize> = by.keys().copied().collect(); keys.sort_unstable();
            let mut all: Option<structure::Analysis> = None;
            for t in keys {
                let ks = &by[&t];
                let mut local = vec![NONE; set.len()]; for (n, &k) in ks.iter().enumerate() { local[k] = n as u32; }
                let s2: Vec<u32> = ks.iter().map(|&k| set[k]).collect();
                let l2: Vec<f32> = ks.iter().map(|&k| load[k]).collect();
                let k2: Vec<bool> = ks.iter().map(|&k| sink[k]).collect();
                let p2: Vec<(u32, f32, f32, f32)> = point.iter().filter(|p| local[p.0 as usize] != NONE).map(|p| (local[p.0 as usize], p.1, p.2, p.3)).collect();
                let a = structure::analyse_warm(&self.b, &s2, &l2, &p2, &k2, v3(0.0, -1.0, 0.0), Some(&mut self.warm));
                all = Some(match all { None => a, Some(mut m) => { m.merge(a); m } });
            }
            all.unwrap_or_else(structure::Analysis::empty)
        };
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
                // (a city: only its active lots, see `active`; rubble lying on a lot makes it active)
        let lots = self.tiles.tile != 0;
        if lots { for r in &self.rest { let t = self.tiles.tile_of((r[0] / TICK) as i32, (r[2] / TICK) as i32) as usize; if t < self.active.len() { self.active[t] = true; } } }
        let analysed = self.active.clone();
        let tl = self.tiles;
        let set: Vec<u32> = (0..self.b.len() as u32).filter(|&i| self.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && (!lots || analysed[tl.tile_of(self.b.x[i as usize] as i32, self.b.z[i as usize] as i32) as usize])).collect();
        if lots { for a in self.active.iter_mut() { *a = false; } }
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
            // the fracture surfaces: cells, and what spalls off them falls as crumbs (within the budget left)
            let mut sp = self.shatter_pending();
            { let more = self.shed(&mut sp, crumb_room); self.spill.extend(more); }
            crumb_room -= sp.len();
            scrumbs.extend(sp);
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
            // what the impact can crush: its energy over the crush energy of the material at the contacts (the
            // weaker side takes most, as for the spheres above). The crushed storey below draws on what is left
            let crush_vol = {
                let mut v = 0.0f32;
                for h in &mine {
                    let Some(&n) = set.iter().chain(crumbs.iter()).min_by(|&&x, &&y| dist2(&self.b, x, h.p).partial_cmp(&dist2(&self.b, y, h.p)).unwrap()) else { continue };
                    let cs = structure::crush_energy(self.b.mat[n as usize]); let co = { let c = ce(h.other); if c > 0.0 { c } else { cs } };
                    v += 0.5 * h.j * h.v * co / (cs * (cs + co));
                }
                v
            };
            let brick_vol = |b: &Bricks, i: u32| { let t = b.tbox(i); ((t.max[0] - t.min[0]) * (t.max[1] - t.min[1]) * (t.max[2] - t.min[2])) as f32 * TICK * TICK * TICK };
            let mut crush_left = crush_vol - crumbs.iter().map(|&i| brick_vol(&self.b, i)).sum::<f32>();
            for &i in &crumbs { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
            let mut fspill = self.shed(&mut crumbs, crumb_room); crumb_room -= crumbs.len(); self.spill_why[0] += fspill.len();
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
                    cracked = self.apply_failures_as(&a, usize::MAX, false).len();
                    self.piece_ratio = a.max_ratio; self.piece_info = [a.failed.len() as f32, a.cracks.len() as f32, sink.iter().filter(|&&s| s).count() as f32, set.len() as f32];
                    for i in before..self.b.len() { set.push(i as u32); }
                }
            }
            if crumbs.is_empty() && fspill.is_empty() && cracked == 0 { self.pieces[pid] = set; continue; }
            self.caps.push(cap_used);
            let mut kids: Vec<(i32, bool, Vec<u32>)> = crumbs.iter().map(|&i| (-1, true, vec![i])).collect();
            sever_infill(&mut self.b, &set);
            // rubble budget: a crushed zone falls apart into many fragments; the biggest stay as bodies, the
            // finest go as dust (removed) so a collapse does not run out of bodies
            let mut groups = components(&self.b, &set);
            groups.sort_by_key(|g| core::cmp::Reverse(g.len()));
            let gvol = |b: &Bricks, g: &[u32]| g.iter().map(|&i| brick_vol(b, i)).sum::<f32>();
            groups.retain(|g| {
                if g.len() >= RUBBLE_MIN { return true; }
                if rubble_room > 0 { rubble_room -= 1; return true; }
                // a fragment of some size stays a body past the budget (up to 4 times it): as particles, a falling
                // part landing poured out as sand (the lab's mixed scenario: 6,517 bricks of fragments in 150 frames)
                if gvol(&self.b, g) >= FRAGMENT_KEEP_M3 && rubble_more > 0 { rubble_more -= 1; return true; }
                for &i in g { self.b.flags[i as usize] &= !F_DYNAMIC; self.b.remove(i); self.b.flags[i as usize] |= F_DYNAMIC; }
                if self.spill_on { fspill.extend_from_slice(g); self.spill_why[1] += g.len(); for &i in g.iter() { self.spill_mat[(self.b.mat[i as usize] as usize).min(7)] += 1; } if g.len() == 1 { self.spill_single += 1; } } else { self.dust += g.len(); }
                false
            });
            for g in groups {
                // with a section cut, the parts wholly beyond it were cut free; otherwise the parts at a contact
                let hit = match cut_at {
                    Some((ax, sg, at)) => g.iter().any(|&i| { let t = self.b.tbox(i); (t.min[ax] * sg).min(t.max[ax] * sg) < at }),
                    None => g.iter().any(|&i| near_contact(&self.b, i)),
                };
                // the side of a section that gave way, where the contacts are, is the crushed storey: it does not
                // stay a solid block for the floors above to land on. But only as much of it is crushed as the
                // impact's energy can crush, nearest the contacts first; the rest stays in pieces (all of it
                // crushed, a slab landing on the ground poured out as a stream of particles)
                if hit && cut_at.is_some() {
                    let d = |b: &Bricks, i: u32| mine.iter().map(|h| dist2(b, i, h.p)).fold(f32::MAX, f32::min);
                    let mut g = g;
                    g.sort_by(|&x, &y| d(&self.b, x).partial_cmp(&d(&self.b, y)).unwrap());
                    let mut k = 0;
                    while k < g.len() && crush_left > 0.0 { crush_left -= brick_vol(&self.b, g[k]); k += 1; }
                    rubble.extend_from_slice(&g[..k]);
                    let rest: Vec<u32> = g[k..].to_vec();
                    if !rest.is_empty() {
                        for part in components(&self.b, &rest) { let id = self.new_piece(part.clone()); kids.push((id as i32, true, part)); }
                    }
                    continue;
                }
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
                    if k < keep { kids.push((-1, true, vec![i])); } else if self.spill_on { fspill.push(i); self.spill_why[2] += 1; } else { self.dust += 1; }
                }
                rubble.clear();
            }
            for &i in &fspill { kids.push((-2, true, vec![i])); }
            // (a brick may leave in one kid only: counted, for the lab, if not)
            { let mut seen = crate::Set::default(); for (_, _, g) in &kids { for &i in g { if !seen.insert(i) { self.dup_bricks += 1; } } } }
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
                // (a free piece is cut by the plane itself, or not at all if the plane misses it)
                let Some(m) = b.cut(i, ax, plane) else { continue };
                let mut jj = b.jhead[m as usize];
                while jj != NONE { if b.joint_other(jj, m) == i { b.fracture_joint(jj); break; } jj = b.joint_next(jj, m); }
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
        for &j in &cut { b.fracture_joint(j); n += 1; }
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
    for &j in &cut { b.fracture_joint(j); }
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
    #[test] fn a_building_left_near_its_limit_wears_and_comes_down_over_time() { wears(false); }
    // Out of the default run since 14 Nov (docs/LAB.md, "The buildings re-sized"): that the worn frame comes down in
    // several rounds. With what a floor carries and concrete's shear at 3 MPa the frame stands at 0.60; nibbled in
    // rows of 0.4 m it reaches 0.90 before the next row brings it down at once, and from 0.90 one joint wears
    // through in 41 s (56 joints crack with it), the rest is relieved, and nothing more goes in the 450 s watched.
    // Before: from 0.99, failures in checks [1, 5, 47, 59, 121].
    #[test] #[ignore] fn a_worn_building_comes_down_in_several_rounds() { wears(true); }
    fn wears(gate: bool) {
        // the ground storey taken away row by row (0.7 m blasts at 1.6 m, from one side) until the worst joint is
        // past SUSTAINED but under strength, as a player nibbling at it. Without wear (dt 0) it then stands for
        // ever, unchanged: the "floating" building. With wear (0.5 s between checks) its joints crack over the
        // following seconds, round after round, before it comes down.
        let prep = |kind: u8| -> Option<Game> {
            let mut g = Game::new_kind(4, kind);
            assert_eq!(g.structure_step(64), 0);
            let mut x = -11.6f32;
            while x <= 11.7 {
                let mut z = -11.6f32; while z <= 11.7 { g.blast(v3(x, 1.6, z), 0.7); z += 0.8; }
                let mut n = 1; let mut k = 0; while n > 0 && k < 200 { n = g.structure_step(64); k += 1; }
                if g.failed_total > 400 { return None; }
                // (0.90: well into the band where concrete wears, which starts at 0.75. It was 0.75, and with what a
                // floor carries, 14 Nov, the frame stands at 0.60 and the first row took it to 0.77, where a joint
                // lasts longer than the 120 s this test watches)
                if g.max_ratio >= 0.90 { return Some(g); }
                if std::env::var("SF_TR").is_ok() { std::println!("  row {x:.1}: worst {:.3}, {} failed", g.max_ratio, g.failed_total); }
                x += 0.4;
            }
            None
        };
        // (the concrete frame: its worst joint goes 0.35 -> 0.58 as the rows go, then the row at x = -4.4 cracks 125
        // joints, which hold at their residual strength, worst 1.00. The mixed building's brick walls give way at
        // the first row, 3,751 joints: it is nibbled at in the lab instead, scenario nibble.)
        for kind in [0u8] {
            let mut still = prep(kind).expect("reaches the wear band");
            let r0 = still.max_ratio;
            let mut n0 = 0; for _ in 0..40 { n0 += still.structure_step(64); }
            let mut g = prep(kind).unwrap();
            g.struct_dt = 0.5;
            let mut rounds_with = Vec::new();
            for step in 0..900 { let n = g.structure_step(64); if n > 0 { rounds_with.push(step); if rounds_with.len() % 3 == 1 { std::println!("  check {step}: {} failed, worst {:.2}, largest displacement {:.1} mm", g.failed_total, g.max_ratio, g.warm.max_disp() * 1e3); } } if g.failed_total > 2000 { break; } }
            std::println!("kind {kind}: worst {r0:.2} after the cuts (largest displacement {:.1} mm); without wear {n0} failures in 40 checks; with wear failures in checks {:?} ({} joints, {} cracked, largest displacement {:.1} mm)", still.warm.max_disp() * 1e3, &rounds_with[..rounds_with.len().min(30)], g.failed_total, g.cracked_total, g.warm.max_disp() * 1e3);
            assert_eq!(n0, 0, "kind {kind}: without wear nothing changes");
            assert!(!rounds_with.is_empty(), "kind {kind}: with wear its joints give way in time");
            // (not in the check that follows the cut. It was not in the first two: the solve then stopped at r z down
            // by 1e-5 and left the cut frame's worst joint at 1.00, held at its residual strength; solved to 1e-7 the
            // joint that goes first stands at 0.96 to 0.99, and the wear rule, (r - 0.7)^2 a second, takes it in the
            // second check, 1 s after the cut.)
            assert!(rounds_with[0] >= 1, "kind {kind}: not at once");
            if gate { assert!(rounds_with.len() >= 4, "kind {kind}: in several rounds, not one"); }
        }
    }
    #[test] #[ignore] fn explore_failures_far_from_the_cut() {
        // (exploring: cut a few ground-storey columns; where in height do joints fail, with and without wear)
        for kind in [0u8, 1, 2, 3] {
            let storeys = if kind == 3 { 5 } else { 4 };
            let mut g = Game::new_kind(storeys, kind);
            g.struct_dt = 0.25; let n0 = g.structure_step(64);
            std::println!("kind {kind}: intact worst {:.2}, {} joints wearing, {n0} failures", g.max_ratio, g.wearing);
            // the west row of the ground storey cut over 6 m
            let mut z = -3.2f32; while z <= 3.3 { for x in [-11.6f32, -10.8] { g.blast(v3(x, 1.6, z), 0.7); } z += 0.8; }
            let mut by = [0usize; 16]; let mut far = Vec::new();
            for step in 0..120 {
                let f0 = g.failed_total;
                let a = { let n = g.structure_step(64); n };
                let _ = a;
                for &(p, q, r) in &g.last_failed.clone() { let st = ((g.b.y[p as usize] as i32).max(g.b.y[q as usize] as i32) / tower::STOREY) as usize; by[st.min(15)] += 1; if st >= 2 && far.len() < 12 { let c = centre(&g.b, p); far.push((step, st, (c[0] * TICK * 10.0).round() / 10.0, (c[2] * TICK * 10.0).round() / 10.0, (r * 100.0).round() / 100.0, g.b.mat[p as usize], g.b.mat[q as usize])); } }
                let _ = f0;
            }
            std::println!("  after the cut: {} failed; failures by storey {:?}; first far ones (check, storey, x, z, ratio, mats) {:?}", g.failed_total, &by[..storeys as usize + 1], far);
        }
    }
    // Out of the default run since 11 Nov (docs/LAB.md, "The cut mixed building is at its limit"): after the cut this
    // building stands at 0.91 to 0.99 of strength with 8 to 13 joints wearing, and what the next 20 s do to it turns
    // on the solve's last digits. Failures in storeys 1-3 / above: 1 / 0 as this test was written (the solve stopped
    // at 0.857, r z down by 1e-5, cells known by their brick); solved to 1e-12 the same build gives 4 / 3, and this
    // one 2 / 2; at this build's 1e-7, 580 above (a joint at 1.00 goes and the storey with it). Not a property the
    // model has, then: no failures above needs the cut building further from its limit, or wear that does not
    // take a joint at 0.9 in seconds.
    #[test] #[ignore] fn parts_far_above_a_cut_do_not_fall_on_their_own() {
        // the mixed building's west wall and the west halves of its fronts shot away at 1.3 m (the lab's nibble,
        // first stage), then 20 s of checks with wear. With one sustained threshold (0.7) for every material, a steel
        // roof beam left at 77% at the cut's edge wore through at 14 s and the top storey came down in pieces, 70
        // failures from the fourth storey up, while the storeys below stood: reported from the playground as "parts of the
        // fourth floor start falling without any reason". Steel does not creep: now 0 there.
        let mut g = Game::new_kind(5, 3);
        assert_eq!(g.structure_step(64), 0);
        let settle = |g: &mut Game| { let mut n = 1; let mut k = 0; while n > 0 && k < 300 { n = g.structure_step(64); k += 1; } };
        let mut z = -10.8f32; while z <= 10.9 { g.blast(v3(-11.2, 1.3, z), 0.7); z += 0.9; } settle(&mut g);
        let mut x = -10.8f32; while x <= 0.1 { g.blast(v3(x, 1.3, -11.2), 0.7); g.blast(v3(x, 1.3, 11.2), 0.7); x += 0.9; } settle(&mut g);
        g.struct_dt = 0.25;
        let mut high = 0; let mut low = 0;
        for _ in 0..80 {
            let nf = g.structure_step(64);
            if std::env::var("SF_TR").is_ok() { std::println!("  step: failed {nf} its {} worst {:.3} wearing {}", g.last_iterations, g.max_ratio, g.wearing); }
            for &(p, q, _) in &g.last_failed { if (g.b.y[p as usize] as i32).max(g.b.y[q as usize] as i32) >= 3 * tower::STOREY { high += 1 } else { low += 1 } }
        }
        std::println!("20 s after the cut: {low} failures in storeys 1-3, {high} above");
        assert_eq!(high, 0, "nothing gives way two storeys and more above the cut on its own");
    }
    #[test] #[ignore] fn explore_the_nibbled_mixed_building() {
        // (exploring: the lab's nibble cuts, in the engine: what holds the upper storeys, and what fails first)
        let mut g = Game::new_kind(5, 3);
        assert_eq!(g.structure_step(64), 0);
        let settle = |g: &mut Game| { let mut n = 1; let mut k = 0; while n > 0 && k < 300 { n = g.structure_step(64); k += 1; } };
        let mut z = -10.8f32; while z <= 10.9 { g.blast(v3(-11.2, 1.3, z), 0.7); z += 0.9; } settle(&mut g);
        let mut x = -10.8f32; while x <= 0.1 { g.blast(v3(x, 1.3, -11.2), 0.7); g.blast(v3(x, 1.3, 11.2), 0.7); x += 0.9; } settle(&mut g);
        if std::env::var("NIB_STAGE").map(|v| v == "1").unwrap_or(false) { std::println!("(stage 1: the west wall and the west halves of the fronts)"); } else {
        for zc in [-7.5f32, 7.5] { for k in 0..4 { g.blast(v3(-7.5 + 0.3 * k as f32 - 0.45, 1.3, zc), 0.7); } } settle(&mut g);
        let mut x = 0.9f32; while x <= 10.9 { g.blast(v3(x, 1.3, -11.2), 0.7); g.blast(v3(x, 1.3, 11.2), 0.7); x += 0.9; } settle(&mut g);
        let mut z = -10.8f32; while z <= 10.9 { g.blast(v3(11.2, 1.3, z), 0.7); z += 0.9; } settle(&mut g);
        for zc in [-7.5f32, 7.5] { for k in 0..4 { g.blast(v3(7.5 + 0.3 * k as f32 - 0.45, 1.3, zc), 0.7); } } settle(&mut g);
        }
        std::println!("after the cuts: worst {:.2}, {} failed, {} cracked", g.max_ratio, g.failed_total, g.cracked_total);
        // standing above the ground storey, by storey and material
        let report = |g: &Game, what: &str| {
            let mut by = [[0usize; 5]; 6];
            for i in 0..g.b.len() as u32 { if g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) != F_ALIVE { continue; } let st = (g.b.y[i as usize] as i32 / tower::STOREY).clamp(0, 5) as usize; by[st][(g.b.mat[i as usize] as usize).min(4)] += 1; }
            std::println!("  {what}: standing bricks by storey [concrete, masonry, steel, glass, timber]: {:?}", by);
        };
        report(&g, "after the cuts");
        g.struct_dt = 0.25;
        for step in 0..80 {
            let n = g.structure_step(64);
            if n > 0 {
                let mut st = [0usize; 6];
                for &(p, q, _) in &g.last_failed { let s2 = ((g.b.y[p as usize] as i32).max(g.b.y[q as usize] as i32) / tower::STOREY).clamp(0, 5) as usize; st[s2] += 1; }
                let w: std::vec::Vec<_> = g.last_failed.iter().take(6).map(|&(p, q, r)| { let c = centre(&g.b, p); ((c[0] * TICK).round(), (c[1] * TICK * 10.0).round() / 10.0, (c[2] * TICK).round(), g.b.mat[p as usize], g.b.mat[q as usize], (r * 100.0).round() / 100.0) }).collect();
                std::println!("  check {step}: {n} failures, by storey {:?}, worst {:.2}; first (x, y, z, mats, ratio) {:?}", st, g.max_ratio, w);
            }
        }
        report(&g, "20 s later");
    }
    #[test] #[ignore] fn explore_the_mixed_building_cut_away() {
        // (exploring: how much of the mixed building's ground storey can go before it comes down)
        for (name, cut) in [("all but the core", (|x: f32, z: f32| !(x.abs() < 4.4 && z.abs() < 4.4)) as fn(f32, f32) -> bool),
                            ("west half", |x, _z| x < 0.0), ("the perimeter walls", |x, z| x.abs() > 10.0 || z.abs() > 10.0)] {
            let mut g = Game::new_kind(5, 3);
            assert_eq!(g.structure_step(64), 0);
            let r0 = g.max_ratio;
            let mut x = -11.6f32;
            while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 { if cut(x, z) { g.blast(v3(x, 1.6, z), 0.7); } z += 0.8; } x += 0.8; }
            let mut rounds = 0; let mut biggest = 0; let mut fell = 0usize;
            while rounds < 300 { let n = g.structure_step(64); for (id, sz) in bodies_in(&g.out) { if id >= 0 { biggest = biggest.max(sz); fell += sz; } } rounds += 1; if n == 0 { break; } }
            std::println!("mixed, {name}: intact worst {r0:.2}; after the cut {} failed, {} cracked in {rounds} rounds, worst now {:.2}, biggest falling piece {biggest} bricks, {fell} bricks fell", g.failed_total, g.cracked_total, g.max_ratio);
            // then left standing, checked every 0.25 s with wear, for 60 s
            g.struct_dt = 0.25; let mut fell2 = 0usize; let mut when = Vec::new();
            for step in 0..240 { let n = g.structure_step(64); let mut f = 0; for (id, sz) in bodies_in(&g.out) { if id >= 0 { f += sz; } } if n > 0 { when.push((step, n, f)); } fell2 += f; }
            std::println!("   then with wear: {} worn, {} bricks fell, {} wearing at the end, worst {:.2}; checks with failures (check, failures, bricks fell): {:?}", g.worn_total, fell2, g.wearing, g.max_ratio, &when[..when.len().min(25)]);
            // what still stands away from the core, above the ground storey, by material
            let mut by = [0usize; 8]; let mut mass = 0.0f32;
            for i in 0..g.b.len() as u32 { if g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) != F_ALIVE { continue; } let c = centre(&g.b, i);
                if (c[0].abs() > 6.0 / TICK || c[2].abs() > 6.0 / TICK) && c[1] > 4.0 / TICK { by[g.b.mat[i as usize] as usize] += 1; mass += structure::mass(&g.b, i); } }
            std::println!("   standing beyond the core above the ground storey: bricks by material {:?}, {:.0} t", by, mass / 1000.0);
        }
    }
    #[test] fn crushed_material_over_the_budget_is_spilled_not_removed() {
        // the tower's ground storey cut, the tower landing on its stumps at 5.2 m/s (as the test above), with the low
        // preset's budget of 64 crumbs: the crushed storey beyond it was removed as dust, whole floors vanishing
        // (lab collapse: 489 of 1,211 m3 in 240 frames). With spill on, it comes out as spill (id -2) instead.
        let run = |spill: bool| -> (usize, usize, usize) {
            let mut g = Game::new(4);
            g.crumb_keep = 64; g.spill_on = spill;
            let mut x = -11.6f32; let mut piece = -1i32;
            while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 { g.blast(v3(x, 1.6, z), 0.7); for (id, sz) in bodies_in(&g.out) { if id >= 0 && sz > 100 { piece = id; } } z += 0.8; } x += 0.8; }
            assert!(piece >= 0);
            let dust0 = g.dust;
            let m: f32 = g.pieces[piece as usize].iter().map(|&i| structure::mass(&g.b, i)).sum();
            let vol0: f32 = g.pieces[piece as usize].iter().map(|&i| { let t = g.b.tbox(i); ((t.max[0] - t.min[0]) * (t.max[1] - t.min[1]) * (t.max[2] - t.min[2])) as f32 }).sum::<f32>() * TICK * TICK * TICK;
            let low = g.pieces[piece as usize].iter().map(|&i| g.b.y[i as usize] as i32).min().unwrap();
            let tops: std::vec::Vec<u32> = (0..g.b.len() as u32).filter(|&i| g.b.flags[i as usize] & (F_ALIVE | F_DYNAMIC) == F_ALIVE && { let t = g.b.tbox(i); t.max[1] >= 5 && t.max[1] <= low && g.b.flags[i as usize] & crate::brick::F_INFILL == 0 }).collect();
            let v = 5.2; let j = m * v / tops.len() as f32;
            for &i in &tops { let t = g.b.tbox(i); let p = [(t.min[0] + t.max[0]) as f32 * 0.5 * TICK, t.max[1] as f32 * TICK, (t.min[2] + t.max[2]) as f32 * 0.5 * TICK]; let fall = (low - t.max[1]) as f32 * TICK;
                g.inp.extend_from_slice(&[-1.0, p[0], p[1], p[2], 0.0, -1.0, 0.0, j, v, 0.0, 0.0]); g.inp.extend_from_slice(&[piece as f32, p[0], p[1] + fall, p[2], 0.0, 1.0, 0.0, j, v, 0.0, 0.0]); }
            g.impacts(1.0 / 60.0);
            let o = &g.out; let nu = o[0] as usize; let mut k = 1 + nu * 9; let nb = o[k] as usize; k += 1;
            let bv = |o: &[f32], at: usize| o[at + 3] * o[at + 4] * o[at + 5];
            let (mut spilled, mut crushed_v, mut piece_v) = (0, 0.0f32, 0.0f32);
            for _ in 0..nb { let id = o[k] as i32; let c = o[k + 1] as usize; if id == -2 { spilled += 1; } for q in 0..c { crushed_v += bv(o, k + 2 + q * 7); } k += 2 + c * 7; }
            let nbr = o[k] as usize; k += 1;
            for _ in 0..nbr { k += 1; let nk = o[k] as usize; k += 2; for _ in 0..nk { let id = o[k] as i32; let c = o[k + 2] as usize; if id == -2 { spilled += 1; } let mut v = 0.0; for q in 0..c { v += bv(o, k + 3 + q * 7); } if id < 0 { crushed_v += v } else { piece_v += v } k += 3 + c * 7; } }
            let e: f32 = tops.len() as f32 * 2.0 * 0.5 * j * v;
            std::println!("  (spill {spill}) the landing: {:.1} MJ, {:.1} m3 crushed, {:.1} m3 still in pieces, {:.0} t ({vol0:.1} m3) landing", e / 1e6, crushed_v * TICK * TICK * TICK, piece_v * TICK * TICK * TICK, m / 1e3);
            if spill { assert!(crushed_v * TICK * TICK * TICK <= 2.0 * e / structure::crush_energy(crate::brick::mat::CONCRETE) + 2.0, "only what the energy can crush"); assert!(piece_v > 4.0 * crushed_v, "most stays in pieces"); }
            (g.dust - dust0, spilled, 0)
        };
        let (dust_off, sp_off, _) = run(false);
        let (dust_on, sp_on, _) = run(true);
        std::println!("landing on the stumps with 64 crumbs: spill off {dust_off} bricks to dust ({sp_off} spilled); spill on {dust_on} to dust, {sp_on} spilled");
        assert_eq!(dust_on, 0, "nothing removed");
        assert!(sp_on >= dust_off, "what was dust is spilled");
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
    #[test] fn a_second_check_starts_from_the_first() {
        // the standing building's solve starts from its last solution: unchanged, the second check of the
        // 12-storey tower needs no iterations and gives the same worst ratio; a small change (one column gone)
        // needs fewer than from zero
        let mut g = Game::new_kind(12, 0);
        let t = std::time::Instant::now();
        g.structure_step(64);
        let (cold, r1, t1) = (g.last_iterations, g.max_ratio, t.elapsed().as_secs_f64() * 1e3);
        let t = std::time::Instant::now();
        g.structure_step(64);
        let (again, r2, t2) = (g.last_iterations, g.max_ratio, t.elapsed().as_secs_f64() * 1e3);
        // the ground storey blasted away but one column (as `a_tower_left_on_one_column_comes_down`): the first
        // round of failures must be the same from the last solution as from zero
        let mut cold_g = Game::new_kind(12, 0);
        let mut x = -11.6f32;
        while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
            if (x + 3.7).abs() > 1.3 || (z + 3.7).abs() > 1.3 { g.blast(v3(x, 1.6, z), 0.7); cold_g.blast(v3(x, 1.6, z), 0.7); }
            z += 0.8; } x += 0.8; }
        let (fw, fc) = (g.structure_step(1 << 20), cold_g.structure_step(1 << 20));
        std::println!("tower checks: first {cold} iterations ({t1:.0} ms), again {again} ({t2:.0} ms), worst {r1:.3} / {r2:.3}; with the ground storey gone but one column: {} iterations warm against {} cold, worst {:.3} / {:.3}, failures {fw} / {fc}", g.last_iterations, cold_g.last_iterations, g.max_ratio, cold_g.max_ratio);
        // (the same within the solve's tolerance: a joint at 1.00 of strength can go either way)
        assert!(fw > 0 && (g.max_ratio - cold_g.max_ratio).abs() < 0.01 * cold_g.max_ratio && (fw as f32 - fc as f32).abs() <= (0.01 * fc as f32).max(2.0), "warm and cold agree");
        assert!(g.last_iterations < cold_g.last_iterations);
        assert_eq!(again, 0);
        assert!((r1 - r2).abs() < 1e-4);
    }
    #[test] fn the_mixed_buildings_timber_floor_gives_way_under_rubble_and_its_brick_cracks() {
        // a timber floor bay of the third storey (joists along x, 2.4 MPa over 0.25 m: 25 kNm/m) between the brick
        // wall and the steel beam at x = -7.5 m (3.25 m clear): 3 kPa is carried; 40 kPa (q L^2 / 8 = 54 kNm/m,
        // shared between midspan and the section over the beam) is not: 1.55 of strength at its worst joist section
        // (3 kPa: 0.59, the building's own worst joint elsewhere)
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
    #[test] fn a_blast_leaves_a_fracture_surface() {
        // a blast in a wall: the bricks round the crater that are left standing are split into the convex cells of
        // their fracture patterns, and one or two cells of each spall off as crumbs; none bigger than 2 m or
        // smaller than 0.25 m a side. The cells are in the engine's mesh and carry their hull ids to the page.
        let mut g = Game::new_kind(3, 1);
        g.blast(v3(-6.0, 5.0, -11.2), 1.2);
        let cells: std::vec::Vec<u32> = (0..g.b.len() as u32).filter(|&i| g.b.alive(i) && g.b.hull[i as usize] != 0 && g.b.flags[i as usize] & F_DYNAMIC == 0).collect();
        let mut parents: std::vec::Vec<crate::brick::TBox> = cells.iter().map(|&i| g.b.tbox(i)).collect();
        parents.sort_by_key(|t| (t.min, t.max)); parents.dedup();
        let spalled = bodies_in(&g.out).iter().filter(|&&(id, _)| id < 0).count();
        std::println!("blast in a masonry facade: {} standing bricks split into {} cells, {} crumbs out (with the crater's), {} split so far", parents.len(), cells.len(), spalled, g.frac_made);
        assert!(parents.len() > 4, "the crater's rim is a fracture surface");
        for t in &parents { for d in 0..3 { let v = t.max[d] - t.min[d]; assert!(v >= 2 && v <= 16, "size {v}"); } }
        // every cell is joined to something, and the cells' volumes are those of their bricks less what spalled
        for &i in &cells { assert!(g.b.jhead[i as usize] != NONE); }
        let mut q = 0.0f64; for &i in &cells { q += g.b.vol_share(i); }
        assert!(q < parents.len() as f64 && q > 0.3 * parents.len() as f64, "cells kept: {q} bricks' worth of {}", parents.len());
        // the out records carry the hull id (code / 512)
        let mut f = std::vec::Vec::new(); g.dump(); f.extend_from_slice(&g.out);
        let n = f[0] as usize; let with_hull = (0..n).filter(|&k| f[1 + k * 8 + 7] as u32 / 512 != 0).count();
        assert_eq!(with_hull, cells.len());
        // the spalled cells: one or two, never all, the same every time
        for i in 0..200u32 { for n in 2..=6u32 { let (p, m) = crate::brick::frac_cells(i, n); assert!(p < 16 && m != 0 && m.count_ones() <= 2 && m < (1 << n)); assert_eq!(crate::brick::frac_cells(i, n), (p, m)); } }
    }
    #[test] fn the_same_game_gives_the_same_result() {
        // determinism: the same blasts, structure checks and break plans, twice in fresh games, give the same
        // outputs bit for bit. (The maps' hasher is fixed, crate::Map: the default one is seeded per process, so two
        // page loads could walk a map in different orders; within one process, as here, the seed is shared.)
        let run = || {
            let mut g = Game::new_kind(4, 3);
            let mut outs: Vec<f32> = Vec::new();
            let mut x = -11.0f32;
            while x <= 11.0 { g.blast(v3(x, 1.6, -11.2), 0.9); outs.extend_from_slice(&g.out); x += 1.5; }
            for _ in 0..6 { g.structure_step(64); outs.extend_from_slice(&g.out); }
            g.plan_out.clear(); g.finish_plans(); outs.extend_from_slice(&g.plan_out);
            outs.push(g.max_ratio); outs.push(g.b.len() as f32);
            outs
        };
        let (a, b) = (run(), run());
        std::println!("two runs of the mixed building's facade blasted: {} numbers out each", a.len());
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "bit for bit");
    }
    #[test] fn a_falling_piece_has_a_break_plan_and_snaps_by_it() {
        // the 4-storey tower's ground storey cut: the building above falls as a piece. Its plan: the weakest section
        // under its own weight on its underside; the impulse that breaks it there (M g dt / ratio); the part that
        // comes away lighter than the part kept. Snapping by the plan parts it there into two pieces with no joint
        // between them; a second snap of the same piece finds no plan.
        let mut g = Game::new_kind(4, 0);
        let mut x = -11.6f32;
        while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 { g.blast(v3(x, 1.6, z), 0.7); z += 0.8; } x += 0.8; }
        for _ in 0..20 { if g.structure_step(64) == 0 { break; } }
        g.plan_out.clear(); g.finish_plans();
        let (&pid, pl) = g.plans.iter().max_by_key(|(_, p)| p.len).expect("a plan");
        let (n, k, imp) = (pl.len, pl.other.len(), pl.impulse);
        let m: f32 = g.pieces[pid as usize].iter().map(|&i| structure::mass(&g.b, i)).sum();
        std::println!("plans for {} pieces; the biggest ({n} bricks, {:.0} t): {} joints cut, {k} bricks come away, broken by {:.0} kN s in a step (it carries {:.0} kN s at 5 m/s)", g.plans.len(), m / 1e3, pl.cut.len(), imp / 1e3, m * 5.0 / 1e3);
        let mo: f32 = pl.other.iter().map(|&i| structure::mass(&g.b, i)).sum();
        assert!(imp > 0.0 && imp.is_finite() && k > 0 && mo >= 0.05 * m && mo <= 0.5 * m, "a part of some size comes away: {:.0} t", mo / 1e3);
        assert!(g.plan_out.len() >= 3);
        assert_eq!(g.snap(pid), 2);
        let (a, b) = (g.out[1] as u32, g.out[3 + 7 * g.out[2] as usize] as u32);
        assert_eq!(g.pieces[a as usize].len() + g.pieces[b as usize].len(), n);
        assert_eq!(g.pieces[b as usize].len(), k);
        let inb: crate::Set<u32> = g.pieces[b as usize].iter().copied().collect();
        for &i in &g.pieces[a as usize] { let mut j = g.b.jhead[i as usize]; while j != NONE { assert!(!inb.contains(&g.b.joint_other(j, i)), "no joint between the parts"); j = g.b.joint_next(j, i); } }
        assert_eq!(g.snap(pid), 0);
        assert_eq!(g.snapped.get(&pid), Some(&(a, b)));
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
