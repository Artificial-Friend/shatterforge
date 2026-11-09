//! First-person walker against the brick store: a 0.6 x 1.8 m box (the capsule of SPEC.md, squared) with
//! gravity, a 0.35 m step-up, and a fly mode for inspecting.
use crate::brick::{Bricks, TICK};
use crate::math::*;
use alloc::vec::Vec;

pub const RADIUS: f32 = 0.3;
pub const HEIGHT: f32 = 1.8;
pub const STEP_UP: f32 = 0.35;

/// A convex obstacle (m): half-spaces n.x <= d, and its bounding box. Debris on the GPU (the page passes the pieces
/// near the player each frame, as oriented boxes from its latest read-back) and the store's free pieces.
#[derive(Clone, Debug)]
pub struct Obstacle { pub planes: Vec<([f32; 3], f32)>, pub lo: [f32; 3], pub hi: [f32; 3] }
impl Obstacle {
    /// an oriented box: centre, rotation (x y z w), half size
    pub fn obb(c: [f32; 3], q: [f32; 4], h: [f32; 3]) -> Obstacle {
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let r = [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                 [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                 [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]];
        let mut planes = Vec::with_capacity(6);
        let (mut lo, mut hi) = ([0.0f32; 3], [0.0f32; 3]);
        for k in 0..3 {
            let a = [r[0][k], r[1][k], r[2][k]];
            let d = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
            planes.push((a, d + h[k])); planes.push(([-a[0], -a[1], -a[2]], -d + h[k]));
        }
        for i in 0..3 { let e = fabs(r[i][0]) * h[0] + fabs(r[i][1]) * h[1] + fabs(r[i][2]) * h[2]; lo[i] = c[i] - e; hi[i] = c[i] + e; }
        Obstacle { planes, lo, hi }
    }
    /// a convex polytope from its corners and faces (outward normals from the corners)
    pub fn convex(v: &[[f64; 3]], faces: &[([f64; 3], Vec<[f64; 3]>)]) -> Obstacle {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in v { for k in 0..3 { lo[k] = lo[k].min(p[k] as f32); hi[k] = hi[k].max(p[k] as f32); } }
        let planes = faces.iter().map(|(n, poly)| { let d = n[0] * poly[0][0] + n[1] * poly[0][1] + n[2] * poly[0][2]; ([n[0] as f32, n[1] as f32, n[2] as f32], d as f32) }).collect();
        Obstacle { planes, lo, hi }
    }
    /// Does it overlap the box lo..hi? Separating axes: the box's three and the obstacle's face normals (edge pairs
    /// left out: a touch more contact near edges, never less).
    fn overlaps(&self, lo: [f32; 3], hi: [f32; 3]) -> bool {
        if (0..3).any(|k| self.hi[k] <= lo[k] || self.lo[k] >= hi[k]) { return false; }
        for (n, d) in &self.planes {
            // the box's corner farthest against n: if even it is outside the half-space, separated
            let m = (0..3).map(|k| n[k] * if n[k] >= 0.0 { lo[k] } else { hi[k] }).sum::<f32>();
            if m >= *d { return false; }
        }
        true
    }
    /// The top of the obstacle over the box's footprint (its centre and corners: the highest point of the obstacle
    /// above any of them, lines through it straight up); its bounding box's top where no such line crosses it.
    fn top(&self, lo: [f32; 3], hi: [f32; 3]) -> f32 {
        let mut best = f32::MIN;
        let cx = 0.5 * (lo[0] + hi[0]); let cz = 0.5 * (lo[2] + hi[2]);
        for (x, z) in [(cx, cz), (lo[0], lo[2]), (lo[0], hi[2]), (hi[0], lo[2]), (hi[0], hi[2])] {
            let (mut y0, mut y1) = (f32::MIN, f32::MAX);
            let mut ok = true;
            for (n, d) in &self.planes {
                let r = d - n[0] * x - n[2] * z;     // n.y y <= r
                if fabs(n[1]) < 1e-6 { if r < 0.0 { ok = false; break; } continue; }
                let yv = r / n[1];
                if n[1] > 0.0 { y1 = y1.min(yv); } else { y0 = y0.max(yv); }
            }
            if ok && y0 <= y1 { best = best.max(y1); }
        }
        if best == f32::MIN { self.hi[1] } else { best }
    }
}

pub struct Player { pub p: V3, pub v: V3, pub on_ground: bool, scratch: Vec<u32>, pub extra: Vec<Obstacle> }

impl Player {
    pub fn new(p: V3) -> Self { Player { p, v: V3::ZERO, on_ground: false, scratch: Vec::new(), extra: Vec::new() } }

    /// Highest top (m) of a live brick or obstacle overlapping the player box at feet position p, if any.
    /// (Bricks are their boxes, free pieces their faces, the debris the page passed in its oriented boxes; one the
    /// player stands inside deeper than a step, debris that moved into the player, is passed through, not a trap.)
    fn hit(&mut self, b: &Bricks, p: V3) -> Option<f32> {
        let e = 1e-4;
        let lo = [p.x - RADIUS + e, p.y + e, p.z - RADIUS + e];
        let hi = [p.x + RADIUS - e, p.y + HEIGHT - e, p.z + RADIUS - e];
        let t = |x: f32| floor(x / TICK) as i32;
        self.scratch.clear();
        b.query([t(lo[0]), t(lo[1]), t(lo[2])], [t(hi[0]) + 1, t(hi[1]) + 1, t(hi[2]) + 1], false, &mut self.scratch);
        let mut top: Option<f32> = None;
        for &i in &self.scratch {
            if let Some(pc) = b.piece(i) {
                let o = Obstacle::convex(&pc.world(), &pc.world_faces());
                if o.overlaps(lo, hi) { let t = o.top(lo, hi); top = Some(top.map_or(t, |v: f32| v.max(t))); }
                continue;
            }
            let tb = b.tbox(i);
            let (mn, mx) = ([tb.min[0] as f32 * TICK, tb.min[1] as f32 * TICK, tb.min[2] as f32 * TICK], [tb.max[0] as f32 * TICK, tb.max[1] as f32 * TICK, tb.max[2] as f32 * TICK]);
            if (0..3).all(|k| mn[k] < hi[k] && mx[k] > lo[k]) { top = Some(top.map_or(mx[1], |v: f32| v.max(mx[1]))); }
        }
        for o in &self.extra {
            if !o.overlaps(lo, hi) { continue; }
            let t = o.top(lo, hi);
            if t - self.p.y > STEP_UP + 0.25 && o.overlaps([self.p.x - RADIUS + e, self.p.y + e, self.p.z - RADIUS + e], [self.p.x + RADIUS - e, self.p.y + HEIGHT - e, self.p.z + RADIUS - e]) { continue; }
            top = Some(top.map_or(t, |v: f32| v.max(t)));
        }
        if p.y < 0.0 { top = Some(top.map_or(0.0, |v: f32| v.max(0.0))); }
        top
    }

    /// Advance by dt. `wish` is the desired velocity (m/s); horizontal when walking, free when flying.
    pub fn step(&mut self, b: &Bricks, dt: f32, wish: V3, jump: bool, fly: bool) {
        if fly { self.p += wish * dt; self.v = V3::ZERO; self.on_ground = false; return; }
        self.v.x = wish.x; self.v.z = wish.z;
        if self.on_ground && jump { self.v.y = 4.2; self.on_ground = false; }
        self.v.y -= 9.81 * dt;
        // horizontal: in small moves, with a step-up over low obstacles (stairs)
        for ax in [0usize, 2] {
            let total = self.v.get(ax) * dt;
            let n = (fabs(total) / 0.04) as i32 + 1;
            for _ in 0..n {
                let mut q = self.p; q.set(ax, q.get(ax) + total / n as f32);
                match self.hit(b, q) {
                    None => self.p = q,
                    Some(top) => {
                        let rise = top - self.p.y;
                        if self.on_ground && rise > 0.0 && rise <= STEP_UP {
                            let up = v3(q.x, top + 0.001, q.z);
                            if self.hit(b, up).is_none() { self.p = up; continue; }
                        }
                        self.v.set(ax, 0.0);
                        break;
                    }
                }
            }
        }
        // vertical
        let total = self.v.y * dt;
        let n = (fabs(total) / 0.04) as i32 + 1;
        self.on_ground = false;
        for _ in 0..n {
            let q = v3(self.p.x, self.p.y + total / n as f32, self.p.z);
            match self.hit(b, q) {
                None => self.p = q,
                Some(top) => {
                    if self.v.y <= 0.0 { self.p.y = top.max(self.p.y.min(top)); self.on_ground = true; }
                    self.v.y = 0.0;
                    break;
                }
            }
        }
        if !self.on_ground {
            // standing exactly on something counts as ground
            if self.hit(b, v3(self.p.x, self.p.y - 0.01, self.p.z)).is_some() && self.v.y <= 0.0 { self.on_ground = true; self.v.y = 0.0; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower;
    extern crate std;

    #[test] fn climbs_the_stair_to_the_next_floor() {
        let b = tower::build(2, 1);
        // into the core through the corridor door, then up flight A (north), across the landing, down flight B
        // direction (south) to the next floor
        let mut pl = Player::new(v3(1.44, 0.26, -4.6));
        let path: [(V3, f32); 4] = [(v3(0.0, 0.0, 1.2), 3.3), (v3(1.2, 0.0, 0.0), 1.3), (v3(0.0, 0.0, -1.2), 3.0), (v3(0.0, 0.0, -1.2), 1.0)];
        let mut ys = std::vec::Vec::new();
        for (dir, secs) in path {
            for _ in 0..(secs * 60.0) as i32 { pl.step(&b, 1.0 / 60.0, dir, false, false); }
            ys.push(pl.p.y);
        }
        std::println!("stair walk: feet heights after each leg {ys:?}, at {:?}", pl.p);
        assert!((pl.p.y - (tower::STOREY + tower::SLAB_T) as f32 * TICK).abs() < 0.05, "reached the floor above: {}", pl.p.y);
    }

    #[test] fn walls_stop_the_player() {
        let b = tower::build(1, 1);
        let mut pl = Player::new(v3(0.0, 0.26, -9.0));   // in the lobby, heading south into the facade
        for _ in 0..240 { pl.step(&b, 1.0 / 60.0, v3(-1.5, 0.0, -1.5), false, false); }
        assert!(pl.p.z > -11.25 + 0.25, "inside the wall line: {:?}", pl.p);
        assert!(pl.on_ground);
    }

    // ---- debris and free pieces (the playground: "if I interact with anything it starts to ignore me and I'm
    // falling through floors": a floor broken loose is a body on the GPU, and the player saw only the store's
    // standing bricks)
    #[test] fn stands_on_debris_the_page_passes() {
        // a 2 x 0.5 x 2 m slab lying 1 m up (a broken floor on the GPU); the player dropped above it lands on it,
        // at 1.25 m; without it the player falls to the ground
        let b = Bricks::new();
        let mut pl = Player::new(v3(0.0, 3.0, 0.0));
        pl.extra.push(Obstacle::obb([0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.25, 1.0]));
        for _ in 0..120 { pl.step(&b, 1.0 / 60.0, V3::ZERO, false, false); }
        std::println!("on a debris slab: feet at {:.3} m", pl.p.y);
        assert!((pl.p.y - 1.25).abs() < 0.01 && pl.on_ground);
        let mut free = Player::new(v3(0.0, 3.0, 0.0));
        for _ in 0..120 { free.step(&b, 1.0 / 60.0, V3::ZERO, false, false); }
        assert!(free.p.y < 0.01);
    }
    #[test] fn stands_on_a_tilted_slab_at_its_surface() {
        // a slab tilted 20 degrees about z: the player stands on its surface over its footprint, not on its
        // bounding box's top (0.70 m higher at the slab's middle)
        let b = Bricks::new();
        let a = 20f32.to_radians();
        let q = [0.0, 0.0, crate::math::sin(a / 2.0), crate::math::cos(a / 2.0)];
        let ob = Obstacle::obb([0.0, 1.0, 0.0], q, [2.0, 0.1, 1.0]);
        let boxtop = ob.hi[1];
        let mut pl = Player::new(v3(0.0, 3.0, 0.0));
        pl.extra.push(ob);
        for _ in 0..120 { pl.step(&b, 1.0 / 60.0, V3::ZERO, false, false); }
        // the surface: 1 + 0.1 / cos 20 at the middle, rising by tan 20 x 0.3 at the footprint's uphill edge
        let want = 1.0 + 0.1 / crate::math::cos(a) + 0.3 * crate::math::sin(a) / crate::math::cos(a);
        std::println!("on a tilted slab: feet at {:.3} m (surface under the footprint {want:.3}, box top {boxtop:.3})", pl.p.y);
        assert!((pl.p.y - want).abs() < 0.02);
    }
    #[test] fn debris_moved_into_the_player_does_not_trap_it() {
        // a block 1 m tall has moved into the player (the page's read-back shows it overlapping): the player walks
        // out of it, it does not stand stuck
        let b = Bricks::new();
        let mut pl = Player::new(v3(0.0, 0.0, 0.0));
        pl.extra.push(Obstacle::obb([0.2, 0.75, 0.0], [0.0, 0.0, 0.0, 1.0], [0.5, 0.75, 0.5]));
        for _ in 0..60 { pl.step(&b, 1.0 / 60.0, v3(-2.0, 0.0, 0.0), false, false); }
        assert!(pl.p.x < -1.0, "walked out: {:?}", pl.p);
    }
    #[test] fn walks_under_an_inclined_free_piece() {
        // a rafter 6 m long rising at 30 degrees from the ground: its bounding box covers 5.2 m x 3 m of air; the
        // player walks under it at x = 4 m (its underside 2.0 m up and more over the player's footprint), where its box
        // stopped the player
        let mut b = Bricks::new();
        let (c, sn) = (crate::math::cos(0.5236) as f64, crate::math::sin(0.5236) as f64);
        let mut pts = std::vec::Vec::new();
        for &o in &[-0.1f64, 0.1] { for &z in &[-0.1f64, 0.1] { let p0 = [o * -sn, o * c, z]; pts.push(p0); pts.push([p0[0] + 6.0 * c, p0[1] + 6.0 * sn, z]); } }
        let n = pts.len() as f64; let cen = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
        let cs = [(cen[0] * 1000.0).round() / 1000.0, (cen[1] * 1000.0).round() / 1000.0, (cen[2] * 1000.0).round() / 1000.0];
        let local: std::vec::Vec<[f64; 3]> = pts.iter().map(|p| [p[0] - cs[0], p[1] - cs[1], p[2] - cs[2]]).collect();
        b.add_piece(cs, [0.0, 0.0, 0.0, 1.0], &local, crate::brick::mat::TIMBER, 0, 1).unwrap();
        let mut pl = Player::new(v3(4.0, 0.0, -2.0));
        for _ in 0..120 { pl.step(&b, 1.0 / 60.0, v3(0.0, 0.0, 2.0), false, false); }
        assert!(pl.p.z > 1.5 && pl.p.y < 0.01, "walked under it: {:?}", pl.p);
    }
}

