//! First-person walker against the brick store: a 0.6 x 1.8 m box (the capsule of SPEC.md, squared) with
//! gravity, a 0.35 m step-up, and a fly mode for inspecting.
use crate::brick::{Bricks, TICK};
use crate::math::*;
use alloc::vec::Vec;

pub const RADIUS: f32 = 0.3;
pub const HEIGHT: f32 = 1.8;
pub const STEP_UP: f32 = 0.35;

pub struct Player { pub p: V3, pub v: V3, pub on_ground: bool, scratch: Vec<u32> }

impl Player {
    pub fn new(p: V3) -> Self { Player { p, v: V3::ZERO, on_ground: false, scratch: Vec::new() } }

    /// Highest top (m) of a live brick overlapping the player box at feet position p, if any.
    fn hit(&mut self, b: &Bricks, p: V3) -> Option<f32> {
        let e = 1e-4;
        let lo = [p.x - RADIUS + e, p.y + e, p.z - RADIUS + e];
        let hi = [p.x + RADIUS - e, p.y + HEIGHT - e, p.z + RADIUS - e];
        let t = |x: f32| floor(x / TICK) as i32;
        self.scratch.clear();
        b.query([t(lo[0]), t(lo[1]), t(lo[2])], [t(hi[0]) + 1, t(hi[1]) + 1, t(hi[2]) + 1], false, &mut self.scratch);
        let mut top: Option<f32> = None;
        for &i in &self.scratch {
            let tb = b.tbox(i);
            let (mn, mx) = ([tb.min[0] as f32 * TICK, tb.min[1] as f32 * TICK, tb.min[2] as f32 * TICK], [tb.max[0] as f32 * TICK, tb.max[1] as f32 * TICK, tb.max[2] as f32 * TICK]);
            if (0..3).all(|k| mn[k] < hi[k] && mx[k] > lo[k]) { top = Some(top.map_or(mx[1], |v: f32| v.max(mx[1]))); }
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
}
