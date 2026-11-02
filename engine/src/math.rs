//! Small, explicit vector / quaternion / 3x3 maths. f32 throughout, laid out for SIMD-friendly loops.
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 { pub x: f32, pub y: f32, pub z: f32 }

#[inline] pub const fn v3(x: f32, y: f32, z: f32) -> V3 { V3 { x, y, z } }

impl V3 {
    pub const ZERO: V3 = v3(0.0, 0.0, 0.0);
    #[inline] pub fn dot(self, o: V3) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    #[inline] pub fn cross(self, o: V3) -> V3 { v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x) }
    #[inline] pub fn len2(self) -> f32 { self.dot(self) }
    #[inline] pub fn len(self) -> f32 { sqrt(self.len2()) }
    #[inline] pub fn norm(self) -> V3 { let l = self.len(); if l > 1e-12 { self * (1.0 / l) } else { V3::ZERO } }
    #[inline] pub fn get(self, i: usize) -> f32 { match i { 0 => self.x, 1 => self.y, _ => self.z } }
    #[inline] pub fn set(&mut self, i: usize, v: f32) { match i { 0 => self.x = v, 1 => self.y = v, _ => self.z = v } }
    #[inline] pub fn abs(self) -> V3 { v3(fabs(self.x), fabs(self.y), fabs(self.z)) }
    #[inline] pub fn min(self, o: V3) -> V3 { v3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z)) }
    #[inline] pub fn max(self, o: V3) -> V3 { v3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z)) }
}
impl Add for V3 { type Output = V3; #[inline] fn add(self, o: V3) -> V3 { v3(self.x + o.x, self.y + o.y, self.z + o.z) } }
impl Sub for V3 { type Output = V3; #[inline] fn sub(self, o: V3) -> V3 { v3(self.x - o.x, self.y - o.y, self.z - o.z) } }
impl Mul<f32> for V3 { type Output = V3; #[inline] fn mul(self, s: f32) -> V3 { v3(self.x * s, self.y * s, self.z * s) } }
impl Neg for V3 { type Output = V3; #[inline] fn neg(self) -> V3 { v3(-self.x, -self.y, -self.z) } }
impl AddAssign for V3 { #[inline] fn add_assign(&mut self, o: V3) { *self = *self + o; } }
impl SubAssign for V3 { #[inline] fn sub_assign(&mut self, o: V3) { *self = *self - o; } }

/// Rotation as a unit quaternion (x, y, z, w).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl Default for Quat { fn default() -> Self { Quat::IDENTITY } }
impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub fn from_axis_angle(a: V3, ang: f32) -> Quat { let a = a.norm(); let (s, c) = (sin(ang * 0.5), cos(ang * 0.5)); Quat { x: a.x * s, y: a.y * s, z: a.z * s, w: c } }
    #[inline] pub fn mul(self, o: Quat) -> Quat {
        Quat { w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
               x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
               y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
               z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w }
    }
    #[inline] pub fn normalized(self) -> Quat { let l = sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w); let k = if l > 0.0 { 1.0 / l } else { 1.0 }; Quat { x: self.x * k, y: self.y * k, z: self.z * k, w: self.w * k } }
    /// q advanced by angular velocity w over time h (first order, renormalised)
    #[inline] pub fn integrate(self, w: V3, h: f32) -> Quat {
        let dq = Quat { x: w.x, y: w.y, z: w.z, w: 0.0 }.mul(self);
        Quat { x: self.x + 0.5 * h * dq.x, y: self.y + 0.5 * h * dq.y, z: self.z + 0.5 * h * dq.z, w: self.w + 0.5 * h * dq.w }.normalized()
    }
    #[inline] pub fn conj(self) -> Quat { Quat { x: -self.x, y: -self.y, z: -self.z, w: self.w } }
    #[inline] pub fn rotate(self, v: V3) -> V3 {
        let u = v3(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }
    pub fn to_m3(self) -> M3 {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        M3 { c: [v3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
                 v3(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
                 v3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))] }
    }
}

/// 3x3 matrix stored as columns; for a rotation the columns are the body's axes in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct M3 { pub c: [V3; 3] }
impl M3 {
    pub const IDENTITY: M3 = M3 { c: [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)] };
    #[inline] pub fn mulv(&self, v: V3) -> V3 { self.c[0] * v.x + self.c[1] * v.y + self.c[2] * v.z }
    /// transpose times v (world -> local for a rotation)
    #[inline] pub fn tmulv(&self, v: V3) -> V3 { v3(self.c[0].dot(v), self.c[1].dot(v), self.c[2].dot(v)) }
    /// R diag(d) R^T for a rotation R (world inverse inertia from the local diagonal)
    #[inline] pub fn sandwich(r: &M3, d: V3) -> M3 {
        let (a, b, c) = (r.c[0] * d.x, r.c[1] * d.y, r.c[2] * d.z);
        let col = |j: usize| a * r.c[0].get(j) + b * r.c[1].get(j) + c * r.c[2].get(j);
        M3 { c: [col(0), col(1), col(2)] }
    }
    pub fn diag(d: V3) -> M3 { M3 { c: [v3(d.x, 0.0, 0.0), v3(0.0, d.y, 0.0), v3(0.0, 0.0, d.z)] } }
}

// Float helpers that work without std: wasm has a sqrt instruction; native builds link std.
#[inline] pub fn sqrt(x: f32) -> f32 {
    #[cfg(target_arch = "wasm32")] { use core::arch::wasm32::*; f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x))) }
    #[cfg(not(target_arch = "wasm32"))] { if x <= 0.0 { 0.0 } else { x.sqrt() } }
}
#[inline] pub fn fabs(x: f32) -> f32 { f32::from_bits(x.to_bits() & 0x7fff_ffff) }
pub fn sin(x: f32) -> f32 {
    // range-reduce to [-pi, pi], then a 7th-order odd polynomial on [-pi/2, pi/2]
    let tau = core::f32::consts::TAU; let pi = core::f32::consts::PI;
    let mut a = x - tau * floor(x / tau + 0.5);
    if a > pi * 0.5 { a = pi - a } else if a < -pi * 0.5 { a = -pi - a }
    let a2 = a * a;
    a * (1.0 - a2 / 6.0 * (1.0 - a2 / 20.0 * (1.0 - a2 / 42.0 * (1.0 - a2 / 72.0))))
}
#[inline] pub fn cos(x: f32) -> f32 { sin(x + core::f32::consts::FRAC_PI_2) }
#[inline] pub fn floor(x: f32) -> f32 { let t = x as i32 as f32; if t > x { t - 1.0 } else { t } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn trig_and_sqrt() {
        for i in -50..50 { let x = i as f32 * 0.37; assert!((sin(x) - x.sin()).abs() < 2e-5, "sin {x}"); assert!((cos(x) - x.cos()).abs() < 2e-5); }
        for i in 1..1000 { let x = i as f32 * 0.731; assert!((sqrt(x) - x.sqrt()).abs() < 1e-4 * x.sqrt().max(1.0)); }
    }
    #[test] fn quat_rotates() {
        let q = Quat::from_axis_angle(v3(0.0, 1.0, 0.0), core::f32::consts::FRAC_PI_2);
        let r = q.to_m3().mulv(v3(1.0, 0.0, 0.0));
        assert!((r - v3(0.0, 0.0, -1.0)).len() < 1e-5, "{r:?}");
    }
}
