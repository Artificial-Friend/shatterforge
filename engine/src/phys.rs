//! Box rigid-body engine: oriented boxes, a hashed-grid broadphase, box-box SAT with face clipping,
//! speculative contacts, and a soft-step solver (sub-stepping, warm start each sub-step, soft contacts,
//! relax pass) in the style of Box2D v3, extended to 3D. Islands sleep when they come to rest.
use crate::math::*;
use alloc::vec::Vec;
use hashbrown::HashMap;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub p: V3, pub q: Quat, pub v: V3, pub w: V3,
    /// half extents (m)
    pub h: V3,
    pub inv_m: f32,
    /// inverse inertia, local diagonal
    pub inv_i: V3,
    pub awake: bool,
    pub sleep_t: f32,
    // per-step scratch
    rot: M3, aabb_min: V3, aabb_max: V3,
}

#[derive(Clone, Copy, Default)]
struct Cp {
    // anchors relative to the body centres (fixed over the step), their moment arms about the normal,
    // and those arms pre-multiplied by the inverse inertia
    ra: V3, rb: V3, la: V3,
    raxn: V3, rbxn: V3, iaxn: V3, ibxn: V3,
    adj: f32, nm: f32, jn: f32,
}

#[derive(Clone, Copy)]
struct Manifold {
    a: u32, b: u32,
    /// solver indices (NONE = static: ground or asleep)
    sa: u32, sb: u32,
    n: V3, t1: V3, t2: V3, mu: f32,
    cnt: u32,
    cp: [Cp; 4],
    // patch friction: two tangents and twist at a friction anchor. The anchor is a point fixed on each body
    // while the patch sticks, so static friction holds position instead of letting it creep.
    fa: V3, fb: V3, la_f: V3, lb_f: V3, e0: V3,
    tm1: f32, tm2: f32, twm: f32, rad: f32,
    jt1: f32, jt2: f32, jtw: f32,
    stick: bool,
    // friction Jacobians: arms crossed with the tangents, and inverse-inertia products (also for twist)
    ca1: V3, ca2: V3, cb1: V3, cb2: V3, ia1: V3, ia2: V3, ib1: V3, ib2: V3, ian: V3, ibn: V3,
    ima: f32, imb: f32,
}

/// Solver body state, struct-of-arrays. Index 0 is a static dummy (ground, sleeping or fixed bodies).
#[derive(Default)]
struct Solver {
    v: Vec<V3>, w: Vec<V3>,
    /// displacement and (linearised) rotation since the start of the step
    dp: Vec<V3>, th: Vec<V3>,
    im: Vec<f32>, ii: Vec<M3>, q: Vec<Quat>, p0: Vec<V3>, body: Vec<u32>,
}
impl Solver {
    fn clear(&mut self) {
        self.v.clear(); self.w.clear(); self.dp.clear(); self.th.clear(); self.im.clear(); self.ii.clear(); self.q.clear(); self.p0.clear(); self.body.clear();
        self.push(V3::ZERO, V3::ZERO, 0.0, M3::diag(V3::ZERO), Quat::IDENTITY, V3::ZERO, NONE);
    }
    fn push(&mut self, v: V3, w: V3, im: f32, ii: M3, q: Quat, p0: V3, body: u32) -> u32 {
        self.v.push(v); self.w.push(w); self.dp.push(V3::ZERO); self.th.push(V3::ZERO); self.im.push(im); self.ii.push(ii);
        self.q.push(q); self.p0.push(p0); self.body.push(body);
        (self.v.len() - 1) as u32
    }
}

pub struct Params {
    pub gravity: V3,
    pub substeps: u32,
    /// contact stiffness; 0 = a sixth of the sub-step rate (a quarter is the soft-step stability limit, but
    /// stacks keep swaying near it).
    /// Tall stacks need it: soft contacts sag by load / (effective mass x omega^2), and a 20-high stack on
    /// 30 Hz contacts is a genuinely unstable inverted pendulum on springs.
    pub contact_hertz: f32,
    pub damping_ratio: f32,
    pub max_push: f32,
    pub friction: f32,
    pub speculative: f32,
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_time: f32,
    pub allow_sleep: bool,
    pub cell: f32,
    /// keep the soft-spring impulse leak of the soft-step formulation (true = Box2D v3 exactly; false =
    /// soft push-out without the leak, so loaded contacts do not sag in proportion to the load above them)
    pub leak: f32,
    pub anchor: bool,
}
impl Default for Params {
    fn default() -> Self {
        Params { gravity: v3(0.0, -9.81, 0.0), substeps: 12, contact_hertz: 0.0, damping_ratio: 10.0, max_push: 3.0,
                 friction: 0.6, speculative: 0.02, sleep_lin: 0.05, sleep_ang: 0.08, sleep_time: 0.5, allow_sleep: true, cell: 1.0, leak: 1.0, anchor: true }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats { pub awake: u32, pub pairs: u32, pub manifolds: u32, pub points: u32,
                   pub t_broad: f64, pub t_narrow: f64, pub t_solve: f64 }

pub struct World {
    pub bodies: Vec<Body>,
    pub prm: Params,
    pub stats: Stats,
    /// host clock in ms for profiling (set by the host; returns 0 when absent)
    pub clock: fn() -> f64,
    man: Vec<Manifold>,
    old: Vec<Manifold>,
    old_map: HashMap<u64, u32>,
    sb: Solver,
    sidx: Vec<u32>,
    // broadphase scratch
    bp_cnt: Vec<u32>, bp_ent: Vec<(u32, [i32; 3])>, bp_tmp: Vec<(u32, u32, [i32; 3])>,
    uf: Vec<u32>,
}

fn no_clock() -> f64 { 0.0 }
#[inline] fn key(a: u32, b: u32) -> u64 { ((a as u64) << 32) | b as u64 }

impl World {
    pub fn new() -> Self {
        World { bodies: Vec::new(), prm: Params::default(), stats: Stats::default(), clock: no_clock,
                man: Vec::new(), old: Vec::new(), old_map: HashMap::new(), sb: Solver::default(), sidx: Vec::new(),
                bp_cnt: Vec::new(), bp_ent: Vec::new(), bp_tmp: Vec::new(), uf: Vec::new() }
    }

    /// Add a box: centre, rotation, half extents (m), density (kg/m^3). Density 0 makes it static.
    pub fn add_box(&mut self, p: V3, q: Quat, h: V3, density: f32) -> u32 {
        let m = density * 8.0 * h.x * h.y * h.z;
        let (inv_m, inv_i) = if m > 0.0 {
            let (x2, y2, z2) = (4.0 * h.x * h.x, 4.0 * h.y * h.y, 4.0 * h.z * h.z);
            (1.0 / m, v3(12.0 / (m * (y2 + z2)), 12.0 / (m * (x2 + z2)), 12.0 / (m * (x2 + y2))))
        } else { (0.0, V3::ZERO) };
        self.bodies.push(Body { p, q, v: V3::ZERO, w: V3::ZERO, h, inv_m, inv_i, awake: m > 0.0, sleep_t: 0.0,
                                rot: q.to_m3(), aabb_min: p, aabb_max: p });
        (self.bodies.len() - 1) as u32
    }

    pub fn awake_count(&self) -> usize { self.bodies.iter().filter(|b| b.awake).count() }

    pub fn step(&mut self, dt: f32) {
        let t0 = (self.clock)();
        let n = self.bodies.len();
        // ---- prepare bodies, AABBs, solver bodies for awake ones
        self.sidx.clear(); self.sidx.resize(n, 0);
        self.sb.clear();
        for i in 0..n {
            let b = &mut self.bodies[i];
            b.rot = b.q.to_m3();
            let e = v3(
                fabs(b.rot.c[0].x) * b.h.x + fabs(b.rot.c[1].x) * b.h.y + fabs(b.rot.c[2].x) * b.h.z,
                fabs(b.rot.c[0].y) * b.h.x + fabs(b.rot.c[1].y) * b.h.y + fabs(b.rot.c[2].y) * b.h.z,
                fabs(b.rot.c[0].z) * b.h.x + fabs(b.rot.c[1].z) * b.h.y + fabs(b.rot.c[2].z) * b.h.z);
            // speculative margin: how far this body can move this step, plus a little
            let m = if b.awake { (b.v.len() + b.w.len() * b.h.len()) * dt + 0.5 * self.prm.speculative } else { 0.5 * self.prm.speculative };
            let e = e + v3(m, m, m);
            b.aabb_min = b.p - e; b.aabb_max = b.p + e;
            if b.awake {
                self.sidx[i] = self.sb.push(b.v, b.w, b.inv_m, M3::sandwich(&b.rot, b.inv_i), b.q, b.p, i as u32);
            }
        }
        // ---- broadphase + narrowphase
        core::mem::swap(&mut self.man, &mut self.old);
        self.man.clear();
        self.stats.pairs = 0;
        self.broadphase_and_collide(dt);
        let t1 = (self.clock)();
        // ---- islands and sleep bookkeeping happen after solving; prepare constraint masses now
        let sub = self.prm.substeps.max(1);
        let h = dt / sub as f32;
        let inv_h = 1.0 / h;
        let hz = if self.prm.contact_hertz > 0.0 { self.prm.contact_hertz } else { inv_h / 6.0 };
        let omega = 2.0 * core::f32::consts::PI * hz;
        let a1 = 2.0 * self.prm.damping_ratio + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        let (bias_rate, mass_scale, imp_scale) = (omega / a1, a2 * a3, a3 * self.prm.leak);
        let mut pts = 0;
        for m in self.man.iter_mut() {
            let (ima, iia) = (self.sb.im[m.sa as usize], self.sb.ii[m.sa as usize]);
            let (imb, iib) = (self.sb.im[m.sb as usize], self.sb.ii[m.sb as usize]);
            m.ima = ima; m.imb = imb;
            let n = m.n;
            for k in 0..m.cnt as usize {
                let c = &mut m.cp[k];
                c.raxn = c.ra.cross(n); c.rbxn = c.rb.cross(n);
                c.iaxn = iia.mulv(c.raxn); c.ibxn = iib.mulv(c.rbxn);
                let kk = ima + imb + c.raxn.dot(c.iaxn) + c.rbxn.dot(c.ibxn);
                c.nm = if kk > 0.0 { 1.0 / kk } else { 0.0 };
            }
            m.ca1 = m.fa.cross(m.t1); m.ca2 = m.fa.cross(m.t2); m.cb1 = m.fb.cross(m.t1); m.cb2 = m.fb.cross(m.t2);
            m.ia1 = iia.mulv(m.ca1); m.ia2 = iia.mulv(m.ca2); m.ib1 = iib.mulv(m.cb1); m.ib2 = iib.mulv(m.cb2);
            let k1 = ima + imb + m.ca1.dot(m.ia1) + m.cb1.dot(m.ib1);
            let k2 = ima + imb + m.ca2.dot(m.ia2) + m.cb2.dot(m.ib2);
            m.tm1 = if k1 > 0.0 { 1.0 / k1 } else { 0.0 }; m.tm2 = if k2 > 0.0 { 1.0 / k2 } else { 0.0 };
            m.ian = iia.mulv(n); m.ibn = iib.mulv(n);
            let kt = n.dot(m.ian) + n.dot(m.ibn);
            m.twm = if kt > 0.0 { 1.0 / kt } else { 0.0 };
            pts += m.cnt;
        }
        self.stats.manifolds = self.man.len() as u32; self.stats.points = pts;
        // ---- sub-steps
        let g = self.prm.gravity;
        let wmax = 0.25 * core::f32::consts::PI * inv_h;
        let nb = self.sb.v.len();
        for _ in 0..sub {
            for i in 1..nb {
                self.sb.v[i] += g * h;
                let w = self.sb.w[i]; let wl = w.len2(); if wl > wmax * wmax { self.sb.w[i] = w * (wmax / sqrt(wl)); }
            }
            self.warm_start();
            self.solve(true, inv_h, bias_rate, mass_scale, imp_scale);
            for i in 1..nb {
                let (v, w) = (self.sb.v[i], self.sb.w[i]);
                self.sb.dp[i] += v * h; self.sb.th[i] += w * h;
                self.sb.q[i] = self.sb.q[i].integrate(w, h);
            }
            self.solve(false, inv_h, bias_rate, mass_scale, imp_scale);
        }
        // ---- write back
        for i in 1..nb {
            let b = &mut self.bodies[self.sb.body[i] as usize];
            b.v = self.sb.v[i]; b.w = self.sb.w[i]; b.p = self.sb.p0[i] + self.sb.dp[i]; b.q = self.sb.q[i];
        }
        let t2 = (self.clock)();
        self.sleep_islands(dt);
        // ---- cache for warm start
        self.old_map.clear();
        for (k, m) in self.man.iter().enumerate() { self.old_map.insert(key(m.a, m.b), k as u32); }
        self.stats.awake = (nb - 1) as u32;
        self.stats.t_broad = t1 - t0; self.stats.t_solve = t2 - t1;
    }

    fn warm_start(&mut self) {
        let sb = &mut self.sb;
        for m in self.man.iter() {
            let (sa, sbi) = (m.sa as usize, m.sb as usize);
            let (mut va, mut wa, mut vb, mut wb) = (sb.v[sa], sb.w[sa], sb.v[sbi], sb.w[sbi]);
            let mut pn = 0.0;
            for k in 0..m.cnt as usize {
                let c = &m.cp[k];
                pn += c.jn; wa -= c.iaxn * c.jn; wb += c.ibxn * c.jn;
            }
            let p = m.n * pn + m.t1 * m.jt1 + m.t2 * m.jt2;
            va -= p * m.ima; vb += p * m.imb;
            wa -= m.ia1 * m.jt1 + m.ia2 * m.jt2 + m.ian * m.jtw;
            wb += m.ib1 * m.jt1 + m.ib2 * m.jt2 + m.ibn * m.jtw;
            sb.v[sa] = va; sb.w[sa] = wa; sb.v[sbi] = vb; sb.w[sbi] = wb;
            sb.v[0] = V3::ZERO; sb.w[0] = V3::ZERO;
        }
    }

    fn solve(&mut self, use_bias: bool, inv_h: f32, bias_rate: f32, mass_scale: f32, imp_scale: f32) {
        let max_push = self.prm.max_push;
        let sb = &mut self.sb;
        for m in self.man.iter_mut() {
            let (sa, sbi) = (m.sa as usize, m.sb as usize);
            let (mut va, mut wa, mut vb, mut wb) = (sb.v[sa], sb.w[sa], sb.v[sbi], sb.w[sbi]);
            let (tha, thb) = (sb.th[sa], sb.th[sbi]);
            let (n, t1, t2) = (m.n, m.t1, m.t2);
            let dpab = sb.dp[sbi] - sb.dp[sa];
            let dn = dpab.dot(n);
            let (ima, imb) = (m.ima, m.imb);
            let mut tot = 0.0;
            for k in 0..m.cnt as usize {
                let c = &mut m.cp[k];
                // current separation from the sub-step motion (rotation linearised over the step)
                let s = c.adj + dn + thb.dot(c.rbxn) - tha.dot(c.raxn);
                let (bias, ms, is) = if s > 0.0 { (s * inv_h, 1.0, 0.0) }
                    else if use_bias { ((bias_rate * s).max(-max_push), mass_scale, imp_scale) }
                    else { (0.0, 1.0, 0.0) };
                let vn = (vb - va).dot(n) + wb.dot(c.rbxn) - wa.dot(c.raxn);
                let imp = -c.nm * ms * (vn + bias) - is * c.jn;
                let nj = (c.jn + imp).max(0.0);
                let imp = nj - c.jn; c.jn = nj; tot += nj;
                va -= n * (imp * ima); wa -= c.iaxn * imp;
                vb += n * (imp * imb); wb += c.ibxn * imp;
            }
            // patch friction at the anchor, soft-anchored to where the patch first stuck
            let lim = m.mu * tot;
            let dv = vb - va;
            let v1 = dv.dot(t1) + wb.dot(m.cb1) - wa.dot(m.ca1);
            let v2 = dv.dot(t2) + wb.dot(m.cb2) - wa.dot(m.ca2);
            let anchored = use_bias && m.stick;
            let (b1, b2, ms, is) = if anchored {
                let e1 = m.e0.dot(t1) + dpab.dot(t1) + thb.dot(m.cb1) - tha.dot(m.ca1);
                let e2 = m.e0.dot(t2) + dpab.dot(t2) + thb.dot(m.cb2) - tha.dot(m.ca2);
                ((bias_rate * e1).max(-max_push).min(max_push), (bias_rate * e2).max(-max_push).min(max_push), mass_scale, imp_scale)
            } else { (0.0, 0.0, 1.0, 0.0) };
            let (o1, o2) = (m.jt1, m.jt2);
            let mut j1 = o1 - m.tm1 * ms * (v1 + b1) - is * o1;
            let mut j2 = o2 - m.tm2 * ms * (v2 + b2) - is * o2;
            let l2 = j1 * j1 + j2 * j2;
            if l2 > lim * lim { let k = lim / sqrt(l2); j1 *= k; j2 *= k; m.stick = false; }
            m.jt1 = j1; m.jt2 = j2;
            let (d1, d2) = (j1 - o1, j2 - o2);
            let p = t1 * d1 + t2 * d2;
            va -= p * ima; wa -= m.ia1 * d1 + m.ia2 * d2;
            vb += p * imb; wb += m.ib1 * d1 + m.ib2 * d2;
            // twist
            let wl = lim * m.rad;
            let ot = m.jtw;
            let jt = (ot - m.twm * (wb - wa).dot(n)).max(-wl).min(wl);
            m.jtw = jt;
            let dt = jt - ot;
            wa -= m.ian * dt; wb += m.ibn * dt;
            sb.v[sa] = va; sb.w[sa] = wa; sb.v[sbi] = vb; sb.w[sbi] = wb;
            sb.v[0] = V3::ZERO; sb.w[0] = V3::ZERO;
        }
    }

    fn sleep_islands(&mut self, dt: f32) {
        let n = self.bodies.len();
        self.uf.clear(); self.uf.extend(0..n as u32);
        fn find(uf: &mut [u32], mut x: u32) -> u32 { while uf[x as usize] != x { let p = uf[uf[x as usize] as usize]; uf[x as usize] = p; x = p; } x }
        for m in self.man.iter() {
            if m.sa == 0 || m.sb == 0 { continue; }
            let (ra, rb) = (find(&mut self.uf, m.a), find(&mut self.uf, m.b));
            if ra != rb { self.uf[ra as usize] = rb; }
        }
        let (l2, a2) = (self.prm.sleep_lin * self.prm.sleep_lin, self.prm.sleep_ang * self.prm.sleep_ang);
        for b in self.bodies.iter_mut() {
            if !b.awake { continue; }
            if b.v.len2() < l2 && b.w.len2() < a2 { b.sleep_t += dt } else { b.sleep_t = 0.0 }
        }
        if !self.prm.allow_sleep { return; }
        // island minimum sleep time, stored on the root
        let mut min_t: Vec<f32> = alloc::vec![f32::MAX; n];
        for i in 0..n { if self.bodies[i].awake { let r = find(&mut self.uf, i as u32) as usize; min_t[r] = min_t[r].min(self.bodies[i].sleep_t); } }
        for i in 0..n {
            if !self.bodies[i].awake { continue; }
            let r = find(&mut self.uf, i as u32) as usize;
            if min_t[r] >= self.prm.sleep_time { let b = &mut self.bodies[i]; b.awake = false; b.v = V3::ZERO; b.w = V3::ZERO; }
        }
    }

    fn broadphase_and_collide(&mut self, dt: f32) {
        let _ = dt;
        let n = self.bodies.len();
        let inv_cell = 1.0 / self.prm.cell;
        let cellf = |x: f32| floor(x * inv_cell) as i32;
        // ground contacts
        for i in 0..n {
            if !self.bodies[i].awake { continue; }
            if self.bodies[i].aabb_min.y < 0.0 { self.collide_ground(i as u32); }
        }
        // count entries
        self.bp_tmp.clear();
        for i in 0..n {
            let b = &self.bodies[i];
            let (c0, c1) = ([cellf(b.aabb_min.x), cellf(b.aabb_min.y), cellf(b.aabb_min.z)], [cellf(b.aabb_max.x), cellf(b.aabb_max.y), cellf(b.aabb_max.z)]);
            for x in c0[0]..=c1[0] { for y in c0[1]..=c1[1] { for z in c0[2]..=c1[2] {
                let h = (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)) as u32;
                self.bp_tmp.push((h, i as u32, [x, y, z]));
            } } }
        }
        let ne = self.bp_tmp.len();
        let mut tbl = 1usize; while tbl < ne * 2 { tbl <<= 1; }
        let mask = (tbl - 1) as u32;
        self.bp_cnt.clear(); self.bp_cnt.resize(tbl + 1, 0);
        for e in self.bp_tmp.iter() { self.bp_cnt[((e.0 & mask) + 1) as usize] += 1; }
        for k in 0..tbl { self.bp_cnt[k + 1] += self.bp_cnt[k]; }
        self.bp_ent.clear(); self.bp_ent.resize(ne, (0, [0; 3]));
        {
            let mut fill: Vec<u32> = self.bp_cnt[..tbl].to_vec();
            for e in self.bp_tmp.iter() { let s = (e.0 & mask) as usize; self.bp_ent[fill[s] as usize] = (e.1, e.2); fill[s] += 1; }
        }
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for s in 0..tbl {
            let (lo, hi) = (self.bp_cnt[s] as usize, self.bp_cnt[s + 1] as usize);
            if hi - lo < 2 { continue; }
            for x in lo..hi { for y in x + 1..hi {
                let (ia, ca) = self.bp_ent[x]; let (ib, cb) = self.bp_ent[y];
                if ca != cb || ia == ib { continue; }
                let (a, b) = (&self.bodies[ia as usize], &self.bodies[ib as usize]);
                if !a.awake && !b.awake { continue; }
                if a.inv_m == 0.0 && b.inv_m == 0.0 { continue; }
                if a.aabb_min.x > b.aabb_max.x || b.aabb_min.x > a.aabb_max.x || a.aabb_min.y > b.aabb_max.y || b.aabb_min.y > a.aabb_max.y
                    || a.aabb_min.z > b.aabb_max.z || b.aabb_min.z > a.aabb_max.z { continue; }
                // report the pair only in the cell holding the max of the two min corners
                let m = a.aabb_min.max(b.aabb_min);
                if [cellf(m.x), cellf(m.y), cellf(m.z)] != ca { continue; }
                pairs.push(if ia < ib { (ia, ib) } else { (ib, ia) });
            } }
        }
        self.stats.pairs = pairs.len() as u32;
        let tn = (self.clock)();
        for (a, b) in pairs { self.collide_pair(a, b); }
        self.stats.t_narrow = (self.clock)() - tn;
    }

    fn wake(&mut self, i: u32) {
        let b = &mut self.bodies[i as usize];
        if b.awake || b.inv_m == 0.0 { return; }
        b.awake = true; b.sleep_t = 0.0;
        // it joins the solver from the next step; this step it is treated as static
    }

    fn collide_ground(&mut self, i: u32) {
        let b = self.bodies[i as usize];
        let mut pts: [(V3, f32); 8] = [(V3::ZERO, 0.0); 8];
        let mut cnt = 0;
        let margin = self.prm.speculative + (-b.v.y).max(0.0) * (1.0 / 60.0);
        for k in 0..8 {
            let s = v3(if k & 1 != 0 { 1.0 } else { -1.0 }, if k & 2 != 0 { 1.0 } else { -1.0 }, if k & 4 != 0 { 1.0 } else { -1.0 });
            let c = b.p + b.rot.mulv(v3(s.x * b.h.x, s.y * b.h.y, s.z * b.h.z));
            if c.y < margin { pts[cnt] = (v3(c.x, 0.5 * c.y, c.z), c.y); cnt += 1; }
        }
        if cnt == 0 { return; }
        let n = v3(0.0, -1.0, 0.0);
        let cnt = reduce4(&mut pts, cnt, n);
        self.push_manifold(i, NONE, n, &pts[..cnt]);
    }

    fn collide_pair(&mut self, ia: u32, ib: u32) {
        let (a, b) = (&self.bodies[ia as usize], &self.bodies[ib as usize]);
        let (a_awake, b_awake, a_still, b_still) = (a.awake, b.awake, a.sleep_t != 0.0, b.sleep_t != 0.0);
        // speculative margin for the pair
        let rel = (b.v - a.v).len() + a.w.len() * a.h.len() + b.w.len() * b.h.len();
        let margin = self.prm.speculative + rel * (1.0 / 60.0);
        let mut pts: [(V3, f32); 8] = [(V3::ZERO, 0.0); 8];
        if let Some((n, cnt)) = collide_boxes(a.p, &a.rot, a.h, b.p, &b.rot, b.h, margin, &mut pts) {
            // waking: a moving awake body touching a sleeping one wakes it
            if a_awake && !b_awake && !a_still { self.wake(ib); }
            if b_awake && !a_awake && !b_still { self.wake(ia); }
            let cnt = reduce4(&mut pts, cnt, n);
            self.push_manifold(ia, ib, n, &pts[..cnt]);
        }
    }

    fn push_manifold(&mut self, a: u32, b: u32, n: V3, pts: &[(V3, f32)]) {
        let ba = &self.bodies[a as usize];
        let (pa, ra) = (ba.p, ba.rot);
        let pb = if b != NONE { self.bodies[b as usize].p } else { V3::ZERO };
        let (sa, sbi) = (self.sidx[a as usize], if b != NONE { self.sidx[b as usize] } else { 0 });
        if sa == 0 && sbi == 0 { return; }
        let (t1, t2) = tangents(n);
        let mut m = Manifold { a, b, sa, sb: sbi, n, t1, t2, mu: self.prm.friction, cnt: pts.len() as u32, cp: [Cp::default(); 4],
            fa: V3::ZERO, fb: V3::ZERO, la_f: V3::ZERO, lb_f: V3::ZERO, e0: V3::ZERO, tm1: 0.0, tm2: 0.0, twm: 0.0, rad: 0.0,
            jt1: 0.0, jt2: 0.0, jtw: 0.0, stick: self.prm.anchor,
            ca1: V3::ZERO, ca2: V3::ZERO, cb1: V3::ZERO, cb2: V3::ZERO, ia1: V3::ZERO, ia2: V3::ZERO, ib1: V3::ZERO, ib2: V3::ZERO,
            ian: V3::ZERO, ibn: V3::ZERO, ima: 0.0, imb: 0.0 };
        let old = self.old_map.get(&key(a, b)).map(|&k| &self.old[k as usize]);
        let rbm = if b != NONE { self.bodies[b as usize].rot } else { M3::IDENTITY };
        let mut cen = V3::ZERO; for &(p, _) in pts { cen += p; } cen = cen * (1.0 / pts.len() as f32);
        let mut rad = 0.0; for &(p, _) in pts { rad += (p - cen).len(); } m.rad = rad / pts.len() as f32;
        // friction anchor: keep last step's anchor while the patch sticks and is still under the contact
        let mut anchored = false;
        if let Some(o) = old {
            if o.stick {
                let wa = pa + ra.mulv(o.la_f);
                let wb = if b != NONE { pb + rbm.mulv(o.lb_f) } else { o.lb_f };
                let lim = 0.05f32.max(m.rad);
                if (wa - cen).len2() < lim * lim && (wb - wa).len2() < 0.03 * 0.03 {
                    m.la_f = o.la_f; m.lb_f = o.lb_f; m.fa = wa - pa; m.fb = wb - pb; m.e0 = wb - wa; anchored = true;
                }
            }
            let f = o.t1 * o.jt1 + o.t2 * o.jt2;
            m.jt1 = f.dot(t1); m.jt2 = f.dot(t2); m.jtw = o.jtw * o.n.dot(n);
        }
        if !anchored {
            m.fa = cen - pa; m.fb = cen - pb; m.la_f = ra.tmulv(m.fa);
            m.lb_f = if b != NONE { rbm.tmulv(m.fb) } else { cen };
            m.e0 = V3::ZERO;
        }
        for (k, &(p, s)) in pts.iter().enumerate() {
            let c = &mut m.cp[k];
            c.ra = p - pa;
            c.rb = if b != NONE { p - pb } else { p };
            c.adj = s;
            c.la = ra.tmulv(c.ra);
            if let Some(o) = old {
                let mut best = 0.03 * 0.03; let mut bi = usize::MAX;
                for q in 0..o.cnt as usize { let d = (o.cp[q].la - c.la).len2(); if d < best { best = d; bi = q; } }
                if bi != usize::MAX { c.jn = o.cp[bi].jn; }
            }
        }
        self.man.push(m);
    }
}

fn tangents(n: V3) -> (V3, V3) {
    let t1 = if fabs(n.x) > 0.57 { v3(n.y, -n.x, 0.0) } else { v3(0.0, n.z, -n.y) }.norm();
    (t1, n.cross(t1))
}

/// Keep at most 4 points: the deepest, the farthest from it, then the two that span the most area.
fn reduce4(pts: &mut [(V3, f32); 8], cnt: usize, n: V3) -> usize {
    if cnt <= 4 { return cnt; }
    let mut i0 = 0; for k in 1..cnt { if pts[k].1 < pts[i0].1 { i0 = k; } }
    let p0 = pts[i0].0;
    let mut i1 = 0; let mut best = -1.0;
    for k in 0..cnt { let d = (pts[k].0 - p0).len2(); if d > best { best = d; i1 = k; } }
    let p1 = pts[i1].0;
    let e = p1 - p0;
    let mut i2 = 0; let mut b2 = f32::MIN;
    for k in 0..cnt { let a = (pts[k].0 - p0).cross(e).dot(n); if a > b2 { b2 = a; i2 = k; } }
    let mut i3 = 0; let mut b3 = f32::MIN;
    for k in 0..cnt { let a = -(pts[k].0 - p0).cross(e).dot(n); if a > b3 { b3 = a; i3 = k; } }
    let sel = [pts[i0], pts[i1], pts[i2], pts[i3]];
    let mut out = 0;
    for (k, s) in sel.iter().enumerate() {
        let idx = [i0, i1, i2, i3][k];
        if [i0, i1, i2, i3][..k].contains(&idx) { continue; }
        pts[out] = *s; out += 1;
    }
    out
}

/// Box-box contact. Returns the normal from A to B and up to 8 points (world point, separation; negative =
/// penetration) within `margin`.
pub fn collide_boxes(pa: V3, ra: &M3, ha: V3, pb: V3, rb: &M3, hb: V3, margin: f32, out: &mut [(V3, f32); 8]) -> Option<(V3, usize)> {
    let t = pb - pa;
    let mut ac = [[0.0f32; 3]; 3];
    for i in 0..3 { for j in 0..3 { ac[i][j] = fabs(ra.c[i].dot(rb.c[j])) + 1e-6; } }
    let (hav, hbv) = ([ha.x, ha.y, ha.z], [hb.x, hb.y, hb.z]);
    // A faces
    let (mut sa, mut ia) = (f32::MIN, 0);
    for i in 0..3 {
        let s = fabs(t.dot(ra.c[i])) - (hav[i] + hbv[0] * ac[i][0] + hbv[1] * ac[i][1] + hbv[2] * ac[i][2]);
        if s > margin { return None; }
        if s > sa { sa = s; ia = i; }
    }
    let (mut sbm, mut jb) = (f32::MIN, 0);
    for j in 0..3 {
        let s = fabs(t.dot(rb.c[j])) - (hbv[j] + hav[0] * ac[0][j] + hav[1] * ac[1][j] + hav[2] * ac[2][j]);
        if s > margin { return None; }
        if s > sbm { sbm = s; jb = j; }
    }
    let (mut se, mut ei, mut ej, mut el) = (f32::MIN, 0, 0, V3::ZERO);
    for i in 0..3 { for j in 0..3 {
        let l = ra.c[i].cross(rb.c[j]);
        let len2 = l.len2();
        if len2 < 1e-8 { continue; }
        let l = l * (1.0 / sqrt(len2));
        let r = hav[0] * fabs(ra.c[0].dot(l)) + hav[1] * fabs(ra.c[1].dot(l)) + hav[2] * fabs(ra.c[2].dot(l))
              + hbv[0] * fabs(rb.c[0].dot(l)) + hbv[1] * fabs(rb.c[1].dot(l)) + hbv[2] * fabs(rb.c[2].dot(l));
        let s = fabs(t.dot(l)) - r;
        if s > margin { return None; }
        if s > se { se = s; ei = i; ej = j; el = l; }
    } }
    let face_tol = 0.005;
    let use_b = sbm > sa + face_tol;
    let face_best = if use_b { sbm } else { sa };
    if se > face_best + 0.01 {
        // edge-edge
        let n = if t.dot(el) < 0.0 { -el } else { el };
        let mut ca = pa; let mut cb = pb;
        for k in 0..3 {
            if k != ei { let s = if ra.c[k].dot(n) > 0.0 { 1.0 } else { -1.0 }; ca += ra.c[k] * (s * hav[k]); }
            if k != ej { let s = if rb.c[k].dot(n) > 0.0 { -1.0 } else { 1.0 }; cb += rb.c[k] * (s * hbv[k]); }
        }
        let (d1, d2) = (ra.c[ei], rb.c[ej]);
        let r = ca - cb;
        let bb = d1.dot(d2); let c = d1.dot(r); let f = d2.dot(r);
        let den = 1.0 - bb * bb;
        let mut s = if den > 1e-6 { (bb * f - c) / den } else { 0.0 };
        s = s.max(-hav[ei]).min(hav[ei]);
        let mut tt = bb * s + f;
        tt = tt.max(-hbv[ej]).min(hbv[ej]);
        s = (bb * tt - c).max(-hav[ei]).min(hav[ei]);
        let (p1, p2) = (ca + d1 * s, cb + d2 * tt);
        let sep = (p2 - p1).dot(n);
        out[0] = ((p1 + p2) * 0.5, sep);
        return Some((n, 1));
    }
    // face contact: reference box R, incident box I
    let (pr, rr, hr, pi, ri, hi, k, flip) = if use_b { (pb, rb, hbv, pa, ra, hav, jb, true) } else { (pa, ra, hav, pb, rb, hbv, ia, false) };
    let mut n = rr.c[k];
    if (pi - pr).dot(n) < 0.0 { n = -n; }
    // incident face: most anti-parallel
    let mut j = 0; let mut bd = -1.0;
    for q in 0..3 { let d = fabs(ri.c[q].dot(n)); if d > bd { bd = d; j = q; } }
    let sgn = if ri.c[j].dot(n) > 0.0 { -1.0 } else { 1.0 };
    let fc = pi + ri.c[j] * (sgn * hi[j]);
    let (u, v) = ((j + 1) % 3, (j + 2) % 3);
    let (eu, ev) = (ri.c[u] * hi[u], ri.c[v] * hi[v]);
    let mut poly: [V3; 8] = [V3::ZERO; 8];
    poly[0] = fc + eu + ev; poly[1] = fc - eu + ev; poly[2] = fc - eu - ev; poly[3] = fc + eu - ev;
    let mut pn = 4;
    let (u2, v2) = ((k + 1) % 3, (k + 2) % 3);
    let mut tmp: [V3; 8] = [V3::ZERO; 8];
    for &(ax, s) in &[(u2, 1.0f32), (u2, -1.0), (v2, 1.0), (v2, -1.0)] {
        let dir = rr.c[ax] * s;
        let off = dir.dot(pr) + hr[ax];
        // keep dot(p, dir) <= off
        let mut tn = 0;
        for q in 0..pn {
            let (p0, p1) = (poly[q], poly[(q + 1) % pn]);
            let (d0, d1) = (p0.dot(dir) - off, p1.dot(dir) - off);
            if d0 <= 0.0 { if tn < 8 { tmp[tn] = p0; tn += 1; } }
            if (d0 <= 0.0) != (d1 <= 0.0) {
                let tt = d0 / (d0 - d1);
                if tn < 8 { tmp[tn] = p0 + (p1 - p0) * tt; tn += 1; }
            }
        }
        pn = tn; poly[..pn].copy_from_slice(&tmp[..pn]);
        if pn == 0 { return None; }
    }
    let rc = pr + n * hr[k];
    let mut cnt = 0;
    for q in 0..pn {
        let p = poly[q];
        let sep = (p - rc).dot(n);
        if sep <= margin { out[cnt] = (p - n * (0.5 * sep), sep); cnt += 1; }
    }
    if cnt == 0 { return None; }
    Some((if flip { -n } else { n }, cnt))
}

/// Throughput scenario shared by the tests, the native bench and the browser bench.
/// Throughput: a pile of n brick-sized boxes dropped in a loose grid so they collide, tumble and stack.
pub fn pile(n: usize) -> World {
    let mut w = World::new();
    let side = (sqrt(n as f32 / 20.0) as usize).max(1);
    let h = v3(0.25, 0.125, 0.125);
    let mut k = 0usize; let mut seed = 1u32;
    let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 8) as f32 / 16777216.0 };
    'o: for layer in 0.. { for i in 0..side { for j in 0..side {
        if k >= n { break 'o; }
        let p = v3(i as f32 * 0.62 + rnd() * 0.05, 0.2 + layer as f32 * 0.3, j as f32 * 0.36 + rnd() * 0.05);
        let q = Quat::from_axis_angle(v3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5), rnd() * 0.6);
        w.add_box(p, q, h, 1900.0);
        k += 1;
    } } }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::time::Instant;

    fn stack(n: usize, sleep: bool) -> (World, f32) {
        let mut w = World::new();
        w.prm.allow_sleep = sleep;
        let h = 0.25;
        for k in 0..n { w.add_box(v3(0.0, h + 2.0 * h * k as f32, 0.0), Quat::IDENTITY, v3(h, h, h), 1900.0); }
        let top0 = w.bodies[n - 1].p.y;
        (w, top0)
    }

    #[test] fn single_box_rests() {
        let mut w = World::new();
        w.add_box(v3(0.0, 1.0, 0.0), Quat::from_axis_angle(v3(1.0, 0.3, 0.2), 0.7), v3(0.25, 0.125, 0.5), 2000.0);
        for _ in 0..300 { w.step(1.0 / 60.0); }
        let b = w.bodies[0];
        assert!(b.p.y > 0.1 && b.p.y < 0.3, "{:?}", b.p);
        assert!(b.v.len() < 0.01, "{:?}", b.v);
    }

    #[test] fn gate_stack_20() {
        for &sleep in &[false, true] {
            let (mut w, top0) = stack(20, sleep);
            let mut maxv = 0.0f32;
            for f in 0..600 { w.step(1.0 / 60.0); if f >= 540 { for b in &w.bodies { maxv = maxv.max(b.v.len()); } } }
            let top = w.bodies[19].p;
            let drift = (top - v3(0.0, top0, 0.0)).len();
            std::println!("stack 20 (sleep {sleep}): top drift {:.2} mm, max speed in last second {:.3} mm/s, awake {}", drift * 1e3, maxv * 1e3, w.awake_count());
            assert!(drift < 0.01, "drift {drift}");
            assert!(maxv < 0.01, "speed {maxv}");
        }
    }

    #[test] fn tilted_stack_falls_and_settles() {
        // a 10-high column with an offset per block topples; everything ends up resting on the ground
        let mut w = World::new();
        for k in 0..10 { w.add_box(v3(0.06 * k as f32, 0.25 + 0.5 * k as f32, 0.0), Quat::IDENTITY, v3(0.25, 0.25, 0.25), 1900.0); }
        for _ in 0..600 { w.step(1.0 / 60.0); }
        let maxy = w.bodies.iter().map(|b| b.p.y).fold(0.0, f32::max);
        let minv = w.bodies.iter().map(|b| b.p.y).fold(9.0, f32::min);
        std::println!("toppled column: highest centre {maxy:.2} m, lowest {minv:.3} m, awake {}", w.awake_count());
        assert!(maxy < 3.0, "column should have toppled: {maxy}");
        assert!(minv > 0.2, "nothing sinks into the ground: {minv}");
        for b in &w.bodies { assert!(b.v.len() < 0.05, "settled"); }
    }

    #[test] #[ignore] fn bench_pile() {
        fn clk() -> f64 { use std::sync::OnceLock; static T: OnceLock<Instant> = OnceLock::new(); T.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3 }
        let only: usize = std::env::var("PILE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for &n in &[2000usize, 20000] {
            if only != 0 && n != only { continue; }
            let mut w = pile(n);
            w.clock = clk;
            let (mut tb, mut ts) = (0.0, 0.0);
            w.prm.allow_sleep = false;
            let mut times = std::vec::Vec::new();
            let mut st = Stats::default();
            for f in 0..240 {
                let t = Instant::now();
                w.step(1.0 / 60.0);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if f >= 60 { times.push(ms); st = w.stats; tb += st.t_broad; ts += st.t_solve; }
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            std::println!("pile {n}: median {:.2} ms, p95 {:.2} ms per 60 Hz step; {} pairs, {} manifolds, {} points",
                times[times.len() / 2], times[times.len() * 95 / 100], st.pairs, st.manifolds, st.points);
            std::println!("   mean collide {:.2} ms, solve {:.2} ms", tb / times.len() as f64, ts / times.len() as f64);
        }
    }
}
