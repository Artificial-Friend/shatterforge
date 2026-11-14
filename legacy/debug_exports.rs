static mut DBG: [f32; 512] = [0.0; 512];
/// debug: per-height histogram of a structure's bricks: [static, detached(dynamic), removed] x NY, plus column x/z extents of detached bricks
#[no_mangle] pub extern "C" fn dbg_hist(s: u32) -> *const f32 {
    let w = world(); let s = s as usize;
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 0..300 { d[k] = 0.0; }
    d[300] = 1e9; d[301] = -1e9; d[302] = 1e9; d[303] = -1e9;
    for i in w.st_lo[s]..w.st_hi[s] {
        if w.sid[i] as usize != s { continue; }
        let y = w.by[i] as usize;
        if w.mat[i] == 0 { d[y * 3 + 2] += 1.0; } else if w.body[i] == STATIC { d[y * 3] += 1.0; } else {
            d[y * 3 + 1] += 1.0;
            let x = w.bx[i] as f32; let z = w.bz[i] as f32;
            if x < d[300] { d[300] = x; } if x > d[301] { d[301] = x; } if z < d[302] { d[302] = z; } if z > d[303] { d[303] = z; }
        }
    }
    d.as_ptr()
}
/// debug: largest debris bodies: (count, vertical extent m, horizontal extent m) for the 8 biggest
#[no_mangle] pub extern "C" fn dbg_bodies() -> *const f32 {
    let w = world();
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 0..30 { d[k] = 0.0; }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        let c = w.bd_count[b] as f32;
        let mut lo = [1e9f32; 3]; let mut hi = [-1e9f32; 3];
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; let p = w.wpos[i]; for a in 0..3 { lo[a] = minf(lo[a], p[a]); hi[a] = maxf(hi[a], p[a]); } }
        let ext = [hi[1] - lo[1], maxf(hi[0] - lo[0], hi[2] - lo[2])];
        // insert into top-8 by count
        let mut pos = 8;
        for t in 0..8 { if c > d[t * 3] { pos = t; break; } }
        if pos < 8 {
            let mut t = 7; while t > pos { d[t * 3] = d[(t - 1) * 3]; d[t * 3 + 1] = d[(t - 1) * 3 + 1]; d[t * 3 + 2] = d[(t - 1) * 3 + 2]; t -= 1; }
            d[pos * 3] = c; d[pos * 3 + 1] = ext[0] + S; d[pos * 3 + 2] = ext[1] + S;
        }
    }
    d.as_ptr()
}
#[no_mangle] pub extern "C" fn dbg_matstats() -> *const f32 {
    let w = world();
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 0..60 { d[k] = 0.0; }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        let mut cnt = [0u32; 9]; let mut tot = 0u32;
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 && w.mat[i] < 9 { cnt[w.mat[i] as usize] += 1; tot += 1; } }
        if tot == 0 { continue; }
        let mut m = 1; for q in 1..9 { if cnt[q] > cnt[m] { m = q; } }
        // per dominant material: bodies, bricks, largest, pieces >= 50 bricks
        d[m * 5] += 1.0; d[m * 5 + 1] += tot as f32; if tot as f32 > d[m * 5 + 2] { d[m * 5 + 2] = tot as f32; } if tot >= 50 { d[m * 5 + 3] += 1.0; } if tot <= 2 { d[m * 5 + 4] += 1.0; }
    }
    d.as_ptr()
}
#[no_mangle] pub extern "C" fn dbg_speed() -> *const f32 {
    let w = world();
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 0..10 { d[k] = 0.0; }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 || w.bd_asleep[b] { continue; }
        let v = w.bd_v[b]; let hs = sqrt(v[0] * v[0] + v[2] * v[2]);
        if hs > d[0] { d[0] = hs; d[1] = w.bd_count[b] as f32; d[2] = v[1]; }
        if hs > 5.0 { d[3] += 1.0; d[4] += w.bd_count[b] as f32; }
        if v[1] > 3.0 { d[5] += 1.0; }
        d[6] += 1.0;
        let tip = len(w.bd_w[b]) * w.bd_rad[b]; if tip > d[7] { d[7] = tip; d[8] = w.bd_count[b] as f32; d[9] = len(w.bd_w[b]); }
    }
    d.as_ptr()
}
#[no_mangle] pub extern "C" fn dbg_bigbody() -> *const f32 {
    // biggest awake-or-not debris body: count, pos, tilt of its local up axis, speed
    let w = world();
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 200..212 { d[k] = 0.0; }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        if w.bd_count[b] as f32 > d[200] {
            d[200] = w.bd_count[b] as f32; d[201] = w.bd_pos[b][0]; d[202] = w.bd_pos[b][1]; d[203] = w.bd_pos[b][2];
            let up = mv(&w.bd_rot[b], [0.0, 1.0, 0.0]); d[204] = acosf_(clampf(up[1], -1.0, 1.0)) * 57.3;
            d[205] = len(w.bd_v[b]); d[206] = len(w.bd_w[b]); d[207] = if w.bd_asleep[b] { 1.0 } else { 0.0 }; d[208] = w.bd_maxr[b]; d[209] = w.bd_resid[b]; d[210] = w.bd_nc[b] as f32; d[211] = w.bd_lt[b];
        }
    }
    unsafe { core::ptr::addr_of!(DBG) as *const f32 }.wrapping_add(200)
}
fn acosf_(x: f32) -> f32 { let mut a = 0.0f32; let mut b = 3.14159f32; for _ in 0..30 { let m = (a + b) * 0.5; if cosf(m) > x { a = m; } else { b = m; } } (a + b) * 0.5 }
#[no_mangle] pub extern "C" fn dbg_tp(x: f32, y: f32, z: f32) { let w = world(); w.pl_pos = [x, y, z]; w.pl_vel = [0.0; 3]; }
#[no_mangle] pub extern "C" fn dbg_top(s: u32) -> *const f32 {
    let w = world(); let s = s as usize;
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DBG) };
    for k in 0..60 { d[k] = 0.0; }
    d[60] = w.st_maxr[s];
    for i in w.st_lo[s]..w.st_hi[s] {
        if w.mat[i] == 0 || w.body[i] != STATIC { continue; }
        let r = w.stress[i];
        let mut pos = 12;
        for t in 0..12 { if r > d[t * 5] { pos = t; break; } }
        if pos < 12 {
            let mut t = 11; while t > pos { for q in 0..5 { d[t * 5 + q] = d[(t - 1) * 5 + q]; } t -= 1; }
            d[pos * 5] = r; d[pos * 5 + 1] = w.bx[i] as f32; d[pos * 5 + 2] = w.by[i] as f32; d[pos * 5 + 3] = w.bz[i] as f32; d[pos * 5 + 4] = w.mat[i] as f32;
        }
    }
    d.as_ptr()
}
#[no_mangle] pub extern "C" fn dbg_ptr() -> *const f32 { unsafe { core::ptr::addr_of!(DBG) as *const f32 } }
#[no_mangle] pub extern "C" fn dbg_wake_all() { let w = world(); for s in 0..w.st_n { st_activate(w, s); } }
