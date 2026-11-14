//! SHATTERFORGE v3 — open-map destruction engine in one no_std WebAssembly module.
//!  * sparse chunked voxel world (320 m x 36 m x 320 m at 0.5 m) holding ~33k bonded bricks;
//!  * structures sleep until disturbed: load-path solve, fatigue and connectivity run only on active ones,
//!    and an unsolved structure is warmed up the first time anything touches it;
//!  * per-structure meshes plus regional meshes for settled debris, rebuilt only when they change;
//!  * compound rigid bodies, Voronoi fracture, ploughing, a 10k shard tier;
//!  * third-person player (character controller, hammer, rockets, sticky remote charges, health);
//!  * raycast-suspension vehicles the player can take over, AI traffic on a road graph with aggro;
//!  * district crowds with flow-field navigation, hazard perception from the stress field and knock-downs.
#![no_std]
#![feature(wasm_numeric_instr)]
#![allow(static_mut_refs)]
#![allow(clippy::all)]
#![allow(dead_code)]

use core::arch::wasm32 as wa;
use core::mem::MaybeUninit;
#[link(wasm_import_module = "env")]
extern "C" { fn now() -> f64; }
#[inline] fn tnow() -> f64 { unsafe { now() } }

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { wa::unreachable() }

// ---------------------------------------------------------------- math
#[inline] fn sqrt(x: f32) -> f32 { wa::f32_sqrt(x) }
#[inline] fn floor(x: f32) -> f32 { wa::f32_floor(x) }
#[inline] fn abs(x: f32) -> f32 { f32::from_bits(x.to_bits() & 0x7fff_ffff) }
#[inline] fn minf(a: f32, b: f32) -> f32 { if a < b { a } else { b } }
#[inline] fn maxf(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
#[inline] fn clampf(x: f32, a: f32, b: f32) -> f32 { minf(maxf(x, a), b) }
#[inline] fn maxi(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
#[inline] fn mini(a: i32, b: i32) -> i32 { if a < b { a } else { b } }
const PI: f32 = 3.14159265;
fn sinf(x: f32) -> f32 {
    let tp = 2.0 * PI;
    let mut x = x - tp * floor((x + PI) / tp);
    if x > 0.5 * PI { x = PI - x; } else if x < -0.5 * PI { x = -PI - x; }
    let x2 = x * x;
    x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))))
}
fn cosf(x: f32) -> f32 { sinf(x + 0.5 * PI) }
fn atan2f(y: f32, x: f32) -> f32 {
    let ax = abs(x); let ay = abs(y);
    let mx = maxf(ax, ay);
    if mx < 1e-12 { return 0.0; }
    let a = minf(ax, ay) / mx;
    let s = a * a;
    let mut r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if ay > ax { r = 0.5 * PI - r; }
    if x < 0.0 { r = PI - r; }
    if y < 0.0 { r = -r; }
    r
}
type V = [f32; 3];
type M = [f32; 9];
const IDM: M = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
#[inline] fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
#[inline] fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
#[inline] fn mul(a: V, s: f32) -> V { [a[0] * s, a[1] * s, a[2] * s] }
#[inline] fn madd(a: V, b: V, s: f32) -> V { [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s] }
#[inline] fn dot(a: V, b: V) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline] fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
#[inline] fn len(a: V) -> f32 { sqrt(dot(a, a)) }
#[inline] fn norm(a: V) -> V { let l = len(a); if l > 1e-9 { mul(a, 1.0 / l) } else { [0.0, 1.0, 0.0] } }
#[inline] fn mv(m: &M, v: V) -> V {
    [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]
}
#[inline] fn mtv(m: &M, v: V) -> V {
    [m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2], m[2] * v[0] + m[5] * v[1] + m[8] * v[2]]
}
fn mmul(a: &M, b: &M) -> M {
    let mut r = [0.0; 9];
    for i in 0..3 { for j in 0..3 { r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; } }
    r
}
fn mt(a: &M) -> M { [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]] }
fn minv(m: &M) -> M {
    let a = m;
    let c0 = a[4] * a[8] - a[5] * a[7];
    let c1 = a[5] * a[6] - a[3] * a[8];
    let c2 = a[3] * a[7] - a[4] * a[6];
    let det = a[0] * c0 + a[1] * c1 + a[2] * c2;
    if abs(det) < 1e-12 { return [0.0; 9]; }
    let id = 1.0 / det;
    [
        c0 * id, (a[2] * a[7] - a[1] * a[8]) * id, (a[1] * a[5] - a[2] * a[4]) * id,
        c1 * id, (a[0] * a[8] - a[2] * a[6]) * id, (a[2] * a[3] - a[0] * a[5]) * id,
        c2 * id, (a[1] * a[6] - a[0] * a[7]) * id, (a[0] * a[4] - a[1] * a[3]) * id,
    ]
}
fn q_to_m(q: [f32; 4]) -> M {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
    ]
}
fn q_int(q: [f32; 4], w: V, dt: f32) -> [f32; 4] {
    let qv = [q[0], q[1], q[2]];
    let d = add(mul(w, q[3]), cross(w, qv));
    let dw = -dot(w, qv);
    let mut r = [q[0] + 0.5 * dt * d[0], q[1] + 0.5 * dt * d[1], q[2] + 0.5 * dt * d[2], q[3] + 0.5 * dt * dw];
    let l = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
    if l > 1e-9 { for k in 0..4 { r[k] /= l; } } else { r = [0.0, 0.0, 0.0, 1.0]; }
    r
}

// ---------------------------------------------------------------- constants
const S: f32 = 0.5;
const H: f32 = 0.25;
const NX: usize = 640;
const NY: usize = 136;
const NZ: usize = 640;
const CXN: usize = NX / 8;
const CYN: usize = NY / 8;
const CZN: usize = NZ / 8;
const NCT: usize = CXN * CYN * CZN;
const MAXCH: usize = 30000;
const MB: usize = 500000;
const MAXBODY: usize = 16000;
const MAXC: usize = 30000;
const MAXP: usize = 5000;
const MAXR: usize = 12;
const MAXEV: usize = 64;
const HSIZE: usize = 32768;
const MAXSH: usize = 10000;
const SHQ: usize = MAXSH / 4;
const SHASH: usize = 8192;
const MAXV: usize = 8;
const NAI: usize = 5;
const MAXA: usize = 48;
const NZONE: usize = 6;
const ZN: usize = 80;
const ZNAV: usize = ZN * ZN;
const NHZ: usize = 6;
const MAXST: usize = 64;
const NBK: usize = 100;
/// static geometry is meshed per structure and per 12 m height band, so a hit only rebuilds the band it touched
const NBAND: usize = 6;
const SSETS: usize = MAXST * NBAND;
const MAXSET: usize = SSETS + NBK;
#[inline] fn band_of(y: i32) -> usize { mini(maxi(y, 0) / 24, NBAND as i32 - 1) as usize }
const NOS: u8 = 255;
const NOB: u16 = 0xFFFF;
const MAXCG: usize = 8;
const STATIC: u16 = 0xFFFF;
const TEMPB: u16 = 0xFFFE;
const NONE: u32 = u32::MAX;
const INF16: u16 = 0xFFFF;
const RB: f32 = 0.27;
const GR: f32 = 9.81;
const SLUG_Y: i32 = NY as i32 - 3;
const VEH_Y: i32 = NY as i32 - 8;
const HALF: f32 = 160.0;
const FS: usize = 52;
const MFS: usize = 100000;
const MFG: usize = 24000;
const MFD: usize = 30000;
const MFDG: usize = 6000;
const FCAPS: [usize; 4] = [MFS, MFG, MFD, MFDG];

const CONC: u8 = 1; const MASON: u8 = 2; const STEEL: u8 = 3; const GLASS: u8 = 4; const WOOD: u8 = 5;
const SLUG: u8 = 6; const VEH: u8 = 7; const VEHH: u8 = 8;
const MASS: [f32; 9] = [0.0, 300.0, 230.0, 250.0, 60.0, 70.0, 2000.0, 8.0, 44.0];
const CAPV: [f32; 9] = [0.0, 140000.0, 50000.0, 150000.0, 1500.0, 9000.0, 1e12, 1e12, 1e12];
const GROUNDK: f32 = 4.0;
const CAPH: [f32; 9] = [0.0, 12000.0, 12000.0, 90000.0, 400.0, 5000.0, 1e12, 1e12, 1e12];
// tension capacity as a fraction of compression (vertical bonds carrying a hanging load)
const TENS: [f32; 9] = [0.0, 0.35, 0.1, 0.9, 0.5, 0.7, 1.0, 1.0, 1.0];
const STIFF: [f32; 9] = [0.0, 1.0, 0.8, 2.0, 0.02, 0.4, 1.0, 1.0, 1.0];
const BLAST: [f32; 9] = [0.0, 0.75, 0.85, 0.35, 1.4, 0.95, 0.0, 0.0, 0.0];
/// Energy (J) to crush one 0.5 m brick of each material. A collision delivers energy at the exact contact
/// (work of the contact impulse, 1/2 * J * v_approach) and it is spent crushing the bricks nearest that contact.
/// Tuned so a concrete frame dropping one storey crushes roughly one storey under it (progressive collapse),
/// a falling chunk of brickwork loses its leading corner, and glass shatters at a touch. Steel bends, it does not crush.
const EB: [f32; 9] = [0.0, 15000.0, 12000.0, 1.5e6, 300.0, 6000.0, 1e12, 1e12, 1e12];
/// share of crushed bricks ground to dust and small fragments; the rest break loose as single-brick rubble
const PULV: [f32; 9] = [0.0, 0.7, 0.55, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0];
/// how fast an overloaded joint gives way (1/s per unit of overload): brittle masonry and glass snap at once,
/// reinforced concrete, timber and steel yield for a moment first
const DUCT: [f32; 9] = [0.0, 6.0, 12.0, 3.0, 40.0, 8.0, 1.0, 1.0, 1.0];
const FRAG: [f32; 9] = [0.0, 1.0, 1.3, 0.25, 4.0, 0.8, 0.0, 0.0, 0.0];
const COL: [[f32; 3]; 9] = [
    [0.0, 0.0, 0.0], [0.63, 0.62, 0.59], [0.64, 0.33, 0.24], [0.36, 0.42, 0.48],
    [0.55, 0.78, 0.90], [0.58, 0.42, 0.26], [0.95, 0.52, 0.12], [0.9, 0.6, 0.2], [0.9, 0.6, 0.2],
];
// paints override the material colour (the material still sets strength and surface pattern)
const P_RED: u8 = 1; const P_BLUE: u8 = 2; const P_GREEN: u8 = 3; const P_ORANGE: u8 = 4; const P_WHITE: u8 = 5;
const P_ROOF: u8 = 6; const P_DARK: u8 = 7; const P_HAZARD: u8 = 8; const P_WARM: u8 = 9;
const PAL: [[f32; 3]; 10] = [
    [0.0, 0.0, 0.0], [0.56, 0.17, 0.12], [0.16, 0.29, 0.46], [0.2, 0.37, 0.25], [0.76, 0.42, 0.13],
    [0.86, 0.85, 0.81], [0.48, 0.2, 0.15], [0.22, 0.24, 0.27], [0.86, 0.66, 0.12], [0.72, 0.67, 0.58],
];
const DIRS: [[i32; 3]; 6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const FACES: [[usize; 4]; 6] = [[1, 3, 7, 5], [0, 4, 6, 2], [2, 6, 7, 3], [0, 1, 5, 4], [4, 5, 7, 6], [0, 2, 3, 1]];
const N8: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
const RN: [f32; 4] = [-120.0, -40.0, 40.0, 120.0];
const ZONES: [[f32; 2]; NZONE] = [[-80.0, 0.0], [0.0, 80.0], [0.0, 0.0], [0.0, -80.0], [-80.0, 80.0], [80.0, -80.0]];
const HUBOFF: [[f32; 2]; NHZ] = [[-15.0, -15.0], [15.0, -15.0], [-15.0, 15.0], [15.0, 15.0], [0.0, -18.0], [0.0, 18.0]];

// vehicles
const WR: f32 = 0.42;
const SREST: f32 = 0.38;
const KS: f32 = 30000.0;
const KD: f32 = 3400.0;
const ENGINE: f32 = 12500.0;
const VMAX: f32 = 30.0;
const BRAKE: f32 = 16000.0;
// player
const PR: f32 = 0.24;
const PHT: f32 = 1.8;
const SPAWN: V = [-4.0, 0.0, 134.0];

#[derive(Clone, Copy)]
struct Contact {
    a: u16, b: u16, bo: u16, on: bool,
    ba: u32, bb: u32,
    n: V, p: V, ra: V, rb: V, t1: V, t2: V,
    depth: f32, mn: f32, mt1: f32, mt2: f32, jn: f32, jt1: f32, jt2: f32, bias: f32, vn0: f32,
}
struct Hit { t: f32, n: V, body: usize, brick: usize }

struct World {
    ctab: [u16; NCT], cpool: [u32; MAXCH * 512], ncpool: usize,
    nb: usize, bcur: usize,
    mat: [u8; MB], bx: [u16; MB], by: [u8; MB], bz: [u16; MB], sid: [u8; MB], paint: [u8; MB], infill: [u8; MB],
    body: [u16; MB], next: [u32; MB],
    local: [V; MB], wpos: [V; MB],
    corner: [[V; 8]; MB], omask: [u8; MB], nb6: [[u32; 6]; MB],
    bond: [[f32; 4]; MB], bmax: [[f32; 4]; MB],
    phi: [f32; MB], ext: [f32; MB], extn: [f32; MB], wload: [f32; MB], stress: [f32; MB], svis: [f32; MB], tint: [f32; MB],
    stamp: [u32; MB], comp: [u32; MB], sup: [u32; MB], frame: u32,
    lst: [u32; MB], lst2: [u32; MB], q: [u32; MB], cnt: [u32; MB],
    // structures
    st_n: usize, cur_sid: u8, cur_paint: u8, cur_infill: bool, hsub: f32, crush_n: u32,
    st_lo: [usize; MAXST], st_hi: [usize; MAXST], st_min: [[i32; 3]; MAXST], st_max: [[i32; 3]; MAXST],
    st_init: [u32; MAXST], st_now: [u32; MAXST], st_top0: [i32; MAXST], st_top: [i32; MAXST],
    st_kind: [u8; MAXST], st_target: [bool; MAXST], st_active: [f32; MAXST], st_settle: [f32; MAXST],
    st_solved: [bool; MAXST], st_conn: [bool; MAXST], st_broke: [bool; MAXST], st_maxr: [f32; MAXST], st_warm: [i32; MAXST], st_resid: [f32; MAXST],
    prof: [f32; 16],
    conn_any: bool,
    set_dirty: [u8; MAXSET], set_since: [f32; MAXSET], set_built: [f32; MAXSET],
    // bodies
    hw: usize, cursor: usize,
    bd_alive: [bool; MAXBODY], bd_first: [u32; MAXBODY], bd_count: [u32; MAXBODY],
    bd_pos: [V; MAXBODY], bd_q: [[f32; 4]; MAXBODY], bd_rot: [M; MAXBODY],
    bd_v: [V; MAXBODY], bd_w: [V; MAXBODY], bd_m: [f32; MAXBODY], bd_im: [f32; MAXBODY],
    bd_iil: [M; MAXBODY], bd_iiw: [M; MAXBODY], bd_sleep: [f32; MAXBODY], bd_asleep: [bool; MAXBODY], bd_born: [f32; MAXBODY], bd_vpre: [V; MAXBODY], bd_wpre: [V; MAXBODY],
    bd_dirty: [bool; MAXBODY], bd_rad: [f32; MAXBODY], bd_nc: [u32; MAXBODY], bd_j: [f32; MAXBODY],
    bd_wake: [bool; MAXBODY], bd_cool: [f32; MAXBODY], bd_crush: [u8; MAXBODY], bd_jmax: [f32; MAXBODY], bd_jp: [V; MAXBODY], bd_lt: [f32; MAXBODY], bd_resid: [f32; MAXBODY], bd_maxr: [f32; MAXBODY], bd_vlast: [V; MAXBODY], bd_wlast: [V; MAXBODY], bd_nmax: [f32; MAXBODY], bd_veh: [u8; MAXBODY], bd_bucket: [u16; MAXBODY], bd_inmesh: [bool; MAXBODY],
    nc: usize, cs: [Contact; MAXC],
    hhead: [u32; HSIZE], hnext: [u32; MB],
    // particles
    pcur: usize,
    p_pos: [V; MAXP], p_vel: [V; MAXP], p_life: [f32; MAXP], p_max: [f32; MAXP], p_size: [f32; MAXP],
    p_col: [V; MAXP], p_kind: [u8; MAXP],
    // shards
    scur: usize, s_on: [bool; MAXSH], s_rest: [bool; MAXSH],
    s_pos: [V; MAXSH], s_vel: [V; MAXSH], s_w: [V; MAXSH], s_q: [[f32; 4]; MAXSH], s_scl: [V; MAXSH],
    s_col: [V; MAXSH], s_t: [f32; MAXSH], s_rad: [f32; MAXSH],
    shead: [u32; SHASH], snext: [u32; MAXSH],
    // rockets
    r_alive: [bool; MAXR], r_pos: [V; MAXR], r_vel: [V; MAXR], r_age: [f32; MAXR],
    // vehicles
    v_alive: [bool; MAXV], v_body: [u16; MAXV], v_in: [[f32; 3]; MAXV], v_steer: [f32; MAXV],
    v_comp: [[f32; 4]; MAXV], v_spin: [f32; MAXV], v_mount: [[V; 4]; MAXV], v_hc: [V; MAXV],
    v_stuck: [f32; MAXV], v_rev: [f32; MAXV], v_flip: [f32; MAXV], v_aggro: [f32; MAXV], v_touch: [u8; MAXV],
    v_ai: [bool; MAXV], v_node: [u8; MAXV], v_prevn: [u8; MAXV], v_respawn: [f32; MAXV],
    ai_on: bool,
    // player
    pl_pos: V, pl_vel: V, pl_in: [f32; 4], pl_ground: bool, pl_hp: f32, pl_hurt: f32, pl_dead: f32, pl_knock: f32,
    pv: i32, pl_jump: bool,
    // remote charges
    cg_st: [u8; MAXCG], cg_pos: [V; MAXCG], cg_vel: [V; MAXCG], cg_brick: [u32; MAXCG], cg_off: [V; MAXCG],
    // agents
    a_on: [bool; MAXA], a_zone: [u8; MAXA], a_pos: [V; MAXA], a_vel: [V; MAXA], a_yaw: [f32; MAXA], a_tilt: [f32; MAXA], a_phase: [f32; MAXA],
    a_state: [u8; MAXA], a_t: [f32; MAXA], a_panic: [f32; MAXA], a_hub: [u8; MAXA], a_flee: [u8; MAXA],
    a_idle: [f32; MAXA], a_scan: [f32; MAXA], a_spd: [f32; MAXA],
    zx0: [i32; NZONE], zz0: [i32; NZONE], znav: [u8; NZONE * ZNAV], zflow: [u16; NZONE * NHZ * ZNAV], zhub: [u32; NZONE * NHZ],
    zdirty: [bool; NZONE], zt: [f32; NZONE], heap: [u32; ZNAV * 8],
    // misc
    rng: u32, boost: i32, time: f32,
    blast_c: V, blast_r: f32, blast_p: f32, blast_t: f32,
    stamp_id: u32, slug_slot: u32, stress_view: bool, breaks: u32, break_c: V, salvage: f32,
    // outputs
    nf: [usize; 4],
    out_so: [f32; MFS * FS], out_sg: [f32; MFG * FS], out_do: [f32; MFD * FS], out_dg: [f32; MFDG * FS],
    out_np: usize, out_p: [f32; MAXP * 4], out_pc: [f32; MAXP * 4], out_rk: [f32; MAXR * 4],
    ns: [u32; 4], out_sh: [f32; MAXSH * 16], out_shc: [f32; MAXSH * 3],
    out_ag: [f32; MAXA * 8], out_vh: [f32; MAXV * 20], out_st: [f32; MAXST * 12], out_pl: [f32; 16], out_cg: [f32; MAXCG * 4],
    nev: usize, ev: [f32; MAXEV * 5],
    stats: [f32; 16], hit: [f32; 8],
}

static mut WORLD: MaybeUninit<World> = MaybeUninit::zeroed();
#[inline] fn world() -> &'static mut World { unsafe { &mut *(core::ptr::addr_of_mut!(WORLD) as *mut World) } }

// ---------------------------------------------------------------- sparse voxel grid
#[inline] fn inb(x: i32, y: i32, z: i32) -> bool { x >= 0 && y >= 0 && z >= 0 && (x as usize) < NX && (y as usize) < NY && (z as usize) < NZ }
#[inline] fn gget(w: &World, x: i32, y: i32, z: i32) -> u32 {
    if !inb(x, y, z) { return 0; }
    let (x, y, z) = (x as usize, y as usize, z as usize);
    let t = w.ctab[((y >> 3) * CZN + (z >> 3)) * CXN + (x >> 3)];
    if t == 0 { return 0; }
    w.cpool[(t as usize - 1) * 512 + (((y & 7) * 8 + (z & 7)) * 8 + (x & 7))]
}
fn gset(w: &mut World, x: i32, y: i32, z: i32, v: u32) {
    if !inb(x, y, z) { return; }
    let (x, y, z) = (x as usize, y as usize, z as usize);
    let ci = ((y >> 3) * CZN + (z >> 3)) * CXN + (x >> 3);
    let mut t = w.ctab[ci];
    if t == 0 {
        if v == 0 || w.ncpool >= MAXCH { return; }
        w.ncpool += 1;
        t = w.ncpool as u16;
        w.ctab[ci] = t;
    }
    w.cpool[(t as usize - 1) * 512 + (((y & 7) * 8 + (z & 7)) * 8 + (x & 7))] = v;
}

// ---------------------------------------------------------------- utils
fn rnd(w: &mut World) -> f32 {
    let mut x = w.rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    w.rng = x;
    (x >> 8) as f32 / 16777216.0
}
fn rndv(w: &mut World) -> V {
    loop {
        let v = [rnd(w) * 2.0 - 1.0, rnd(w) * 2.0 - 1.0, rnd(w) * 2.0 - 1.0];
        let l = dot(v, v);
        if l > 0.01 && l <= 1.0 { return mul(v, 1.0 / sqrt(l)); }
    }
}
#[inline] fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6b343) ^ (y as u32).wrapping_mul(0xd8163841) ^ (z as u32).wrapping_mul(0xcb1ab31f);
    h ^= h >> 13; h = h.wrapping_mul(0x5bd1e995); h ^= h >> 15; h
}
#[inline] fn h01(h: u32, k: u32) -> f32 {
    let mut x = h.wrapping_add(k.wrapping_mul(0x9E3779B9));
    x ^= x >> 16; x = x.wrapping_mul(0x7feb352d); x ^= x >> 15; x = x.wrapping_mul(0x846ca68b); x ^= x >> 16;
    (x >> 8) as f32 / 16777216.0
}
#[inline] fn lat_pos(x: i32, y: i32, z: i32) -> V {
    [(x - (NX / 2) as i32) as f32 * S + H, y as f32 * S + H, (z - (NZ / 2) as i32) as f32 * S + H]
}
#[inline] fn world_cell(p: V) -> (i32, i32, i32) {
    (floor(p[0] / S) as i32 + (NX / 2) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32 + (NZ / 2) as i32)
}
#[inline] fn cm(m: f32) -> i32 { floor(m / S) as i32 + (NX / 2) as i32 }
#[inline] fn blat(w: &World, i: usize) -> (i32, i32, i32) { (w.bx[i] as i32, w.by[i] as i32, w.bz[i] as i32) }
#[inline] fn bpos(w: &World, i: usize) -> V {
    if w.body[i] == STATIC { let (x, y, z) = blat(w, i); lat_pos(x, y, z) } else { w.wpos[i] }
}
#[inline] fn static_at(w: &World, x: i32, y: i32, z: i32) -> bool {
    let g = gget(w, x, y, z);
    if g == 0 { return false; }
    let j = g as usize - 1;
    w.mat[j] != 0 && w.body[j] == STATIC
}
#[inline] fn static_brick(w: &World, x: i32, y: i32, z: i32) -> usize {
    let g = gget(w, x, y, z);
    if g == 0 { return usize::MAX; }
    let j = g as usize - 1;
    if w.mat[j] != 0 && w.body[j] == STATIC { j } else { usize::MAX }
}
#[inline] fn nbr(w: &World, i: usize, d: usize) -> usize {
    let j = w.nb6[i][d];
    if j == NONE { usize::MAX } else { j as usize }
}
fn nbr_grid(w: &World, i: usize, d: usize) -> usize {
    let (x, y, z) = blat(w, i);
    let g = gget(w, x + DIRS[d][0], y + DIRS[d][1], z + DIRS[d][2]);
    if g == 0 { usize::MAX } else { g as usize - 1 }
}
#[inline] fn linked(w: &World, i: usize, d: usize) -> (usize, f32) {
    let j = nbr(w, i, d);
    if j == usize::MAX || w.mat[j] == 0 || w.body[j] != w.body[i] { return (usize::MAX, 0.0); }
    let hp = if d & 1 == 0 { w.bond[i][d >> 1] } else { w.bond[j][d >> 1] };
    if hp > 0.0 { (j, hp) } else { (usize::MAX, 0.0) }
}
fn event(w: &mut World, t: f32, p: V, m: f32) {
    if w.nev >= MAXEV { return; }
    let k = w.nev * 5;
    w.ev[k] = t; w.ev[k + 1] = p[0]; w.ev[k + 2] = p[1]; w.ev[k + 3] = p[2]; w.ev[k + 4] = m;
    w.nev += 1;
}
fn spawn_p(w: &mut World, p: V, v: V, life: f32, size: f32, col: V, kind: u8) {
    let k = w.pcur;
    w.pcur = (w.pcur + 1) % MAXP;
    w.p_pos[k] = p; w.p_vel[k] = v; w.p_life[k] = life; w.p_max[k] = life; w.p_size[k] = size; w.p_col[k] = col; w.p_kind[k] = kind;
}
fn base_col(w: &World, i: usize) -> V {
    let p = w.paint[i];
    if p != 0 { PAL[p as usize] } else { COL[w.mat[i] as usize] }
}
fn brick_col(w: &World, i: usize) -> V { mul(base_col(w, i), w.tint[i]) }
fn dust_burst(w: &mut World, p: V, col: V, n: u32, spread: f32) {
    for _ in 0..n {
        let d = rndv(w);
        let s = rnd(w);
        let c = mul(col, 0.75 + 0.3 * rnd(w));
        let life = 1.6 + 2.0 * rnd(w);
        let sz = 0.5 + 0.7 * rnd(w);
        spawn_p(w, madd(p, d, 0.2), mul(d, spread * (0.3 + s)), life, sz, c, 0);
    }
}
fn mark_st(w: &mut World, s: usize, urgent: bool) { for k in 0..NBAND { mark_set(w, s * NBAND + k, urgent); } }
fn mark_brick(w: &mut World, s: usize, y: i32, urgent: bool) {
    let (a, b) = (band_of(y - 1), band_of(y + 1));
    for k in a..=b { mark_set(w, s * NBAND + k, urgent); }
}
fn mark_set(w: &mut World, id: usize, urgent: bool) {
    if id >= MAXSET { return; }
    let lvl = if urgent { 2 } else { 1 };
    if w.set_dirty[id] == 0 { w.set_since[id] = w.time; }
    if w.set_dirty[id] < lvl { w.set_dirty[id] = lvl; }
}
fn bucket_of(p: V) -> usize {
    let bx = clampf((p[0] + HALF) / 32.0, 0.0, 9.0) as usize;
    let bz = clampf((p[2] + HALF) / 32.0, 0.0, 9.0) as usize;
    bz * 10 + bx
}
fn zone_touch(w: &mut World, x: i32, z: i32) {
    for zn in 0..NZONE {
        if x >= w.zx0[zn] - 2 && x < w.zx0[zn] + ZN as i32 + 2 && z >= w.zz0[zn] - 2 && z < w.zz0[zn] + ZN as i32 + 2 { w.zdirty[zn] = true; }
    }
}

// ---------------------------------------------------------------- shards
fn spawn_shard(w: &mut World, p: V, v: V, size: f32, col: V, m: u8) {
    let k = w.scur;
    w.scur = (k + 1) % MAXSH;
    w.s_on[k] = true; w.s_rest[k] = false; w.s_t[k] = 0.0;
    w.s_pos[k] = p; w.s_vel[k] = v;
    let sp = 4.0 + 10.0 * rnd(w);
    w.s_w[k] = mul(rndv(w), sp);
    let q = [rnd(w) * 2.0 - 1.0, rnd(w) * 2.0 - 1.0, rnd(w) * 2.0 - 1.0, rnd(w) * 2.0 - 1.0];
    let l = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    w.s_q[k] = if l > 1e-4 { [q[0] / l, q[1] / l, q[2] / l, q[3] / l] } else { [0.0, 0.0, 0.0, 1.0] };
    let (a, b, c) = (rnd(w), rnd(w), rnd(w));
    let s = size;
    w.s_scl[k] = match m {
        WOOD => [s * (0.35 + 0.2 * a), s * (0.35 + 0.2 * b), s * (1.8 + 1.2 * c)],
        GLASS => [s * (0.9 + 0.6 * a), s * 0.14, s * (0.7 + 0.6 * c)],
        STEEL => [s * 0.45, s * (0.25 + 0.2 * b), s * (1.4 + 0.8 * c)],
        _ => [s * (0.7 + 0.6 * a), s * (0.5 + 0.5 * b), s * (0.7 + 0.6 * c)],
    };
    let sc = w.s_scl[k];
    w.s_rad[k] = 0.3 * (sc[0] + sc[1] + sc[2]) / 3.0 + 0.01;
    let shade = 0.78 + 0.32 * rnd(w);
    w.s_col[k] = mul(col, shade);
}
fn shards_from(w: &mut World, p: V, i: usize, n: u32, speed: f32, base_v: V) {
    let m = w.mat[i];
    let col = brick_col(w, i);
    for _ in 0..n {
        let d = rndv(w);
        let s = rnd(w);
        let v = add(base_v, mul(add(d, [0.0, 0.5, 0.0]), speed * (0.3 + s)));
        let size = if m == GLASS { 0.08 + 0.1 * rnd(w) } else { 0.09 + 0.17 * rnd(w) };
        let pp = madd(p, d, 0.15);
        spawn_shard(w, pp, v, size, col, m);
    }
}
fn solid_at(w: &World, q: V, r: f32) -> bool {
    if q[1] < 0.0 { return true; }
    let (x, y, z) = world_cell(q);
    if static_at(w, x, y, z) { return true; }
    let (hx, hy, hz) = (floor(q[0] / S) as i32, floor(q[1] / S) as i32, floor(q[2] / S) as i32);
    for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
        let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
        while m != NONE {
            let j = m as usize; m = w.hnext[j];
            if w.mat[j] == 0 { continue; }
            let d = sub(q, w.wpos[j]);
            if dot(d, d) < (RB + r) * (RB + r) { return true; }
        }
    }}}
    false
}
fn shards_step(w: &mut World, dt: f32) {
    for k in 0..SHASH { w.shead[k] = NONE; }
    for k in 0..MAXSH {
        if !w.s_on[k] || !w.s_rest[k] { continue; }
        let p = w.s_pos[k];
        let h = hkey(floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32) & (SHASH - 1);
        w.snext[k] = w.shead[h]; w.shead[h] = k as u32;
    }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_asleep[b] { continue; }
        if dot(w.bd_v[b], w.bd_v[b]) + dot(w.bd_w[b], w.bd_w[b]) < 0.3 { continue; }
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize; k = w.next[i];
            let p = w.wpos[i];
            let (hx, hy, hz) = (floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
            let vb = add(w.bd_v[b], cross(w.bd_w[b], sub(p, w.bd_pos[b])));
            for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                let mut m = w.shead[hkey(hx + dx, hy + dy, hz + dz) & (SHASH - 1)];
                while m != NONE {
                    let s = m as usize; m = w.snext[s];
                    if !w.s_rest[s] { continue; }
                    let d = sub(w.s_pos[s], p);
                    let rr = RB + w.s_rad[s];
                    if dot(d, d) < rr * rr {
                        w.s_rest[s] = false; w.s_t[s] = 0.0;
                        w.s_vel[s] = madd(vb, norm(d), 1.5);
                        let sp = rndv(w);
                        w.s_w[s] = mul(sp, 6.0);
                    }
                }
            }}}
        }
    }
    let chk = (w.frame % 8) as usize;
    for k in 0..MAXSH {
        if !w.s_on[k] { continue; }
        let rr = w.s_rad[k];
        if w.s_rest[k] {
            if k % 8 != chk { continue; }
            let p = w.s_pos[k];
            if !solid_at(w, [p[0], p[1] - rr - 0.08, p[2]], 0.04) { w.s_rest[k] = false; w.s_t[k] = 0.0; }
            continue;
        }
        let mut v = w.s_vel[k];
        v[1] -= GR * dt;
        let pp = w.s_pos[k];
        let mut p = madd(pp, v, dt);
        let mut sup = false;
        if p[1] < rr {
            p[1] = rr;
            if v[1] < 0.0 { v[1] = -v[1] * 0.25; }
            v[0] *= 0.7; v[2] *= 0.7; sup = true;
            w.s_w[k] = mul(w.s_w[k], 0.6);
        }
        let (cx, cy, cz) = world_cell(p);
        if static_at(w, cx, cy, cz) {
            let (px, py, pz) = world_cell(pp);
            if py != cy {
                if py > cy {
                    p[1] = (cy + 1) as f32 * S + rr;
                    if v[1] < 0.0 { v[1] = -v[1] * 0.25; }
                    v[0] *= 0.7; v[2] *= 0.7; sup = true;
                    w.s_w[k] = mul(w.s_w[k], 0.6);
                } else {
                    p[1] = cy as f32 * S - rr;
                    if v[1] > 0.0 { v[1] = -v[1] * 0.2; }
                }
            } else if px != cx {
                let x0 = (cx - (NX / 2) as i32) as f32 * S;
                p[0] = if px > cx { x0 + S + rr } else { x0 - rr };
                v[0] = -v[0] * 0.3;
            } else if pz != cz {
                let z0 = (cz - (NZ / 2) as i32) as f32 * S;
                p[2] = if pz > cz { z0 + S + rr } else { z0 - rr };
                v[2] = -v[2] * 0.3;
            } else {
                let mut y = cy;
                while y < NY as i32 - 1 && static_at(w, cx, y, cz) { y += 1; }
                p[1] = y as f32 * S + rr;
                v = mul(v, 0.3);
            }
        }
        let (hx, hy, hz) = (floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
        for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
            let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
            while m != NONE {
                let j = m as usize; m = w.hnext[j];
                if w.mat[j] == 0 { continue; }
                let d = sub(p, w.wpos[j]);
                let dd = dot(d, d);
                let lim = RB + rr;
                if dd >= lim * lim || dd < 1e-8 { continue; }
                let n = mul(d, 1.0 / sqrt(dd));
                p = madd(w.wpos[j], n, lim);
                let b = w.body[j] as usize;
                let vb = add(w.bd_v[b], cross(w.bd_w[b], sub(w.wpos[j], w.bd_pos[b])));
                let mut rel = sub(v, vb);
                let vn = dot(rel, n);
                if vn < 0.0 { rel = madd(rel, n, -vn * 1.3); rel = mul(rel, 0.85); v = add(vb, rel); }
                if n[1] > 0.5 { sup = true; }
            }
        }}}
        w.s_q[k] = q_int(w.s_q[k], w.s_w[k], dt);
        w.s_w[k] = mul(w.s_w[k], 1.0 - minf(dt * 0.8, 0.5));
        w.s_pos[k] = p; w.s_vel[k] = v;
        if sup && dot(v, v) < 0.2 {
            w.s_t[k] += dt;
            if w.s_t[k] > 0.35 { w.s_rest[k] = true; w.s_vel[k] = [0.0; 3]; w.s_w[k] = [0.0; 3]; }
        } else { w.s_t[k] = 0.0; }
        if p[1] < -5.0 || abs(p[0]) > 200.0 || abs(p[2]) > 200.0 { w.s_on[k] = false; }
    }
}
fn shard_out(w: &mut World) {
    let mut cnt = [0usize; 4];
    for k in 0..MAXSH {
        if !w.s_on[k] { continue; }
        let v = k & 3;
        let slot = v * SHQ + cnt[v];
        cnt[v] += 1;
        let r = q_to_m(w.s_q[k]);
        let s = w.s_scl[k];
        let p = w.s_pos[k];
        let o = slot * 16;
        for c in 0..3 {
            for rr in 0..3 { w.out_sh[o + c * 4 + rr] = r[rr * 3 + c] * s[c]; }
            w.out_sh[o + c * 4 + 3] = 0.0;
        }
        w.out_sh[o + 12] = p[0]; w.out_sh[o + 13] = p[1]; w.out_sh[o + 14] = p[2]; w.out_sh[o + 15] = 1.0;
        let co = slot * 3;
        let c = w.s_col[k];
        w.out_shc[co] = c[0]; w.out_shc[co + 1] = c[1]; w.out_shc[co + 2] = c[2];
    }
    for v in 0..4 { w.ns[v] = cnt[v] as u32; }
}

// ---------------------------------------------------------------- bricks
fn alloc_brick(w: &mut World) -> usize {
    if w.nb < MB { w.nb += 1; return w.nb - 1; }
    for k in 0..MB {
        let i = (w.bcur + k) % MB;
        if w.mat[i] == 0 && w.body[i] == STATIC && w.sid[i] == NOS { w.bcur = (i + 1) % MB; return i; }
    }
    usize::MAX
}
fn add_brick(w: &mut World, x: i32, y: i32, z: i32, m: u8) -> usize {
    if !inb(x, y, z) || gget(w, x, y, z) != 0 { return usize::MAX; }
    let i = alloc_brick(w);
    if i == usize::MAX { return usize::MAX; }
    w.mat[i] = m; w.bx[i] = x as u16; w.by[i] = y as u8; w.bz[i] = z as u16;
    w.sid[i] = w.cur_sid; w.paint[i] = w.cur_paint;
    w.infill[i] = if m == GLASS || w.cur_infill { 1 } else { 0 };
    w.body[i] = STATIC; w.next[i] = NONE;
    gset(w, x, y, z, (i + 1) as u32);
    let t = rnd(w);
    w.tint[i] = 0.97 + 0.06 * t;
    w.phi[i] = 0.0; w.stress[i] = 0.0; w.svis[i] = 0.0; w.ext[i] = 0.0; w.extn[i] = 0.0; w.wload[i] = 0.0;
    w.bond[i] = [0.0; 4]; w.bmax[i] = [0.0; 4];
    w.corner[i] = [[0.0; 3]; 8]; w.omask[i] = 0;
    w.local[i] = lat_pos(x, y, z); w.wpos[i] = w.local[i];
    i
}
fn bond_cap(mi: u8, mj: u8, axis: usize) -> f32 {
    let (a, b) = if axis == 1 { (CAPV[mi as usize], CAPV[mj as usize]) } else { (CAPH[mi as usize], CAPH[mj as usize]) };
    minf(a, b)
}
fn vjit(w: &World, vx: i32, vy: i32, vz: i32) -> V {
    let mut o = [0u8; 8];
    for k in 0..8usize {
        let g = gget(w, vx - 1 + (k & 1) as i32, vy - 1 + ((k >> 1) & 1) as i32, vz - 1 + ((k >> 2) & 1) as i32);
        if g != 0 { let m = w.mat[g as usize - 1]; o[k] = if m >= SLUG { 0 } else { m }; }
    }
    let mut free = [true; 3];
    for k in 0..8usize {
        for a in 0..3 { let bit = 1usize << a; if k & bit == 0 && o[k] != o[k | bit] { free[a] = false; } }
    }
    let h = hash3(vx, vy, vz);
    let amp = 0.17;
    [
        if free[0] { amp * (h01(h, 1) * 2.0 - 1.0) } else { 0.0 },
        if free[1] { amp * (h01(h, 2) * 2.0 - 1.0) } else { 0.0 },
        if free[2] { amp * (h01(h, 3) * 2.0 - 1.0) } else { 0.0 },
    ]
}
fn finalize_brick(w: &mut World, i: usize) {
    if w.mat[i] == 0 { return; }
    for d in 0..6 { let j = nbr_grid(w, i, d); w.nb6[i][d] = if j == usize::MAX { NONE } else { j as u32 }; }
    for a in 0..3 {
        let j = nbr(w, i, a * 2);
        // infill (glazing, stud partitions) stands on the structure but is not built into it:
        // no bond up into a structural brick above, none sideways into structure
        let gap = j != usize::MAX && w.infill[i] != w.infill[j] && (a != 1 || w.infill[i] == 1);
        if j != usize::MAX && !gap && w.mat[j] != 0 && w.body[j] == w.body[i] && w.sid[j] == w.sid[i] {
            let c = bond_cap(w.mat[i], w.mat[j], a);
            w.bond[i][a] = c; w.bmax[i][a] = c;
        } else { w.bond[i][a] = 0.0; w.bmax[i][a] = 0.0; }
    }
    if w.by[i] == 0 && w.body[i] == STATIC {
        let c = CAPV[w.mat[i] as usize] * 2.0;
        w.bond[i][3] = c; w.bmax[i][3] = c;
    } else { w.bond[i][3] = 0.0; w.bmax[i][3] = 0.0; }
    let mut om = 0u8;
    for d in 0..6 { let j = nbr(w, i, d); if j != usize::MAX && w.mat[j] != 0 { om |= 1 << d; } }
    w.omask[i] = om;
    let (x, y, z) = blat(w, i);
    let flat = w.mat[i] >= SLUG;
    for c in 0..8usize {
        let (dx, dy, dz) = ((c & 1) as i32, ((c >> 1) & 1) as i32, ((c >> 2) & 1) as i32);
        let base = [if dx == 1 { H } else { -H }, if dy == 1 { H } else { -H }, if dz == 1 { H } else { -H }];
        let j = if flat { [0.0; 3] } else { vjit(w, x + dx, y + dy, z + dz) };
        w.corner[i][c] = add(base, j);
    }
}
fn st_static_change(w: &mut World, i: usize, urgent: bool) {
    let s = w.sid[i] as usize;
    if s < w.st_n {
        st_activate(w, s);
        let y = w.by[i] as i32;
        mark_brick(w, s, y, urgent);
    }
}
fn remove_brick(w: &mut World, i: usize) {
    if w.mat[i] == 0 { return; }
    let (x, y, z) = blat(w, i);
    if gget(w, x, y, z) as usize == i + 1 { gset(w, x, y, z, 0); }
    let m = w.mat[i];
    w.mat[i] = 0;
    if w.body[i] == STATIC {
        w.salvage += MASS[m as usize];
        let s = w.sid[i] as usize;
        if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; }
        st_static_change(w, i, true);
        if y < 4 { zone_touch(w, x, z); }
    } else { let b = w.body[i] as usize; w.bd_dirty[b] = true; }
}
fn remove_brick_raw(w: &mut World, i: usize) {
    let (x, y, z) = blat(w, i);
    if gget(w, x, y, z) as usize == i + 1 { gset(w, x, y, z, 0); }
    w.mat[i] = 0;
    w.body[i] = STATIC;
}
fn damage_brick(w: &mut World, i: usize, frac: f32, absolute: f32) -> u32 {
    let mut broke = 0;
    for d in 0..6 {
        let j = nbr(w, i, d);
        if j == usize::MAX || w.mat[j] == 0 { continue; }
        let (o, a) = if d & 1 == 0 { (i, d >> 1) } else { (j, d >> 1) };
        if w.bond[o][a] > 0.0 {
            w.bond[o][a] -= w.bmax[o][a] * frac + absolute;
            if w.bond[o][a] <= w.bmax[o][a] * 0.03 { w.bond[o][a] = 0.0; broke += 1; }
        }
    }
    if w.bond[i][3] > 0.0 {
        w.bond[i][3] -= w.bmax[i][3] * frac * 0.5 + absolute * 0.5;
        if w.bond[i][3] <= 0.0 { w.bond[i][3] = 0.0; broke += 1; }
    }
    if w.body[i] == STATIC {
        st_static_change(w, i, false);
        if broke > 0 { let s = w.sid[i] as usize; if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; w.st_broke[s] = true; } }
    } else if broke > 0 { let b = w.body[i] as usize; w.bd_dirty[b] = true; }
    broke
}
fn sever_brick(w: &mut World, i: usize) {
    w.bond[i] = [0.0; 4];
    for d in [1usize, 3, 5] {
        let j = nbr(w, i, d);
        if j != usize::MAX { w.bond[j][d >> 1] = 0.0; }
    }
}

// ---------------------------------------------------------------- bodies
fn alloc_body(w: &mut World) -> usize {
    for k in 0..MAXBODY {
        let b = (w.cursor + k) % MAXBODY;
        if !w.bd_alive[b] {
            w.cursor = (b + 1) % MAXBODY;
            if b + 1 > w.hw { w.hw = b + 1; }
            w.bd_veh[b] = 0; w.bd_bucket[b] = NOB; w.bd_inmesh[b] = false;
            return b;
        }
    }
    usize::MAX
}
fn unbucket(w: &mut World, b: usize) {
    let k = w.bd_bucket[b];
    if k != NOB { mark_set(w, SSETS + k as usize, w.bd_inmesh[b]); w.bd_bucket[b] = NOB; }
    w.bd_inmesh[b] = false;
}
fn wake_body(w: &mut World, b: usize) {
    if w.bd_asleep[b] { w.bd_asleep[b] = false; unbucket(w, b); }
    w.bd_sleep[b] = 0.0;
}
/// tiny debris (a handful of bricks) crumbles to dust 20 s after it broke off, once it has come to rest
/// a resting crumb (a single loose brick) that nothing else lies on
fn is_crumb(w: &World, b: usize, min_age: f32) -> bool {
    if !w.bd_alive[b] || w.bd_veh[b] != 0 || w.bd_count[b] > 1 || !w.bd_asleep[b] { return false; }
    if w.time - w.bd_born[b] < min_age { return false; }
    let f = w.bd_first[b];
    if f == NONE || w.mat[f as usize] == 0 || w.mat[f as usize] >= SLUG { return false; }
    // anything of another body (or the static world) in the cells right above it means it is load-bearing rubble
    let mut k = f;
    while k != NONE {
        let i = k as usize; k = w.next[i];
        let p = w.wpos[i];
        let (cx, cy, cz) = (floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
        for dz in -1..=1 { for dx in -1..=1 { for dy in 1..=2 {
            let (x, y, z) = (cx + dx, cy + dy, cz + dz);
            if inb(x, y, z) && gget(w, x, y, z) != 0 { let j = gget(w, x, y, z) as usize - 1; if w.body[j] as usize != b && w.mat[j] != 0 { return false; } }
            let mut h = w.hhead[hkey(x, y, z)];
            while h != NONE {
                let j = h as usize; h = w.hnext[j];
                if w.body[j] as usize == b || w.mat[j] == 0 { continue; }
                let q = w.wpos[j];
                if floor(q[0] / S) as i32 == x && floor(q[1] / S) as i32 == y && floor(q[2] / S) as i32 == z { return false; }
            }
        }}}
    }
    true
}
fn crumb_to_dust(w: &mut World, b: usize) {
    let f = w.bd_first[b] as usize;
    let p = w.bd_pos[b];
    let c = brick_col(w, f);
    dust_burst(w, p, c, 1, 0.3);
    kill_body(w, b);
}
/// single loose bricks turn to dust 20 s after they broke off; anything bigger, or anything holding debris up, stays for good
fn clear_rubble(w: &mut World) {
    for b in 0..w.hw { if is_crumb(w, b, 20.0) { crumb_to_dust(w, b); } }
}
/// make room in the body pool by dusting the oldest resting crumbs; false if there is no room to be had
fn ensure_bodies(w: &mut World, need: usize) -> bool {
    let mut free = 0usize;
    for b in 0..MAXBODY { if !w.bd_alive[b] { free += 1; if free >= need { return true; } } }
    let mut pass = 0;
    while free < need && pass < 3 {
        let age = [8.0, 3.0, 0.5][pass];
        for b in 0..w.hw { if free >= need { break; } if is_crumb(w, b, age) { crumb_to_dust(w, b); free += 1; } }
        pass += 1;
    }
    free >= need
}
fn kill_body(w: &mut World, b: usize) {
    unbucket(w, b);
    let mut k = w.bd_first[b];
    while k != NONE {
        let ki = k as usize;
        let nx = w.next[ki];
        if w.body[ki] as usize == b {
            let (x, y, z) = blat(w, ki);
            if gget(w, x, y, z) as usize == ki + 1 { gset(w, x, y, z, 0); }
            w.mat[ki] = 0;
            w.body[ki] = STATIC;
        }
        k = nx;
    }
    w.bd_alive[b] = false;
    w.bd_first[b] = NONE;
    w.bd_veh[b] = 0;
}
fn setup_body(w: &mut World, b: usize, s: usize, e: usize, ppos: V, prot: M, pq: [f32; 4], pv: V, pw: V) {
    w.bd_born[b] = w.time; w.bd_lt[b] = 0.0; w.bd_resid[b] = 1.0; w.bd_maxr[b] = 0.0;
    let mut mass = 0.0;
    let mut cl = [0.0; 3];
    for t in s..e {
        let i = w.lst2[t] as usize;
        let m = MASS[w.mat[i] as usize];
        mass += m;
        cl = madd(cl, w.local[i], m);
    }
    cl = mul(cl, 1.0 / mass);
    let mut it = [0.0f32; 9];
    let mut rad = 0.0f32;
    let cube = S * S / 6.0;
    for t in s..e {
        let i = w.lst2[t] as usize;
        let m = MASS[w.mat[i] as usize];
        let r = sub(w.local[i], cl);
        w.local[i] = r;
        let rr = dot(r, r);
        rad = maxf(rad, rr);
        it[0] += m * (rr - r[0] * r[0] + cube); it[4] += m * (rr - r[1] * r[1] + cube); it[8] += m * (rr - r[2] * r[2] + cube);
        it[1] -= m * r[0] * r[1]; it[2] -= m * r[0] * r[2]; it[5] -= m * r[1] * r[2];
        w.body[i] = b as u16;
        w.next[i] = if t + 1 < e { w.lst2[t + 1] } else { NONE };
    }
    it[3] = it[1]; it[6] = it[2]; it[7] = it[5];
    let iil = minv(&it);
    let off = mv(&prot, cl);
    w.bd_alive[b] = true;
    w.bd_first[b] = w.lst2[s];
    w.bd_count[b] = (e - s) as u32;
    w.bd_pos[b] = add(ppos, off);
    w.bd_q[b] = pq; w.bd_rot[b] = prot;
    w.bd_v[b] = add(pv, cross(pw, off));
    w.bd_w[b] = pw;
    w.bd_vlast[b] = w.bd_v[b]; w.bd_wlast[b] = pw; w.bd_nmax[b] = 1.0;
    w.bd_m[b] = mass; w.bd_im[b] = 1.0 / mass;
    w.bd_iil[b] = iil;
    w.bd_iiw[b] = mmul(&mmul(&prot, &iil), &mt(&prot));
    w.bd_sleep[b] = 0.0; w.bd_asleep[b] = false; w.bd_dirty[b] = false; w.bd_wake[b] = false; w.bd_cool[b] = 0.15;
    w.bd_bucket[b] = NOB; w.bd_inmesh[b] = false;
    w.bd_rad[b] = sqrt(rad) + H * 1.8;
    for t in s..e {
        let i = w.lst2[t] as usize;
        w.wpos[i] = add(w.bd_pos[b], mv(&prot, w.local[i]));
    }
}
fn impulse_at(w: &mut World, b: usize, p: V, j: V) {
    w.bd_v[b] = madd(w.bd_v[b], j, w.bd_im[b]);
    let r = sub(p, w.bd_pos[b]);
    let dw = mv(&w.bd_iiw[b], cross(r, j));
    w.bd_w[b] = add(w.bd_w[b], dw);
}
fn apply_blast_to_body(w: &mut World, b: usize) {
    if w.blast_t <= 0.0 { return; }
    let mut k = w.bd_first[b];
    while k != NONE {
        let i = k as usize;
        let p = w.wpos[i];
        let d = sub(p, w.blast_c);
        let dist = len(d);
        if dist < w.blast_r {
            let f = w.blast_p * 1600.0 * (1.0 - dist / w.blast_r);
            let dir = if dist > 1e-3 { mul(d, 1.0 / dist) } else { [0.0, 1.0, 0.0] };
            let dir = add(dir, [0.0, 0.35, 0.0]);
            let mm = maxf(MASS[w.mat[i] as usize], 25.0);
            impulse_at(w, b, p, mul(dir, f * mm / 300.0));
        }
        k = w.next[i];
    }
    clamp_vel(w, b);
}
fn clamp_vel(w: &mut World, b: usize) {
    let v = len(w.bd_v[b]);
    if v > 60.0 { w.bd_v[b] = mul(w.bd_v[b], 60.0 / v); }
    // tumbling debris: rim speed of a spinning piece stays below ~16 m/s (a 2 m chunk at most ~8 rev/s)
    let a = len(w.bd_w[b]);
    let amax = minf(25.0, 16.0 / maxf(w.bd_rad[b], 0.3));
    if a > amax { w.bd_w[b] = mul(w.bd_w[b], amax / a); }
}
fn wake_near(w: &mut World, p: V, r: f32) {
    for b in 0..w.hw {
        if w.bd_alive[b] && w.bd_asleep[b] && len(sub(w.bd_pos[b], p)) < r + w.bd_rad[b] { wake_body(w, b); }
    }
}
fn rebuild_body(w: &mut World, id: usize) {
    w.bd_dirty[id] = false;
    unbucket(w, id);
    let mut n = 0usize;
    let mut k = w.bd_first[id];
    while k != NONE {
        let ki = k as usize;
        let nx = w.next[ki];
        if w.body[ki] as usize == id {
            if w.mat[ki] != 0 { w.lst[n] = k; n += 1; } else { w.body[ki] = STATIC; }
        }
        k = nx;
    }
    if n == 0 { w.bd_alive[id] = false; w.bd_first[id] = NONE; w.bd_veh[id] = 0; return; }
    w.stamp_id += 1;
    let sid = w.stamp_id;
    let mut ncomp = 0u32;
    for t in 0..n {
        let s = w.lst[t] as usize;
        if w.stamp[s] == sid { continue; }
        w.stamp[s] = sid; w.comp[s] = ncomp;
        let mut qh = 0; let mut qt = 0;
        w.q[qt] = s as u32; qt += 1;
        while qh < qt {
            let i = w.q[qh] as usize; qh += 1;
            for d in 0..6 {
                let (j, _) = linked(w, i, d);
                if j != usize::MAX && w.stamp[j] != sid { w.stamp[j] = sid; w.comp[j] = ncomp; w.q[qt] = j as u32; qt += 1; }
            }
        }
        ncomp += 1;
    }
    let mut nc = ncomp as usize;
    // no room for the new pieces: keep the body whole for now instead of deleting anything
    if nc > 1 && !ensure_bodies(w, nc - 1) { for t in 0..n { w.comp[w.lst[t] as usize] = 0; } nc = 1; }
    for c in 0..=nc { w.cnt[c] = 0; }
    for t in 0..n { let c = w.comp[w.lst[t] as usize] as usize; w.cnt[c + 1] += 1; }
    for c in 0..nc { w.cnt[c + 1] += w.cnt[c]; }
    for t in 0..n {
        let i = w.lst[t] as usize;
        let c = w.comp[i] as usize;
        let at = w.cnt[c] as usize;
        w.lst2[at] = i as u32;
        w.cnt[c] += 1;
    }
    let (ppos, prot, pq, pv, pw) = (w.bd_pos[id], w.bd_rot[id], w.bd_q[id], w.bd_v[id], w.bd_w[id]);
    let veh = w.bd_veh[id];
    for c in 0..nc {
        let s = if c == 0 { 0 } else { w.cnt[c - 1] as usize };
        let e = w.cnt[c] as usize;
        // in a big collapse, lone bricks knocked off a crushing piece are mostly small fragments: they go to the
        // particle debris instead of each becoming a rigid body (keeps thousands of crumbs from swamping the solver)
        if c > 0 && e - s == 1 && w.stats[3] > 1500.0 {
            let i = w.lst2[s] as usize;
            let m = w.mat[i];
            if m != STEEL && rnd(w) < minf((w.stats[3] - 1500.0) / 2000.0, 0.85) {
                let p = w.wpos[i];
                shards_from(w, p, i, 2, 1.5, pv);
                w.salvage += MASS[m as usize];
                remove_brick_raw(w, i);
                continue;
            }
        }
        let b = if c == 0 { id } else { alloc_body(w) };
        if b == usize::MAX {
            for t in s..e { let i = w.lst2[t] as usize; let p = w.wpos[i]; shards_from(w, p, i, 2, 1.0, [0.0; 3]); remove_brick_raw(w, i); }
            continue;
        }
        let (keep_cool, keep_lt) = (w.bd_cool[id], w.bd_lt[id]);
        setup_body(w, b, s, e, ppos, prot, pq, pv, pw);
        if c == 0 { w.bd_veh[b] = veh; w.bd_cool[b] = keep_cool; w.bd_lt[b] = keep_lt; }
    }
}

// ---------------------------------------------------------------- structures
fn sor_brick(w: &mut World, i: usize, omega: f32, fr: u32) -> f32 {
    let mi = w.mat[i];
    let mut sk = 0.0; let mut skp = 0.0;
    for d in 0..6 {
        let (j, _) = linked(w, i, d);
        if j == usize::MAX { continue; }
        let k = cond_w(w, i, j, d >> 1);
        sk += k; skp += k * w.phi[j];
    }
    // the ground under a footing is far stiffer than a brick-to-brick joint: load goes straight down instead of spreading along the base course
    if w.body[i] == STATIC { if w.by[i] == 0 && w.bond[i][3] > 0.0 { sk += GROUNDK * STIFF[mi as usize]; } }
    // a resting contact counts as a support for a moment after it was last seen: contacts flicker from substep to
    // substep, and a load field that loses its supports every frame never settles
    else if fr.wrapping_sub(w.sup[i]) < 10 { sk += GROUNDK * STIFF[mi as usize]; }
    if sk > 0.0 {
        let target = (MASS[mi as usize] + w.ext[i] + skp) / sk;
        let d = omega * (target - w.phi[i]);
        w.phi[i] += d;
        return abs(d);
    }
    0.0
}
#[inline] fn in_struct(w: &World, i: usize, s: usize) -> bool { w.sid[i] as usize == s && w.mat[i] != 0 && w.mat[i] < SLUG && w.body[i] == STATIC }
/// one SOR sweep; records the largest update relative to the potential scale (convergence measure)
fn sor_struct(w: &mut World, s: usize, omega: f32) {
    let fr = w.frame;
    let mut md = 0.0f32; let mut mp = 1.0f32;
    for i in w.st_lo[s]..w.st_hi[s] {
        if in_struct(w, i, s) {
            let d = sor_brick(w, i, omega, fr);
            if d > md { md = d; }
            if w.phi[i] > mp { mp = w.phi[i]; }
        }
    }
    w.st_resid[s] = md / mp;
}
#[inline] fn body_loaded(w: &World, b: usize) -> bool {
    w.bd_alive[b] && w.bd_count[b] >= 8 && !w.bd_asleep[b] && w.bd_nc[b] > 0 && w.bd_veh[b] == 0 && w.mat[w.bd_first[b] as usize] < SLUG
}
/// one sweep over a debris piece; returns the largest update relative to its potential scale (convergence measure)
fn sor_body(w: &mut World, b: usize, omega: f32) -> f32 {
    let fr = w.frame;
    let mut md = 0.0f32; let mut mp = 1.0f32;
    let mut k = w.bd_first[b];
    while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 { let d = sor_brick(w, i, omega, fr); if d > md { md = d; } if w.phi[i] > mp { mp = w.phi[i]; } } }
    md / mp
}
#[allow(dead_code)]
fn sor_bodies(w: &mut World, omega: f32) {
    let fr = w.frame;
    for b in 0..w.hw {
        if !body_loaded(w, b) { continue; }
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 { let _ = sor_brick(w, i, omega, fr); } }
    }
}
fn warm_struct(w: &mut World, s: usize) {
    if w.st_solved[s] { return; }
    w.st_solved[s] = true;
    w.st_warm[s] = 560;
    w.st_settle[s] = 1e9;
}
/// spread load-path warm-up sweeps over frames (budget in brick-updates)
fn warm_step(w: &mut World) {
    let mut budget: i64 = 120_000;
    for s in 0..w.st_n {
        while w.st_warm[s] > 0 && budget > 0 {
            let om = if w.st_warm[s] > 60 { 1.9 } else { 1.0 };
            sor_struct(w, s, om);
            w.st_warm[s] -= 1;
            budget -= (w.st_hi[s] - w.st_lo[s]) as i64;
            if w.st_warm[s] == 0 { w.st_settle[s] = 0.3; }
        }
    }
}
fn st_activate(w: &mut World, s: usize) {
    if s >= w.st_n { return; }
    if !w.st_solved[s] { warm_struct(w, s); }
    if w.st_active[s] < 4.0 { w.st_active[s] = 4.0; }
}
fn conn_struct(w: &mut World, s: usize) {
    w.stamp_id += 1;
    let sid = w.stamp_id;
    let mut qh = 0; let mut qt = 0;
    for i in w.st_lo[s]..w.st_hi[s] {
        if in_struct(w, i, s) && w.by[i] == 0 && w.bond[i][3] > 0.0 { w.stamp[i] = sid; w.q[qt] = i as u32; qt += 1; }
    }
    while qh < qt {
        let i = w.q[qh] as usize; qh += 1;
        for d in 0..6 {
            let (j, _) = linked(w, i, d);
            if j != usize::MAX && w.stamp[j] != sid { w.stamp[j] = sid; w.q[qt] = j as u32; qt += 1; }
        }
    }
    let mut detached = 0u32;
    let mut retry = false;
    for st in w.st_lo[s]..w.st_hi[s] {
        if !in_struct(w, st, s) || w.stamp[st] == sid { continue; }
        let mut n = 0;
        w.stamp[st] = sid; w.lst2[n] = st as u32; n += 1;
        let mut h = 0;
        while h < n {
            let i = w.lst2[h] as usize; h += 1;
            for d in 0..6 {
                let (j, _) = linked(w, i, d);
                if j != usize::MAX && w.stamp[j] != sid { w.stamp[j] = sid; w.lst2[n] = j as u32; n += 1; }
            }
        }
        for t in 0..n { let i = w.lst2[t] as usize; w.salvage += MASS[w.mat[i] as usize]; }
        if !ensure_bodies(w, 1) { retry = true; continue; }
        let b = alloc_body(w);
        if b == usize::MAX { retry = true; continue; }
        for t in 0..n { let i = w.lst2[t] as usize; let (x, y, z) = blat(w, i); w.local[i] = lat_pos(x, y, z); w.phi[i] = 0.0; w.ext[i] = 0.0; w.extn[i] = 0.0; w.wload[i] = 0.0; }
        setup_body(w, b, 0, n, [0.0; 3], IDM, [0.0, 0.0, 0.0, 1.0], [0.0; 3], [0.0; 3]);
        apply_blast_to_body(w, b);
        let (p, r) = (w.bd_pos[b], w.bd_rad[b]);
        wake_near(w, p, r + 1.5);
        detached += n as u32;
        if n > 20 { event(w, 2.0, p, n as f32); panic_from(w, p, 10.0 + r); }
    }
    if retry { w.st_conn[s] = true; w.conn_any = true; }
    let mut cnt = 0u32; let mut top = 0i32;
    for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) { cnt += 1; top = maxi(top, w.by[i] as i32 + 1); } }
    w.st_now[s] = cnt; w.st_top[s] = top;
    if detached > 0 {
        w.boost = 50;
        mark_st(w, s, true);
        let (a, b) = (w.st_min[s], w.st_max[s]);
        let mut x = a[0]; while x <= b[0] { let mut z = a[2]; while z <= b[2] { zone_touch(w, x, z); z += 20; } x += 20; }
        zone_touch(w, b[0], b[2]);
    }
}
fn connectivity(w: &mut World) {
    if !w.conn_any { return; }
    w.conn_any = false;
    for s in 0..w.st_n { if w.st_conn[s] { w.st_conn[s] = false; conn_struct(w, s); } }
}
fn cond(mi: u8, mj: u8, axis: usize) -> f32 { minf(STIFF[mi as usize], STIFF[mj as usize]) * if axis == 1 { 1.0 } else { 0.3 } }
/// world-vertical component of a body's lattice axis (a tumbling piece's "vertical" bonds may now lie flat)
#[inline] fn axis_up(w: &World, i: usize, a: usize) -> f32 { if w.body[i] == STATIC { if a == 1 { 1.0 } else { 0.0 } } else { w.bd_rot[w.body[i] as usize][3 + a] } }
/// conductance of a bond as it is oriented right now: bearing (vertical) joints carry load more stiffly than shear ones
#[inline] fn cond_w(w: &World, i: usize, j: usize, a: usize) -> f32 {
    let k = minf(STIFF[w.mat[i] as usize], STIFF[w.mat[j] as usize]);
    if w.body[i] == STATIC { return k * if a == 1 { 1.0 } else { 0.3 }; }
    let u = abs(axis_up(w, i, a));
    k * (0.3 + 0.7 * u)
}
/// stress and fatigue for the three positive-axis bonds (and foundation) of brick i
fn stress_brick(w: &mut World, i: usize, dt: f32, live: bool) -> f32 {
    let mi = w.mat[i];
    let dynb = w.body[i] != STATIC;
    let s = w.sid[i] as usize;
    let mut maxr = 0.0f32;
    for a in 0..3 {
        let (j, hp) = linked(w, i, a * 2);
        if j == usize::MAX { continue; }
        let f = cond_w(w, i, j, a) * (w.phi[i] - w.phi[j]);
        // load flowing upward through a bond = the lower brick hangs from the upper one: tension, much weaker for masonry.
        // For a tumbling piece the bond's present orientation decides how much of it is bearing, shear or tension.
        let u = axis_up(w, i, a);
        let up = if f > 0.0 { maxf(u, 0.0) } else { maxf(-u, 0.0) };
        let tens = minf(TENS[mi as usize], TENS[w.mat[j] as usize]);
        let capm = 1.0 - (1.0 - tens) * up;
        let hpw = if dynb {
            let (cv, ch) = (minf(CAPV[mi as usize], CAPV[w.mat[j] as usize]), minf(CAPH[mi as usize], CAPH[w.mat[j] as usize]));
            let lat = if a == 1 { cv } else { ch };
            let au = abs(u);
            hp * (cv * au + ch * (1.0 - au)) / maxf(lat, 1.0)
        } else { hp };
        let r = abs(f) / (hpw * capm);
        if r > w.stress[i] { w.stress[i] = r; }
        if r > w.stress[j] { w.stress[j] = r; }
        if r > maxr { maxr = r; }
        if live && r > 1.0 {
            // time to failure falls with the overload: a joint at 1.2x its strength creaks for seconds, one at
            // 20x gives way within a frame or two (judged only on a settled load field, so this is a real overload)
            // Standing structures have reserve (rebar, redundancy, arching) and creak before they go; loose pieces under
            // impact fail at their material's pace: masonry and glass snap, concrete and timber hang on for a moment.
            let nhp = if dynb { hp - hp * dt * DUCT[mi as usize] * minf(r - 0.85, 40.0) } else { hp - hp * dt * 2.2 * minf(r - 0.85, 2.5) };
            if nhp < w.bmax[i][a] * 0.08 {
                // A bearing joint failing in compression crushes material - the weaker brick turns to rubble and
                // dust and the load above sinks onto what is below. Only tension and shear failures open a crack
                // (two pieces that can separate). Without this, a broken column would still stand as a stack of
                // loose bricks and hold the building up.
                if up == 0.0 && abs(u) > 0.5 && w.crush_n < 300 {
                    w.crush_n += 1;
                    let k = if CAPV[w.mat[j] as usize] < CAPV[mi as usize] { j } else { i };
                    let p = bpos(w, k); let c = brick_col(w, k);
                    dust_burst(w, p, c, 2, 1.0);
                    shards_from(w, p, k, 2, 1.5, if dynb { mul(w.bd_v[w.body[i] as usize], 0.5) } else { [0.0; 3] });
                    w.salvage += MASS[w.mat[k] as usize];
                    remove_brick(w, k);
                    w.breaks += 1; w.break_c = p;
                    return maxr.max(r);
                }
                w.bond[i][a] = 0.0;
                if dynb { let b = w.body[i] as usize; w.bd_dirty[b] = true; }
                else if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; w.st_broke[s] = true; }
                w.boost = maxi(w.boost, 30);
                let p = mul(add(bpos(w, i), bpos(w, j)), 0.5);
                let c = brick_col(w, i);
                dust_burst(w, p, c, 3, 1.2);
                shards_from(w, p, i, 2, 2.0, [0.0; 3]);
                w.breaks += 1; w.break_c = p;
            } else {
                w.bond[i][a] = nhp;
                if !dynb && s < w.st_n && w.frame % 12 == 0 { let y = w.by[i] as i32; mark_brick(w, s, y, false); }
                if rnd(w) < 0.02 { let p = bpos(w, i); let c = brick_col(w, i); dust_burst(w, p, c, 1, 0.4); }
            }
        }
    }
    if !dynb && w.by[i] == 0 && w.bond[i][3] > 0.0 {
        let f = GROUNDK * STIFF[mi as usize] * w.phi[i];
        let r = f / w.bond[i][3];
        if r > w.stress[i] { w.stress[i] = r; }
        if r > maxr { maxr = r; }
        if live && r > 1.0 {
            w.bond[i][3] -= w.bond[i][3] * dt * 2.2 * (r - 0.85);
            if w.bond[i][3] < w.bmax[i][3] * 0.08 { w.bond[i][3] = 0.0; if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; w.st_broke[s] = true; } }
        }
    }
    maxr
}
fn structural(w: &mut World, dt: f32) {
    let mut act = [0u8; MAXST];
    let mut na = 0;
    for s in 0..w.st_n { if w.st_active[s] > 0.0 && w.st_solved[s] { act[na] = s as u8; na += 1; } }
    // loads from tyres and resting rubble
    for t in 0..na { let s = act[t] as usize; for i in w.st_lo[s]..w.st_hi[s] { w.extn[i] = w.wload[i]; w.wload[i] = 0.0; } }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        let mut ns = 0u32; let mut nground = 0u32;
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize;
            let p = w.wpos[i];
            if p[1] < RB + 0.05 { nground += 1; }
            let (cx, cy, cz) = world_cell([p[0], p[1] - RB - 0.02, p[2]]);
            let j = static_brick(w, cx, cy, cz);
            if j != usize::MAX && (w.sid[j] as usize) < w.st_n && w.st_active[w.sid[j] as usize] > 0.0 { ns += 1; }
            k = w.next[i];
        }
        if ns == 0 { continue; }
        let share = w.bd_m[b] / (ns + nground) as f32;
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize;
            let p = w.wpos[i];
            let (cx, cy, cz) = world_cell([p[0], p[1] - RB - 0.02, p[2]]);
            let j = static_brick(w, cx, cy, cz);
            if j != usize::MAX && (w.sid[j] as usize) < w.st_n && w.st_active[w.sid[j] as usize] > 0.0 { w.extn[j] += share; }
            k = w.next[i];
        }
    }
    for t in 0..na { let s = act[t] as usize; for i in w.st_lo[s]..w.st_hi[s] { w.ext[i] = w.ext[i] * 0.8 + w.extn[i] * 0.2; } }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_count[b] < 8 || w.bd_veh[b] != 0 { continue; }
        let mut any = false;
        let mut k = w.bd_first[b];
        while k != NONE { if w.frame.wrapping_sub(w.sup[k as usize]) < 10 { any = true; break; } k = w.next[k as usize]; }
        if !any { let mut k = w.bd_first[b]; while k != NONE { w.phi[k as usize] = 0.0; k = w.next[k as usize]; } }
    }
    if w.boost > 0 { w.boost -= 1; }
    let mut budget: i64 = 160_000;
    for t in 0..na {
        let s = act[t] as usize;
        let size = (w.st_hi[s] - w.st_lo[s]) as i64;
        // big buildings: fewer forced sweeps once the field has converged, the rest of the budget carries on next frame
        let nmin = if w.st_resid[s] < 1e-4 { if (w.frame as usize + s) % 3 == 0 { 1 } else { 0 } } else if size > 15_000 { 1 } else { 6 };
        let mut n = 0;
        if nmin == 0 { continue; }
        loop {
            sor_struct(w, s, 1.85);
            n += 1; budget -= size;
            if n >= nmin && (w.st_resid[s] < 4e-4 || n >= 90 || budget <= 0) { break; }
        }
    }
    // debris pieces get a converged load field too (adaptive sweeps from a shared budget): a section of building that
    // lands on rubble must feel its full weight on the bricks it rests on, or it can never crush
    // Inertial loading. A piece's bricks are loaded by the effective gravity they feel, g - a: nothing in free fall,
    // their weight when resting, and many times their weight in the instant a falling piece is stopped. That is
    // what snaps a tower when it hits the ground (the part that stops is hit by the momentum of the rest), so
    // pieces break on impact where the dynamic load exceeds their strength, and stay whole when merely lying.
    let idt = 1.0 / maxf(dt, 1e-3);
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        let al = mul(sub(w.bd_v[b], w.bd_vlast[b]), idt);
        let aa = mul(sub(w.bd_w[b], w.bd_wlast[b]), idt);
        w.bd_vlast[b] = w.bd_v[b]; w.bd_wlast[b] = w.bd_w[b];
        if !body_loaded(w, b) { continue; }
        let pb = w.bd_pos[b];
        let mut nm = 0.0f32;
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize; k = w.next[i];
            if w.mat[i] == 0 { continue; }
            let a = add(al, cross(aa, sub(w.wpos[i], pb)));
            let ge = [-a[0], -GR - a[1], -a[2]];
            let n = minf(len(ge) / GR, 150.0);
            if n > nm { nm = n; }
            w.ext[i] = MASS[w.mat[i] as usize] * (n - 1.0);
        }
        w.bd_nmax[b] = nm;
    }
    let mut bbudget: i64 = 160_000;
    for b in 0..w.hw {
        if !body_loaded(w, b) || w.bd_count[b] > 60000 { continue; }
        // a settled piece only needs a sweep now and then
        if w.bd_resid[b] < 2e-4 && w.bd_nmax[b] < 1.5 && (w.frame as usize + b) % 4 != 0 { continue; }
        let size = w.bd_count[b] as i64;
        let mut n = 0;
        loop {
            let r = sor_body(w, b, 1.8);
            w.bd_resid[b] = r;
            n += 1; bbudget -= size;
            if n >= 2 && (r < 4e-4 || n >= 60 || bbudget <= 0) { break; }
        }
    }
    let mut gmax = 0.0f32;
    for t in 0..na {
        let s = act[t] as usize;
        for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) { w.stress[i] = 0.0; } }
        // judge bonds only on a settled load field: transient potentials after a change are not real loads
        let live = w.st_settle[s] <= 0.0 && w.st_resid[s] < 1.5e-3;
        let mut mr = 0.0f32;
        for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) { let r = stress_brick(w, i, dt, live); if r > mr { mr = r; } } }
        w.st_maxr[s] = mr;
        if mr > gmax { gmax = mr; }
        for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) { w.svis[i] = w.svis[i] * 0.85 + w.stress[i] * 0.15; } }
        if w.st_settle[s] > 0.0 && w.st_warm[s] == 0 { w.st_settle[s] -= dt; }
        if w.st_warm[s] == 0 { w.st_active[s] -= dt; }
        if w.st_broke[s] { w.st_broke[s] = false; w.st_active[s] = 4.0; }
        if mr > 0.98 && w.st_active[s] < 1.0 { w.st_active[s] = 1.0; }
        if w.stress_view && w.frame % 6 == 0 { mark_st(w, s, false); }
    }
    for b in 0..w.hw {
        if !body_loaded(w, b) || w.bd_count[b] > 60000 { continue; }
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 { w.stress[i] = 0.0; } }
        // like standing structures, a debris piece is only judged on a settled load field: its potentials need a
        // few dozen sweeps after it lands (more for a big piece) before they mean anything
        w.bd_lt[b] += dt;
        let settled = w.bd_resid[b] < 2e-4 && w.bd_nmax[b] < 1.5;
        if settled && (w.frame as usize + b) % 4 != 0 { continue; }
        let dt = if settled { dt * 4.0 } else { dt };
        // a debris field starts from zero potential, so before it settles it under- rather than over-states loads:
        // judging it a little early is on the safe side
        // judged on a settled field, or at once during a hard impact (the impact loads are real, and the field
        // starts from zero so it can only understate them)
        let live = (w.bd_lt[b] > 0.3 && (w.bd_resid[b] < 1e-2 || w.bd_lt[b] > 0.6)) || w.bd_nmax[b] > 2.5;
        let mut bm = 0.0f32;
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 { let r = stress_brick(w, i, dt, live); if r > bm { bm = r; } } }
        w.bd_maxr[b] = bm;
        if bm > gmax { gmax = bm; }
    }
    // contacts flicker; only a piece that has really been in the air for a while starts its load field over
    for b in 0..w.hw { if w.bd_alive[b] && !body_loaded(w, b) { w.bd_lt[b] = maxf(w.bd_lt[b] - 2.0 * dt, 0.0); } }
    w.stats[6] = gmax;
    w.stats[13] = na as f32;
}

// ---------------------------------------------------------------- rigid-body dynamics
fn update_cache(w: &mut World) {
    for b in 0..w.hw {
        if !w.bd_alive[b] || (w.bd_asleep[b] && w.bd_veh[b] == 0) { continue; }
        let r = q_to_m(w.bd_q[b]);
        w.bd_rot[b] = r;
        w.bd_iiw[b] = mmul(&mmul(&r, &w.bd_iil[b]), &mt(&r));
        let pos = w.bd_pos[b];
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; w.wpos[i] = add(pos, mv(&r, w.local[i])); k = w.next[i]; }
    }
}
#[inline] fn hkey(x: i32, y: i32, z: i32) -> usize {
    ((x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)) as u32 as usize) & (HSIZE - 1)
}
fn build_hash(w: &mut World) {
    for k in 0..HSIZE { w.hhead[k] = NONE; }
    for b in 0..w.hw {
        if !w.bd_alive[b] { continue; }
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize;
            let p = w.wpos[i];
            let h = hkey(floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
            w.hnext[i] = w.hhead[h]; w.hhead[h] = i as u32;
            k = w.next[i];
        }
    }
}
fn push_contact(w: &mut World, a: usize, b: u16, bo: u16, ba: usize, bb: u32, p: V, n: V, depth: f32) {
    if w.nc >= MAXC { return; }
    let c = &mut w.cs[w.nc];
    c.a = a as u16; c.b = b; c.bo = bo; c.ba = ba as u32; c.bb = bb; c.p = p; c.n = n; c.depth = depth; c.on = true;
    c.jn = 0.0; c.jt1 = 0.0; c.jt2 = 0.0; c.vn0 = 0.0;
    w.nc += 1;
    w.bd_nc[a] += 1;
    if b != STATIC { w.bd_nc[b as usize] += 1; }
}
fn gen_contacts(w: &mut World) {
    w.nc = 0;
    for b in 0..w.hw { w.bd_nc[b] = 0; w.bd_j[b] = 0.0; w.bd_crush[b] = 0; w.bd_jmax[b] = 0.0; }
    for a in 0..w.hw {
        if !w.bd_alive[a] || w.bd_asleep[a] { continue; }
        let rot = w.bd_rot[a];
        let speed = len(w.bd_v[a]) + len(w.bd_w[a]) * w.bd_rad[a] * 0.5;
        let mut k = w.bd_first[a];
        while k != NONE {
            let i = k as usize;
            let p = w.wpos[i];
            if p[1] < 0.5 {
                for c in 0..8 {
                    let l = [if c & 1 == 0 { -H } else { H }, if c & 2 == 0 { -H } else { H }, if c & 4 == 0 { -H } else { H }];
                    let cp = add(p, mv(&rot, l));
                    if cp[1] < 0.0 { push_contact(w, a, STATIC, STATIC, i, NONE, cp, [0.0, 1.0, 0.0], -cp[1]); w.sup[i] = w.frame; }
                }
            }
            let (cx, cy, cz) = world_cell(p);
            for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                let j = static_brick(w, x, y, z);
                if j == usize::MAX { continue; }
                let c = lat_pos(x, y, z);
                let q = [clampf(p[0], c[0] - H, c[0] + H), clampf(p[1], c[1] - H, c[1] + H), clampf(p[2], c[2] - H, c[2] + H)];
                let d = sub(p, q);
                let dist = len(d);
                if dist >= RB { continue; }
                if dist > 1e-4 {
                    if d[1] > 0.3 * dist { w.sup[i] = w.frame; }
                    push_contact(w, a, STATIC, STATIC, i, j as u32, q, mul(d, 1.0 / dist), RB - dist);
                } else {
                    let r = sub(p, c);
                    let mut ax = 0; for t in 1..3 { if abs(r[t]) > abs(r[ax]) { ax = t; } }
                    let mut n = [0.0; 3]; n[ax] = if r[ax] >= 0.0 { 1.0 } else { -1.0 };
                    push_contact(w, a, STATIC, STATIC, i, j as u32, p, n, H + RB - abs(r[ax]));
                }
            }}}
            let (hx, hy, hz) = (floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
            for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
                while m != NONE {
                    let j = m as usize;
                    m = w.hnext[j];
                    let bj = w.body[j] as usize;
                    if bj == a || w.mat[j] == 0 { continue; }
                    let other_awake = !w.bd_asleep[bj];
                    if other_awake && j < i { continue; }
                    let d = sub(p, w.wpos[j]);
                    let dd = dot(d, d);
                    if dd >= 4.0 * RB * RB || dd < 1e-10 { continue; }
                    let dist = sqrt(dd);
                    let n = mul(d, 1.0 / dist);
                    let cp = madd(w.wpos[j], n, dist * 0.5);
                    if n[1] > 0.3 { w.sup[i] = w.frame; } else if n[1] < -0.3 { w.sup[j] = w.frame; }
                    if other_awake {
                        push_contact(w, a, bj as u16, bj as u16, i, j as u32, cp, n, 2.0 * RB - dist);
                    } else {
                        if speed > 0.7 { w.bd_wake[bj] = true; }
                        push_contact(w, a, STATIC, bj as u16, i, j as u32, cp, n, 2.0 * RB - dist);
                    }
                }
            }}}
            k = w.next[i];
        }
    }
}
#[inline] fn vel_at(w: &World, b: u16, r: V) -> V {
    if b == STATIC { return [0.0; 3]; }
    let b = b as usize;
    add(w.bd_v[b], cross(w.bd_w[b], r))
}
fn eff_mass(w: &World, a: usize, b: u16, ra: V, rb: V, n: V) -> f32 {
    let mut k = w.bd_im[a] + dot(n, cross(mv(&w.bd_iiw[a], cross(ra, n)), ra));
    if b != STATIC { let b = b as usize; k += w.bd_im[b] + dot(n, cross(mv(&w.bd_iiw[b], cross(rb, n)), rb)); }
    if k > 1e-12 { 1.0 / k } else { 0.0 }
}
fn apply_imp(w: &mut World, a: usize, b: u16, ra: V, rb: V, p: V) {
    w.bd_v[a] = madd(w.bd_v[a], p, w.bd_im[a]);
    let dw = mv(&w.bd_iiw[a], cross(ra, p));
    w.bd_w[a] = add(w.bd_w[a], dw);
    if b != STATIC {
        let b = b as usize;
        w.bd_v[b] = madd(w.bd_v[b], p, -w.bd_im[b]);
        let dw = mv(&w.bd_iiw[b], cross(rb, p));
        w.bd_w[b] = sub(w.bd_w[b], dw);
    }
}
fn plough(w: &mut World) {
    for k in 0..w.nc {
        let c = w.cs[k];
        if c.b != STATIC || c.bb == NONE || c.bo != STATIC { continue; }
        let j = c.bb as usize;
        if w.mat[j] == 0 || w.body[j] != STATIC { w.cs[k].on = false; continue; }
        let a = c.a as usize;
        let isveh = w.bd_veh[a] != 0;
        let ra = sub(c.p, w.bd_pos[a]);
        let vn = -dot(vel_at(w, c.a, ra), c.n);
        if vn < 2.0 { continue; }
        let mom = if isveh { w.bd_m[a] * vn * 0.25 } else {
            let meff = w.bd_m[a] * 3.0 / maxf(w.bd_nc[a] as f32, 3.0);
            minf(meff, w.bd_m[a]) * vn
        };
        let mj = w.mat[j];
        if mj >= SLUG || mom < CAPV[mj as usize] * 0.35 { continue; }
        if !isveh && w.bd_count[a] > 40 {
            let f = clampf(mom / (CAPV[mj as usize] * 1.5), 0.15, 0.9);
            let br = damage_brick(w, j, f, 0.0);
            if br > 0 {
                let p = bpos(w, j); let col = brick_col(w, j);
                dust_burst(w, p, col, 2, 1.5);
                shards_from(w, p, j, 2, 3.0, [0.0; 3]);
                w.breaks += br; w.break_c = p;
            }
            continue;
        }
        if !ensure_bodies(w, 1) { continue; }
        sever_brick(w, j);
        let s = w.sid[j] as usize;
        if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; }
        st_static_change(w, j, true);
        w.salvage += MASS[mj as usize];
        let b = alloc_body(w);
        if b == usize::MAX { w.cs[k].on = false; continue; }
        let (x, y, z) = blat(w, j);
        w.local[j] = lat_pos(x, y, z);
        w.lst2[0] = j as u32;
        let rv = rndv(w);
        let v = madd(mul(w.bd_v[a], 0.75), rv, 2.0);
        let ww = mul(rndv(w), 6.0);
        // spin is set after setup: passed in, it would be applied about the world origin (v += w x position)
        setup_body(w, b, 0, 1, [0.0; 3], IDM, [0.0, 0.0, 0.0, 1.0], v, [0.0; 3]);
        w.bd_w[b] = ww;
        let m = w.bd_m[a];
        let f = m / (m + MASS[mj as usize] * 1.5);
        w.bd_v[a] = mul(w.bd_v[a], f);
        let p = w.wpos[j];
        let col = brick_col(w, j);
        dust_burst(w, p, col, 2, 2.0);
        shards_from(w, p, j, if mj == GLASS { 6 } else { 3 }, 4.0, mul(v, 0.5));
        if mj == GLASS { event(w, 4.0, p, 1.0); }
        w.breaks += 1; w.break_c = p;
        w.cs[k].on = false;
        for t in (k + 1)..w.nc { if w.cs[t].bb == j as u32 { w.cs[t].on = false; } }
    }
}
fn prep_contacts(w: &mut World, h: f32) {
    for k in 0..w.nc {
        let mut c = w.cs[k];
        if !c.on { continue; }
        let a = c.a as usize;
        c.ra = sub(c.p, w.bd_pos[a]);
        c.rb = if c.b != STATIC { sub(c.p, w.bd_pos[c.b as usize]) } else { [0.0; 3] };
        c.mn = eff_mass(w, a, c.b, c.ra, c.rb, c.n);
        let dv = sub(vel_at(w, c.a, c.ra), vel_at(w, c.b, c.rb));
        let vn = dot(dv, c.n);
        c.vn0 = maxf(-vn, 0.0);
        let vt = sub(dv, mul(c.n, vn));
        let lt = len(vt);
        c.t1 = if lt > 1e-4 { mul(vt, 1.0 / lt) } else {
            let t = if abs(c.n[0]) < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let x = cross(c.n, t); mul(x, 1.0 / len(x))
        };
        c.t2 = cross(c.n, c.t1);
        c.mt1 = eff_mass(w, a, c.b, c.ra, c.rb, c.t1);
        c.mt2 = eff_mass(w, a, c.b, c.ra, c.rb, c.t2);
        // penetration is recovered gently: a hard push-out on a heavy body would look like (and count as) a huge impact
        let mut bias = minf(0.2 / h * maxf(c.depth - 0.01, 0.0), 1.0);
        if vn < -2.0 { bias = maxf(bias, -0.12 * vn); }
        c.bias = bias;
        w.cs[k] = c;
    }
}
const MU: f32 = 0.65;
fn solve(w: &mut World) {
    for k in 0..w.nc {
        let mut c = w.cs[k];
        if !c.on { continue; }
        let a = c.a as usize;
        let dv = sub(vel_at(w, c.a, c.ra), vel_at(w, c.b, c.rb));
        let vn = dot(dv, c.n);
        let lam = c.mn * (-vn + c.bias);
        let old = c.jn;
        c.jn = maxf(old + lam, 0.0);
        let d = c.jn - old;
        apply_imp(w, a, c.b, c.ra, c.rb, mul(c.n, d));
        let lim = MU * c.jn;
        let dv = sub(vel_at(w, c.a, c.ra), vel_at(w, c.b, c.rb));
        let l1 = -c.mt1 * dot(dv, c.t1);
        let o1 = c.jt1; c.jt1 = clampf(o1 + l1, -lim, lim);
        let l2 = -c.mt2 * dot(dv, c.t2);
        let o2 = c.jt2; c.jt2 = clampf(o2 + l2, -lim, lim);
        let p = add(mul(c.t1, c.jt1 - o1), mul(c.t2, c.jt2 - o2));
        apply_imp(w, a, c.b, c.ra, c.rb, p);
        w.cs[k] = c;
    }
}
/// Collisions crush material where they happen, and only there (the Geo-Mod rule: damage is local, structure
/// decides the rest). For every contact the energy it absorbed this substep is the work of its impulse,
/// E = 1/2 * J * v_approach - zero for resting or sliding contacts, so loads that are merely carried never chip
/// anything. That energy is split between the two sides in proportion to how easily each crushes, and each side
/// spends it on its bricks nearest the contact point, growing outward brick by brick (crush_at). What is left
/// over cracks the next layer. Whether anything then breaks off is decided by connectivity and by the load solver,
/// so a falling block loses its leading edge, a block dropping a storey crushes about a storey and keeps going,
/// and the rest of it stays whole.
fn impact_damage(w: &mut World) {
    let mut budget = 2500u32;
    for k in 0..w.nc {
        let c = w.cs[k];
        if !c.on { continue; }
        let a = c.a as usize;
        w.bd_j[a] += c.jn;
        if c.b != STATIC { w.bd_j[c.b as usize] += c.jn; }
        let ia = c.ba as usize;
        let ma = w.mat[ia];
        if ma == 0 || ma >= SLUG { continue; }
        // a heavy hit on a standing structure wakes its load solver
        if c.bb != NONE && c.bo == STATIC && c.jn * c.vn0 > 4000.0 {
            let j = c.bb as usize;
            if w.mat[j] != 0 && w.body[j] == STATIC { let s = w.sid[j] as usize; st_activate(w, s); }
        }
        if w.mat[ia] == 0 { continue; }
        if c.vn0 < 1.0 { continue; }
        let e = 0.5 * c.jn * c.vn0;
        if e < 150.0 { continue; }
        if e > 3.0e5 { event(w, 3.0, c.p, e * 0.02); }
        let jb = if c.bb != NONE { c.bb as usize } else { usize::MAX };
        let mb = if jb != usize::MAX { w.mat[jb] } else { 0 };
        let soft_a = if w.bd_veh[a] != 0 { 0.0 } else { 1.0 / EB[ma as usize] };
        let soft_b = if jb == usize::MAX || mb == 0 || mb >= SLUG || (c.b != STATIC && w.bd_veh[c.b as usize] != 0) { 0.0 } else { 1.0 / EB[mb as usize] };
        if soft_a + soft_b <= 0.0 { continue; }
        let ea = e * soft_a / (soft_a + soft_b);
        if ea > 0.0 && budget > 0 { budget -= mini(crush_at(w, ia, ea, c.p, budget) as i32, budget as i32) as u32; }
        if soft_b > 0.0 && budget > 0 { budget -= mini(crush_at(w, jb, e - ea, c.p, budget) as i32, budget as i32) as u32; }
    }
}
/// Spend energy e crushing the bricks of one body (or one standing structure) nearest point p, starting at brick
/// `start` and growing outward through intact joints. Returns how many bricks were crushed.
fn crush_at(w: &mut World, start: usize, mut e: f32, p: V, cap: u32) -> u32 {
    let m0 = w.mat[start];
    if m0 == 0 || m0 >= SLUG { return 0; }
    if e < EB[m0 as usize] * 0.15 { return 0; }
    let body = w.body[start];
    let sid = w.sid[start];
    w.stamp_id += 1;
    let st = w.stamp_id;
    // breadth-first ring growth from the contact brick; a ring is ordered by distance to the contact point
    w.q[0] = start as u32; let mut qn = 1usize; w.stamp[start] = st;
    let mut head = 0usize;
    let mut crushed = 0u32;
    let maxn = mini(cap as i32, 600) as u32;
    let mut ncr = 0usize;
    while head < qn && crushed < maxn {
        // pick the nearest unvisited candidate among the next few in the queue (keeps the crushed region round)
        let mut best = head; let mut bd = 1e18f32;
        let lim = mini((head + 24) as i32, qn as i32) as usize;
        for t in head..lim { let i = w.q[t] as usize; let d = sub(bpos(w, i), p); let dd = dot(d, d); if dd < bd { bd = dd; best = t; } }
        w.q.swap(head, best);
        let i = w.q[head] as usize; head += 1;
        let m = w.mat[i];
        if m == 0 { continue; }
        let cost = EB[m as usize];
        if e < cost {
            // not enough left to crush this one: crack its joints in proportion, and stop
            damage_brick(w, i, 0.6 * e / cost, 0.0);
            break;
        }
        e -= cost;
        w.lst2[ncr] = i as u32; ncr += 1; crushed += 1;
        for d in 0..6 {
            let (j, _) = linked(w, i, d);
            if j == usize::MAX || w.stamp[j] == st || w.body[j] != body || w.sid[j] != sid { continue; }
            if qn >= w.q.len() { break; }
            w.stamp[j] = st; w.q[qn] = j as u32; qn += 1;
        }
    }
    if ncr == 0 { return 0; }
    // apply: part is ground to dust and fragments, part breaks loose as rubble
    let vb = if body != STATIC { w.bd_v[body as usize] } else { [0.0; 3] };
    for t in 0..ncr {
        let i = w.lst2[t] as usize;
        let m = w.mat[i];
        if m == 0 { continue; }
        let q = bpos(w, i);
        if t % 4 == 0 { let col = brick_col(w, i); dust_burst(w, q, col, 1, 0.8); }
        if rnd(w) < PULV[m as usize] {
            shards_from(w, q, i, 2, 1.2, mul(vb, 0.6));
            w.salvage += MASS[m as usize];
            remove_brick(w, i);
        } else {
            sever_brick(w, i);
            if body == STATIC { let s = w.sid[i] as usize; if s < w.st_n { w.st_conn[s] = true; w.conn_any = true; } st_static_change(w, i, true); }
            else { w.bd_dirty[body as usize] = true; }
        }
    }
    w.breaks += crushed; w.break_c = p;
    crushed
}
fn integrate(w: &mut World, h: f32) {
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_asleep[b] { continue; }
        clamp_vel(w, b);
        if w.bd_cool[b] > 0.0 { w.bd_cool[b] -= h; }
        w.bd_pos[b] = madd(w.bd_pos[b], w.bd_v[b], h);
        w.bd_q[b] = q_int(w.bd_q[b], w.bd_w[b], h);
        // air drag only; a big piece pivoting on its foot must be free to topple, so no extra spin damping for it
        w.bd_v[b] = mul(w.bd_v[b], 1.0 - 0.01 * h);
        w.bd_w[b] = mul(w.bd_w[b], 1.0 - 0.05 * h);
        if w.bd_veh[b] == 0 {
            let small = w.bd_count[b] <= 3;
            if w.bd_nc[b] > 0 && small {
                // rolling resistance of loose bricks on rubble
                w.bd_w[b] = mul(w.bd_w[b], 1.0 - minf(7.0 * h, 0.5));
                w.bd_v[b] = mul(w.bd_v[b], 1.0 - minf(2.0 * h, 0.3));
            }
            let v2 = dot(w.bd_v[b], w.bd_v[b]);
            // sleep on rim speed, not angular rate: a 30 m tower just starting to tip turns slowly but must not freeze
            let rr = w.bd_rad[b];
            let w2 = dot(w.bd_w[b], w.bd_w[b]) * rr * rr;
            let (vt, wt) = if small { (0.16, 0.3) } else { (0.06, 0.04) };
            if v2 < vt && w2 < wt && w.bd_nc[b] > 0 && w.bd_maxr[b] < 0.95 {
                w.bd_sleep[b] += h;
                if w.bd_sleep[b] > 0.45 {
                    w.bd_asleep[b] = true; w.bd_v[b] = [0.0; 3]; w.bd_w[b] = [0.0; 3];
                    let k = bucket_of(w.bd_pos[b]);
                    w.bd_bucket[b] = k as u16; w.bd_inmesh[b] = false;
                    mark_set(w, SSETS + k, false);
                    if w.bd_pos[b][1] < 2.0 { let (x, _, z) = world_cell(w.bd_pos[b]); zone_touch(w, x, z); }
                }
            } else { w.bd_sleep[b] = 0.0; }
        }
        let p = w.bd_pos[b];
        if p[1] < -15.0 || abs(p[0]) > 190.0 || abs(p[2]) > 190.0 || p[1] > 120.0 { kill_body(w, b); }
    }
    for b in 0..w.hw {
        if w.bd_wake[b] { w.bd_wake[b] = false; if w.bd_alive[b] { wake_body(w, b); } }
    }
}
/// A light piece caught between heavy ones gets an enormous impulse from the contact solver and would shoot out
/// like a squeezed seed. In reality it is crushed: small pieces whose velocity jumps by more than ~100 g in one
/// substep are pulverised, and the velocity change any piece can take from contacts in one substep is limited.
fn pinch_check(w: &mut World) {
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_asleep[b] || w.bd_veh[b] != 0 { continue; }
        // spin picked up from off-centre contact impulses is limited the same way (as a speed at the piece's rim)
        let dw = sub(w.bd_w[b], w.bd_wpre[b]);
        let tip = len(dw) * w.bd_rad[b];
        if tip > 6.0 { w.bd_w[b] = madd(w.bd_wpre[b], dw, 6.0 / tip); }
        let dv = sub(w.bd_v[b], w.bd_vpre[b]);
        let m = len(dv);
        if m < 8.0 { continue; }
        let f = w.bd_first[b];
        if f == NONE || w.mat[f as usize] >= SLUG { continue; }
        if w.bd_count[b] <= 6 && w.blast_t <= 0.0 {
            let mut k = f;
            while k != NONE { let i = k as usize; k = w.next[i]; if w.mat[i] != 0 { let p = w.wpos[i]; shards_from(w, p, i, 2, 1.5, mul(w.bd_vpre[b], 0.5)); } }
            let p = w.bd_pos[b]; let c = brick_col(w, f as usize);
            dust_burst(w, p, c, 1, 0.6);
            kill_body(w, b);
        } else {
            w.bd_v[b] = madd(w.bd_vpre[b], dv, 8.0 / m);
        }
    }
}
fn substep(w: &mut World, h: f32) {
    w.hsub = h;
    update_cache(w);
    for b in 0..w.hw { if w.bd_alive[b] && !w.bd_asleep[b] { w.bd_v[b][1] -= GR * h; } }
    build_hash(w);
    vehicles_substep(w, h);
    gen_contacts(w);
    plough(w);
    prep_contacts(w, h);
    for b in 0..w.hw { if w.bd_alive[b] { w.bd_vpre[b] = w.bd_v[b]; w.bd_wpre[b] = w.bd_w[b]; } }
    for _ in 0..10 { solve(w); }
    pinch_check(w);
    impact_damage(w);
    integrate(w, h);
}

// ---------------------------------------------------------------- vehicles
fn veh_ok(w: &World, vi: usize) -> bool {
    if vi >= MAXV || !w.v_alive[vi] { return false; }
    let b = w.v_body[vi] as usize;
    b < MAXBODY && w.bd_alive[b] && w.bd_veh[b] as usize == vi + 1
}
fn veh_pos(w: &World, vi: usize) -> V { let b = w.v_body[vi] as usize; add(w.bd_pos[b], mv(&w.bd_rot[b], w.v_hc[vi])) }
fn wheel_ray(w: &World, p: V, d: V, maxt: f32, skip: usize) -> Hit {
    let mut h = Hit { t: maxt, n: [0.0, 1.0, 0.0], body: usize::MAX, brick: usize::MAX };
    if d[1] < -1e-4 { let t = -p[1] / d[1]; if t >= 0.0 && t < h.t { h.t = t; } }
    let mut prev = world_cell(p);
    let mut t = 0.0;
    while t < h.t {
        let q = madd(p, d, t);
        let c = world_cell(q);
        let j = static_brick(w, c.0, c.1, c.2);
        if j != usize::MAX {
            h.t = t; h.brick = j;
            h.n = if prev.1 != c.1 { [0.0, if prev.1 > c.1 { 1.0 } else { -1.0 }, 0.0] }
                else if prev.0 != c.0 { [if prev.0 > c.0 { 1.0 } else { -1.0 }, 0.0, 0.0] }
                else if prev.2 != c.2 { [0.0, 0.0, if prev.2 > c.2 { 1.0 } else { -1.0 }] }
                else { [0.0, 1.0, 0.0] };
            break;
        }
        prev = c;
        t += 0.035;
    }
    let mid = madd(p, d, maxt * 0.5);
    let (hx, hy, hz) = (floor(mid[0] / S) as i32, floor(mid[1] / S) as i32, floor(mid[2] / S) as i32);
    for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
        let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
        while m != NONE {
            let j = m as usize; m = w.hnext[j];
            if w.mat[j] == 0 { continue; }
            let bj = w.body[j] as usize;
            if bj == skip { continue; }
            let oc = sub(p, w.wpos[j]);
            let bq = dot(oc, d);
            let cq = dot(oc, oc) - RB * RB;
            let disc = bq * bq - cq;
            if disc < 0.0 { continue; }
            let tt = -bq - sqrt(disc);
            if tt >= 0.0 && tt < h.t { h.t = tt; h.n = norm(sub(madd(p, d, tt), w.wpos[j])); h.body = bj; h.brick = usize::MAX; }
        }
    }}}
    h
}
fn vehicles_substep(w: &mut World, h: f32) {
    for vi in 0..MAXV {
        if !veh_ok(w, vi) { continue; }
        let b = w.v_body[vi] as usize;
        let r = w.bd_rot[b];
        let up = [r[1], r[4], r[7]];
        let fwd = [r[2], r[5], r[8]];
        let right = cross(fwd, up);
        let inp = w.v_in[vi];
        let fs = dot(w.bd_v[b], fwd);
        let st = inp[1] * 0.6 / (1.0 + abs(fs) * 0.05);
        w.v_steer[vi] += (st - w.v_steer[vi]) * minf(1.0, h * 7.0);
        let sa = w.v_steer[vi];
        let (cs, sn) = (cosf(sa), sinf(sa));
        let down = mul(up, -1.0);
        let mut touch = 0u8;
        let mut spin = 0.0;
        for k in 0..4 {
            let mp = add(w.bd_pos[b], mv(&r, w.v_mount[vi][k]));
            let hit = wheel_ray(w, mp, down, SREST + WR, b);
            if hit.t >= SREST + WR - 1e-4 { w.v_comp[vi][k] = SREST; continue; }
            touch += 1;
            let cp = madd(mp, down, hit.t);
            let ra = sub(cp, w.bd_pos[b]);
            let hb = hit.body;
            let mut vc = add(w.bd_v[b], cross(w.bd_w[b], ra));
            if hb != usize::MAX { let rb = sub(cp, w.bd_pos[hb]); vc = sub(vc, add(w.bd_v[hb], cross(w.bd_w[hb], rb))); }
            let x = SREST + WR - hit.t;
            let rate = dot(vc, down);
            let f = clampf(KS * x + KD * rate, 0.0, 60000.0);
            let jn = f * h;
            impulse_at(w, b, cp, mul(up, jn));
            if hb != usize::MAX { impulse_at(w, hb, cp, mul(up, -jn)); wake_body(w, hb); }
            if hit.brick != usize::MAX {
                w.wload[hit.brick] += f / GR * 0.5;
                let s = w.sid[hit.brick] as usize;
                if s < w.st_n && w.st_active[s] < 0.5 { st_activate(w, s); }
            }
            let n = hit.n;
            let heading = if k < 2 { add(mul(fwd, cs), mul(right, sn)) } else { fwd };
            let hd = norm(sub(heading, mul(n, dot(heading, n))));
            let sd = norm(cross(n, hd));
            let mut vc = add(w.bd_v[b], cross(w.bd_w[b], ra));
            if hb != usize::MAX { let rb = sub(cp, w.bd_pos[hb]); vc = sub(vc, add(w.bd_v[hb], cross(w.bd_w[hb], rb))); }
            let vlong = dot(vc, hd);
            let vlat = dot(vc, sd);
            let ml = eff_mass(w, b, STATIC, ra, ra, hd);
            let ms = eff_mass(w, b, STATIC, ra, ra, sd);
            let mut jlat = -vlat * ms * 0.6;
            let mut jlong = -vlong * ml * 0.0003;
            let thr = inp[0];
            let hand = inp[2] > 0.5;
            if thr > 0.0 {
                if fs > -1.0 { jlong += thr * ENGINE * 0.25 * maxf(0.0, 1.0 - fs / VMAX) * h; }
                else { let bm = BRAKE * 0.25 * h; jlong += clampf(-vlong * ml, -bm, bm); }
            } else if thr < 0.0 {
                if fs > 1.0 { let bm = BRAKE * 0.25 * h * -thr; jlong += clampf(-vlong * ml, -bm, bm); }
                else { jlong += thr * ENGINE * 0.25 * 0.55 * maxf(0.0, 1.0 + fs / 9.0) * h; }
            } else if abs(fs) < 1.5 { jlong += -vlong * ml * 0.25; }
            if hand && k >= 2 { let bm = BRAKE * 0.35 * h; jlong += clampf(-vlong * ml, -bm, bm); }
            let mu = if hand && k >= 2 { 0.45 } else { 1.15 };
            let lim = mu * jn;
            let mag = sqrt(jlong * jlong + jlat * jlat);
            if mag > lim && mag > 1e-9 { let s = lim / mag; jlong *= s; jlat *= s; }
            let jt = add(mul(hd, jlong), mul(sd, jlat));
            let com_h = dot(sub(w.bd_pos[b], cp), up);
            let ap = madd(cp, up, com_h * 0.7);
            impulse_at(w, b, ap, jt);
            if hb != usize::MAX { impulse_at(w, hb, cp, mul(jt, -1.0)); }
            w.v_comp[vi][k] = hit.t - WR;
            spin += vlong;
        }
        w.v_touch[vi] = touch;
        if touch > 0 { w.v_spin[vi] += spin / touch as f32 * h / WR; }
        let v = w.bd_v[b];
        let sp = len(v);
        w.bd_v[b] = mul(v, 1.0 - minf(0.0003 * sp * h, 0.05));
        if touch >= 2 {
            let wf = dot(w.bd_w[b], fwd);
            w.bd_w[b] = madd(w.bd_w[b], fwd, -wf * minf(1.0, h * 3.0));
        }
    }
}
fn spawn_vehicle(w: &mut World, vi: usize, pos: V, yaw: f32) {
    if veh_ok(w, vi) { let b = w.v_body[vi] as usize; kill_body(w, b); }
    w.v_alive[vi] = false;
    let (x0, y0, z0) = (2 + vi as i32 * 6, VEH_Y, 2);
    for y in 0..3 { for z in 0..8 { for x in 0..4 {
        let g = gget(w, x0 + x, y0 + y, z0 + z);
        if g != 0 {
            let e = g as usize - 1;
            gset(w, x0 + x, y0 + y, z0 + z, 0);
            if w.mat[e] != 0 {
                let bb = w.body[e];
                w.mat[e] = 0;
                if bb == STATIC || bb == TEMPB { w.body[e] = STATIC; } else { w.bd_dirty[bb as usize] = true; }
            }
        }
    }}}
    let (os, op) = (w.cur_sid, w.cur_paint);
    w.cur_sid = NOS; w.cur_paint = 0;
    let mut n = 0;
    for y in 0..3 { for z in 0..8 { for x in 0..4 {
        if y == 2 && !(z >= 2 && z < 6) { continue; }
        let m = if y == 0 { VEHH } else { VEH };
        let idx = add_brick(w, x0 + x, y0 + y, z0 + z, m);
        if idx == usize::MAX { continue; }
        w.body[idx] = TEMPB;
        w.local[idx] = [(x as f32 - 1.5) * S, y as f32 * S, (z as f32 - 3.5) * S];
        w.lst2[n] = idx as u32; n += 1;
    }}}
    w.cur_sid = os; w.cur_paint = op;
    if n == 0 { return; }
    for t in 0..n { let i = w.lst2[t] as usize; finalize_brick(w, i); }
    let b = alloc_body(w);
    if b == usize::MAX { for t in 0..n { let i = w.lst2[t] as usize; remove_brick_raw(w, i); } return; }
    let q = [0.0, sinf(yaw * 0.5), 0.0, cosf(yaw * 0.5)];
    let rm = q_to_m(q);
    setup_body(w, b, 0, n, pos, rm, q, [0.0; 3], [0.0; 3]);
    w.bd_veh[b] = vi as u8 + 1;
    let mut hc = [0.0f32; 3];
    let mut nh = 0.0;
    for t in 0..n { let i = w.lst2[t] as usize; if w.mat[i] == VEHH { hc = add(hc, w.local[i]); nh += 1.0; } }
    hc = mul(hc, 1.0 / maxf(nh, 1.0));
    w.v_hc[vi] = hc;
    let (mx, mz) = (0.95, 1.35);
    w.v_mount[vi] = [add(hc, [mx, 0.0, mz]), add(hc, [-mx, 0.0, mz]), add(hc, [mx, 0.0, -mz]), add(hc, [-mx, 0.0, -mz])];
    w.v_body[vi] = b as u16; w.v_alive[vi] = true;
    w.v_comp[vi] = [SREST; 4]; w.v_steer[vi] = 0.0; w.v_spin[vi] = 0.0; w.v_in[vi] = [0.0; 3];
    w.v_stuck[vi] = 0.0; w.v_rev[vi] = 0.0; w.v_flip[vi] = 0.0;
}
#[inline] fn node_pos(n: usize) -> [f32; 2] { [RN[n % 4], RN[n / 4]] }
fn node_next(w: &mut World, cur: usize, prev: usize) -> usize {
    let (i, j) = ((cur % 4) as i32, (cur / 4) as i32);
    let mut opts = [0usize; 4]; let mut n = 0;
    for &(di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)].iter() {
        let (a, b) = (i + di, j + dj);
        if a < 0 || b < 0 || a > 3 || b > 3 { continue; }
        let c = (b * 4 + a) as usize;
        if c == prev { continue; }
        opts[n] = c; n += 1;
    }
    if n == 0 { return prev; }
    let k = (rnd(w) * n as f32) as usize % n;
    opts[k]
}
fn lane_target(w: &World, vi: usize) -> V {
    let a = node_pos(w.v_prevn[vi] as usize);
    let b = node_pos(w.v_node[vi] as usize);
    let d = norm([b[0] - a[0], 0.0, b[1] - a[1]]);
    let r = [-d[2], 0.0, d[0]];
    [b[0] + r[0] * 2.4, 0.0, b[1] + r[2] * 2.4]
}
fn respawn_ai(w: &mut World, vi: usize) {
    let a = (rnd(w) * 16.0) as usize % 16;
    let b = node_next(w, a, 99);
    let pa = node_pos(a); let pb = node_pos(b);
    let d = norm([pb[0] - pa[0], 0.0, pb[1] - pa[1]]);
    let r = [-d[2], 0.0, d[0]];
    let t = 0.2 + 0.5 * rnd(w);
    let pos = [pa[0] + (pb[0] - pa[0]) * t + r[0] * 2.4, 1.0, pa[1] + (pb[1] - pa[1]) * t + r[2] * 2.4];
    // keep clear of the player
    let pp = w.pl_pos;
    if (pos[0] - pp[0]) * (pos[0] - pp[0]) + (pos[2] - pp[2]) * (pos[2] - pp[2]) < 900.0 { w.v_respawn[vi] = 1.0; return; }
    spawn_vehicle(w, vi, pos, atan2f(d[0], d[2]));
    w.v_prevn[vi] = a as u8; w.v_node[vi] = b as u8; w.v_aggro[vi] = 0.0; w.v_ai[vi] = true;
}
fn probe(w: &World, o: V, d: V, l: f32) -> f32 {
    let mut t = 0.0;
    while t < l { let (x, y, z) = world_cell(madd(o, d, t)); if static_at(w, x, y, z) { return t; } t += 0.4; }
    l
}
fn player_target(w: &World) -> (V, V) {
    if w.pv >= 0 && veh_ok(w, w.pv as usize) { let b = w.v_body[w.pv as usize] as usize; (w.bd_pos[b], w.bd_v[b]) } else { (w.pl_pos, w.pl_vel) }
}
fn ai_vehicles(w: &mut World, dt: f32) {
    let (pp, pvel) = player_target(w);
    let alive = w.pl_dead <= 0.0;
    for vi in 0..NAI {
        if !w.ai_on {
            if veh_ok(w, vi) && w.v_ai[vi] { let b = w.v_body[vi] as usize; kill_body(w, b); w.v_alive[vi] = false; }
            continue;
        }
        if !veh_ok(w, vi) {
            w.v_respawn[vi] -= dt;
            if w.v_respawn[vi] <= 0.0 { respawn_ai(w, vi); }
            continue;
        }
        if !w.v_ai[vi] || w.pv == vi as i32 { continue; }
        let b = w.v_body[vi] as usize;
        let r = w.bd_rot[b];
        let up = [r[1], r[4], r[7]];
        let f3 = [r[2], r[5], r[8]];
        let fl = norm([f3[0], 0.0, f3[2]]);
        let rt = [-fl[2], 0.0, fl[0]];
        let pos = w.bd_pos[b];
        let fs = dot(w.bd_v[b], f3);
        if up[1] < 0.35 { w.v_flip[vi] += dt; if w.v_flip[vi] > 2.5 { let bb = w.v_body[vi] as usize; kill_body(w, bb); w.v_alive[vi] = false; w.v_respawn[vi] = 2.0; continue; } } else { w.v_flip[vi] = 0.0; }
        if w.v_aggro[vi] > 0.0 { w.v_aggro[vi] -= dt; }
        if w.pv >= 0 {
            let d = sub(pos, pp);
            let rv = sub(w.bd_v[b], pvel);
            if dot(d, d) < 4.5 * 4.5 && len(rv) > 3.0 { w.v_aggro[vi] = 25.0; }
        }
        let chase = alive && w.v_aggro[vi] > 0.0;
        let mut desired;
        let tp;
        if chase { tp = madd(pp, pvel, 0.5); desired = 22.0; }
        else {
            let mut t = lane_target(w, vi);
            let dx = t[0] - pos[0]; let dz = t[2] - pos[2];
            if dx * dx + dz * dz < 64.0 {
                let cur = w.v_node[vi] as usize; let prev = w.v_prevn[vi] as usize;
                let nx = node_next(w, cur, prev);
                w.v_prevn[vi] = cur as u8; w.v_node[vi] = nx as u8;
                t = lane_target(w, vi);
            }
            tp = t; desired = 15.0;
        }
        let to = norm([tp[0] - pos[0], 0.0, tp[2] - pos[2]]);
        let ang = atan2f(dot(to, rt), dot(to, fl));
        let tdx = tp[0] - pos[0]; let tdz = tp[2] - pos[2];
        let tdist = sqrt(tdx * tdx + tdz * tdz);
        if !chase && tdist < 22.0 { desired = minf(desired, 8.0); }
        if abs(ang) > 0.4 { desired = minf(desired, 8.0); }
        let mut steer = clampf(ang * 2.0, -1.0, 1.0);
        let o = madd([pos[0], 0.9, pos[2]], fl, 2.2);
        let ln = 5.0 + maxf(fs, 0.0) * 0.7;
        let dl = norm(add(mul(fl, 0.9), mul(rt, -0.45)));
        let dr = norm(add(mul(fl, 0.9), mul(rt, 0.45)));
        let pc = probe(w, o, fl, ln);
        let pleft = probe(w, o, dl, ln * 0.8);
        let pright = probe(w, o, dr, ln * 0.8);
        if pc < ln && !(chase && tdist < pc + 2.0) {
            steer = if pleft > pright { -1.0 } else { 1.0 };
            desired = minf(desired, maxf(pc * 1.2, 4.0));
        }
        if pleft < ln * 0.4 { steer += 0.5; }
        if pright < ln * 0.4 { steer -= 0.5; }
        steer = clampf(steer, -1.0, 1.0);
        let mut blocked = false;
        if !chase {
            for oj in 0..MAXV {
                if oj == vi || !veh_ok(w, oj) { continue; }
                let ob = w.v_body[oj] as usize;
                let rel = sub(w.bd_pos[ob], pos);
                let ahead = dot(rel, fl); let side = dot(rel, rt);
                if ahead > 0.0 && ahead < 5.0 + maxf(fs, 0.0) * 0.9 && abs(side) < 2.6 { blocked = true; }
            }
            for a in 0..MAXA {
                if !w.a_on[a] || w.a_state[a] > 1 { continue; }
                let rel = sub(w.a_pos[a], pos);
                let ahead = dot(rel, fl); let side = dot(rel, rt);
                if ahead > 0.0 && ahead < 3.5 + maxf(fs, 0.0) * 0.6 && abs(side) < 1.6 { blocked = true; }
            }
            if w.pv < 0 {
                let rel = sub(w.pl_pos, pos);
                let ahead = dot(rel, fl); let side = dot(rel, rt);
                if ahead > 0.0 && ahead < 4.0 + maxf(fs, 0.0) * 0.7 && abs(side) < 1.8 { blocked = true; }
            }
        }
        let mut thr = if fs < desired { 1.0 } else { 0.0 };
        if blocked { thr = if fs > 0.5 { -1.0 } else { 0.0 }; }
        if abs(ang) > 1.2 && fs > 9.0 { thr = -1.0; }
        if w.v_rev[vi] > 0.0 { w.v_rev[vi] -= dt; thr = -1.0; steer = -steer; }
        else if thr > 0.0 && fs < 0.8 && !blocked {
            w.v_stuck[vi] += dt;
            if w.v_stuck[vi] > 1.6 { w.v_rev[vi] = 1.4; w.v_stuck[vi] = 0.0; }
        } else { w.v_stuck[vi] = maxf(0.0, w.v_stuck[vi] - dt); }
        w.v_in[vi] = [thr, steer, 0.0];
    }
}

// ---------------------------------------------------------------- player
fn box_free(w: &World, p: V) -> bool {
    if p[1] < -0.001 { return false; }
    let (x0, y0, z0) = world_cell([p[0] - PR, p[1] + 0.001, p[2] - PR]);
    let (x1, y1, z1) = world_cell([p[0] + PR, p[1] + PHT, p[2] + PR]);
    for y in y0..=y1 { for z in z0..=z1 { for x in x0..=x1 { if static_at(w, x, y, z) { return false; } } } }
    true
}
fn hurt(w: &mut World, dmg: f32, kv: V) {
    if w.pl_dead > 0.0 { return; }
    let dmg = if w.pv >= 0 { dmg * 0.3 } else { dmg };
    w.pl_hp -= dmg;
    w.pl_hurt = 0.0;
    if w.pv < 0 && dot(kv, kv) > 0.01 { w.pl_vel = kv; w.pl_knock = 0.5; w.pl_ground = false; }
    let p = w.pl_pos;
    event(w, 8.0, p, dmg);
    if w.pl_hp <= 0.0 {
        w.pl_hp = 0.0; w.pl_dead = 3.5;
        if w.pv >= 0 { let vi = w.pv as usize; w.v_in[vi] = [0.0; 3]; w.pv = -1; }
        event(w, 9.0, p, 0.0);
    }
}
fn respawn_player(w: &mut World) {
    w.pl_pos = SPAWN; w.pl_vel = [0.0; 3]; w.pl_hp = 100.0; w.pl_dead = 0.0; w.pl_knock = 0.0; w.pv = -1; w.pl_hurt = 10.0;
}
fn player_step(w: &mut World, dt: f32) {
    if w.pl_dead > 0.0 { w.pl_dead -= dt; if w.pl_dead <= 0.0 { respawn_player(w); } return; }
    w.pl_hurt += dt;
    if w.pl_hurt > 4.0 { w.pl_hp = minf(100.0, w.pl_hp + 15.0 * dt); }
    if w.pv >= 0 {
        let vi = w.pv as usize;
        if veh_ok(w, vi) { w.pl_pos = veh_pos(w, vi); let b = w.v_body[vi] as usize; w.pl_vel = w.bd_v[b]; return; }
        w.pv = -1;
        w.pl_pos[1] = maxf(w.pl_pos[1], 0.0);
    }
    let mut v = w.pl_vel;
    let control = w.pl_knock <= 0.0;
    if w.pl_knock > 0.0 { w.pl_knock -= dt; }
    if control {
        let sprint = w.pl_in[3] > 0.5;
        let spd = if sprint { 7.2 } else { 4.4 };
        let tv = [w.pl_in[0] * spd, 0.0, w.pl_in[1] * spd];
        let acc = if w.pl_ground { 14.0 } else { 3.5 };
        let k = minf(1.0, acc * dt);
        v[0] += (tv[0] - v[0]) * k; v[2] += (tv[2] - v[2]) * k;
    }
    if w.pl_jump && w.pl_ground && control { v[1] = 5.4; w.pl_ground = false; }
    w.pl_jump = false;
    v[1] -= GR * dt;
    let mut p = w.pl_pos;
    let was_ground = w.pl_ground;
    for ax in [0usize, 2].iter() {
        let mut np = p;
        np[*ax] += v[*ax] * dt;
        if box_free(w, np) { p = np; }
        else {
            let cy = floor(p[1] / S) as i32 + 1;
            let sy = cy as f32 * S + 0.002;
            let up = [np[0], sy, np[2]];
            if was_ground && sy - p[1] <= 0.56 && box_free(w, up) { p = up; } else { v[*ax] = 0.0; }
        }
    }
    let mut ground = false;
    let mut np = p;
    np[1] += v[1] * dt;
    if np[1] <= 0.0 { np[1] = 0.0; if v[1] < 0.0 { v[1] = 0.0; } ground = true; }
    if box_free(w, np) { p = np; }
    else if v[1] < 0.0 {
        let cy = floor(np[1] / S) as i32 + 1;
        let ty = cy as f32 * S + 0.001;
        if box_free(w, [p[0], ty, p[2]]) { p[1] = ty; }
        v[1] = 0.0; ground = true;
    } else { v[1] = 0.0; }
    if !ground && v[1] <= 0.0 && !box_free(w, [p[0], p[1] - 0.04, p[2]]) { ground = true; v[1] = 0.0; }
    // rubble, debris and vehicles
    let (hx, hz) = (floor(p[0] / S) as i32, floor(p[2] / S) as i32);
    let y0 = floor(p[1] / S) as i32;
    let mut dmg = 0.0f32; let mut kv = [0.0f32; 3];
    for dz in -1..=1 { for dx in -1..=1 { for dy in -1..5 {
        let mut m = w.hhead[hkey(hx + dx, y0 + dy, hz + dz)];
        while m != NONE {
            let j = m as usize; m = w.hnext[j];
            if w.mat[j] == 0 { continue; }
            let b = w.body[j] as usize;
            if w.pv >= 0 && w.bd_veh[b] as i32 == w.pv + 1 { continue; }
            let q = w.wpos[j];
            let cy = clampf(q[1], p[1] + PR, p[1] + PHT - PR);
            let d = [p[0] - q[0], cy - q[1], p[2] - q[2]];
            let dist = len(d);
            let lim = RB + PR;
            if dist >= lim || dist < 1e-5 { continue; }
            let n = mul(d, 1.0 / dist);
            let pen = lim - dist;
            let bv = add(w.bd_v[b], cross(w.bd_w[b], sub(q, w.bd_pos[b])));
            let closing = -dot(sub(bv, v), n);
            if closing > 4.5 {
                let dd = (closing - 4.0) * if w.bd_veh[b] != 0 { 9.0 } else { 5.0 };
                if dd > dmg { dmg = dd; kv = add(mul(bv, 0.6), [0.0, 3.0, 0.0]); }
            }
            if n[1] > 0.55 && cy <= p[1] + PR + 0.01 {
                p[1] += pen * n[1];
                if v[1] < 0.0 { v[1] = 0.0; }
                ground = true;
            } else {
                p[0] += n[0] * pen; p[2] += n[2] * pen;
                if w.bd_veh[b] == 0 && w.bd_m[b] < 1500.0 {
                    impulse_at(w, b, q, mul([n[0], 0.0, n[2]], -minf(pen * 600.0, 80.0)));
                    wake_body(w, b);
                }
            }
        }
    }}}
    p[0] = clampf(p[0], -HALF + 1.0, HALF - 1.0);
    p[2] = clampf(p[2], -HALF + 1.0, HALF - 1.0);
    w.pl_pos = p; w.pl_vel = v; w.pl_ground = ground;
    if dmg > 0.0 { hurt(w, dmg, kv); }
}

// ---------------------------------------------------------------- remote charges
fn charges_step(w: &mut World, dt: f32) {
    for k in 0..MAXCG {
        match w.cg_st[k] {
            1 => {
                let n = 4;
                let h = dt / n as f32;
                for _ in 0..n {
                    w.cg_vel[k][1] -= GR * h;
                    let p = w.cg_pos[k];
                    let np = madd(p, w.cg_vel[k], h);
                    if np[1] < 0.06 { w.cg_pos[k] = [np[0], 0.06, np[2]]; w.cg_st[k] = 4; break; }
                    let (x, y, z) = world_cell(np);
                    let j = static_brick(w, x, y, z);
                    if j != usize::MAX {
                        let (bx, by, bz) = blat(w, j);
                        w.cg_st[k] = 2; w.cg_brick[k] = j as u32; w.cg_off[k] = sub(p, lat_pos(bx, by, bz)); w.cg_pos[k] = p; break;
                    }
                    let (hx, hy, hz) = (floor(np[0] / S) as i32, floor(np[1] / S) as i32, floor(np[2] / S) as i32);
                    let mut stuck = usize::MAX;
                    for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                        let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
                        while m != NONE {
                            let j = m as usize; m = w.hnext[j];
                            if w.mat[j] == 0 { continue; }
                            let d = sub(np, w.wpos[j]);
                            if dot(d, d) < (RB + 0.08) * (RB + 0.08) { stuck = j; }
                        }
                    }}}
                    if stuck != usize::MAX {
                        let b = w.body[stuck] as usize;
                        w.cg_st[k] = 3; w.cg_brick[k] = stuck as u32; w.cg_off[k] = mtv(&w.bd_rot[b], sub(p, w.wpos[stuck])); w.cg_pos[k] = p; break;
                    }
                    w.cg_pos[k] = np;
                }
            }
            2 => {
                let j = w.cg_brick[k] as usize;
                if w.mat[j] == 0 { w.cg_st[k] = 1; w.cg_vel[k] = [0.0; 3]; }
                else if w.body[j] != STATIC {
                    let b = w.body[j] as usize;
                    let (bx, by, bz) = blat(w, j);
                    let wp = add(lat_pos(bx, by, bz), w.cg_off[k]);
                    w.cg_off[k] = mtv(&w.bd_rot[b], sub(wp, w.wpos[j]));
                    w.cg_st[k] = 3;
                }
            }
            3 => {
                let j = w.cg_brick[k] as usize;
                if w.mat[j] == 0 || w.body[j] == STATIC { w.cg_st[k] = 1; w.cg_vel[k] = [0.0; 3]; }
                else { let b = w.body[j] as usize; w.cg_pos[k] = add(w.wpos[j], mv(&w.bd_rot[b], w.cg_off[k])); }
            }
            _ => {}
        }
        let o = k * 4;
        let p = w.cg_pos[k];
        w.out_cg[o] = p[0]; w.out_cg[o + 1] = p[1]; w.out_cg[o + 2] = p[2]; w.out_cg[o + 3] = w.cg_st[k] as f32;
    }
}

// ---------------------------------------------------------------- districts & crowds
#[inline] fn zidx(lx: i32, lz: i32) -> usize { lz as usize * ZN + lx as usize }
#[inline] fn zl(w: &World, zn: usize, p: V) -> (i32, i32) { let (x, _, z) = world_cell(p); (x - w.zx0[zn], z - w.zz0[zn]) }
#[inline] fn zok(w: &World, zn: usize, lx: i32, lz: i32) -> bool { lx >= 0 && lz >= 0 && (lx as usize) < ZN && (lz as usize) < ZN && w.znav[zn * ZNAV + zidx(lx, lz)] != 0 }
#[inline] fn zctr(w: &World, zn: usize, lx: i32, lz: i32) -> V { let c = lat_pos(lx + w.zx0[zn], 0, lz + w.zz0[zn]); [c[0], 0.0, c[2]] }
fn heap_push(w: &mut World, n: &mut usize, key: u32) {
    if *n >= w.heap.len() { return; }
    let mut i = *n; *n += 1;
    w.heap[i] = key;
    while i > 0 { let p = (i - 1) / 2; if w.heap[p] <= w.heap[i] { break; } w.heap.swap(p, i); i = p; }
}
fn heap_pop(w: &mut World, n: &mut usize) -> u32 {
    let top = w.heap[0];
    *n -= 1;
    if *n > 0 {
        w.heap[0] = w.heap[*n];
        let mut i = 0;
        loop {
            let l = 2 * i + 1; let r = l + 1;
            let mut m = i;
            if l < *n && w.heap[l] < w.heap[m] { m = l; }
            if r < *n && w.heap[r] < w.heap[m] { m = r; }
            if m == i { break; }
            w.heap.swap(i, m); i = m;
        }
    }
    top
}
fn dijkstra(w: &mut World, zn: usize, h: usize) {
    let base = (zn * NHZ + h) * ZNAV;
    for c in 0..ZNAV { w.zflow[base + c] = INF16; }
    let s = w.zhub[zn * NHZ + h];
    if s == NONE { return; }
    w.zflow[base + s as usize] = 0;
    let mut n = 0usize;
    heap_push(w, &mut n, s);
    while n > 0 {
        let key = heap_pop(w, &mut n);
        let d = key >> 13;
        let c = (key & 0x1FFF) as usize;
        if d > w.zflow[base + c] as u32 { continue; }
        let (cx, cz) = ((c % ZN) as i32, (c / ZN) as i32);
        for k in 0..8 {
            let (dx, dz) = N8[k];
            let (nx, nz) = (cx + dx, cz + dz);
            if !zok(w, zn, nx, nz) { continue; }
            if dx != 0 && dz != 0 && (!zok(w, zn, cx + dx, cz) || !zok(w, zn, cx, cz + dz)) { continue; }
            let nd = d + if dx != 0 && dz != 0 { 14 } else { 10 };
            let nc = zidx(nx, nz);
            if nd < 65000 && nd < w.zflow[base + nc] as u32 {
                w.zflow[base + nc] = nd as u16;
                heap_push(w, &mut n, (nd << 13) | nc as u32);
            }
        }
    }
}
fn build_zone(w: &mut World, zn: usize) {
    let base = zn * ZNAV;
    for c in 0..ZNAV { w.znav[base + c] = 1; }
    let (x0, z0) = (w.zx0[zn], w.zz0[zn]);
    let (x1, z1) = (x0 + ZN as i32, z0 + ZN as i32);
    for s in 0..w.st_n {
        let (a, b) = (w.st_min[s], w.st_max[s]);
        if b[0] < x0 || a[0] >= x1 || b[2] < z0 || a[2] >= z1 { continue; }
        for i in w.st_lo[s]..w.st_hi[s] {
            if !in_struct(w, i, s) || w.by[i] > 3 { continue; }
            let (lx, lz) = (w.bx[i] as i32 - x0, w.bz[i] as i32 - z0);
            if lx >= 0 && lz >= 0 && lx < ZN as i32 && lz < ZN as i32 { w.znav[base + zidx(lx, lz)] = 0; }
        }
    }
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 || !w.bd_asleep[b] { continue; }
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize; k = w.next[i];
            let p = w.wpos[i];
            if p[1] < 0.9 {
                let (lx, lz) = zl(w, zn, p);
                if lx >= 0 && lz >= 0 && lx < ZN as i32 && lz < ZN as i32 { w.znav[base + zidx(lx, lz)] = 0; }
            }
        }
    }
    for h in 0..NHZ {
        let c = ZONES[zn];
        let hp = [c[0] + HUBOFF[h][0], 0.0, c[1] + HUBOFF[h][1]];
        let (hx, hz) = zl(w, zn, hp);
        let mut found = NONE;
        'search: for r in 0..16i32 {
            for dz in -r..=r { for dx in -r..=r {
                if maxi(dx.abs(), dz.abs()) != r { continue; }
                if zok(w, zn, hx + dx, hz + dz) { found = zidx(hx + dx, hz + dz) as u32; break 'search; }
            }}
        }
        w.zhub[zn * NHZ + h] = found;
        dijkstra(w, zn, h);
    }
}
#[inline] fn zflow(w: &World, zn: usize, h: usize, c: usize) -> u16 { w.zflow[(zn * NHZ + h) * ZNAV + c] }
fn pick_hub(w: &mut World, zn: usize, c: usize) -> u8 {
    let start = (rnd(w) * NHZ as f32) as usize % NHZ;
    for t in 0..NHZ {
        let h = (start + t) % NHZ;
        let f = zflow(w, zn, h, c);
        if f != INF16 && f > 40 { return h as u8; }
    }
    start as u8
}
fn flee_hub(w: &World, zn: usize, c: usize, from: V) -> u8 {
    let mut best = 0usize; let mut bd = -1.0f32;
    for h in 0..NHZ {
        if zflow(w, zn, h, c) == INF16 { continue; }
        let z = ZONES[zn];
        let dx = z[0] + HUBOFF[h][0] - from[0]; let dz = z[1] + HUBOFF[h][1] - from[2];
        let d = dx * dx + dz * dz;
        if d > bd { bd = d; best = h; }
    }
    best as u8
}
fn panic_from(w: &mut World, c: V, r: f32) {
    for a in 0..MAXA {
        if !w.a_on[a] || w.a_state[a] > 1 { continue; }
        let p = w.a_pos[a];
        let dx = p[0] - c[0]; let dz = p[2] - c[2];
        if dx * dx + dz * dz > r * r { continue; }
        w.a_state[a] = 1;
        w.a_panic[a] = 4.0 + 2.0 * rnd(w);
        w.a_idle[a] = 0.0;
        let zn = w.a_zone[a] as usize;
        let (lx, lz) = zl(w, zn, p);
        if zok(w, zn, lx, lz) { let f = flee_hub(w, zn, zidx(lx, lz), c); w.a_flee[a] = f; }
    }
}
fn knock_agent(w: &mut World, a: usize, v: V) {
    if w.a_state[a] == 2 { w.a_vel[a] = add(w.a_vel[a], mul(v, 0.5)); return; }
    w.a_state[a] = 2;
    w.a_vel[a] = v;
    w.a_pos[a][1] = maxf(w.a_pos[a][1], 0.05);
    w.a_idle[a] = 0.0;
    let p = w.a_pos[a];
    event(w, 7.0, p, len(v));
}
fn spawn_agents(w: &mut World) {
    for a in 0..MAXA { w.a_on[a] = false; }
    let per = MAXA / NZONE;
    for zn in 0..NZONE {
        let mut placed = 0; let mut tries = 0;
        while placed < per && tries < 3000 {
            tries += 1;
            let lx = (rnd(w) * ZN as f32) as i32; let lz = (rnd(w) * ZN as f32) as i32;
            if !zok(w, zn, lx, lz) { continue; }
            let c = zidx(lx, lz);
            let h = pick_hub(w, zn, c);
            if zflow(w, zn, h as usize, c) == INF16 { continue; }
            let a = zn * per + placed;
            w.a_on[a] = true; w.a_zone[a] = zn as u8;
            let jx = rnd(w) - 0.5; let jz = rnd(w) - 0.5;
            w.a_pos[a] = add(zctr(w, zn, lx, lz), [jx * 0.3, 0.0, jz * 0.3]);
            w.a_vel[a] = [0.0; 3];
            w.a_yaw[a] = rnd(w) * 2.0 * PI;
            w.a_tilt[a] = 0.0; w.a_phase[a] = rnd(w) * 6.0; w.a_state[a] = 0; w.a_t[a] = 0.0; w.a_panic[a] = 0.0;
            w.a_hub[a] = h; w.a_flee[a] = h; w.a_idle[a] = rnd(w) * 2.0; w.a_scan[a] = rnd(w) * 0.4;
            w.a_spd[a] = 1.15 + 0.4 * rnd(w);
            placed += 1;
        }
    }
}
fn creak(w: &World, p: V) -> bool {
    let (cx, _, cz) = world_cell(p);
    for &(dx, dz) in [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)].iter() {
        for y in 1..50 {
            let j = static_brick(w, cx + dx, y, cz + dz);
            if j != usize::MAX && w.svis[j] > 0.8 { return true; }
        }
    }
    false
}
fn agent_walk(w: &mut World, a: usize, dt: f32) {
    let zn = w.a_zone[a] as usize;
    let mut p = w.a_pos[a];
    let (cx, cz) = zl(w, zn, p);
    if !zok(w, zn, cx, cz) {
        let mut best = (0i32, 0i32, 1e9f32);
        for r in 1..6i32 {
            for dz in -r..=r { for dx in -r..=r {
                if !zok(w, zn, cx + dx, cz + dz) { continue; }
                let d = (dx * dx + dz * dz) as f32;
                if d < best.2 { best = (cx + dx, cz + dz, d); }
            }}
            if best.2 < 1e8 { break; }
        }
        if best.2 < 1e8 {
            let t = zctr(w, zn, best.0, best.1);
            let d = sub([t[0], 0.0, t[2]], [p[0], 0.0, p[2]]);
            let l = len(d);
            let step = minf(l, 3.0 * dt);
            if l > 1e-4 { p = madd(p, d, step / l); }
            w.a_pos[a] = p;
            w.a_vel[a] = mul(d, 3.0 / maxf(l, 0.1));
        }
        return;
    }
    let c = zidx(cx, cz);
    if w.a_state[a] == 1 {
        w.a_panic[a] -= dt;
        if w.a_panic[a] <= 0.0 { w.a_state[a] = 0; let h = pick_hub(w, zn, c); w.a_hub[a] = h; }
    }
    let panic = w.a_state[a] == 1;
    let hub = if panic { w.a_flee[a] } else { w.a_hub[a] } as usize;
    let base = (zn * NHZ + hub) * ZNAV;
    let here = w.zflow[base + c];
    let mut desired = [0.0f32; 3];
    if here == INF16 {
        let h = pick_hub(w, zn, c);
        w.a_hub[a] = h;
        if panic { let f = flee_hub(w, zn, c, p); w.a_flee[a] = f; }
    } else if here < 14 {
        if !panic && w.a_idle[a] <= 0.0 { w.a_idle[a] = 1.0 + 3.0 * rnd(w); }
    } else {
        let mut tx = cx; let mut tz = cz;
        for _ in 0..2 {
            let cur = w.zflow[base + zidx(tx, tz)];
            let mut best = cur; let mut bx = tx; let mut bz = tz;
            for k in 0..8 {
                let (dx, dz) = N8[k];
                let (nx, nz) = (tx + dx, tz + dz);
                if !zok(w, zn, nx, nz) { continue; }
                if dx != 0 && dz != 0 && (!zok(w, zn, tx + dx, tz) || !zok(w, zn, tx, tz + dz)) { continue; }
                let f = w.zflow[base + zidx(nx, nz)];
                if f < best { best = f; bx = nx; bz = nz; }
            }
            tx = bx; tz = bz;
        }
        let t = zctr(w, zn, tx, tz);
        desired = norm([t[0] - p[0], 0.0, t[2] - p[2]]);
        if tx == cx && tz == cz { desired = [0.0; 3]; }
    }
    if w.a_idle[a] > 0.0 {
        w.a_idle[a] -= dt;
        desired = [0.0; 3];
        if w.a_idle[a] <= 0.0 { let h = pick_hub(w, zn, c); w.a_hub[a] = h; }
    }
    for o in 0..MAXA {
        if o == a || !w.a_on[o] || w.a_state[o] > 1 { continue; }
        let d = [p[0] - w.a_pos[o][0], 0.0, p[2] - w.a_pos[o][2]];
        let dd = dot(d, d);
        if dd < 0.64 && dd > 1e-6 { let l = sqrt(dd); desired = madd(desired, d, (0.8 - l) * 2.5 / l); }
    }
    for vi in 0..MAXV {
        if !veh_ok(w, vi) { continue; }
        let b = w.v_body[vi] as usize;
        let vv = [w.bd_v[b][0], 0.0, w.bd_v[b][2]];
        let sp2 = dot(vv, vv);
        if sp2 < 4.0 { continue; }
        let rel = [p[0] - w.bd_pos[b][0], 0.0, p[2] - w.bd_pos[b][2]];
        if dot(rel, rel) > 18.0 * 18.0 { continue; }
        let t = clampf(dot(rel, vv) / sp2, 0.0, 2.0);
        let cl = madd(rel, vv, -t);
        if len(cl) < 2.6 && t < 1.6 {
            let dodge = if len(cl) > 0.1 { norm(cl) } else { norm([-vv[2], 0.0, vv[0]]) };
            desired = madd(desired, dodge, 3.0);
            if w.a_state[a] == 0 {
                w.a_state[a] = 1; w.a_panic[a] = 2.0;
                let vp = w.bd_pos[b];
                let f = flee_hub(w, zn, c, vp); w.a_flee[a] = f;
            }
        }
    }
    let spd = if w.a_state[a] == 1 { 3.9 + w.a_spd[a] } else { w.a_spd[a] };
    let dl = len(desired);
    if dl > 1.0 { desired = mul(desired, 1.0 / dl); }
    let tv = mul(desired, spd);
    let mut v = w.a_vel[a];
    let k = minf(1.0, dt * 7.0);
    v[0] += (tv[0] - v[0]) * k; v[2] += (tv[2] - v[2]) * k; v[1] = 0.0;
    let nxp = p[0] + v[0] * dt;
    let (tx, _) = zl(w, zn, [nxp, 0.0, p[2]]);
    if zok(w, zn, tx, cz) { p[0] = nxp; } else { v[0] = 0.0; }
    let nzp = p[2] + v[2] * dt;
    let (cx2, tz) = zl(w, zn, [p[0], 0.0, nzp]);
    if zok(w, zn, cx2, tz) { p[2] = nzp; } else { v[2] = 0.0; }
    p[1] = 0.0;
    w.a_pos[a] = p; w.a_vel[a] = v;
    let sp = sqrt(v[0] * v[0] + v[2] * v[2]);
    if sp > 0.15 {
        let ty = atan2f(v[0], v[2]);
        let mut d = ty - w.a_yaw[a];
        d -= 2.0 * PI * floor((d + PI) / (2.0 * PI));
        w.a_yaw[a] += d * minf(1.0, dt * 10.0);
    }
    w.a_phase[a] += sp * dt * 4.2;
    w.a_scan[a] -= dt;
    if w.a_scan[a] <= 0.0 {
        w.a_scan[a] = 0.35 + 0.1 * rnd(w);
        if creak(w, p) { panic_from(w, p, 7.0); }
    }
}
fn agent_air(w: &mut World, a: usize, dt: f32) {
    let p = w.a_pos[a];
    let mut v = w.a_vel[a];
    v[1] -= GR * dt;
    let mut np = madd(p, v, dt);
    let (x, y, z) = world_cell([np[0], np[1] + 0.5, np[2]]);
    if static_at(w, x, y, z) { np[0] = p[0]; np[2] = p[2]; v[0] = 0.0; v[2] = 0.0; if v[1] > 0.0 { v[1] = 0.0; } }
    if np[1] <= 0.0 {
        np[1] = 0.0;
        w.a_state[a] = 3;
        w.a_t[a] = 2.2 + 2.0 * rnd(w);
        v = [0.0; 3];
    }
    w.a_tilt[a] = minf(1.0, w.a_tilt[a] + dt * 4.0);
    w.a_yaw[a] += dt * 3.0;
    w.a_pos[a] = np; w.a_vel[a] = v;
}
fn agent_contacts(w: &mut World, a: usize) {
    let p = w.a_pos[a];
    let (hx, hz) = (floor(p[0] / S) as i32, floor(p[2] / S) as i32);
    let y0 = floor(p[1] / S) as i32;
    let mut knock: Option<V> = None;
    let mut push = [0.0f32; 3];
    for dz in -1..=1 { for dx in -1..=1 { for dy in 0..4 {
        let mut m = w.hhead[hkey(hx + dx, y0 + dy, hz + dz)];
        while m != NONE {
            let j = m as usize; m = w.hnext[j];
            if w.mat[j] == 0 { continue; }
            let q = w.wpos[j];
            let cy = clampf(q[1], p[1] + 0.3, p[1] + 1.5);
            let d = [p[0] - q[0], cy - q[1], p[2] - q[2]];
            let dist = len(d);
            let lim = RB + 0.28;
            if dist >= lim { continue; }
            let b = w.body[j] as usize;
            let bv = add(w.bd_v[b], cross(w.bd_w[b], sub(q, w.bd_pos[b])));
            let bs = len(bv);
            if bs > 2.8 && w.a_state[a] != 2 {
                let dir = norm([d[0], 0.0, d[2]]);
                knock = Some(add(add(mul(bv, 0.55), mul(dir, 1.5)), [0.0, 2.0 + 0.2 * bs, 0.0]));
            } else if w.a_state[a] <= 1 && q[1] < p[1] + 1.6 {
                let dh = [d[0], 0.0, d[2]];
                let l = len(dh);
                if l > 1e-4 { push = madd(push, dh, (lim - dist) / l); }
            }
        }
    }}}
    if let Some(v) = knock { knock_agent(w, a, v); return; }
    if w.a_state[a] <= 1 {
        let zn = w.a_zone[a] as usize;
        let np = add(p, push);
        let (cx, cz) = zl(w, zn, np);
        if zok(w, zn, cx, cz) { w.a_pos[a] = np; }
    }
}
fn agents_step(w: &mut World, dt: f32) {
    for zn in 0..NZONE {
        w.zt[zn] -= dt;
        if w.zdirty[zn] && w.zt[zn] <= 0.0 { build_zone(w, zn); w.zdirty[zn] = false; w.zt[zn] = 0.6; break; }
    }
    let (mut up, mut down, mut tot) = (0, 0, 0);
    for a in 0..MAXA {
        let o = a * 8;
        if !w.a_on[a] || !w.ai_on { w.out_ag[o + 6] = -1.0; continue; }
        match w.a_state[a] {
            0 | 1 => agent_walk(w, a, dt),
            2 => agent_air(w, a, dt),
            3 => { w.a_t[a] -= dt; if w.a_t[a] <= 0.0 { w.a_state[a] = 4; w.a_t[a] = 0.9; } }
            _ => {
                w.a_t[a] -= dt;
                w.a_tilt[a] = maxf(0.0, w.a_t[a] / 0.9);
                if w.a_t[a] <= 0.0 {
                    w.a_tilt[a] = 0.0; w.a_state[a] = 1; w.a_panic[a] = 3.0;
                    let p = w.a_pos[a];
                    let zn = w.a_zone[a] as usize;
                    let (lx, lz) = zl(w, zn, p);
                    if zok(w, zn, lx, lz) { let f = flee_hub(w, zn, zidx(lx, lz), p); w.a_flee[a] = f; }
                }
            }
        }
        if w.a_state[a] != 2 { agent_contacts(w, a); }
        tot += 1;
        if w.a_state[a] <= 1 { up += 1; } else { down += 1; }
        let p = w.a_pos[a];
        let v = w.a_vel[a];
        w.out_ag[o] = p[0]; w.out_ag[o + 1] = p[1]; w.out_ag[o + 2] = p[2];
        w.out_ag[o + 3] = w.a_yaw[a]; w.out_ag[o + 4] = w.a_tilt[a]; w.out_ag[o + 5] = w.a_phase[a];
        w.out_ag[o + 6] = w.a_state[a] as f32; w.out_ag[o + 7] = sqrt(v[0] * v[0] + v[2] * v[2]);
    }
    w.stats[8] = up as f32; w.stats[9] = tot as f32; w.stats[10] = down as f32;
}

// ---------------------------------------------------------------- weapons
fn bodies_near(w: &World, b: usize, c: V, r: f32) -> bool { w.bd_alive[b] && len(sub(w.bd_pos[b], c)) < r + w.bd_rad[b] }
fn struct_near(w: &World, s: usize, c: V, r: f32) -> bool {
    let (x0, y0, z0) = world_cell(sub(c, [r, r, r]));
    let (x1, y1, z1) = world_cell(add(c, [r, r, r]));
    let (a, b) = (w.st_min[s], w.st_max[s]);
    !(x1 < a[0] || x0 > b[0] || y1 < a[1] || y0 > b[1] || z1 < a[2] || z0 > b[2])
}
fn blast_brick(w: &mut World, i: usize, c: V, r: f32, power: f32, glass: &mut u32) {
    let m = w.mat[i];
    if m == 0 || m >= SLUG { return; }
    let p = bpos(w, i);
    let d = len(sub(p, c));
    let rr = r * BLAST[m as usize];
    if d < rr * 0.5 {
        let out = mul(sub(p, c), 1.0 / maxf(d, 0.1));
        let ns = if m == GLASS { 5 } else { 3 };
        shards_from(w, p, i, ns, 8.0 * power, mul(out, 7.0 * power));
        if rnd(w) < 0.5 { let col = brick_col(w, i); dust_burst(w, p, col, 1, 3.0); }
        if m == GLASS { *glass += 1; }
        remove_brick(w, i);
    } else if d < rr * 1.35 {
        let f = 1.0 - d / (rr * 1.35);
        damage_brick(w, i, power * f * f * 1.3, 0.0);
    }
}
fn explode_at(w: &mut World, c: V, r: f32, power: f32) {
    event(w, 1.0, c, r * power);
    w.blast_c = c; w.blast_r = r * 2.2; w.blast_p = power; w.blast_t = 0.25;
    let reach = r * 1.4 * 1.35 + 0.5;
    let mut glass = 0u32;
    for s in 0..w.st_n {
        if !struct_near(w, s, c, reach) { continue; }
        for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) { blast_brick(w, i, c, r, power, &mut glass); } }
    }
    for b in 0..w.hw {
        if !bodies_near(w, b, c, reach) || w.bd_veh[b] != 0 { continue; }
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; blast_brick(w, i, c, r, power, &mut glass); }
    }
    if glass > 0 { event(w, 4.0, c, glass as f32); }
    for b in 0..w.hw {
        if !bodies_near(w, b, c, w.blast_r) { continue; }
        wake_body(w, b);
        apply_blast_to_body(w, b);
    }
    let sr = r * 2.5;
    for k in 0..MAXSH {
        if !w.s_on[k] { continue; }
        let d = sub(w.s_pos[k], c);
        let dist = len(d);
        if dist > sr { continue; }
        w.s_rest[k] = false; w.s_t[k] = 0.0;
        let f = 10.0 * (1.0 - dist / sr) * power;
        w.s_vel[k] = add(madd(w.s_vel[k], norm(d), f), [0.0, 2.0, 0.0]);
    }
    for a in 0..MAXA {
        if !w.a_on[a] { continue; }
        let pa = add(w.a_pos[a], [0.0, 0.9, 0.0]);
        let d = sub(pa, c);
        let dist = len(d);
        let ar = r * 1.7;
        if dist < ar {
            let f = 1.0 - dist / ar;
            let v = add(mul(norm([d[0], 0.0, d[2]]), 10.0 * f + 2.0), [0.0, 3.0 + 4.0 * f, 0.0]);
            knock_agent(w, a, v);
        }
    }
    let pd = len(sub(add(w.pl_pos, [0.0, 0.9, 0.0]), c));
    if pd < r * 2.0 {
        let f = 1.0 - pd / (r * 2.0);
        let kv = add(mul(norm(sub(w.pl_pos, c)), 9.0 * f), [0.0, 4.0 * f, 0.0]);
        hurt(w, 75.0 * f * power, kv);
    }
    panic_from(w, c, 22.0);
    for vi in 0..NAI {
        if !veh_ok(w, vi) { continue; }
        if len(sub(veh_pos(w, vi), c)) < 16.0 { w.v_aggro[vi] = 25.0; }
    }
    w.boost = 50;
    for _ in 0..(50.0 * power) as u32 {
        let d = rndv(w); let s = rnd(w);
        let v = mul(d, (6.0 + 10.0 * s) * r * 0.5);
        let life = 0.35 + 0.5 * rnd(w);
        let sz = 0.6 + 0.8 * rnd(w); let g = 0.75 + 0.2 * rnd(w);
        spawn_p(w, c, v, life, sz, [1.0, g, 0.35], 1);
    }
    for _ in 0..(30.0 * power) as u32 {
        let d = rndv(w);
        let life = 2.5 + 2.5 * rnd(w);
        let (a1, a2, a3) = (rnd(w), rnd(w), rnd(w));
        spawn_p(w, madd(c, d, r * 0.4 * a1), mul(d, 1.5 + 2.0 * a2), life, 1.2 + 1.3 * a3, [0.32, 0.30, 0.29], 3);
    }
}

// ---------------------------------------------------------------- particles & rockets
fn particles(w: &mut World, dt: f32) {
    let mut n = 0;
    for k in 0..MAXP {
        if w.p_life[k] <= 0.0 { continue; }
        w.p_life[k] -= dt;
        if w.p_life[k] <= 0.0 { continue; }
        let mut v = w.p_vel[k];
        let mut p = w.p_pos[k];
        let t = w.p_life[k] / w.p_max[k];
        let mut col = w.p_col[k];
        let alpha;
        let mut size = w.p_size[k];
        match w.p_kind[k] {
            0 | 3 => {
                v = mul(v, 1.0 - minf(1.8 * dt, 1.0));
                v[1] += 0.35 * dt;
                w.p_size[k] += dt * 0.5;
                size = w.p_size[k];
                alpha = t * if w.p_kind[k] == 3 { 0.6 } else { 0.45 };
            }
            _ => {
                v = mul(v, 1.0 - minf(3.0 * dt, 1.0));
                v[1] += 2.0 * dt;
                col = [col[0], col[1] * t, col[2] * t * t];
                alpha = t;
                size = size * (0.6 + 0.6 * t);
            }
        }
        p = madd(p, v, dt);
        w.p_pos[k] = p; w.p_vel[k] = v;
        let o = n * 4;
        w.out_p[o] = p[0]; w.out_p[o + 1] = p[1]; w.out_p[o + 2] = p[2]; w.out_p[o + 3] = size;
        w.out_pc[o] = col[0]; w.out_pc[o + 1] = col[1]; w.out_pc[o + 2] = col[2]; w.out_pc[o + 3] = alpha;
        n += 1;
    }
    w.out_np = n;
}
fn point_hits(w: &World, p: V, skip: i32) -> bool {
    if p[1] < 0.05 { return true; }
    let (x, y, z) = world_cell(p);
    if static_at(w, x, y, z) { return true; }
    let (hx, hy, hz) = (floor(p[0] / S) as i32, floor(p[1] / S) as i32, floor(p[2] / S) as i32);
    for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
        let mut m = w.hhead[hkey(hx + dx, hy + dy, hz + dz)];
        while m != NONE {
            let j = m as usize; m = w.hnext[j];
            if w.mat[j] == 0 || w.body[j] == STATIC { continue; }
            if skip >= 0 && w.bd_veh[w.body[j] as usize] as i32 == skip + 1 { continue; }
            let d = sub(p, w.wpos[j]);
            if dot(d, d) < (RB + 0.1) * (RB + 0.1) { return true; }
        }
    }}}
    false
}
fn rockets(w: &mut World, dt: f32) {
    for r in 0..MAXR {
        if !w.r_alive[r] { continue; }
        w.r_age[r] += dt;
        let sub_n = 6;
        let h = dt / sub_n as f32;
        let skip = if w.r_age[r] < 0.3 { w.pv } else { -1 };
        for _ in 0..sub_n {
            let p = madd(w.r_pos[r], w.r_vel[r], h);
            w.r_vel[r][1] -= 1.5 * h;
            if point_hits(w, p, skip) {
                let back = mul(w.r_vel[r], -0.004);
                w.r_alive[r] = false;
                explode_at(w, add(p, back), 2.1, 1.0);
                break;
            }
            w.r_pos[r] = p;
            if abs(p[0]) > 190.0 || abs(p[2]) > 190.0 || p[1] > 80.0 || w.r_age[r] > 5.0 { w.r_alive[r] = false; break; }
        }
        if w.r_alive[r] {
            let p = w.r_pos[r];
            let v = w.r_vel[r];
            let j = rndv(w);
            let lf = 0.9 + 0.6 * rnd(w);
            spawn_p(w, madd(p, v, -0.012), mul(j, 0.4), lf, 0.35, [0.55, 0.55, 0.56], 3);
            spawn_p(w, madd(p, v, -0.008), madd(mul(v, -0.08), j, 1.0), 0.18, 0.45, [1.0, 0.8, 0.4], 1);
        }
    }
    for r in 0..MAXR {
        let o = r * 4;
        if w.r_alive[r] { w.out_rk[o] = w.r_pos[r][0]; w.out_rk[o + 1] = w.r_pos[r][1]; w.out_rk[o + 2] = w.r_pos[r][2]; w.out_rk[o + 3] = 1.0; }
        else { w.out_rk[o + 3] = 0.0; }
    }
}

// ---------------------------------------------------------------- mesher
fn put_face(w: &mut World, t: usize, v: &[V; 4], n: V, r: &[V; 4], col: V, mf: f32) {
    let k = w.nf[t];
    if k >= FCAPS[t] { return; }
    w.nf[t] = k + 1;
    let buf: &mut [f32] = match t { 0 => &mut w.out_so[..], 1 => &mut w.out_sg[..], 2 => &mut w.out_do[..], _ => &mut w.out_dg[..] };
    let o = k * FS;
    for q in 0..4 {
        let b = o + q * 13;
        buf[b] = v[q][0]; buf[b + 1] = v[q][1]; buf[b + 2] = v[q][2];
        buf[b + 3] = n[0]; buf[b + 4] = n[1]; buf[b + 5] = n[2];
        buf[b + 6] = r[q][0]; buf[b + 7] = r[q][1]; buf[b + 8] = r[q][2];
        buf[b + 9] = col[0]; buf[b + 10] = col[1]; buf[b + 11] = col[2];
        buf[b + 12] = mf;
    }
}
fn emit_brick(w: &mut World, i: usize, sv: bool, dynamic: bool) {
    let m = w.mat[i];
    if m == 0 || m >= VEH { return; }
    let b = w.body[i];
    let (x, y, z) = blat(w, i);
    let rc = lat_pos(x, y, z);
    let isrot = b != STATIC;
    let (ctr, rot) = if isrot { (w.wpos[i], w.bd_rot[b as usize]) } else { (rc, IDM) };
    let mut cw = [[0.0f32; 3]; 8];
    let mut cr = [[0.0f32; 3]; 8];
    for c in 0..8 {
        let o = w.corner[i][c];
        cw[c] = if isrot { add(ctr, mv(&rot, o)) } else { add(ctr, o) };
        cr[c] = add(rc, o);
    }
    let mut col = brick_col(w, i);
    let mut hmin = 1.0f32;
    for a in 0..4 { if w.bmax[i][a] > 0.0 && w.bmax[i][a] < 1e11 { let f = w.bond[i][a] / w.bmax[i][a]; if f < hmin { hmin = f; } } }
    col = mul(col, 0.55 + 0.45 * maxf(hmin, 0.0));
    if sv {
        if b == STATIC {
            let s = clampf(w.svis[i], 0.0, 1.2);
            col = if s < 0.35 { let t = s / 0.35; [0.04 + 0.08 * t, 0.10 + 0.55 * t, 0.42 - 0.02 * t] }
                else if s < 0.75 { let t = (s - 0.35) / 0.4; [0.15 + 0.85 * t, 0.70 + 0.2 * t, 0.45 - 0.35 * t] }
                else { let t = minf((s - 0.75) / 0.35, 1.0); [1.0, 0.9 - 0.75 * t, 0.10] };
        } else { col = mul(col, 0.35); }
    }
    let glass = m == GLASS && !sv;
    let target = (if dynamic { 2 } else { 0 }) + if glass { 1 } else { 0 };
    let pf = if w.paint[i] != 0 && !sv { 128.0 } else { 0.0 };
    for d in 0..6 {
        let j = nbr(w, i, d);
        if j != usize::MAX && w.mat[j] != 0 && w.body[j] == b {
            let nglass = w.mat[j] == GLASS && !sv;
            if !nglass || glass { continue; }
        }
        let f = FACES[d];
        let v = [cw[f[0]], cw[f[1]], cw[f[2]], cw[f[3]]];
        let r = [cr[f[0]], cr[f[1]], cr[f[2]], cr[f[3]]];
        let n = norm(cross(sub(v[2], v[0]), sub(v[3], v[1])));
        let inner = (w.omask[i] >> d) & 1;
        let mf = m as f32 + 8.0 * inner as f32 + 16.0 * (d >> 1) as f32 + pf;
        put_face(w, target, &v, n, &r, col, mf);
    }
}

// ---------------------------------------------------------------- map builders (cell coordinates)
fn begin_st(w: &mut World, kind: u8, target: bool) {
    let s = w.st_n;
    w.st_lo[s] = w.nb; w.cur_sid = s as u8; w.st_kind[s] = kind; w.st_target[s] = target; w.cur_paint = 0;
}
fn end_st(w: &mut World) {
    let s = w.st_n;
    w.st_hi[s] = w.nb;
    let mut mn = [i32::MAX; 3]; let mut mx = [i32::MIN; 3]; let mut cnt = 0u32;
    for i in w.st_lo[s]..w.st_hi[s] {
        if w.mat[i] == 0 { continue; }
        let (x, y, z) = blat(w, i);
        mn = [mini(mn[0], x), mini(mn[1], y), mini(mn[2], z)];
        mx = [maxi(mx[0], x), maxi(mx[1], y), maxi(mx[2], z)];
        cnt += 1;
    }
    w.st_min[s] = mn; w.st_max[s] = mx;
    w.st_init[s] = cnt; w.st_now[s] = cnt; w.st_top0[s] = mx[1] + 1; w.st_top[s] = mx[1] + 1;
    w.st_n += 1; w.cur_sid = NOS; w.cur_paint = 0;
}
fn boxf(w: &mut World, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32, m: u8) {
    for y in y0..y1 { for z in z0..z1 { for x in x0..x1 { add_brick(w, x, y, z, m); } } }
}
fn tower(w: &mut World, x0: i32, z0: i32, wd: i32, floors: i32) {
    let fh = 6;
    let cols = [0, wd / 2 - 1, wd - 2];
    for f in 0..floors {
        let yb = f * fh;
        for &cx in cols.iter() { for &cz in cols.iter() { boxf(w, x0 + cx, yb, z0 + cz, x0 + cx + 2, yb + fh - 1, z0 + cz + 2, CONC); } }
        boxf(w, x0, yb + fh - 1, z0, x0 + wd, yb + fh, z0 + wd, CONC);
        for y in yb..(yb + fh - 1) {
            for t in 0..wd {
                let door = f == 0 && y < yb + 4 && (t - wd / 2).abs() <= 1;
                if !door { add_brick(w, x0 + t, y, z0 + wd - 1, GLASS); }
                add_brick(w, x0 + t, y, z0, GLASS);
                add_brick(w, x0, y, z0 + t, GLASS);
                add_brick(w, x0 + wd - 1, y, z0 + t, GLASS);
            }
        }
    }
    let yt = floors * fh;
    let c = x0 + wd / 2 - 2;
    let cz = z0 + wd / 2 - 2;
    w.cur_paint = P_WHITE;
    for y in yt..yt + 3 { for z in cz..cz + 4 { for x in c..c + 4 {
        if x == c || x == c + 3 || z == cz || z == cz + 3 || y == yt + 2 { add_brick(w, x, y, z, STEEL); }
    }}}
    w.cur_paint = 0;
}
fn bridge(w: &mut World, xa: i32, xb: i32, zc: i32, dy: i32, piers: &[i32]) {
    let hw = 4;
    for &px in piers { boxf(w, px, 0, zc - hw + 1, px + 6, dy, zc + hw - 1, CONC); boxf(w, px - 1, dy - 2, zc - hw, px + 7, dy, zc + hw, CONC); }
    boxf(w, xa, dy, zc - hw, xb, dy + 1, zc + hw, CONC);
    w.cur_paint = P_DARK;
    for &zs in [zc - hw - 1, zc + hw].iter() {
        for x in xa..xb { add_brick(w, x, dy, zs, STEEL); }
        for x in (xa + 2)..(xb - 2) { add_brick(w, x, dy + 7, zs, STEEL); }
        let mut xv = xa + 2;
        while xv < xb - 2 {
            for y in dy + 1..dy + 7 { add_brick(w, xv, y, zs, STEEL); }
            if xv + 8 < xb - 2 { for k in 1..7 { add_brick(w, xv + k, dy + k, zs, STEEL); add_brick(w, xv + k, dy + k + 1, zs, STEEL); } }
            xv += 8;
        }
        for y in dy + 1..dy + 7 { add_brick(w, xb - 3, y, zs, STEEL); }
    }
    let mut xv = xa + 2;
    while xv < xb - 2 { for z in zc - hw - 1..zc + hw + 1 { add_brick(w, xv, dy + 7, z, STEEL); } xv += 16; }
    w.cur_paint = 0;
}
fn castle(w: &mut World, x0: i32, z0: i32, ln: i32) {
    let hgt = 28; let th = 3; let tw = 10; let tth = 40; let ar = 6.1f32;
    let xc = x0 + ln / 2;
    for y in 0..hgt { for z in z0..z0 + th { for x in x0..x0 + ln {
        let dx = (x - xc) as f32 + 0.5;
        let dyf = y as f32 - 10.0;
        let open = abs(dx) < ar && (y < 10 || dx * dx + dyf * dyf < ar * ar);
        let slit = z == z0 && (y == 18 || y == 19) && (x - x0) % 8 == 4;
        if !open && !slit { add_brick(w, x, y, z, MASON); }
    }}}
    // wall walk with crenellations
    for x in (x0..x0 + ln).step_by(2) { for z in z0..z0 + th { add_brick(w, x, hgt, z, MASON); } add_brick(w, x, hgt + 1, z0, MASON); }
    for &tx in [x0 - tw, x0 + ln].iter() {
        let tz = z0 - 3;
        for y in 0..tth { for z in tz..tz + tw { for x in tx..tx + tw {
            let wall = x == tx || x == tx + tw - 1 || z == tz || z == tz + tw - 1;
            let floor_ = y % 8 == 7;
            let slit = (y % 8 == 3 || y % 8 == 4) && ((x - tx) == tw / 2 || (z - tz) == tw / 2);
            if (wall && !slit) || floor_ { add_brick(w, x, y, z, if floor_ && !wall { WOOD } else { MASON }); }
        }}}
        for x in tx..tx + tw { for z in tz..tz + tw {
            if (x == tx || x == tx + tw - 1 || z == tz || z == tz + tw - 1) && (x + z) % 2 == 0 { add_brick(w, x, tth, z, MASON); add_brick(w, x, tth + 1, z, MASON); }
        }}
    }
    // keep behind the wall
    let (kx, kz) = (xc - 9, z0 - 26);
    for y in 0..44 { for z in kz..kz + 18 { for x in kx..kx + 18 {
        let wall = x <= kx + 1 || x >= kx + 16 || z <= kz + 1 || z >= kz + 16;
        let floor_ = y % 11 == 10;
        let win = (y % 11 == 5 || y % 11 == 6) && ((x - kx) % 5 == 2 || (z - kz) % 5 == 2) && y > 8;
        let pillar = (x - kx == 6 || x - kx == 11) && (z - kz == 6 || z - kz == 11);
        if pillar && !wall { add_brick(w, x, y, z, MASON); continue; }
        let door = z >= kz + 16 && (x - kx - 8).abs() <= 1 && y < 6;
        if door { continue; }
        if wall { add_brick(w, x, y, z, if win { GLASS } else { MASON }); } else if floor_ { add_brick(w, x, y, z, WOOD); }
    }}}
    for x in kx..kx + 18 { for z in kz..kz + 18 { if (x == kx || x == kx + 17 || z == kz || z == kz + 17) && (x + z) % 2 == 0 { add_brick(w, x, 44, z, MASON); } } }
}
fn watchtower(w: &mut World, x0: i32, z0: i32) {
    let top = 16;
    for &(lx, lz) in [(0, 0), (5, 0), (0, 5), (5, 5)].iter() { for y in 0..top { add_brick(w, x0 + lx, y, z0 + lz, WOOD); } }
    for &yy in [5, 11].iter() {
        for t in 0..6 { add_brick(w, x0 + t, yy, z0, WOOD); add_brick(w, x0 + t, yy, z0 + 5, WOOD); add_brick(w, x0, yy, z0 + t, WOOD); add_brick(w, x0 + 5, yy, z0 + t, WOOD); }
    }
    boxf(w, x0, top, z0, x0 + 6, top + 1, z0 + 6, WOOD);
    for t in 0..6 { for &(a, b) in [(t, 0), (t, 5), (0, t), (5, t)].iter() { if (a + b) % 2 == 0 { add_brick(w, x0 + a, top + 1, z0 + b, WOOD); } } }
    for &(lx, lz) in [(0, 0), (5, 0), (0, 5), (5, 5)].iter() { for y in top + 1..top + 4 { add_brick(w, x0 + lx, y, z0 + lz, WOOD); } }
    w.cur_paint = P_ROOF;
    boxf(w, x0, top + 4, z0, x0 + 6, top + 5, z0 + 6, WOOD);
    w.cur_paint = 0;
}
fn house(w: &mut World, x0: i32, z0: i32, wd: i32, dp: i32, hg: i32) {
    for y in 0..hg { for z in z0..z0 + dp { for x in x0..x0 + wd {
        let wall = x == x0 || x == x0 + wd - 1 || z == z0 || z == z0 + dp - 1;
        let window = y >= 2 && y <= 3 && ((x - x0) % 3 == 1) && (z == z0 || z == z0 + dp - 1);
        let door = y < 4 && z == z0 + dp - 1 && x - x0 == wd / 2;
        if wall && !door { add_brick(w, x, y, z, if window { GLASS } else { MASON }); }
    }}}
    w.cur_paint = P_ROOF;
    boxf(w, x0, hg, z0, x0 + wd, hg + 1, z0 + dp, WOOD);
    boxf(w, x0 + 1, hg + 1, z0 + 1, x0 + wd - 1, hg + 2, z0 + dp - 1, WOOD);
    w.cur_paint = 0;
}
fn apartment(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, floors: i32) {
    let fh = 6;
    for f in 0..floors {
        let yb = f * fh;
        for y in yb..yb + fh - 1 { for z in z0..z0 + wz { for x in x0..x0 + wx {
            let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
            if !edge { continue; }
            let along = if z == z0 || z == z0 + wz - 1 { x - x0 } else { z - z0 };
            let win = (y - yb) >= 2 && (y - yb) <= 3 && along % 4 >= 1 && along % 4 <= 2 && along > 1 && along < (if z == z0 || z == z0 + wz - 1 { wx } else { wz }) - 2;
            let door = f == 0 && (y - yb) < 4 && z == z0 + wz - 1 && (x - x0 - wx / 2).abs() <= 1;
            if door { continue; }
            add_brick(w, x, y, z, if win { GLASS } else { MASON });
        }}}
        let (cx, cz) = (x0 + wx / 2 - 1, z0 + wz / 2 - 1);
        for y in yb..yb + fh - 1 { for z in cz..cz + 3 { for x in cx..cx + 3 {
            if x == cx + 1 && z == cz + 1 { continue; }
            add_brick(w, x, y, z, CONC);
        }}}
        boxf(w, x0, yb + fh - 1, z0, x0 + wx, yb + fh, z0 + wz, CONC);
    }
    w.cur_paint = P_WARM;
    let yt = floors * fh;
    for z in z0..z0 + wz { for x in x0..x0 + wx {
        if x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1 { add_brick(w, x, yt, z, CONC); }
    }}
    w.cur_paint = 0;
}
fn warehouse(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, h: i32) {
    for y in 0..=h { for z in z0..z0 + wz { for x in x0..x0 + wx {
        let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
        if !edge { continue; }
        let along = if z == z0 || z == z0 + wz - 1 { x - x0 } else { z - z0 };
        let door = z == z0 + wz - 1 && (x - x0 - wx / 2).abs() <= 3 && y < 6;
        if door { continue; }
        let col = along % 6 == 0;
        let win = !col && y == h - 2 && along % 3 == 1;
        add_brick(w, x, y, z, if col { CONC } else if win { GLASS } else { MASON });
    }}}
    w.cur_paint = P_DARK;
    let mut x = x0 + 6;
    while x < x0 + wx - 1 { for z in z0 + 1..z0 + wz - 1 { add_brick(w, x, h, z, STEEL); } x += 6; }
    w.cur_paint = P_ROOF;
    boxf(w, x0, h + 1, z0, x0 + wx, h + 2, z0 + wz, WOOD);
    w.cur_paint = 0;
}
fn tank(w: &mut World, cx: i32, cz: i32, r: i32, h: i32, paint: u8) {
    w.cur_paint = paint;
    let rf = r as f32;
    for y in 0..=h { for dz in -r..=r { for dx in -r..=r {
        let d = sqrt((dx * dx + dz * dz) as f32);
        if d >= rf + 0.5 { continue; }
        if y == h || d >= rf - 0.5 { add_brick(w, cx + dx, y, cz + dz, STEEL); }
    }}}
    w.cur_paint = P_DARK;
    for y in 0..=h + 1 { add_brick(w, cx + r + 1, y, cz, STEEL); }
    w.cur_paint = 0;
}
fn chimney(w: &mut World, cx: i32, cz: i32, h: i32) {
    w.cur_paint = 0;
    boxf(w, cx - 8, 0, cz - 8, cx + 9, 3, cz + 9, CONC);
    let (r0, r1) = (6, 4);
    for y in 3..h {
        let lower = y < h / 2;
        let hs = if lower { r0 } else { r1 };
        for z in cz - hs..=cz + hs { for x in cx - hs..=cx + hs {
            let d = maxi((x - cx).abs(), (z - cz).abs());
            // two-brick-thick shell in the lower half where the load is highest
            if d == hs || (lower && d == hs - 1) || (!lower && y < h * 3 / 4 && d == hs - 1) { add_brick(w, x, y, z, MASON); }
        }}
        if y == h / 2 || y == h / 2 + 1 { for z in cz - r0..=cz + r0 { for x in cx - r0..=cx + r0 { if maxi((x - cx).abs(), (z - cz).abs()) >= r1 - 1 { add_brick(w, x, y, z, MASON); } } } }
    }
    w.cur_paint = P_RED;
    for y in h - 6..h { for z in cz - r1..=cz + r1 { for x in cx - r1..=cx + r1 {
        if maxi((x - cx).abs(), (z - cz).abs()) == r1 && (y < h - 4 || y >= h - 2) { add_brick(w, x, y, z, MASON); }
    }}}
    w.cur_paint = 0;
}
fn pipe_rack(w: &mut World, x0: i32, z0: i32, ln: i32) {
    w.cur_paint = P_DARK;
    let mut x = x0;
    while x <= x0 + ln {
        for y in 0..16 { add_brick(w, x, y, z0, STEEL); add_brick(w, x, y, z0 + 9, STEEL); }
        for z in z0..=z0 + 9 { add_brick(w, x, 16, z, STEEL); add_brick(w, x, 8, z, STEEL); }
        x += 8;
    }
    w.cur_paint = P_WHITE;
    for x in x0..=x0 + ln { for &z in [z0 + 2, z0 + 4, z0 + 7].iter() { add_brick(w, x, 17, z, STEEL); } add_brick(w, x, 9, z0 + 3, STEEL); add_brick(w, x, 9, z0 + 6, STEEL); }
    w.cur_paint = 0;
}
fn mast(w: &mut World, cx: i32, cz: i32, h: i32) {
    boxf(w, cx - 3, 0, cz - 3, cx + 9, 2, cz + 9, CONC);
    for y in 2..h {
        w.cur_paint = if (y / 12) % 2 == 0 { P_RED } else { P_WHITE };
        let sp = if y < h / 2 { 6 } else { 4 };
        let o = (6 - sp) / 2;
        for &(dx, dz) in [(0, 0), (sp, 0), (0, sp), (sp, sp)].iter() { add_brick(w, cx + o + dx, y, cz + o + dz, STEEL); }
        if y % 5 == 0 || y == h / 2 { for t in 0..=sp { add_brick(w, cx + o + t, y, cz + o, STEEL); add_brick(w, cx + o + t, y, cz + o + sp, STEEL); add_brick(w, cx + o, y, cz + o + t, STEEL); add_brick(w, cx + o + sp, y, cz + o + t, STEEL); } }
    }
    w.cur_paint = P_WHITE;
    boxf(w, cx, h, cz, cx + 7, h + 1, cz + 7, STEEL);
    for &(dx, dz) in [(-1, 3), (7, 3), (3, -1), (3, 7)].iter() { for y in h - 6..h - 2 { add_brick(w, cx + dx, y, cz + dz, STEEL); } }
    w.cur_paint = P_RED;
    for y in h + 1..h + 8 { add_brick(w, cx + 3, y, cz + 3, STEEL); }
    w.cur_paint = 0;
}
fn water_tower(w: &mut World, cx: i32, cz: i32) {
    w.cur_paint = P_DARK;
    let (sp, lh) = (14, 36);
    let legs = [(0, 0), (sp, 0), (0, sp), (sp, sp)];
    for &(dx, dz) in legs.iter() { for y in 0..lh { for q in 0..2 { add_brick(w, cx + dx + q, y, cz + dz, STEEL); add_brick(w, cx + dx, y, cz + dz + q, STEEL); } } }
    for &yy in [12, 24].iter() { for t in 0..=sp { add_brick(w, cx + t, yy, cz, STEEL); add_brick(w, cx + t, yy, cz + sp, STEEL); add_brick(w, cx, yy, cz + t, STEEL); add_brick(w, cx + sp, yy, cz + t, STEEL); } }
    w.cur_paint = P_WHITE;
    let (a, b) = (cx - 2, cx + sp + 3);
    let (c, d) = (cz - 2, cz + sp + 3);
    for y in lh..lh + 12 { for z in c..d { for x in a..b {
        if y == lh || y == lh + 11 || x == a || x == b - 1 || z == c || z == d - 1 { add_brick(w, x, y, z, STEEL); }
    }}}
    w.cur_paint = 0;
}
fn bunker(w: &mut World, x0: i32, z0: i32) {
    for y in 0..5 { for z in z0..z0 + 8 { for x in x0..x0 + 10 {
        let edge = x == x0 || x == x0 + 9 || z == z0 || z == z0 + 7 || y == 4;
        let slot = y == 2 && z == z0 && x > x0 + 2 && x < x0 + 7;
        let door = z == z0 + 7 && x >= x0 + 4 && x <= x0 + 5 && y < 3;
        if edge && !slot && !door { add_brick(w, x, y, z, CONC); }
    }}}
}
fn container(w: &mut World, x0: i32, y0: i32, z0: i32, alongx: bool, paint: u8) {
    w.cur_paint = paint;
    let (lx, lz) = if alongx { (12, 5) } else { (5, 12) };
    for y in y0..y0 + 5 { for z in z0..z0 + lz { for x in x0..x0 + lx {
        let shell = x == x0 || x == x0 + lx - 1 || z == z0 || z == z0 + lz - 1 || y == y0 || y == y0 + 4;
        if shell { add_brick(w, x, y, z, STEEL); }
    }}}
    w.cur_paint = 0;
}
fn gantry(w: &mut World, x0: i32, z0: i32, sx: i32, sz: i32, h: i32) {
    w.cur_paint = P_HAZARD;
    for &(dx, dz) in [(0, 0), (sx - 3, 0), (0, sz - 3), (sx - 3, sz - 3)].iter() { boxf(w, x0 + dx, 0, z0 + dz, x0 + dx + 3, h, z0 + dz + 3, STEEL); }
    boxf(w, x0, h, z0, x0 + sx, h + 3, z0 + 3, STEEL);
    boxf(w, x0, h, z0 + sz - 3, x0 + sx, h + 3, z0 + sz, STEEL);
    boxf(w, x0, h, z0 + 3, x0 + 3, h + 3, z0 + sz - 3, STEEL);
    boxf(w, x0 + sx - 3, h, z0 + 3, x0 + sx, h + 3, z0 + sz - 3, STEEL);
    // knee braces
    for &(dx, sgn) in [(0, 1), (sx - 3, -1)].iter() { for k in 1..8 { for &zz in [z0 + 1, z0 + sz - 2].iter() { add_brick(w, x0 + dx + 1 + sgn * k, h - 8 + k, zz, STEEL); } } }
    w.cur_paint = P_DARK;
    let tx = x0 + sx / 2 - 4;
    boxf(w, tx, h + 3, z0, tx + 8, h + 7, z0 + sz, STEEL);
    w.cur_paint = P_HAZARD;
    boxf(w, tx + 2, h + 7, z0 + sz / 2 - 3, tx + 6, h + 11, z0 + sz / 2 + 3, STEEL);
    w.cur_paint = 0;
}
fn bell_tower(w: &mut World, x0: i32, z0: i32) {
    let (h, s) = (78, 10);
    for y in 0..h { for z in z0..z0 + s { for x in x0..x0 + s {
        let wall = x == x0 || x == x0 + s - 1 || z == z0 || z == z0 + s - 1;
        let mid = (x - x0 - s / 2).abs() <= 1 || (z - z0 - s / 2).abs() <= 1;
        let open = y >= 62 && y < 70 && mid;
        let slit = y % 16 >= 6 && y % 16 <= 9 && (x - x0 == s / 2 || z - z0 == s / 2) && y < 60;
        let door = y < 7 && z == z0 + s - 1 && (x - x0 - s / 2).abs() <= 1;
        let fl = y % 16 == 15;
        if (wall && !open && !door) || fl { add_brick(w, x, y, z, if fl && !wall { WOOD } else if slit { GLASS } else { MASON }); }
    }}}
    w.cur_paint = P_ROOF;
    for k in 0..(s / 2) {
        let y = h + k * 2;
        for yy in y..y + 2 { boxf(w, x0 + k, yy, z0 + k, x0 + s - k, yy + 1, z0 + s - k, WOOD); }
    }
    w.cur_paint = P_DARK;
    for y in h + s..h + s + 4 { add_brick(w, x0 + s / 2, y, z0 + s / 2, STEEL); }
    w.cur_paint = 0;
}

// ---- complex buildings -------------------------------------------------------------------------
/// 6 x 8 cell concrete shear-wall core with switchback stairs from the ground to a roof penthouse.
/// Each storey has a two-cell door in the east wall (x = cx + 5) opening onto rows cz + 1 .. cz + 2.
fn stair_core(w: &mut World, cx: i32, cz: i32, floors: i32) {
    let top = floors * 6 + 5;
    let p = w.cur_paint;
    w.cur_paint = 0;
    for y in 0..=top { for z in cz..cz + 8 { for x in cx..cx + 6 {
        let wall = x == cx || x == cx + 5 || z == cz || z == cz + 7;
        if !wall && y != top { continue; }
        let door = x == cx + 5 && (z == cz + 1 || z == cz + 2) && y % 6 <= 3 && y < top;
        if door { continue; }
        add_brick(w, x, y, z, CONC);
    }}}
    for f in 0..=floors {
        let l = f * 6;
        if f > 0 { for x in cx + 3..cx + 5 { add_brick(w, x, l - 1, cz + 1, CONC); } }
        if f == floors { break; }
        // thin (one cell) flights so a 1.8 m tall walker clears the flight above
        for k in 1..=3 { for x in cx + 1..=cx + 2 { add_brick(w, x, l + k - 1, cz + 1 + k, CONC); } }
        for r in 4..=5 { for x in cx + 1..cx + 5 { add_brick(w, x, l + 2, cz + 1 + r, CONC); } }
        for k in 1..=3 { let r = 4 - k; for x in cx + 3..=cx + 4 { add_brick(w, x, l + 2 + k, cz + 1 + r, CONC); } }
    }
    w.cur_paint = p;
}
fn hvac(w: &mut World, x: i32, y: i32, z: i32) {
    let p = w.cur_paint; w.cur_paint = P_WHITE;
    boxf(w, x, y, z, x + 4, y + 2, z + 3, STEEL);
    w.cur_paint = p;
}
/// reinforced-concrete frame office: column grid, slabs, curtain wall or ribbon windows, stair core,
/// a corridor with office partitions and doors on every floor, roof parapet and plant
fn office(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, floors: i32, glassy: bool, spandrel: u8) {
    let cx = x0 + 1; let cz = z0 + wz / 2 - 3;
    stair_core(w, cx, cz, floors);
    let (zc0, zc1) = (cz + 1, cz + 2);
    for f in 0..floors {
        let l = f * 6;
        // towers get a 4 m column grid with heavy 1.5 m columns, low-rise blocks a 3 m grid with 1 m columns
        let (sp, cs) = if floors >= 8 { (8, 3) } else { (6, 2) };
        for xi in 0..=(wx / sp + 1) { for zi in 0..=(wz / sp + 1) {
            let x = mini(x0 + xi * sp, x0 + wx - cs); let z = mini(z0 + zi * sp, z0 + wz - cs);
            if x + cs - 1 >= cx && x <= cx + 5 && z + cs - 1 >= cz && z <= cz + 7 { continue; }
            if x > cx + 5 && z + cs - 1 >= zc0 - 1 && z <= zc1 + 1 { continue; }
            if z + cs >= z0 + wz && (x - (x0 + wx / 2)).abs() <= 2 { continue; }
            for y in l..l + 5 { for dz in 0..cs { for dx in 0..cs { add_brick(w, x + dx, y, z + dz, CONC); } } }
        }}
        for y in l..l + 5 {
            let rel = y - l;
            for z in z0..z0 + wz { for x in x0..x0 + wx {
                let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
                if !edge { continue; }
                let entrance = f == 0 && z == z0 + wz - 1 && (x - (x0 + wx / 2)) >= -2 && (x - (x0 + wx / 2)) <= 1 && rel <= 3;
                if entrance { continue; }
                let win = if glassy { rel >= 1 } else { rel >= 2 && rel <= 3 };
                if win { add_brick(w, x, y, z, GLASS); } else { w.cur_paint = spandrel; add_brick(w, x, y, z, CONC); w.cur_paint = 0; }
            }}
        }
        w.cur_paint = P_WHITE; w.cur_infill = true;
        for &zw in [zc0 - 1, zc1 + 1].iter() {
            for x in cx + 6..x0 + wx - 1 { for y in l..l + 5 {
                let dx = (x - cx - 6) % 8;
                if dx >= 3 && dx <= 5 && y - l <= 3 { continue; }
                add_brick(w, x, y, zw, WOOD);
            }}
        }
        let mut x = cx + 14;
        while x < x0 + wx - 2 {
            for y in l..l + 5 {
                for z in z0 + 1..zc0 - 1 { add_brick(w, x, y, z, WOOD); }
                // keep the ground-floor lobby behind the entrance open
                if !(f == 0 && (x - (x0 + wx / 2)).abs() <= 6) { for z in zc1 + 2..z0 + wz - 1 { add_brick(w, x, y, z, WOOD); } }
            }
            x += 8;
        }
        w.cur_paint = 0; w.cur_infill = false;
        for z in z0..z0 + wz { for x in x0..x0 + wx {
            if x > cx && x < cx + 5 && z > cz && z < cz + 7 { continue; }
            add_brick(w, x, l + 5, z, CONC);
        }}
    }
    let yr = floors * 6;
    w.cur_paint = spandrel;
    for z in z0..z0 + wz { for x in x0..x0 + wx { if x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1 { add_brick(w, x, yr, z, CONC); } } }
    w.cur_paint = 0;
    hvac(w, x0 + wx - 7, yr, z0 + 2);
    if wx > 18 { hvac(w, x0 + wx / 2, yr, z0 + wz - 5); }
}
/// load-bearing masonry apartment block: flats off a central corridor, stair core, cantilevered balconies
fn apartment2(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, floors: i32) {
    let cx = x0 + 1; let cz = z0 + wz / 2 - 3;
    stair_core(w, cx, cz, floors);
    let (zc0, zc1) = (cz + 1, cz + 2);
    for f in 0..floors {
        let l = f * 6;
        for y in l..l + 5 {
            let rel = y - l;
            for z in z0..z0 + wz { for x in x0..x0 + wx {
                let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
                if !edge { continue; }
                let side = z == z0 || z == z0 + wz - 1;
                let along = if side { x - x0 } else { z - z0 };
                let len = if side { wx } else { wz };
                let bay = along % 6;
                let front = z == z0 + wz - 1;
                let door = f == 0 && front && (x - (x0 + wx / 2)).abs() <= 1 && rel <= 3;
                let bdoor = f > 0 && front && (bay == 2 || bay == 3) && rel <= 3 && (along / 6) % 2 == 0 && along > 1 && along < len - 2;
                if door || bdoor { continue; }
                let win = rel >= 2 && rel <= 3 && (bay == 2 || bay == 3) && along > 1 && along < len - 2;
                add_brick(w, x, y, z, if win { GLASS } else { MASON });
            }}
        }
        w.cur_paint = P_WHITE;
        for &zw in [zc0 - 1, zc1 + 1].iter() {
            for x in cx + 6..x0 + wx - 1 { for y in l..l + 5 {
                let dx = (x - cx - 6) % 6;
                if dx >= 2 && dx <= 4 && y - l <= 3 { continue; }
                add_brick(w, x, y, zw, MASON);
            }}
        }
        let mut x = cx + 11;
        while x < x0 + wx - 2 {
            for y in l..l + 5 {
                for z in z0 + 1..zc0 - 1 { add_brick(w, x, y, z, MASON); }
                for z in zc1 + 2..z0 + wz - 1 { if !(z == zc1 + 4 && y - l <= 3) { add_brick(w, x, y, z, MASON); } }
            }
            x += 6;
        }
        w.cur_paint = 0;
        for z in z0..z0 + wz { for x in x0..x0 + wx {
            if x > cx && x < cx + 5 && z > cz && z < cz + 7 { continue; }
            add_brick(w, x, l + 5, z, CONC);
        }}
        if f + 1 < floors {
            let yb = l + 5;
            let mut bx = x0 + 1;
            while bx + 5 < x0 + wx {
                if ((bx - x0) / 6) % 2 == 0 {
                    for z in z0 + wz..z0 + wz + 2 { for x in bx..bx + 5 { add_brick(w, x, yb, z, CONC); } }
                    w.cur_paint = P_DARK;
                    for x in bx..bx + 5 { add_brick(w, x, yb + 1, z0 + wz + 1, STEEL); }
                    add_brick(w, bx, yb + 1, z0 + wz, STEEL); add_brick(w, bx + 4, yb + 1, z0 + wz, STEEL);
                    w.cur_paint = 0;
                }
                bx += 6;
            }
        }
    }
    let yr = floors * 6;
    for z in z0..z0 + wz { for x in x0..x0 + wx { if x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1 { add_brick(w, x, yr, z, MASON); } } }
    w.cur_paint = P_DARK;
    for y in yr..yr + 3 { for z in z0 + 2..z0 + 5 { for x in x0 + wx - 6..x0 + wx - 3 { add_brick(w, x, y, z, STEEL); } } }
    w.cur_paint = 0;
}
/// two-storey terraced houses: party walls, timber upper floor with a stair, pitched tiled roof with gables
/// three-storey terraced houses: party walls, timber floors with two stair flights, pitched tiled roof with gables
fn rowhouses(w: &mut World, x0: i32, z0: i32, units: i32, uw: i32, d: i32) {
    let h = 18;
    for u in 0..units {
        let ux = x0 + u * (uw - 1);
        for y in 0..h { for z in z0..z0 + d { for x in ux..ux + uw {
            let edge = x == ux || x == ux + uw - 1 || z == z0 || z == z0 + d - 1;
            if !edge { continue; }
            let rel = y % 6;
            let front = z == z0 + d - 1; let back = z == z0;
            let along = x - ux;
            let door = front && y < 4 && (along == 2 || along == 3);
            let win = (front || back) && rel >= 1 && rel <= 3 && (along == 6 || along == 7 || along == 9 || (y >= 6 && (along == 2 || along == 3)));
            if door { continue; }
            add_brick(w, x, y, z, if win { GLASS } else { MASON });
        }}}
        // floors at 5 and 11 with stairwell openings; ceiling at h-1
        for &(fy, sx) in [(5, ux + 1), (11, ux + uw - 3)].iter() {
            for z in z0 + 1..z0 + d - 1 { for x in ux + 1..ux + uw - 1 {
                if (x == sx || x == sx + 1) && z >= z0 + 7 && z <= z0 + 13 { continue; }
                add_brick(w, x, fy, z, WOOD);
            }}
        }
        for k in 1..=6 { let z = z0 + 13 - k; for x in ux + 1..=ux + 2 { add_brick(w, x, k - 1, z, WOOD); } }
        for k in 1..=6 { let z = z0 + 13 - k; for x in ux + uw - 3..=ux + uw - 2 { add_brick(w, x, 5 + k, z, WOOD); } }
        for z in z0 + 1..z0 + d - 1 { for x in ux + 1..ux + uw - 1 { add_brick(w, x, h - 1, z, WOOD); } }
        w.cur_paint = P_DARK;
        for y in h - 1..h + d / 2 + 3 { for x in ux + 5..ux + 7 { add_brick(w, x, y, z0 + 2, MASON); } }
        w.cur_paint = 0;
        for k in 0..(d / 2) {
            let y = h + k; let za = z0 + k; let zb = z0 + d - 1 - k;
            for z in za..=zb { for x in ux..ux + uw {
                let slope = z == za || z == zb || z == za + 1 || z == zb - 1;
                let gable = x == ux || x == ux + uw - 1;
                if slope { w.cur_paint = P_ROOF; add_brick(w, x, y, z, WOOD); w.cur_paint = 0; }
                else if gable { add_brick(w, x, y, z, MASON); }
            }}
        }
    }
}
/// nave with tall windows, a rose window over the door, timber pews, a pitched roof and a bell tower
/// nave with tall windows, a rose window over the door, timber pews, a pitched roof and a bell tower
fn church(w: &mut World, x0: i32, z0: i32) {
    let (wx, wz, h) = (28, 52, 30);
    for y in 0..h { for z in z0..z0 + wz { for x in x0..x0 + wx {
        let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
        if !edge { continue; }
        let sidewall = x == x0 || x == x0 + wx - 1;
        let bay = (z - z0) % 6;
        let win = sidewall && (bay == 2 || bay == 3) && y >= 7 && y <= 22 && z > z0 + 2 && z < z0 + wz - 3;
        let front = z == z0 + wz - 1;
        let dx = x - (x0 + wx / 2);
        let door = front && (dx >= -2 && dx <= 1) && y < 10;
        let rdy = y - 20;
        let rose = front && (dx * dx + rdy * rdy) <= 12;
        let apse = z == z0 && (dx == 0 || dx == -1) && y >= 8 && y <= 22;
        if door { continue; }
        add_brick(w, x, y, z, if win || rose || apse { GLASS } else { MASON });
    }}}
    // buttresses between the windows
    let mut z = z0 + 5;
    while z < z0 + wz - 3 { for y in 0..h - 6 { let out = if y < 12 { 2 } else { 1 }; for k in 1..=out { add_brick(w, x0 - k, y, z, MASON); add_brick(w, x0 + wx - 1 + k, y, z, MASON); } } z += 6; }
    for k in 0..(wx / 2) {
        let y = h + k; let xa = x0 + k; let xb = x0 + wx - 1 - k;
        for x in xa..=xb { for z in z0..z0 + wz {
            let slope = x == xa || x == xb || x == xa + 1 || x == xb - 1;
            let gable = z == z0 || z == z0 + wz - 1;
            if slope { w.cur_paint = P_DARK; add_brick(w, x, y, z, WOOD); w.cur_paint = 0; }
            else if gable { add_brick(w, x, y, z, MASON); }
        }}
    }
    let mut z = z0 + 6;
    while z < z0 + wz - 8 { for x in x0 + 3..x0 + wx - 3 { if (x - (x0 + wx / 2)).abs() > 2 { add_brick(w, x, 0, z, WOOD); add_brick(w, x, 1, z + 1, WOOD); } } z += 4; }
    bell_tower(w, x0 + wx + 2, z0 + wz - 12);
    boxf(w, x0 + wx, 0, z0 + wz - 10, x0 + wx + 2, 14, z0 + wz - 4, MASON);
}
/// open parking decks on a column grid, perimeter curbs, stair core in the corner
fn garage(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, decks: i32) {
    let (cx, cz) = (x0 + 1, z0 + 1);
    stair_core(w, cx, cz, decks);
    for f in 0..decks {
        let l = f * 6;
        let mut x = x0;
        while x < x0 + wx { let mut z = z0; while z < z0 + wz {
            if !(x + 1 >= cx && x <= cx + 5 && z >= cz && z <= cz + 7) { for y in l..l + 5 { add_brick(w, x, y, z, CONC); add_brick(w, x + 1, y, z, CONC); } }
            z += 8; } x += 8; }
        for z in z0..z0 + wz { for x in x0..x0 + wx {
            if x > cx && x < cx + 5 && z > cz && z < cz + 7 { continue; }
            add_brick(w, x, l + 5, z, CONC);
        }}
        w.cur_paint = P_HAZARD;
        for z in z0..z0 + wz { for x in x0..x0 + wx {
            if x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1 { add_brick(w, x, l + 6, z, CONC); }
        }}
        w.cur_paint = 0;
    }
}
fn shops(w: &mut World, x0: i32, z0: i32, wx: i32, wz: i32, sign: u8) {
    let h = 9;
    for y in 0..h { for z in z0..z0 + wz { for x in x0..x0 + wx {
        let edge = x == x0 || x == x0 + wx - 1 || z == z0 || z == z0 + wz - 1;
        let party = (x - x0) % 14 == 0;
        if !edge && !party { continue; }
        let front = z == z0 + wz - 1;
        if front && !party && x != x0 + wx - 1 {
            let k = (x - x0) % 14;
            let door = k == 6 || k == 7 || k == 8;
            if door && y <= 4 { continue; }
            add_brick(w, x, y, z, if y >= 1 && y <= 6 { GLASS } else { CONC });
        } else { add_brick(w, x, y, z, if edge { CONC } else { MASON }); }
    }}}
    boxf(w, x0, h, z0, x0 + wx, h + 1, z0 + wz, CONC);
    w.cur_paint = sign;
    for x in x0..x0 + wx { for y in h + 1..h + 4 { add_brick(w, x, y, z0 + wz - 1, CONC); } }
    // awnings
    w.cur_paint = P_WHITE;
    for x in x0 + 1..x0 + wx - 1 { if (x - x0) % 14 != 0 { add_brick(w, x, 7, z0 + wz, STEEL); add_brick(w, x, 7, z0 + wz + 1, STEEL); } }
    w.cur_paint = 0;
}
fn gas_station(w: &mut World, x0: i32, z0: i32) {
    w.cur_paint = P_DARK;
    for &(dx, dz) in [(0, 0), (15, 0), (0, 9), (15, 9)].iter() { for y in 0..8 { add_brick(w, x0 + dx, y, z0 + dz, STEEL); } }
    w.cur_paint = P_WHITE;
    boxf(w, x0 - 2, 8, z0 - 2, x0 + 18, 9, z0 + 12, STEEL);
    w.cur_paint = P_RED;
    for x in x0 - 2..x0 + 18 { add_brick(w, x, 9, z0 - 2, STEEL); add_brick(w, x, 9, z0 + 11, STEEL); }
    w.cur_paint = 0;
    for &dx in [5, 10].iter() { boxf(w, x0 + dx, 0, z0 + 4, x0 + dx + 1, 3, z0 + 6, CONC); }
    shops(w, x0 + 22, z0, 11, 8, P_RED);
}
fn silo(w: &mut World, cx: i32, cz: i32, r: i32, h: i32) {
    w.cur_paint = P_WARM;
    let rf = r as f32;
    for y in 0..=h { for dz in -r..=r { for dx in -r..=r {
        let d = sqrt((dx * dx + dz * dz) as f32);
        if d >= rf + 0.5 { continue; }
        if y == h || d >= rf - 0.5 { add_brick(w, cx + dx, y, cz + dz, CONC); }
    }}}
    w.cur_paint = 0;
}
fn billboard(w: &mut World, x0: i32, z0: i32) {
    w.cur_paint = P_DARK;
    for y in 0..10 { add_brick(w, x0, y, z0, STEEL); add_brick(w, x0 + 9, y, z0, STEEL); }
    for y in 10..16 { for x in x0 - 2..x0 + 12 {
        w.cur_paint = if y - 10 < 2 { P_ORANGE } else if (x - x0 + 2) % 4 < 2 { P_WHITE } else { P_BLUE };
        add_brick(w, x, y, z0, STEEL);
    }}
    w.cur_paint = 0;
}

fn build_map(w: &mut World) {
    // C — downtown: a 16-storey admin tower, two more office towers, a hotel block and a shopping parade
    begin_st(w, 1, true); office(w, cm(-12.0), cm(-10.0), 48, 36, 16, true, P_DARK); end_st(w);
    begin_st(w, 0, false); office(w, cm(14.0), cm(-35.0), 40, 28, 10, false, P_WARM); end_st(w);
    begin_st(w, 0, false); office(w, cm(-35.0), cm(13.0), 36, 28, 8, true, P_BLUE); end_st(w);
    begin_st(w, 0, false); apartment2(w, cm(-35.0), cm(-35.0), 36, 24, 8); end_st(w);
    begin_st(w, 0, false); shops(w, cm(-8.0), cm(23.0), 84, 20, P_ORANGE); end_st(w);
    begin_st(w, 0, false); billboard(w, cm(18.0), cm(-15.0)); end_st(w);
    // N — Fort Kessel with its keep, watchtowers, barracks and HQ
    begin_st(w, 2, true); castle(w, cm(-18.0), cm(-62.0), 72); end_st(w);
    begin_st(w, 0, false); watchtower(w, cm(-34.0), cm(-110.0)); watchtower(w, cm(30.0), cm(-110.0)); end_st(w);
    begin_st(w, 0, false); warehouse(w, cm(-35.0), cm(-104.0), 30, 20, 12); end_st(w);
    begin_st(w, 0, false); office(w, cm(14.0), cm(-106.0), 36, 24, 4, false, P_WARM); end_st(w);
    // NE — refinery
    begin_st(w, 0, false);
    tank(w, cm(58.0), cm(-100.0), 16, 26, P_WHITE); tank(w, cm(58.0), cm(-66.0), 16, 26, P_WHITE); tank(w, cm(80.0), cm(-104.0), 10, 20, P_ORANGE);
    pipe_rack(w, cm(68.0), cm(-86.0), 64);
    end_st(w);
    begin_st(w, 3, true); chimney(w, cm(102.0), cm(-64.0), 112); end_st(w);
    begin_st(w, 0, false); silo(w, cm(98.0), cm(-106.0), 9, 60); silo(w, cm(108.5), cm(-106.0), 9, 60); end_st(w);
    // E — viaduct over the canal, fuel stop, parking garage
    begin_st(w, 4, true); bridge(w, cm(45.0), cm(115.0), cm(0.0), 28, &[cm(46.0), cm(62.0), cm(78.0), cm(94.0), cm(110.0)]); end_st(w);
    begin_st(w, 0, false); gas_station(w, cm(50.0), cm(-30.0)); end_st(w);
    begin_st(w, 0, false); billboard(w, cm(100.0), cm(-14.0)); bunker(w, cm(96.0), cm(-32.0)); end_st(w);
    begin_st(w, 0, false); garage(w, cm(50.0), cm(9.0), 68, 52, 4); end_st(w);
    // W — old town
    begin_st(w, 5, true); church(w, cm(-92.0), cm(-35.0)); end_st(w);
    begin_st(w, 0, false); rowhouses(w, cm(-115.0), cm(14.0), 4, 12, 16); end_st(w);
    begin_st(w, 0, false); rowhouses(w, cm(-80.0), cm(14.0), 4, 12, 16); end_st(w);
    begin_st(w, 0, false); house(w, cm(-114.0), cm(-30.0), 14, 12, 10); house(w, cm(-62.0), cm(-32.0), 14, 12, 10); end_st(w);
    // NW — depot
    begin_st(w, 6, true); warehouse(w, cm(-115.0), cm(-112.0), 60, 32, 16); end_st(w);
    begin_st(w, 0, false); warehouse(w, cm(-80.0), cm(-112.0), 52, 28, 14); end_st(w);
    begin_st(w, 0, false); office(w, cm(-112.0), cm(-84.0), 32, 24, 4, false, P_WARM); end_st(w);
    begin_st(w, 0, false); watchtower(w, cm(-56.0), cm(-60.0)); bunker(w, cm(-80.0), cm(-66.0)); end_st(w);
    // SW — container yard under a big gantry
    begin_st(w, 7, true); gantry(w, cm(-114.0), cm(54.0), 64, 46, 48); end_st(w);
    begin_st(w, 0, false);
    let pals = [P_RED, P_BLUE, P_GREEN, P_ORANGE];
    let mut k = 0;
    for row in 0..4 {
        let z = cm(57.0) + row * 7;
        for col in 0..3 {
            let x = cm(-110.0) + col * 13;
            container(w, x, 0, z, true, pals[k % 4]); k += 1;
            if (row + col) % 2 == 0 { container(w, x, 5, z, true, pals[k % 4]); k += 1; if row == 1 { container(w, x, 10, z, true, pals[k % 4]); k += 1; } }
        }
    }
    for c in 0..3 { let x = cm(-74.0) + c * 7; container(w, x, 0, cm(60.0), false, pals[(k + c as usize) % 4]); if c != 1 { container(w, x, 5, cm(60.0), false, pals[(k + c as usize + 1) % 4]); } }
    end_st(w);
    begin_st(w, 0, false); office(w, cm(-76.0), cm(94.0), 28, 24, 2, false, P_WHITE); end_st(w);
    begin_st(w, 0, false); watchtower(w, cm(-52.0), cm(104.0)); end_st(w);
    // S — housing estate, school, corner shops
    begin_st(w, 0, false); apartment2(w, cm(-35.0), cm(46.0), 48, 28, 8); end_st(w);
    begin_st(w, 0, false); apartment2(w, cm(5.0), cm(46.0), 48, 28, 8); end_st(w);
    begin_st(w, 0, false); office(w, cm(-34.0), cm(76.0), 60, 32, 3, false, P_WARM); end_st(w);
    begin_st(w, 0, false); shops(w, cm(6.0), cm(88.0), 56, 20, P_BLUE); end_st(w);
    // SE — comms hill
    begin_st(w, 8, true); mast(w, cm(82.0), cm(80.0), 112); end_st(w);
    begin_st(w, 0, false); water_tower(w, cm(100.0), cm(52.0)); end_st(w);
    begin_st(w, 0, false); bunker(w, cm(62.0), cm(100.0)); house(w, cm(96.0), cm(98.0), 14, 12, 10); end_st(w);
    begin_st(w, 0, false); apartment2(w, cm(48.0), cm(48.0), 36, 24, 6); end_st(w);
}

fn reset_world(w: &mut World) {
    for c in 0..NCT { w.ctab[c] = 0; }
    for c in 0..w.ncpool * 512 { w.cpool[c] = 0; }
    w.ncpool = 0;
    for i in 0..MB { w.mat[i] = 0; w.next[i] = NONE; w.body[i] = STATIC; w.sid[i] = NOS; }
    w.nb = 0; w.bcur = 0; w.hw = 0; w.cursor = 0; w.nc = 0;
    for b in 0..MAXBODY { w.bd_alive[b] = false; w.bd_first[b] = NONE; w.bd_veh[b] = 0; w.bd_bucket[b] = NOB; }
    for k in 0..MAXP { w.p_life[k] = 0.0; }
    for k in 0..MAXSH { w.s_on[k] = false; }
    for r in 0..MAXR { w.r_alive[r] = false; }
    for v in 0..MAXV { w.v_alive[v] = false; w.v_in[v] = [0.0; 3]; w.v_aggro[v] = 0.0; w.v_ai[v] = false; w.v_respawn[v] = 0.0; }
    for a in 0..MAXA { w.a_on[a] = false; }
    for k in 0..MAXCG { w.cg_st[k] = 0; }
    for k in 0..HSIZE { w.hhead[k] = NONE; }
    for s in 0..MAXST { w.st_warm[s] = 0; w.st_active[s] = 0.0; w.st_solved[s] = false; w.st_conn[s] = false; w.st_broke[s] = false; w.st_settle[s] = 0.0; w.st_maxr[s] = 0.0; }
    for k in 0..MAXSET { w.set_dirty[k] = 0; w.set_built[k] = -10.0; }
    w.st_n = 0; w.cur_sid = NOS; w.cur_paint = 0; w.conn_any = false;
    w.pcur = 0; w.scur = 0; w.boost = 0; w.time = 0.0; w.blast_t = 0.0; w.nev = 0; w.breaks = 0; w.salvage = 0.0;
    w.nf = [0; 4]; w.ns = [0; 4];
}

// ---------------------------------------------------------------- exports
#[no_mangle]
pub extern "C" fn init() -> u32 {
    let w = world();
    if w.rng == 0 { w.rng = 0x9E3779B9; w.ai_on = true; }
    reset_world(w);
    build_map(w);
    let nb = w.nb;
    for i in 0..nb { finalize_brick(w, i); }
    for s in 0..w.st_n { mark_st(w, s, true); }
    for zn in 0..NZONE {
        let (cx, _, cz) = world_cell([ZONES[zn][0], 0.0, ZONES[zn][1]]);
        w.zx0[zn] = cx - (ZN / 2) as i32; w.zz0[zn] = cz - (ZN / 2) as i32;
        build_zone(w, zn); w.zdirty[zn] = false; w.zt[zn] = 0.0;
    }
    respawn_player(w);
    if w.ai_on { spawn_agents(w); for vi in 0..NAI { w.v_respawn[vi] = 0.0; respawn_ai(w, vi); } }
    spawn_vehicle(w, 5, [3.0, 1.0, 134.0], PI);
    spawn_vehicle(w, 6, [9.0, 1.0, 134.0], PI);
    w.v_ai[5] = false; w.v_ai[6] = false;
    update_cache(w);
    build_hash(w);
    write_structs(w);
    write_player(w);
    w.nb as u32
}
#[no_mangle]
pub extern "C" fn step(dt: f32) {
    let w = world();
    let dt = clampf(dt, 0.0, 1.0 / 30.0);
    w.time += dt;
    w.breaks = 0;
    w.crush_n = 0;
    w.frame = w.frame.wrapping_add(1);
    if w.blast_t > 0.0 { w.blast_t -= dt; }
    let t0 = tnow();
    ai_vehicles(w, dt);
    for _ in 0..2 { substep(w, dt * 0.5); }
    let t1 = tnow();
    for b in 0..w.hw { if w.bd_alive[b] && w.bd_dirty[b] { rebuild_body(w, b); } }
    connectivity(w);
    update_cache(w);
    let t2 = tnow();
    structural(w, dt);
    warm_step(w);
    let t3 = tnow();
    connectivity(w);
    update_cache(w);
    if w.frame % 30 == 0 { clear_rubble(w); }
    build_hash(w);
    rockets(w, dt);
    charges_step(w, dt);
    particles(w, dt);
    let t4 = tnow();
    shards_step(w, dt);
    let t5 = tnow();
    player_step(w, dt);
    agents_step(w, dt);
    let t6 = tnow();
    let ph = [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5];
    for k in 0..6 { w.prof[k] = w.prof[k] * 0.9 + ph[k] as f32 * 0.1; }
    // background-solve the nearest structure that has never been loaded
    let pending = (0..w.st_n).any(|s| w.st_warm[s] > 0);
    if !pending && w.frame % 10 == 0 {
        let mut best = usize::MAX; let mut bd = 120.0f32 * 120.0;
        for s in 0..w.st_n {
            if w.st_solved[s] { continue; }
            let c = lat_pos((w.st_min[s][0] + w.st_max[s][0]) / 2, 0, (w.st_min[s][2] + w.st_max[s][2]) / 2);
            let d = (c[0] - w.pl_pos[0]) * (c[0] - w.pl_pos[0]) + (c[2] - w.pl_pos[2]) * (c[2] - w.pl_pos[2]);
            if d < bd { bd = d; best = s; }
        }
        if best != usize::MAX { warm_struct(w, best); }
    }
    if w.breaks > 0 { let c = w.break_c; let m = w.breaks as f32; event(w, 2.0, c, m); }
    // vehicle hits on the player on foot are handled in player_step; outputs below
    let (mut dynb, mut awake, mut bodies) = (0, 0, 0);
    for b in 0..w.hw { if w.bd_alive[b] && w.bd_veh[b] == 0 { bodies += 1; dynb += w.bd_count[b]; if !w.bd_asleep[b] { awake += 1; } } }
    let mut shards = 0;
    for k in 0..MAXSH { if w.s_on[k] { shards += 1; } }
    let mut chasing = 0;
    for vi in 0..MAXV {
        let o = vi * 20;
        if !veh_ok(w, vi) { w.out_vh[o] = 0.0; continue; }
        let b = w.v_body[vi] as usize;
        let r = w.bd_rot[b];
        let hp = veh_pos(w, vi);
        let q = w.bd_q[b];
        let fwd = [r[2], r[5], r[8]];
        w.out_vh[o] = 1.0;
        w.out_vh[o + 1] = hp[0]; w.out_vh[o + 2] = hp[1]; w.out_vh[o + 3] = hp[2];
        w.out_vh[o + 4] = q[0]; w.out_vh[o + 5] = q[1]; w.out_vh[o + 6] = q[2]; w.out_vh[o + 7] = q[3];
        for k in 0..4 { w.out_vh[o + 8 + k] = w.v_comp[vi][k]; }
        w.out_vh[o + 12] = w.v_spin[vi];
        w.out_vh[o + 13] = w.v_steer[vi];
        let ai_driven = w.v_ai[vi] && w.pv != vi as i32;
        let mode = if !ai_driven { 0.0 } else if w.v_aggro[vi] > 0.0 && w.pl_dead <= 0.0 { chasing += 1; 2.0 } else { 1.0 };
        w.out_vh[o + 14] = mode;
        w.out_vh[o + 15] = dot(w.bd_v[b], fwd);
        w.out_vh[o + 16] = w.v_touch[vi] as f32;
        w.out_vh[o + 17] = w.v_in[vi][0];
        w.out_vh[o + 18] = if w.v_ai[vi] { 1.0 } else { 0.0 };
        w.out_vh[o + 19] = if w.pv == vi as i32 { 1.0 } else { 0.0 };
    }
    let (stat, unsolved) = write_structs(w);
    write_player(w);
    w.stats[0] = stat as f32; w.stats[1] = dynb as f32; w.stats[2] = bodies as f32; w.stats[3] = awake as f32;
    w.stats[4] = w.nc as f32; w.stats[5] = w.out_np as f32; w.stats[7] = shards as f32; w.stats[11] = chasing as f32;
    w.stats[12] = w.salvage / 1000.0; w.stats[14] = unsolved as f32; w.stats[15] = w.st_n as f32;
}
fn write_player(w: &mut World) {
    let pl = w.pl_pos;
    w.out_pl = [pl[0], pl[1], pl[2], w.pl_hp, if w.pl_dead > 0.0 { 2.0 } else if w.pv >= 0 { 1.0 } else { 0.0 },
        w.pl_vel[0], w.pl_vel[1], w.pl_vel[2], if w.pl_ground { 1.0 } else { 0.0 }, w.pl_dead, w.pv as f32, w.salvage, 0.0, 0.0, 0.0, 0.0];
    let mut ncg = 0.0; for k in 0..MAXCG { if w.cg_st[k] != 0 { ncg += 1.0; } }
    w.out_pl[12] = ncg;
    // nearest enterable vehicle
    let mut nv = -1.0; let mut nd = 16.0f32;
    if w.pv < 0 { for vi in 0..MAXV { if veh_ok(w, vi) { let d = sub(veh_pos(w, vi), w.pl_pos); let dd = d[0] * d[0] + d[2] * d[2]; if dd < nd { nd = dd; nv = vi as f32; } } } }
    w.out_pl[13] = nv;
}
fn write_structs(w: &mut World) -> (u32, u32) {
    let mut stat = 0; let mut unsolved = 0;
    for s in 0..w.st_n {
        let o = s * 12;
        let (a, b) = (w.st_min[s], w.st_max[s]);
        let pa = lat_pos(a[0], a[1], a[2]); let pb = lat_pos(b[0], b[1], b[2]);
        w.out_st[o] = pa[0] - H; w.out_st[o + 1] = pa[2] - H; w.out_st[o + 2] = pb[0] + H; w.out_st[o + 3] = pb[2] + H;
        w.out_st[o + 4] = w.st_init[s] as f32; w.out_st[o + 5] = w.st_now[s] as f32;
        w.out_st[o + 6] = w.st_top0[s] as f32; w.out_st[o + 7] = w.st_top[s] as f32;
        w.out_st[o + 8] = if w.st_target[s] { 1.0 } else { 0.0 }; w.out_st[o + 9] = w.st_kind[s] as f32;
        w.out_st[o + 10] = if w.st_active[s] > 0.0 { 1.0 } else { 0.0 }; w.out_st[o + 11] = w.st_top0[s] as f32 * S;
        stat += w.st_now[s];
        if !w.st_solved[s] { unsolved += 1; }
    }
    (stat, unsolved)
}
/// returns the dirty mesh set to rebuild next (urgent first, then lazy ones that have waited, nearest first), or -1
#[no_mangle]
pub extern "C" fn next_dirty(px: f32, pz: f32, lazy_ok: u32) -> i32 {
    let w = world();
    let mut best = -1i32; let mut bd = f32::MAX;
    for id in 0..MAXSET {
        let lvl = w.set_dirty[id];
        if lvl == 0 { continue; }
        if id < SSETS && id / NBAND >= w.st_n { w.set_dirty[id] = 0; continue; }
        if lvl == 1 && (lazy_ok == 0 || w.time - w.set_since[id] < 0.6) { continue; }
        if w.time - w.set_built[id] < 0.12 && w.time >= w.set_built[id] { continue; }
        let c = if id < SSETS { let s = id / NBAND; lat_pos((w.st_min[s][0] + w.st_max[s][0]) / 2, 0, (w.st_min[s][2] + w.st_max[s][2]) / 2) }
            else { let k = id - SSETS; [(k % 10) as f32 * 32.0 - HALF + 16.0, 0.0, (k / 10) as f32 * 32.0 - HALF + 16.0] };
        let mut d = (c[0] - px) * (c[0] - px) + (c[2] - pz) * (c[2] - pz);
        if lvl == 2 { d -= 1e9; }
        if d < bd { bd = d; best = id as i32; }
    }
    best
}
#[no_mangle]
pub extern "C" fn build_set(id: u32) -> u32 {
    let w = world();
    let id = id as usize;
    if id >= MAXSET { return 0; }
    w.set_dirty[id] = 0;
    w.set_built[id] = w.time;
    w.nf[0] = 0; w.nf[1] = 0;
    let sv = w.stress_view;
    if id < SSETS {
        let (s, band) = (id / NBAND, id % NBAND);
        if s < w.st_n { for i in w.st_lo[s]..w.st_hi[s] { if in_struct(w, i, s) && band_of(w.by[i] as i32) == band { emit_brick(w, i, sv, false); } } }
    } else {
        let k = (id - SSETS) as u16;
        for b in 0..w.hw {
            if !w.bd_alive[b] || !w.bd_asleep[b] || w.bd_bucket[b] != k || w.bd_veh[b] != 0 { continue; }
            w.bd_inmesh[b] = true;
            let mut j = w.bd_first[b];
            while j != NONE { let i = j as usize; j = w.next[i]; emit_brick(w, i, sv, false); }
        }
    }
    w.nf[0] as u32
}
#[no_mangle]
pub extern "C" fn render() {
    let w = world();
    let sv = w.stress_view;
    w.nf[2] = 0; w.nf[3] = 0;
    for b in 0..w.hw {
        if !w.bd_alive[b] || w.bd_veh[b] != 0 { continue; }
        if w.bd_asleep[b] && w.bd_inmesh[b] { continue; }
        let mut k = w.bd_first[b];
        while k != NONE { let i = k as usize; k = w.next[i]; emit_brick(w, i, sv, true); }
    }
    shard_out(w);
}
#[no_mangle]
pub extern "C" fn fire_rocket(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
    let w = world();
    if w.pl_dead > 0.0 { return; }
    for r in 0..MAXR {
        if !w.r_alive[r] {
            w.r_alive[r] = true; w.r_age[r] = 0.0;
            let d = [dx, dy, dz];
            w.r_pos[r] = madd([ox, oy, oz], d, 0.4);
            w.r_vel[r] = mul(d, 46.0);
            return;
        }
    }
}
#[no_mangle]
pub extern "C" fn raycast(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32, maxd: f32) -> u32 {
    let w = world();
    let o = [ox, oy, oz]; let d = [dx, dy, dz];
    let mut best = maxd;
    if dy < -1e-4 { best = minf(best, -oy / dy); }
    let mut t = 0.0;
    while t < best {
        let (x, y, z) = world_cell(madd(o, d, t));
        if static_at(w, x, y, z) { best = t; break; }
        t += 0.05;
    }
    for b in 0..w.hw {
        if !w.bd_alive[b] { continue; }
        if w.pv >= 0 && w.bd_veh[b] as i32 == w.pv + 1 { continue; }
        let oc = sub(o, w.bd_pos[b]);
        let tb = dot(oc, d);
        let dd = dot(oc, oc) - tb * tb;
        if dd > w.bd_rad[b] * w.bd_rad[b] || -tb > best + w.bd_rad[b] { continue; }
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize; k = w.next[i];
            if w.mat[i] == 0 { continue; }
            let oc = sub(o, w.wpos[i]);
            let bq = dot(oc, d);
            let c = dot(oc, oc) - RB * RB;
            let disc = bq * bq - c;
            if disc < 0.0 { continue; }
            let tt = -bq - sqrt(disc);
            if tt > 0.0 && tt < best { best = tt; }
        }
    }
    if best >= maxd { w.hit[3] = 0.0; return 0; }
    let p = madd(o, d, best);
    w.hit[0] = p[0]; w.hit[1] = p[1]; w.hit[2] = p[2]; w.hit[3] = best;
    1
}
#[no_mangle]
pub extern "C" fn explode(x: f32, y: f32, z: f32, r: f32, p: f32) { explode_at(world(), [x, y, z], r, p); }
/// sledgehammer swing from (o) along (d); returns material hit (0 = air)
#[no_mangle]
pub extern "C" fn hammer(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> u32 {
    let w = world();
    if w.pl_dead > 0.0 || w.pv >= 0 { return 0; }
    let o = [ox, oy, oz]; let d = norm([dx, dy, dz]);
    let reach = 2.6;
    let mut ts = reach; let mut js = usize::MAX;
    let mut t = 0.0;
    while t < reach {
        let (x, y, z) = world_cell(madd(o, d, t));
        let j = static_brick(w, x, y, z);
        if j != usize::MAX { ts = t; js = j; break; }
        t += 0.05;
    }
    let mut td = ts; let mut jd = usize::MAX;
    for b in 0..w.hw {
        if !bodies_near(w, b, o, reach) { continue; }
        let mut k = w.bd_first[b];
        while k != NONE {
            let i = k as usize; k = w.next[i];
            if w.mat[i] == 0 { continue; }
            let oc = sub(o, w.wpos[i]);
            let bq = dot(oc, d);
            let disc = bq * bq - (dot(oc, oc) - (RB + 0.15) * (RB + 0.15));
            if disc < 0.0 { continue; }
            let tt = -bq - sqrt(disc);
            if tt > 0.0 && tt < td { td = tt; jd = i; }
        }
    }
    let hitp = madd(o, d, minf(ts, td));
    for a in 0..MAXA {
        if !w.a_on[a] || w.a_state[a] == 2 { continue; }
        let pa = add(w.a_pos[a], [0.0, 0.9, 0.0]);
        let rel = sub(pa, o);
        let along = dot(rel, d);
        if along > 0.0 && along < 2.2 && len(sub(rel, mul(d, along))) < 0.9 { knock_agent(w, a, add(mul(d, 6.0), [0.0, 3.0, 0.0])); }
    }
    if jd != usize::MAX {
        let b = w.body[jd] as usize;
        let m = w.mat[jd];
        let p = w.wpos[jd];
        wake_body(w, b);
        impulse_at(w, b, p, mul(d, if w.bd_veh[b] != 0 { 2500.0 } else { 900.0 }));
        let vi = w.bd_veh[b] as usize;
        if vi > 0 && vi - 1 < NAI { w.v_aggro[vi - 1] = 25.0; }
        if m < SLUG { let br = damage_brick(w, jd, 0.0, 6000.0 * FRAG[m as usize]); if br > 0 { shards_from(w, p, jd, 2, 3.0, mul(d, 3.0)); } }
        event(w, 10.0, hitp, m as f32);
        return m as u32;
    }
    if js != usize::MAX {
        let m = w.mat[js];
        let p = bpos(w, js);
        match m {
            GLASS | WOOD | MASON => {
                shards_from(w, p, js, if m == GLASS { 6 } else { 3 }, 3.5, mul(d, 4.0));
                let col = brick_col(w, js);
                dust_burst(w, p, col, 2, 1.0);
                if m == GLASS { event(w, 4.0, p, 1.0); }
                remove_brick(w, js);
                for k in 0..6 { let j = nbr(w, js, k); if j != usize::MAX && w.mat[j] != 0 && w.body[j] == STATIC { damage_brick(w, j, 0.25, 0.0); } }
            }
            CONC => {
                let br = damage_brick(w, js, 0.5, 0.0);
                let col = brick_col(w, js);
                dust_burst(w, p, col, 2, 0.8);
                shards_from(w, p, js, 2, 2.5, mul(d, 3.0));
                if br >= 3 { remove_brick(w, js); }
            }
            _ => { damage_brick(w, js, 0.18, 0.0); shards_from(w, p, js, 1, 2.0, mul(d, 2.0)); }
        }
        event(w, 10.0, hitp, m as f32);
        return m as u32;
    }
    0
}
#[no_mangle]
pub extern "C" fn throw_charge(ox: f32, oy: f32, oz: f32, vx: f32, vy: f32, vz: f32) -> u32 {
    let w = world();
    if w.pl_dead > 0.0 { return 0; }
    for k in 0..MAXCG {
        if w.cg_st[k] == 0 { w.cg_st[k] = 1; w.cg_pos[k] = [ox, oy, oz]; w.cg_vel[k] = [vx, vy, vz]; return 1; }
    }
    0
}
#[no_mangle]
pub extern "C" fn detonate() -> u32 {
    let w = world();
    let mut n = 0;
    for k in 0..MAXCG {
        if w.cg_st[k] == 0 { continue; }
        let p = w.cg_pos[k];
        w.cg_st[k] = 0;
        explode_at(w, p, 3.0, 1.4);
        n += 1;
    }
    n
}
#[no_mangle]
pub extern "C" fn set_move(mx: f32, mz: f32, jump: u32, sprint: u32) {
    let w = world();
    let l = sqrt(mx * mx + mz * mz);
    let s = if l > 1.0 { 1.0 / l } else { 1.0 };
    w.pl_in = [mx * s, mz * s, 0.0, if sprint != 0 { 1.0 } else { 0.0 }];
    if jump != 0 { w.pl_jump = true; }
}
/// enter the nearest vehicle or step out of the current one
#[no_mangle]
pub extern "C" fn use_vehicle() -> i32 {
    let w = world();
    if w.pl_dead > 0.0 { return -1; }
    if w.pv >= 0 {
        let vi = w.pv as usize;
        w.v_in[vi] = [0.0; 3];
        w.pv = -1;
        if veh_ok(w, vi) {
            let b = w.v_body[vi] as usize;
            let r = w.bd_rot[b];
            let right = cross([r[2], r[5], r[8]], [r[1], r[4], r[7]]);
            let hp = veh_pos(w, vi);
            let mut p = [hp[0] - right[0] * 2.0, 0.0, hp[2] - right[2] * 2.0];
            if !box_free(w, p) { p = [hp[0] + right[0] * 2.0, 0.0, hp[2] + right[2] * 2.0]; }
            if !box_free(w, p) { p = [hp[0], hp[1] + 1.6, hp[2]]; }
            w.pl_pos = p; w.pl_vel = [0.0; 3];
        }
        return -1;
    }
    let mut best = -1i32; let mut bd = 16.0f32;
    for vi in 0..MAXV {
        if !veh_ok(w, vi) { continue; }
        let d = sub(veh_pos(w, vi), w.pl_pos);
        let dd = d[0] * d[0] + d[2] * d[2];
        if dd < bd { bd = dd; best = vi as i32; }
    }
    if best >= 0 {
        let vi = best as usize;
        w.pv = best; w.v_ai[vi] = false; w.v_aggro[vi] = 0.0; w.v_in[vi] = [0.0; 3];
        let b = w.v_body[vi] as usize;
        let r = w.bd_rot[b];
        if r[4] < 0.3 { let hp = veh_pos(w, vi); let yaw = atan2f(r[2], r[8]); spawn_vehicle(w, vi, [hp[0], maxf(hp[1], 0.0) + 1.4, hp[2]], yaw); }
    }
    best
}
#[no_mangle]
pub extern "C" fn set_input(t: f32, s: f32, hb: f32) {
    let w = world();
    if w.pv >= 0 { let vi = w.pv as usize; w.v_in[vi] = [clampf(t, -1.0, 1.0), clampf(s, -1.0, 1.0), hb]; }
}
#[no_mangle]
pub extern "C" fn flip_vehicle() {
    let w = world();
    if w.pv < 0 { return; }
    let vi = w.pv as usize;
    if !veh_ok(w, vi) { return; }
    let b = w.v_body[vi] as usize;
    let r = w.bd_rot[b];
    let hp = veh_pos(w, vi);
    let yaw = atan2f(r[2], r[8]);
    spawn_vehicle(w, vi, [hp[0], maxf(hp[1], 0.0) + 1.6, hp[2]], yaw);
}
#[no_mangle]
pub extern "C" fn set_ai(on: u32) {
    let w = world();
    let was = w.ai_on;
    w.ai_on = on != 0;
    if w.ai_on && !was { spawn_agents(w); for vi in 0..NAI { if w.v_ai[vi] || !veh_ok(w, vi) { w.v_respawn[vi] = 0.0; } } }
    if !w.ai_on { for a in 0..MAXA { w.a_on[a] = false; } }
}
#[no_mangle] pub extern "C" fn set_stress_view(on: u32) { let w = world(); w.stress_view = on != 0; for s in 0..w.st_n { mark_st(w, s, false); for k in 0..NBAND { w.set_since[s * NBAND + k] = -10.0; } } }
#[no_mangle] pub extern "C" fn clear_events() { world().nev = 0; }
#[no_mangle] pub extern "C" fn n_events() -> u32 { world().nev as u32 }
#[no_mangle] pub extern "C" fn ptr_events() -> *const f32 { world().ev.as_ptr() }
#[no_mangle] pub extern "C" fn face_cap(k: u32) -> u32 { FCAPS[(k & 3) as usize] as u32 }
#[no_mangle] pub extern "C" fn n_faces(k: u32) -> u32 { world().nf[(k & 3) as usize] as u32 }
#[no_mangle] pub extern "C" fn ptr_faces(k: u32) -> *const f32 {
    let w = world();
    match k & 3 { 0 => w.out_so.as_ptr(), 1 => w.out_sg.as_ptr(), 2 => w.out_do.as_ptr(), _ => w.out_dg.as_ptr() }
}
#[no_mangle] pub extern "C" fn n_particles() -> u32 { world().out_np as u32 }
#[no_mangle] pub extern "C" fn ptr_particles() -> *const f32 { world().out_p.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_pcolors() -> *const f32 { world().out_pc.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_rockets() -> *const f32 { world().out_rk.as_ptr() }
#[no_mangle] pub extern "C" fn max_rockets() -> u32 { MAXR as u32 }
#[no_mangle] pub extern "C" fn ptr_stats() -> *const f32 { world().stats.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_hit() -> *const f32 { world().hit.as_ptr() }
#[no_mangle] pub extern "C" fn shard_cap() -> u32 { MAXSH as u32 }
#[no_mangle] pub extern "C" fn n_shards(v: u32) -> u32 { world().ns[(v & 3) as usize] }
#[no_mangle] pub extern "C" fn ptr_shards() -> *const f32 { world().out_sh.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_shcolors() -> *const f32 { world().out_shc.as_ptr() }
#[no_mangle] pub extern "C" fn max_agents() -> u32 { MAXA as u32 }
#[no_mangle] pub extern "C" fn ptr_agents() -> *const f32 { world().out_ag.as_ptr() }
#[no_mangle] pub extern "C" fn max_vehicles() -> u32 { MAXV as u32 }
#[no_mangle] pub extern "C" fn ptr_vehicles() -> *const f32 { world().out_vh.as_ptr() }
#[no_mangle] pub extern "C" fn max_particles() -> u32 { MAXP as u32 }
#[no_mangle] pub extern "C" fn n_structs() -> u32 { world().st_n as u32 }
#[no_mangle] pub extern "C" fn max_sets() -> u32 { MAXSET as u32 }
#[no_mangle] pub extern "C" fn ptr_structs() -> *const f32 { world().out_st.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_player() -> *const f32 { world().out_pl.as_ptr() }
#[no_mangle] pub extern "C" fn ptr_charges() -> *const f32 { world().out_cg.as_ptr() }
#[no_mangle] pub extern "C" fn max_charges() -> u32 { MAXCG as u32 }
#[no_mangle] pub extern "C" fn ptr_prof() -> *const f32 { world().prof.as_ptr() }
