//! The barn: an aisled timber barn on the pattern of the Barley Barn at Cressing Temple (Essex, about 1220), five
//! bays of 4.5 m, 12.5 m wide, its ridge 10.75 m up.
//!
//! - Six frames across: two arcade posts 0.5 m square on stone pads, 6 m apart, a tie beam on their heads, a
//!   brace from each post up to the tie, a collar between the rafters high up; a low aisle each side, 3 m wide,
//!   its wall post tied to the arcade post.
//! - Along the barn: an arcade plate on the tie beams' ends over each row of posts, a wall plate on each aisle
//!   wall's posts, the aisle walls on a stone plinth.
//! - One roof slope each side from the aisle's eaves (2.75 m) over the arcade plate to the ridge, at 4 in 3:
//!   rafters every 1.5 m, boards and tiles on them.
//! - Weatherboards on the aisle walls and the gable ends; the great door in the south gable, 3 m wide.
//! - Inside, loose: bales, a cart.
use crate::brick::{mat, Bricks, F_INFILL};
use crate::kit::*;
use alloc::vec::Vec;

pub const BAYS: usize = 5;
pub const BAY: f64 = 4.5;
/// the arcade posts' and the aisle walls' lines (x), the posts' top, the eaves
pub const NAVE: f64 = 3.0;
pub const AISLE: f64 = 6.0;
const POST_TOP: f64 = 5.625;
const TIE_TOP: f64 = 6.0;
const EAVE: f64 = 2.75;
const SLOPE: f64 = 4.0 / 3.0;
/// half the rafters' depth, measured upright
const DV: f64 = 0.125;

#[derive(Default, Clone, Debug)]
pub struct Parts { pub posts: Vec<u32>, pub ties: Vec<u32>, pub braces: Vec<Vec<u32>>, pub rafters: Vec<Vec<u32>>, pub boards: Vec<Vec<u32>>, pub collars: Vec<Vec<u32>>, pub cladding: Vec<u32>, pub contents: Vec<u32> }

pub fn frame_z(k: usize) -> f64 { (k as f64 - BAYS as f64 / 2.0) * BAY }
/// the rafters' underside at x
fn under(x: f64) -> f64 { EAVE - DV + (AISLE - x.abs()) * SLOPE }

pub fn build() -> Bricks { build_parts().0 }
pub fn build_parts() -> (Bricks, Parts) {
    let mut b = Bricks::new();
    let mut p = Parts::default();
    let mut sid: u16 = 100;
    let tb = |b: &mut Bricks, lo: P3, hi: P3| bx(b, lo, hi, mat::TIMBER, P_DARKWOOD);
    let (z0, z1) = (frame_z(0), frame_z(BAYS));
    let up = [0.0, 1.0, 0.0];
    // ---- the frames
    for k in 0..=BAYS {
        let z = frame_z(k);
        for &s in &[-1.0f64, 1.0] {
            let x = s * NAVE;
            // a stone pad and the arcade post on it
            bx(&mut b, [x - 0.5, 0.0, z - 0.5], [x + 0.5, 0.5, z + 0.5], mat::MASONRY, P_TRAVERTINE);
            p.posts.push(tb(&mut b, [x - 0.25, 0.5, z - 0.25], [x + 0.25, POST_TOP, z + 0.25]));
            // the aisle: its wall post (on the plinth) and the tie from it to the arcade post, under the eaves
            let xa = s * AISLE;
            p.posts.push(tb(&mut b, [xa - 0.125, 0.5, z - 0.125], [xa + 0.125, EAVE - 0.25, z + 0.125]));
            let (ta, tc) = if s < 0.0 { (xa + 0.125, x - 0.25) } else { (x + 0.25, xa - 0.125) };
            tb(&mut b, [ta, EAVE - 0.5, z - 0.125], [tc, EAVE - 0.25, z + 0.125]);
            // the brace from the post's inner face up to the tie beam's underside, at 45 degrees, 1.5 m out
            let foot = [x - s * 0.25, POST_TOP - 1.5, z];
            let head = [x - s * (0.25 + 1.5), POST_TOP, z];
            let ax = unit(sub(head, foot));
            let nn = [-ax[1] * -s, ax[0] * -s, 0.0];
            p.braces.push(member(&mut b, foot, head, unit(nn), [0.0, 0.0, 1.0], 0.25, 0.25, [1.0, 0.0, 0.0], up, 3.0, mat::TIMBER, P_DARKWOOD, &mut sid));
        }
        // the tie beam on the posts' heads
        p.ties.push(tb(&mut b, [-NAVE - 0.25, POST_TOP, z - 0.25], [NAVE + 0.25, TIE_TOP, z + 0.25]));
    }
    // ---- along the barn: the arcade plates on the ties' ends (their tops the rafters' underside), the wall plates,
    // the plinths; a length to a bay, meeting over the frames
    for &s in &[-1.0f64, 1.0] {
        let (xa, xb) = (s * (NAVE - 0.25), s * (NAVE + 0.25));
        let poly = [[xa, TIE_TOP], [xb, TIE_TOP], [xb, under(xb)], [xa, under(xa)]];
        for k in 0..BAYS {
            let (za, zb) = (if k == 0 { z0 - 0.25 } else { frame_z(k) }, if k + 1 == BAYS { z1 + 0.25 } else { frame_z(k + 1) });
            sid += 1; prism_z(&mut b, &poly, za, zb, mat::TIMBER, P_DARKWOOD, sid);
            let x = s * AISLE;
            tb(&mut b, [x - 0.125, EAVE - 0.25, za.max(z0 - 0.125)], [x + 0.125, EAVE, zb.min(z1 + 0.125)]);
            bx(&mut b, [x - 0.25, 0.0, za.max(z0 - 0.125)], [x + 0.25, 0.5, zb.min(z1 + 0.125)], mat::MASONRY, P_TRAVERTINE);
            // weatherboards outside the posts, from the plinth to the plate
            let (ca, cb) = if s < 0.0 { (x - 0.25, x - 0.125) } else { (x + 0.125, x + 0.25) };
            let i = bx(&mut b, [ca, 0.5, za.max(z0 - 0.125)], [cb, EAVE, zb.min(z1 + 0.125)], mat::TIMBER, P_DARKWOOD);
            b.flags[i as usize] |= F_INFILL; p.cladding.push(i);
        }
    }
    // ---- the roof: rafters every 1.5 m, feet level on the wall plates, over the arcade plates, heads together at
    // the ridge; boards and tiles between them; a collar under each frame's pair
    let cosr = 1.0 / crate::math::sqrt64(1.0 + SLOPE * SLOPE);
    let depth = 2.0 * DV * cosr;
    let ridge = EAVE + AISLE * SLOPE;
    let nr = ((z1 - z0) / 1.5 + 0.5) as usize + 1;
    for &s in &[-1.0f64, 1.0] {
        let a = unit([-s, SLOPE, 0.0]);
        let nn = unit(sub(up, mul(a, dot(up, a))));
        for k in 0..nr {
            let z = z0 + 1.5 * k as f64;
            p.rafters.push(member(&mut b, [s * AISLE, EAVE, z], [0.0, ridge, z], nn, [0.0, 0.0, 1.0], depth, 0.125, up, [1.0, 0.0, 0.0], 3.5, mat::TIMBER, P_DARKWOOD, &mut sid));
        }
        let lift = depth / 2.0 + 0.03125;
        for k in 0..nr - 1 {
            let zm = z0 + 1.5 * k as f64 + 0.75;
            let (f, h) = ([s * AISLE, EAVE, zm], [0.0, ridge, zm]);
            let (bf, bh) = (add(f, mul(nn, lift)), add(h, mul(nn, lift)));
            let t_top = -(depth / 2.0) * nn[1] / a[1];
            p.boards.push(member(&mut b, add(bf, mul(a, t_top)), bh, nn, [0.0, 0.0, 1.0], 0.0625, 1.5, mul(a, -1.0), [1.0, 0.0, 0.0], 3.5, mat::TIMBER, P_TILE, &mut sid));
        }
    }
    // the collars, 8.5 m up, from rafter underside to rafter underside (at the frames: every third rafter)
    {
        let y = 8.5;
        let half = AISLE - (y - (EAVE - DV)) / SLOPE;   // where the underside is at y
        // (the slopes' undersides: y - S x = c in the west, y + S x = c in the east)
        let nl = unit([-SLOPE, 1.0, 0.0]);
        let nrn = unit([SLOPE, 1.0, 0.0]);
        for k in 0..=BAYS {
            let z = frame_z(k);
            p.collars.push(member(&mut b, [-half, y, z], [half, y, z], up, [0.0, 0.0, 1.0], 0.25, 0.125, nl, nrn, 3.0, mat::TIMBER, P_DARKWOOD, &mut sid));
        }
    }
    // ---- the gable ends: boards between the posts, and above the ties up to the roof (the south one with the door)
    for (g, &z) in [z0, z1].iter().enumerate() {
        // (outside the frame, against the arcade posts' and the tie beam's faces; standing on the ground)
        let (za, zb) = if g == 0 { (z - 0.375, z - 0.25) } else { (z + 0.25, z + 0.375) };
        let mut board = |b: &mut Bricks, x0: f64, x1: f64, y0: f64, y1: f64| { let i = bx(b, [x0, y0, za], [x1, y1, zb], mat::TIMBER, P_DARKWOOD); b.flags[i as usize] |= F_INFILL; p.cladding.push(i); };
        // the aisles' ends
        for &s in &[-1.0f64, 1.0] { let (xa, xb) = if s < 0.0 { (-AISLE - 0.125, -NAVE + 0.25) } else { (NAVE - 0.25, AISLE + 0.125) }; board(&mut b, xa, xb, 0.0, EAVE - 0.25); }
        // the nave's end: whole in the north, round the door (3 m wide, 4 m high) in the south
        if g == 0 {
            board(&mut b, -NAVE + 0.25, -1.5, 0.0, TIE_TOP); board(&mut b, 1.5, NAVE - 0.25, 0.0, TIE_TOP); board(&mut b, -1.5, 1.5, 4.5, TIE_TOP);
        } else { board(&mut b, -NAVE + 0.25, 0.0, 0.0, TIE_TOP); board(&mut b, 0.0, NAVE - 0.25, 0.0, TIE_TOP); }
        // above: the gable's boards, their tops the rafters' underside, standing on the tie beam and on the aisles' ties
        let (ga, gb) = if g == 0 { (z - 0.25, z - 0.125) } else { (z + 0.125, z + 0.25) };
        for &s in &[-1.0f64, 1.0] {
            let nave = [[0.0, TIE_TOP], [s * (NAVE - 0.25), TIE_TOP], [s * (NAVE - 0.25), under(NAVE - 0.25)], [0.0, under(0.0)]];
            sid += 1; let i = prism_z(&mut b, &nave, ga, gb, mat::TIMBER, P_DARKWOOD, sid); b.flags[i as usize] |= F_INFILL; p.cladding.push(i);
            let aisle = [[s * (NAVE + 0.25), EAVE - 0.25], [s * (AISLE - 0.125), EAVE - 0.25], [s * (AISLE - 0.125), under(AISLE - 0.125)], [s * (NAVE + 0.25), under(NAVE + 0.25)]];
            sid += 1; let i = prism_z(&mut b, &aisle, ga, gb, mat::TIMBER, P_DARKWOOD, sid); b.flags[i as usize] |= F_INFILL; p.cladding.push(i);
        }
    }
    // ---- inside, loose: bales stacked in the north bays, a cart in the middle
    for &(x, z, n) in &[(-2.25, 6.0, 3), (-1.0, 6.0, 2), (0.5, 6.5, 3), (-2.25, 8.0, 2), (1.5, 8.5, 1), (-4.75, 2.0, 2), (4.25, -3.0, 2), (4.25, 5.0, 1)] {
        for k in 0..n { sid += 1; let y = 0.5 * k as f64; loose(&mut b, &mut p.contents, [x, y, z], [x + 1.0, y + 0.5, z + 0.5], mat::TIMBER, P_PARTITION, sid); }
    }
    {
        sid += 1; let s = sid;
        loose(&mut b, &mut p.contents, [-0.75, 0.625, -4.0], [0.75, 0.75, -1.0], mat::TIMBER, P_WOOD, s);
        for &(x, z) in &[(-0.875, -3.75), (0.75, -3.75), (-0.875, -1.875), (0.75, -1.875)] { loose(&mut b, &mut p.contents, [x, 0.0, z], [x + 0.125, 0.625, z + 0.625], mat::TIMBER, P_WOOD, s); }
        loose(&mut b, &mut p.contents, [-0.75, 0.75, -4.0], [-0.625, 1.25, -1.0], mat::TIMBER, P_WOOD, s);
        loose(&mut b, &mut p.contents, [0.625, 0.75, -4.0], [0.75, 1.25, -1.0], mat::TIMBER, P_WOOD, s);
    }
    b.build_joints();
    b.log.clear();
    (b, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_barn_is_what_it_should_be_and_stands() {
        let (b, p) = build_parts();
        assert_eq!(p.ties.len(), BAYS + 1); assert_eq!(p.posts.len(), 4 * (BAYS + 1));
        assert_eq!(p.braces.len(), 2 * (BAYS + 1)); assert_eq!(p.collars.len(), BAYS + 1);
        assert_eq!(p.rafters.len(), 32); assert_eq!(p.boards.len(), 30);
        let r = stands("the barn", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
