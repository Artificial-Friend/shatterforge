//! Two works of the industrial town: the water tower and the mill chimney.
//!
//! The water tower: a riveted steel tank on a braced steel tower, as railways and towns built them about 1900. Four
//! legs on a 5 m square, 15 m tall in four tiers; a ring of struts at each tier and one diagonal on each face of
//! each tier, leaning the other way from the tier below; under the tank two pairs of beams that run out past the
//! legs; the tank eight-sided, 6.5 m across and 4.5 m high, of plates in two courses, on a plate floor, under a low
//! pointed roof; a riser pipe up the middle. It stands empty.
//!
//! The chimney: a square brick stack 24 m tall on a plinth, 2.5 m at the foot and 1.75 m at the top in four
//! stages, hollow (the flue 1.5 m at the foot), its walls 0.5 m thick below and 0.375 m above, a cap of two
//! oversailing courses.
use crate::brick::{mat, Bricks};
use crate::kit::*;
use alloc::vec::Vec;

pub const LEG: f64 = 2.5;
pub const TIER: f64 = 3.75;
pub const TIERS: usize = 4;
pub const TANK_R: f64 = 3.25;
pub const TANK_H: f64 = 4.5;

#[derive(Default, Clone, Debug)]
pub struct Tower { pub legs: Vec<Vec<u32>>, pub struts: Vec<u32>, pub braces: Vec<Vec<u32>>, pub tank: Vec<u32>, pub floor: u32, pub roof: Vec<u32>, pub riser: Vec<u32> }

pub fn water_tower() -> Bricks { water_tower_parts().0 }
pub fn water_tower_parts() -> (Bricks, Tower) {
    let mut b = Bricks::new();
    let mut p = Tower::default();
    let mut sid: u16 = 100;
    let st = |b: &mut Bricks, lo: P3, hi: P3| bx(b, lo, hi, mat::STEEL, P_OXIDE);
    let up = [0.0, 1.0, 0.0];
    let top = TIER * TIERS as f64;
    let c = 0.125;   // the legs' half width
    // ---- the legs, a length to a tier, on concrete pads
    for &x in &[-LEG, LEG] { for &z in &[-LEG, LEG] {
        bx(&mut b, [x - 0.5, 0.0, z - 0.5], [x + 0.5, 0.25, z + 0.5], mat::CONCRETE, P_SLAB);
        let mut leg = Vec::new();
        for t in 0..TIERS { let (y0, y1) = (if t == 0 { 0.25 } else { TIER * t as f64 }, TIER * (t + 1) as f64); leg.push(st(&mut b, [x - c, y0, z - c], [x + c, y1, z + c])); }
        p.legs.push(leg);
    } }
    // ---- the struts: a ring at each tier's top, between the legs' faces, their tops level with the tier
    for t in 0..TIERS {
        let y = TIER * (t + 1) as f64;
        if t + 1 < TIERS {
            for &z in &[-LEG, LEG] { p.struts.push(st(&mut b, [-LEG + c, y - 0.25, z - c], [LEG - c, y, z + c])); }
            for &x in &[-LEG, LEG] { p.struts.push(st(&mut b, [x - c, y - 0.25, -LEG + c], [x + c, y, LEG - c])); }
        }
    }
    // under the tank: two beams 0.375 m deep along z over the legs' heads, running out to the tank's edge, and two
    // along x on those
    for &x in &[-LEG, LEG] { p.struts.push(st(&mut b, [x - c, top, -TANK_R], [x + c, top + 0.375, TANK_R])); }
    for &z in &[-LEG, LEG] { p.struts.push(st(&mut b, [-TANK_R, top + 0.375, z - c], [TANK_R, top + 0.75, z + c])); }
    // ---- the diagonals: on each face of each tier one, from the foot of a leg (beside it, on the pad or on the
    // strut below) to the next leg's face under the strut above; leaning the other way on the next tier
    for t in 0..TIERS {
        let y0 = if t == 0 { 0.25 } else { TIER * t as f64 };
        let y1 = TIER * (t + 1) as f64 - if t + 1 < TIERS { 0.25 } else { 0.0 };
        for f in 0..4 {
            // the face's line: along x at z = -LEG and +LEG, along z at x = -LEG and +LEG
            let s = if (t + f) % 2 == 0 { 1.0 } else { -1.0 };
            let (foot, head, nx, out): (P3, P3, P3, P3) = match f {
                0 => ([-s * (LEG - c - 0.3), y0, -LEG], [s * (LEG - c), y1 - 0.25, -LEG], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                1 => ([-s * (LEG - c - 0.3), y0, LEG], [s * (LEG - c), y1 - 0.25, LEG], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                2 => ([-LEG, y0, -s * (LEG - c - 0.3)], [-LEG, y1 - 0.25, s * (LEG - c)], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
                _ => ([LEG, y0, -s * (LEG - c - 0.3)], [LEG, y1 - 0.25, s * (LEG - c)], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            };
            let ax = unit(sub(head, foot));
            let nn = unit(sub(up, mul(ax, dot(up, ax))));
            p.braces.push(member(&mut b, foot, head, nn, out, 0.1875, 0.1875, up, nx, 2.75, mat::STEEL, P_OXIDE, &mut sid));
        }
    }
    // ---- the tank on the beams: a floor plate, eight sides of plates in two courses, a roof of eight plates to a point
    let ty = top + 0.75;
    sid += 1;
    p.floor = prism(&mut b, &ngon(0.0, 0.0, TANK_R, 8), ty, ty + 0.125, mat::STEEL, P_OXIDE, sid);
    let (outer, inner) = (ngon(0.0, 0.0, TANK_R, 8), ngon(0.0, 0.0, TANK_R - 0.125, 8));
    for course in 0..2 {
        let (y0, y1) = (ty + 0.125 + TANK_H / 2.0 * course as f64, ty + 0.125 + TANK_H / 2.0 * (course + 1) as f64);
        for k in 0..8 { let q = (k + 1) % 8; sid += 1; p.tank.push(prism(&mut b, &[outer[k], outer[q], inner[q], inner[k]], y0, y1, mat::STEEL, P_OXIDE, sid)); }
    }
    let ry = ty + 0.125 + TANK_H;
    for k in 0..8 {
        let q = (k + 1) % 8;
        // (a level rim on the wall's top, then up to the point)
        let pts = [[outer[k][0], ry, outer[k][1]], [outer[q][0], ry, outer[q][1]], [inner[k][0], ry, inner[k][1]], [inner[q][0], ry, inner[q][1]],
            [outer[k][0], ry + 0.125, outer[k][1]], [outer[q][0], ry + 0.125, outer[q][1]], [0.0, ry + 1.5, 0.0], [0.0, ry + 1.625, 0.0]];
        sid += 1; p.roof.push(piece(&mut b, &pts, mat::STEEL, P_OXIDE, sid));
    }
    // the riser: a 0.5 m pipe from the ground to the tank's floor, in four lengths
    for t in 0..TIERS { let (y0, y1) = (TIER * t as f64, if t + 1 == TIERS { ty } else { TIER * (t + 1) as f64 }); p.riser.push(st(&mut b, [-0.25, y0, -0.25], [0.25, y1, 0.25])); }
    b.build_joints();
    b.log.clear();
    (b, p)
}

/// The chimney, its middle at (0, 0). Returns its courses, from the plinth up.
pub fn chimney() -> Bricks { let mut b = Bricks::new(); chimney_into(&mut b, 0.0, 0.0); b.build_joints(); b.log.clear(); b }
pub fn chimney_into(b: &mut Bricks, cx: f64, cz: f64) -> Vec<Vec<u32>> {
    let br = |b: &mut Bricks, lo: P3, hi: P3| bx(b, [cx + lo[0], lo[1], cz + lo[2]], [cx + hi[0], hi[1], cz + hi[2]], mat::MASONRY, P_BRICK);
    let mut courses = Vec::new();
    // the plinth: solid but for the flue, 3.25 m square and 1.5 m high, in two courses
    let ring = |b: &mut Bricks, h: f64, t: f64, y0: f64, y1: f64, turn: bool| -> Vec<u32> {
        // four walls: two the full width, two between them; which two alternates course by course (the bond)
        let mut out = Vec::new();
        if !turn {
            out.push(br(b, [-h, y0, -h], [h, y1, -h + t])); out.push(br(b, [-h, y0, h - t], [h, y1, h]));
            out.push(br(b, [-h, y0, -h + t], [-h + t, y1, h - t])); out.push(br(b, [h - t, y0, -h + t], [h, y1, h - t]));
        } else {
            out.push(br(b, [-h, y0, -h], [-h + t, y1, h])); out.push(br(b, [h - t, y0, -h], [h, y1, h]));
            out.push(br(b, [-h + t, y0, -h], [h - t, y1, -h + t])); out.push(br(b, [-h + t, y0, h - t], [h - t, y1, h]));
        }
        out
    };
    courses.push(ring(b, 1.625, 0.875, 0.0, 0.75, false));
    courses.push(ring(b, 1.625, 0.875, 0.75, 1.5, true));
    // four stages of 6 m (eight courses of 0.75 m each): half widths 1.25, 1.125, 1.0, 0.875; walls 0.5 m, then 0.375 m
    let mut n = 0;
    for (stage, &(h, t)) in [(1.25, 0.5), (1.125, 0.5), (1.0, 0.375), (0.875, 0.375)].iter().enumerate() {
        for c in 0..8 { let y0 = 1.5 + 6.0 * stage as f64 + 0.75 * c as f64; courses.push(ring(b, h, t, y0, y0 + 0.75, n % 2 == 1)); n += 1; }
    }
    // the cap: two oversailing courses of 0.25 m
    courses.push(ring(b, 1.0, 0.5, 25.5, 25.75, false));
    courses.push(ring(b, 1.125, 0.625, 25.75, 26.0, true));
    courses
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    #[test] fn the_water_tower_is_what_it_should_be_and_stands() {
        let (b, p) = water_tower_parts();
        assert_eq!(p.legs.len(), 4); assert_eq!(p.braces.len(), 16); assert_eq!(p.tank.len(), 16); assert_eq!(p.roof.len(), 8);
        let r = stands("the water tower", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
    #[test] fn the_chimney_is_what_it_should_be_and_stands() {
        let b = chimney();
        assert_eq!(b.alive_count(), 4 * 36);
        let r = stands("the chimney", &b);
        assert!(r < 0.9, "worst joint {r}");
    }
}
