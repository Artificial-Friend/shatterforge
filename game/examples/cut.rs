// Cut the whole ground storey with a sweep of blasts: does the tower above detach? If not, show what holds it.
use sf_engine::math::*;
fn main() {
    let mut g = sf_demo::tower::new(12);
    let mut x = -11.6f32;
    let mut big = 0;
    while x <= 11.7 { let mut z = -11.6f32; while z <= 11.7 {
        g.blast(v3(x, 1.6, z), 0.7);
        // parse: updates, then bodies
        let o = &g.out; let nu = o[0] as usize; let mut k = 1 + nu * 9; let nb = o[k] as usize; k += 1;
        for _ in 0..nb { let n = o[k] as usize; if n > 1 { big = big.max(n); } k += 1 + n * 7; }
        z += 0.8; } x += 0.8; }
    println!("alive {}, biggest detached piece {} bricks", g.b.alive_count(), big);
    // path from a roof brick to the ground
    let n = g.b.len();
    let start = (0..n as u32).filter(|&i| g.b.alive(i)).max_by_key(|&i| g.b.y[i as usize]).unwrap();
    let mut prev = vec![u32::MAX; n]; let mut seen = vec![false; n];
    let mut q = std::collections::VecDeque::new(); q.push_back(start); seen[start as usize] = true;
    let mut found = None;
    while let Some(i) = q.pop_front() {
        if g.b.y[i as usize] <= 0 { found = Some(i); break; }
        let mut j = g.b.jhead[i as usize];
        while j != u32::MAX { let o = g.b.joint_other(j, i); if !seen[o as usize] { seen[o as usize] = true; prev[o as usize] = i; q.push_back(o); } j = g.b.joint_next(j, i); }
    }
    let mut i = found.unwrap(); let mut path = vec![];
    while i != u32::MAX { path.push(i); i = prev[i as usize]; }
    for &i in path.iter().take(8) { let t = g.b.tbox(i); println!("  #{i} mat {} min {:?} max {:?} (m: y {:.3}..{:.3})", g.b.mat[i as usize], t.min, t.max, t.min[1] as f32 * 0.125, t.max[1] as f32 * 0.125); }
}
