import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports;
const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const sts = () => { const n = e.n_structs(); const a = F(e.ptr_structs(), n * 12); const r = []; for (let s = 0; s < n; s++) r.push(Array.from(a.slice(s * 12, s * 12 + 12))); return r; };
const stats = () => Array.from(F(e.ptr_stats(), 16));
function find(x, z) { const S = sts(); for (let s = 0; s < S.length; s++) { const b = S[s]; if (x >= b[0] && x <= b[2] && z >= b[1] && z <= b[3]) return s; } return -1; }
function step(n) { let w = 0; for (let i = 0; i < n; i++) { const t = performance.now(); e.step(1 / 60); e.render(); for (;;) { const id = e.next_dirty(0, 0, 1); if (id < 0) break; e.build_set(id); } e.clear_events(); w = Math.max(w, performance.now() - t); } return w; }
const cases = [
  ['admin16: one front column blown', 0, 0, [[-3.2,0.6,7.2]]],
  ['admin16: two front columns blown', 0, 0, [[-3.2,0.6,7.2],[4.8,0.6,7.2]]],
  ['admin16: front row (6) blown', 0, 0, [[-11.2,0.6,7.2],[-7.2,0.6,7.2],[-3.2,0.6,7.2],[4.8,0.6,7.2],[8.8,0.6,7.2],[11.2,0.6,7.2]]],
  ['admin16: four corners blown', 0, 0, [[-11.2,1,-9.2],[11.2,1,7.2],[-11.2,1,7.2],[11.2,1,-9.2]]],
  ['admin16: 8 charges, front two rows', 0, 0, [[-11.2,0.6,7.2],[-5.2,0.6,7.2],[4.8,0.6,7.2],[11.2,0.6,7.2],[-7.2,0.6,3.2],[-1.2,0.6,3.2],[4.8,0.6,3.2],[10.8,0.6,3.2]]],
];
for (const [label, px, pz, booms] of cases) {
  e.init(); step(30);
  const s = find(px, pz);
  const before = sts()[s][5];
  for (const b of booms) e.explode(b[0], b[1], b[2], +(process.env.R||3.0), +(process.env.P||1.4));
  let worst = 0;
  for (let k = 0; k < 20; k++) worst = Math.max(worst, step(60));
  const after = sts()[s][5];
  const big = F(e.dbg_bodies(), 24);
  const top = []; for (let t = 0; t < 4; t++) if (big[t * 3]) top.push(`${big[t*3]}br ${big[t*3+1].toFixed(1)}m tall ${big[t*3+2].toFixed(1)}m wide`);
  const st = stats();
  console.log(`${label.padEnd(36)} lost ${((before - after) / before * 100).toFixed(0)}% | height ${sts()[s][7]}/${sts()[s][6]} | bodies ${st[2]} shards ${st[7]} | biggest: ${top.join('; ')} | worst ${worst.toFixed(0)}ms`);
}
