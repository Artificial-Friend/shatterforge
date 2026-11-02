import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports; const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const sts = () => { const n = e.n_structs(); const a = F(e.ptr_structs(), n * 12); const r = []; for (let s = 0; s < n; s++) r.push(Array.from(a.slice(s * 12, s * 12 + 12))); return r; };
function find(x, z) { const S = sts(); for (let s = 0; s < S.length; s++) { const b = S[s]; if (x >= b[0] && x <= b[2] && z >= b[1] && z <= b[3]) return s; } return -1; }
const MN = ['', 'concrete', 'masonry', 'steel', 'glass', 'wood'];
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
for (const [label, px, pz, booms, secs] of cases) {
  e.set_ai(process.env.AI ? 1 : 0); e.init(); for (let i = 0; i < 30; i++) e.step(1 / 60);
  const s = find(px, pz); const before = sts()[s][5];
  for (const b of booms) e.explode(b[0], b[1], b[2], b[3] || 3.0, 1.4);
  let worst = 0;
  for (let i = 0; i < (secs || 15) * 60; i++) { const t = performance.now(); e.step(1 / 60); e.clear_events(); worst = Math.max(worst, performance.now() - t); }
  const after = sts()[s][5]; const d = F(e.dbg_matstats(), 50);
  const parts = []; for (let m = 1; m <= 5; m++) if (d[m * 5]) parts.push(`${MN[m]}: ${d[m*5]} pieces avg ${(d[m*5+1]/d[m*5]).toFixed(1)} max ${d[m*5+2]} big(>=50) ${d[m*5+3]} crumbs ${d[m*5+4]}`);
  console.log(`${label} | lost ${((before - after) / before * 100).toFixed(0)}% top ${sts()[s][7]}/${sts()[s][6]} | worst ${worst.toFixed(0)}ms\n   ${parts.join('\n   ')}`);
}
if (process.env.BIG) { const big = F(e.dbg_bodies(), 24); const top = []; for (let t = 0; t < 8; t++) if (big[t * 3]) top.push(`${big[t*3]}br ${big[t*3+1].toFixed(1)}m tall ${big[t*3+2].toFixed(1)}m wide`); console.log('   biggest:', top.join('; ')); }
