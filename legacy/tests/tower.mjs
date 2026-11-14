import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports; const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const sts = () => { const n = e.n_structs(); const a = F(e.ptr_structs(), n * 12); const r = []; for (let s = 0; s < n; s++) r.push(Array.from(a.slice(s * 12, s * 12 + 12))); return r; };
e.set_ai(0); e.init(); for (let i = 0; i < 30; i++) e.step(1 / 60);
const booms = JSON.parse(process.argv[2]); const sid = +process.argv[4];
for (const b of booms) e.explode(b[0], b[1], b[2], b[3] || 3, 1.4);
let worst = 0;
for (let k = 0; k < (+process.argv[3] || 60); k++) {
  for (let i = 0; i < 15; i++) { const t = performance.now(); e.step(1 / 60); e.clear_events(); worst = Math.max(worst, performance.now() - t); }
  const d = F(e.dbg_speed(), 10); const g = F(e.dbg_bigbody(), 12); const st = sts()[sid];
  console.log(((k + 1) * 0.25).toFixed(2), `static ${st[5]} top ${st[7]} | big ${g[0]}br @y${g[2].toFixed(1)} tilt ${g[4].toFixed(0)}° v${g[5].toFixed(1)} w${g[6].toFixed(2)} ${g[7] ? 'asleep' : ''} maxr ${g[8].toFixed(2)} res ${g[9].toExponential(1)} nc ${g[10]} lt ${g[11].toFixed(2)} | maxH ${d[0].toFixed(1)} awake ${d[6]} | ${worst.toFixed(0)}ms`);
}
{ const big = F(e.dbg_bodies(), 24); const top = []; for (let t = 0; t < 8; t++) if (big[t * 3]) top.push(`${big[t*3]}br ${big[t*3+1].toFixed(1)}x${big[t*3+2].toFixed(1)}m`); console.log('biggest pieces:', top.join('; ')); }
