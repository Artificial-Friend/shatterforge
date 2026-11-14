import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports;
const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const sts = () => { const n = e.n_structs(); const a = F(e.ptr_structs(), n * 12); const r = []; for (let s = 0; s < n; s++) r.push(Array.from(a.slice(s * 12, s * 12 + 12))); return r; };
let t = performance.now(); const nb = e.init();
console.log('bricks', nb, 'structs', e.n_structs(), 'init ms', (performance.now() - t).toFixed(0));
const S0 = sts();
e.dbg_wake_all();
t = performance.now();
for (let i = 0; i < 1200; i++) { e.step(1 / 60); if (i % 60 == 59) e.dbg_wake_all(); e.clear_events(); }
console.log('20s ms/frame', ((performance.now() - t) / 1200).toFixed(1));
const S1 = sts(); const st = Array.from(F(e.ptr_stats(), 16));
S0.forEach((s, i) => { const lost = s[5] - S1[i][5]; console.log(i, `[${s[0].toFixed(0)},${s[1].toFixed(0)}]`, 'bricks', s[4], 'lost', lost, lost ? '<<<<' : ''); });
console.log('bodies', st[2]);
