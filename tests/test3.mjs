import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../target/wasm32-unknown-unknown/release/shatterforge.wasm', import.meta.url)), { env: { now: () => performance.now() } });
const e = instance.exports;
const F = (p, n) => new Float32Array(e.memory.buffer, p, n);
const stats = () => Array.from(F(e.ptr_stats(), 16));
const pl = () => Array.from(F(e.ptr_player(), 16));
const sts = () => { const n = e.n_structs(); const a = F(e.ptr_structs(), n * 12); const r = []; for (let s = 0; s < n; s++) r.push(Array.from(a.slice(s * 12, s * 12 + 12))); return r; };
let meshFaces = 0, meshBuilds = 0, meshMs = 0;
function pumpMeshes(budget = 3) {
  const p = pl();
  for (let k = 0; k < budget; k++) {
    const id = e.next_dirty(p[0], p[2], k === 0 ? 1 : 0);
    if (id < 0) break;
    const t = performance.now();
    const nf = e.build_set(id);
    const a = F(e.ptr_faces(0), nf * 52);
    for (let i = 0; i < a.length; i += 97) if (!Number.isFinite(a[i])) { console.log('NaN in set', id); break; }
    meshMs += performance.now() - t; meshFaces += nf + e.n_faces(1); meshBuilds++;
  }
}
let worstAll = 0;
function run(frames, label, input) {
  let worst = 0, total = 0;
  meshFaces = 0; meshBuilds = 0; meshMs = 0;
  for (let i = 0; i < frames; i++) {
    if (input) input(i);
    const t = performance.now();
    e.step(1 / 60); e.render(); pumpMeshes(); e.clear_events();
    const d = performance.now() - t; total += d; worst = Math.max(worst, d);
  }
  worstAll = Math.max(worstAll, worst);
  const s = stats(), p = pl();
  const pr = Array.from(F(e.ptr_prof(), 6)).map(x => x.toFixed(1)).join('/');
  console.log(`${label.padEnd(26)} [${pr}] static=${s[0]} bodies=${s[2]} awake=${s[3]} shards=${s[7]} crew=${s[8]}/${s[9]} active=${s[13]} unsolved=${s[14]} salvage=${s[12].toFixed(1)}t dynF=${e.n_faces(2)} | pl ${p.slice(0, 3).map(x => x.toFixed(1)).join(',')} hp=${p[3].toFixed(0)} st=${p[4]} | avg ${(total / frames).toFixed(1)}ms worst ${worst.toFixed(1)}ms meshes ${meshBuilds} (${meshMs.toFixed(0)}ms)`);
}
let t = performance.now();
const nb = e.init();
console.log(`map: ${nb} bricks, ${e.n_structs()} structures, init ${(performance.now() - t).toFixed(0)}ms`);
// initial mesh build of everything
t = performance.now(); let tf = 0, n = 0;
for (;;) { const id = e.next_dirty(0, 130, 1); if (id < 0) break; tf += e.build_set(id) + e.n_faces(1); n++; }
console.log(`initial meshes: ${n} sets, ${tf} faces, ${(performance.now() - t).toFixed(0)}ms`);
run(120, 'idle 2s');
run(40, 'walk to car', () => e.set_move(1, 0, 0, 0));
run(1, 'stop', () => e.set_move(0, 0, 0, 0));
run(30, 'jump', (i) => e.set_move(0, 0, i === 0 ? 1 : 0, 0));
console.log('use vehicle ->', e.use_vehicle());
run(30, 'in car');
run(240, 'drive north 4s', () => e.set_input(1, 0, 0));
console.log('exit ->', e.use_vehicle());
run(30, 'on foot again', () => e.set_move(0, 0, 0, 0));
// demolish targets with explosions
const S = sts();
const names = ['', 'Admin tower', 'Fort', 'Smokestack', 'Viaduct', 'Bell tower', 'Depot', 'Gantry', 'Mast'];
for (const st of S) {
  if (!st[8]) continue;
  const cx = (st[0] + st[2]) / 2, cz = (st[1] + st[3]) / 2;
  const kind = st[9];
  if (kind === 4) { for (const x of [48, 64, 80, 96, 114]) e.explode(x, 2, 0, 3.4, 1.6); }
  else if (kind === 2) { e.explode(cx - 6, 1, cz, 3.4, 1.6); e.explode(cx + 6, 1, cz, 3.4, 1.6); e.explode(cx, 1, cz, 3.4, 1.6); }
  else { e.explode(st[0] + 0.5, 1, st[1] + 0.5, 3.4, 1.6); e.explode(st[2] - 0.5, 1, st[3] - 0.5, 3.4, 1.6); e.explode(st[0] + 0.5, 1, st[3] - 0.5, 3.4, 1.6); e.explode(st[2] - 0.5, 1, st[1] + 0.5, 3.4, 1.6); }
  run(60, `boom ${names[kind]}`);
}
for (let k = 0; k < 6; k++) run(60, `collapse t+${k + 1}s`);
run(600, 'settle 10s');
run(1200, 'rubble +20s');
for (const st of sts()) if (st[8]) console.log(names[st[9]].padEnd(12), 'integrity', (st[5] / st[4] * 100).toFixed(0) + '%', 'height', (st[7] / st[6] * 100).toFixed(0) + '%');
console.log('worst frame overall', worstAll.toFixed(1), 'ms');
