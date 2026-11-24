// The GPU world's shaders through a second WGSL compiler: naga (the one Firefox uses; Chrome's is Tint).
// Usage: node engine/gpu/test_wgsl.cjs      (needs `naga` on the PATH, or NAGA=/path/to/naga: cargo install naga-cli)
// No browser and no GPU: the three modules are read out of gpu_world.js, validated, and translated to SPIR-V
// (Vulkan), HLSL (Direct3D) and MSL (Metal).
//
// What it reproduces (24 Nov): `find`, a function ending in a loop that always returns from inside, was refused by
// naga ("Returning None where Some([2]) is expected"), so the main and the particle modules did not compile, the
// GPU world was not made, and the page ran without it: holes, nothing falling, no ball. Before the fix: 2 of 3
// modules refused. After: 3 of 3 valid, 9 of 9 translations.
const vm = require('vm'), fs = require('fs'), path = require('path'), cp = require('child_process');
const root = path.resolve(__dirname, '..', '..'), out = path.join(root, 'lab-results', 'wgsl');
const ctx = { console, performance }; ctx.globalThis = ctx; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'gpu_world.js'), 'utf8') + ';globalThis.__m = { main: WGSL, args: WGSL_ARGS, particles: WGSL_P };', ctx);
fs.mkdirSync(out, { recursive: true });
const naga = process.env.NAGA || 'naga';
if (cp.spawnSync(naga, ['--version']).error) { console.log(`FAIL wgsl: no naga to run (cargo install naga-cli, or NAGA=/path/to/naga)`); process.exit(2); }
let bad = 0; const res = {};
for (const [name, code] of Object.entries(ctx.__m)) {
  const f = path.join(out, name + '.wgsl'); fs.writeFileSync(f, code);
  const steps = [['valid', [f]], ...['spv', 'hlsl', 'metal'].map(e => [e, [f, path.join(out, name + '.' + e)]])];
  res[name] = {};
  for (const [what, args] of steps) {
    const r = cp.spawnSync(naga, args, { encoding: 'utf8' }), ok = r.status === 0;
    res[name][what] = ok;
    if (!ok) { bad++; console.log(`  ${name} ${what}: ` + (r.stderr + r.stdout).replace(/\x1b\[[0-9;]*m/g, '').split('\n').filter(Boolean).slice(0, 6).join('\n    ')); }
  }
}
fs.writeFileSync(path.join(root, 'lab-results', 'wgsl.json'), JSON.stringify({ naga: cp.spawnSync(naga, ['--version'], { encoding: 'utf8' }).stdout.trim(), results: res, date: new Date().toISOString() }));
console.log(`${bad ? 'FAIL' : 'PASS'} wgsl ${JSON.stringify(res)}`);
process.exit(bad ? 1 : 0);
