// Compare a lab scenario with the dynamic structure and solved at rest (docs/DYNPLAN.md, stage 0).
//   node tools/compare.cjs <scenario> [--frames n] [--keep]     (--keep: use the reports already in lab-results/cmp)
// Runs game/lab.cjs twice (the second with SF_STATIC=1), into lab-results/cmp/{dyn,static}, and prints the table
// "better than at rest" is measured by. Software-GPU times compare only with each other.
const { spawnSync } = require('child_process'), fs = require('fs'), path = require('path');
const args = process.argv.slice(2), name = args[0], keep = args.includes('--keep');
if (!name) { console.log('usage: node tools/compare.cjs <scenario> [--frames n] [--keep]'); process.exit(2); }
const root = path.join(__dirname, '..'), rest = args.slice(1).filter(a => a !== '--keep');
const R = {};
for (const mode of ['dyn', 'static']) {
  const out = path.join(root, 'lab-results', 'cmp', mode);
  if (!keep) spawnSync('node', [path.join(root, 'game', 'lab.cjs'), name, ...rest, '--out', out], { env: { ...process.env, ...(mode === 'static' ? { SF_STATIC: '1' } : {}) }, stdio: 'ignore' });
  R[mode] = JSON.parse(fs.readFileSync(path.join(out, name + '.json'), 'utf8'));
}
const row = (r) => {
  const T = (r.timeline || []).filter(t => t.op === 'structure'), slow = (r.timeline || []).slice().sort((a, b) => (b.ms || 0) - (a.ms || 0))[0] || {};
  return {
    failed: r.checks.filter(c => !c.pass).map(c => `${c.name.replace(/ \(.*/, '')} ${c.value}`),
    eng: r.perf.engine_ms, slow: `${slow.op || '?'} at frame ${slow.f} (${slow.ms} ms)`,
    steps: T.length, capped: T.filter(t => t.dyn && t.dyn.capped).length, its: T.reduce((a, t) => a + (t.dyn ? t.dyn.its : 0), 0),
    joints: T.reduce((a, t) => a + t.n, 0), bodies: T.reduce((a, t) => a + t.new_bodies, 0), frames: r.frames, full: (r.timeline || []).length < 400,
  };
};
const d = row(R.dyn), s = row(R.static);
const verdict = [
  ['lab checks failed: no more than at rest', d.failed.length <= s.failed.length],
  ['slowest engine job: no slower than at rest', d.eng.max <= s.eng.max],
  ['engine jobs p95: a quarter of at rest at most', d.eng.p95 <= s.eng.p95 / 4 || d.eng.p95 <= 20],
  ['structure steps at the cap: under 5%', d.capped <= 0.05 * Math.max(1, d.steps)],
];
console.log(`\n${name}, ${d.frames} frames (software GPU${d.full ? '' : '; timeline cut at 400 jobs, structure counts are of those'})\n`);
console.log('| | Dynamic | At rest |\n| --- | --- | --- |');
console.log(`| Failed checks | ${d.failed.length ? d.failed.join('; ') : 'none'} | ${s.failed.length ? s.failed.join('; ') : 'none'} |`);
console.log(`| Engine jobs: median / p95 / max (ms) | ${d.eng.median} / ${d.eng.p95} / ${d.eng.max} | ${s.eng.median} / ${s.eng.p95} / ${s.eng.max} |`);
console.log(`| Slowest job | ${d.slow} | ${s.slow} |`);
console.log(`| Structure steps (at the cap), iterations | ${d.steps} (${d.capped}), ${d.its} | ${s.steps}, - |`);
console.log(`| Joints failed, pieces sent off | ${d.joints}, ${d.bodies} | ${s.joints}, ${s.bodies} |`);
console.log('');
for (const [what, ok] of verdict) console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`);
process.exit(verdict.every(v => v[1]) ? 0 : 1);
