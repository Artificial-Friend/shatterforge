// The lab, headless: node game/lab/lab.cjs [quick|full|scenario ...] [--read-lag N] [--job-lag N] [--out DIR]
// Runs each scenario of the page's lab (game/lab/lab.js) in headless Chrome on the software GPU, on the lab's
// virtual clock (so slowness changes nothing but the wall time), prints the checks and writes the full reports
// (events, time series) as JSON. Exit code 1 when a check fails.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); if (i < 0) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const readLag = +opt('--read-lag', 2), jobLag = +opt('--job-lag', 4), outDir = path.resolve(opt('--out', path.join(__dirname, '..', '..', 'lab-results')));
const frames = opt('--frames', null), trace = opt('--trace', null);
// --twice: run each scenario twice and compare them frame by frame (the lab must be deterministic)
const twice = args.includes('--twice'); if (twice) args.splice(args.indexOf('--twice'), 1);
const root = path.join(__dirname, '..', '..');
const page0 = fs.readFileSync(path.join(root, 'dist/tower.html'), 'utf8');
const SUITES = { quick: ['collapse', 'ball'], full: ['collapse', 'ball', 'masonry', 'timber', 'mixed', 'complexstand', 'complex', 'complexcol', 'complexroom', 'keystone', 'rubble', 'nibble'], complex: ['complexstand', 'complex', 'complexcol', 'complexroom', 'keystone'], city: ['citystand', 'city'] };
const names = args.length ? args.flatMap(a => a.startsWith('=') ? [a.slice(1)] : SUITES[a] || [a]) : SUITES.quick;   // (=name: the scenario, where a suite has its name)
(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const html = path.join(outDir, 'lab_page.html');
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + page0 + '</body></html>');
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const results = []; let failed = false;
  const runs = twice ? names.flatMap(n => [n, n]) : names;
  const digests = {};
  for (const name of runs) {
    const p = await b.newPage({ viewport: { width: 800, height: 500 } });
    p.on('pageerror', e => console.log(`[${name}] page error: ${e.message}`));
    let threeJs; try { threeJs = fs.readFileSync(require.resolve('three/build/three.min.js'), 'utf8'); } catch (e) { threeJs = fs.readFileSync(path.join(process.env.SF_NODE_MODULES || '', 'three/build/three.min.js'), 'utf8'); }
    await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
    // one scenario per page load, queued the way the page's own button queues them (no render: nothing looks)
    await p.addInitScript((nf) => { if (nf) window.__noFracCells = true; }, !!process.env.SF_NO_FRAC_CELLS);
    await p.addInitScript((ns) => { if (ns) window.__noSnap = true; }, !!process.env.SF_NO_SNAP);
    await p.addInitScript((d) => { if (d) window.__detDump = true; }, !!process.env.SF_DET_DUMP);
    // (SF_STATIC: the standing structure solved at rest after each change, as before docs/DYNAMIC.md)
    await p.addInitScript((d) => { if (d) window.__static = true; }, !!process.env.SF_STATIC);
    await p.addInitScript((d) => { if (d) window.__detAt = d; }, +(process.env.SF_DET_AT || 0));
    await p.addInitScript((d) => { if (d) window.__colR = d; }, process.env.SF_COLR || '');
    await p.addInitScript((d) => { if (d) window.__colCheck = true; }, !!process.env.SF_COLCHECK);
    await p.addInitScript((d) => { if (d) window.__seed = d; }, +(process.env.SF_SEED || 0));
    await p.addInitScript((d) => { if (d) window.__preset = d; }, process.env.SF_PRESET || '');
    await p.addInitScript((d) => { if (d) window.__shots = d.split(',').map(Number); }, process.env.SF_SHOTS || '');
    await p.addInitScript((d) => { if (d) window.__shotView = d.split(',').map(Number); }, process.env.SF_SHOT_VIEW || '');
    await p.addInitScript((d) => { if (d) window.__pinch = true; }, !!process.env.SF_PINCH);
    await p.addInitScript((d) => { if (d) window.__oldMasonry = true; }, !!process.env.SF_OLD_MASONRY);
    await p.addInitScript((d) => { if (d) window.__parkDbg = true; }, !!process.env.SF_PARK_DBG);
    await p.addInitScript((d) => { if (d) window.__traceBodies = d; }, process.env.SF_TRACE_BODIES || '');   // (with --trace: these bodies, every frame)
    await p.addInitScript((d) => { if (d) window.__detRows = true; }, !!process.env.SF_DET_ROWS);   // (compare without breaking on the GPU)   // (compare without irregular fracture)
    await p.addInitScript(([name, readLag, jobLag, frames, trace]) => {
      if (sessionStorage.getItem('lab-set')) return; sessionStorage.setItem('lab-set', '1');
      localStorage.setItem('sf-lab', JSON.stringify({ queue: [name], done: [], results: [], readLag, jobLag, render: false, external: true, frames, trace }));
    }, [name, readLag, jobLag, frames ? +frames : null, trace ? trace.split(':').map(Number) : null]);
    const t0 = Date.now();
    await p.goto('file://' + html);
    let last = '';
    for (;;) {
      await new Promise(r => setTimeout(r, process.env.SF_SHOTS ? 500 : 5000));
      const st = await p.evaluate(() => ({ done: !!window.LAB_RESULT, err: window.LAB_ERROR || null, s: LAB.status || null, shot: window.__shotReady || 0 }));
      if (st.shot) { await p.screenshot({ path: path.join(outDir, `${name}_f${st.shot}.png`) }); await p.evaluate(() => { window.__shotReady = 0; }); continue; }
      if (st.err) { console.log(`[${name}] lab error: ${st.err}`); failed = true; break; }
      if (st.done) break;
      const line = st.s ? `[${name}] frame ${st.s.f} of ${st.s.frames} · ${((Date.now() - t0) / 1000).toFixed(0)} s` : '';
      if (line && Date.now() - (this.lastPrint || 0) > 60000) { console.log(line); this.lastPrint = Date.now(); }
    }
    const rep = await p.evaluate(() => window.LAB_RESULT || null);
    if (rep) {
      results.push(rep); if (!rep.pass) failed = true;
      console.log(await p.evaluate((r) => labSummary(r), rep));
      if (process.env.SF_COLCHECK) console.log(`  colour conflicts: ${rep.colConflicts} (first at frame ${rep.colConflictAt})`);
      if (twice && digests[name]) {
        const a = digests[name], d = rep.digest || [], n = Math.min(a.length, d.length);
        const eq = (x, y) => JSON.stringify(x) === JSON.stringify(y);
        let k = 0; while (k < n && eq(a[k], d[k])) k++;
        const same = k === n && a.length === d.length;
        const parts = ['poses', 'particles', 'bodies', 'breaks', 'dust', 'bricks', 'joints', 'contacts', 'particles made', 'awake', 'events', 'jobs', 'inputs', 'gpu frame', 'skipped', 'colour limit', 'sub-steps', 'shapes', 'particle slots', 'stats', 'pending inherit', 'GPU bodies', 'GPU solver', 'GPU shapes', 'GPU colours', 'GPU contacts'];
        const what = same ? '' : parts.filter((_, q) => a[k] && d[k] && a[k][q] !== d[k][q]).join(', ');
        console.log(`  ${same ? 'ok  ' : 'FAIL'} deterministic: ${same ? `the two runs agree in all ${n} frames` : `the two runs differ from frame ${k}: ${what}`}`);
        if (!same) for (let q = Math.max(0, k - 3); q < Math.min(n, k + 3); q++) console.log(`    frame ${q}: ${JSON.stringify(a[q])} / ${JSON.stringify(d[q])}`);
        if (!same) failed = true;
        if (rep.dumpRaw && digests[name + ':raw']) fs.writeFileSync(path.join(outDir, 'det_raw.json'), JSON.stringify([digests[name + ':raw'], rep.dumpRaw]));
      } else if (twice) { digests[name] = rep.digest; digests[name + ':raw'] = rep.dumpRaw; const { digest, dumpRaw, ...keep } = rep; fs.writeFileSync(path.join(outDir, `${name}_run1.json`), JSON.stringify(keep, null, 1)); }
      { const { digest, dumpRaw, ...keep } = rep; fs.writeFileSync(path.join(outDir, `${name}.json`), JSON.stringify(keep, null, 1)); }
    }
    await p.close();
  }
  await b.close();
  fs.writeFileSync(path.join(outDir, 'all.json'), JSON.stringify(results));
  console.log(`\n${results.filter(r => r.pass).length} of ${names.length} scenarios passed · reports in ${outDir}`);
  process.exit(failed ? 1 : 0);
})();
