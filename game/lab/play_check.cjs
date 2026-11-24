// The game as a player has it, checked: node game/lab/play_check.cjs [case ...]   (default: blast nogpu)
//   blast     default settings: a 0.7 m blast, a 1.2 m blast, then the ball, from where the player starts
//   low       the same with the settings a player on the Low preset has saved (and the debris budget: furniture
//             that has not been touched is not debris)
//   hz144     the same on a 144 Hz screen (more frames than steps)
//   noworker  the same with no Worker (the engine inline in the page)
//   nogpu     no WebGPU: the page must say so in play, on screen
//   nocreate  the shaders refused (as Firefox's compiler refused them until 24 Nov): the page must say so
//   lost      the GPU device lost in play: the page must say so
// The page runs its own loop, worker, settings and budgets on a virtual clock (game/lab/play_harness.js), on the
// software GPU. SF_NODE_MODULES: a node_modules folder with three; Playwright installed. One case at a time: two
// headless browsers with this page do not fit in 4 GB.
//
// Why (24 Nov): the user's report, "when I break something it just disappears and the steel ball doesn't work".
// Every check until then ran the lab (engine inline, no debris limit, fresh settings) or looked at one thing
// (page_check). None asked of the player's page: is what left the building still there, drawn, some seconds later;
// did the ball fly; and when the GPU world is not there, does the page say so. What each case showed before the
// fixes is in docs/LAB.md, "The player's page".
const { chromium } = require(process.env.PW || 'playwright'); const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..', '..'), out = path.join(root, 'lab-results');
const LOW = { phys: 'low', gfx: 'low', debris: 1500, crumbs: 64, rubble: 16, substeps: 6, colors: 10, gShadows: 0, gMap: 0, gPx: 1 };
const acts = (k) => [{ f: Math.round(300 * k), code: 'setRadius(0.7); doBlast();' }, { f: Math.round(420 * k), code: 'setRadius(1.2); doBlast();' }, { f: Math.round(560 * k), code: "setRadius('ball'); fireBall();" }];
const CASES = {
  blast: { cfg: { script: acts(1), stopAt: 760 }, expect: 'play' },
  low: { cfg: { script: acts(1), stopAt: 760, saved: LOW }, expect: 'play', budget: 1500 },
  hz144: { cfg: { script: acts(2.4), stopAt: 1820, hz: 144, every: 24, jobLag: 4, readLag: 4 }, expect: 'play' },
  noworker: { cfg: { script: acts(1), stopAt: 760, noWorker: 1 }, expect: 'play', worker: false },
  nogpu: { cfg: { script: [{ f: 40, code: 'setRadius(1.2); doBlast();' }, { f: 80, code: "setRadius('ball'); fireBall();" }], stopAt: 140, noGpu: 1 }, expect: 'said', key: 'gpu' },
  nocreate: { cfg: { script: [{ f: 40, code: 'setRadius(1.2); doBlast();' }], stopAt: 100, failCompile: 1 }, expect: 'said', key: 'gpu' },
  lost: { cfg: { script: [{ f: 200, code: 'dev.destroy();' }, { f: 230, code: 'setRadius(1.2); doBlast();' }], stopAt: 300 }, expect: 'said', key: 'lost' },
};
const names = process.argv.slice(2).length ? process.argv.slice(2) : ['blast', 'nogpu'];
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const html = path.join(out, 'play_page.html');
  fs.writeFileSync(html, '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + fs.readFileSync(path.join(root, 'dist/tower.html'), 'utf8') + '</body></html>');
  let threeJs; try { threeJs = fs.readFileSync(require.resolve('three/build/three.min.js'), 'utf8'); } catch (e) { threeJs = fs.readFileSync(path.join(process.env.SF_NODE_MODULES || '', 'three/build/three.min.js'), 'utf8'); }
  let failed = false;
  for (const name of names) {
    const cs = CASES[name]; if (!cs) { console.log(`FAIL play ${name}: no such case`); failed = true; continue; }
    const cfg = { fast: 1, noRender: 1, jobLag: 2, readLag: 2, every: 10, ...cs.cfg };
    const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
    const p = await b.newPage({ viewport: { width: 400, height: 250 } });
    const errs = []; p.on('pageerror', e => errs.push(e.message));
    await p.route('**/three.min.js', r => r.fulfill({ body: threeJs, contentType: 'text/javascript' }));
    await p.addInitScript(([c]) => { window.__HCFG = c; if (c.saved) localStorage.setItem('sf-settings', JSON.stringify(c.saved)); }, [cfg]);
    await p.addInitScript({ path: path.join(__dirname, 'play_harness.js') });
    let res = { case: name, pass: false, why: [] };
    try {
      await p.goto('file://' + html);
      for (let i = 0; i < 600; i++) { await new Promise(r => setTimeout(r, 1000)); if (await p.evaluate(() => !$('go').disabled)) break; }
      const start = await p.evaluate(() => ({ gpu: !!gpu, worker: !!worker, warn: $('warn').textContent, debrisMax, selfTest: health.selfTest }));
      await p.evaluate(() => { __H.install(); $('go').click(); locked = true; });
      for (let i = 0; i < 4000; i++) { await new Promise(r => setTimeout(r, 2000)); if (await p.evaluate(() => __H.done)) break; }
      const end = await p.evaluate(() => ({ done: __H.done, samples: __H.samples, log: __H.log, state: diagnosis(), moved: [...debrisBorn].filter(([b, f]) => !f.orig).slice(0, 6).map(([b, f]) => [b, f.p.map(x => +x.toFixed(1))]), alert: $('alert').hidden ? '' : $('alert').textContent, hud: $('hud').textContent.slice(0, 400) }));
      const S = end.samples, last = S[S.length - 1] || {}, why = res.why, tot = (x) => x.stat + x.bod + x.par, drawn = (x) => x.statDrawn + x.drawnB + x.drawnP;
      const need = (ok, text) => { if (!ok) why.push(text); };
      need(end.done, 'the run did not end'); need(!errs.length, 'page errors: ' + errs.slice(0, 2).join(' | ')); need(!end.log.length, 'harness log: ' + end.log.slice(0, 3).join(' | '));
      if (cs.expect === 'play') {
        const T0 = S.length ? tot(S[0]) : 0, a = cs.cfg.script.map(s => s.f), at = (f) => S.filter(x => x.f <= f).pop() || S[0];
        const before = at(a[0] - 1), mid = at(a[2] - 1);
        // nothing lost: structure + pieces + crumbs is what it was, at every sample, and nothing was removed
        const worst = Math.max(...S.map(x => Math.abs(tot(x) - T0)));
        // drawn is what exists: at the end, and never behind for two samples running
        let behind = 0, run = 0; const behindAt = []; for (const x of S) { const off = Math.abs(drawn(x) - tot(x)) > 0.05; run = off ? run + 1 : 0; behind = Math.max(behind, run); if (off) behindAt.push([x.f, +(drawn(x) - tot(x)).toFixed(2), x.pendCh, x.busy, x.read]); }
        const ballMoved = last.ballP && S.find(x => x.ballP) ? Math.hypot(...last.ballP.map((v, k) => v - S.find(x => x.ballP).ballP[k])) : 0;
        Object.assign(res, { gpu: start.gpu, worker: start.worker, self_test: start.selfTest, total_m3: +T0.toFixed(2), worst_change_m3: +worst.toFixed(3), left_structure_m3: +(before.stat - last.stat).toFixed(2), pieces_and_crumbs_m3: +((last.bod + last.par) - (before.bod + before.par)).toFixed(2), drawn_end_m3: +drawn(last).toFixed(2), exists_end_m3: +tot(last).toFixed(2), samples_drawn_behind: behind, behind_at: behindAt.slice(0, 8), moved_at_rest: end.moved, removed: { budget: last.capped, far: last.pFar + last.bFar, crushed: last.pCrushed, dust: last.dust, lost: last.lost }, ball: { live: last.balls, moved_m: +ballMoved.toFixed(1), look_ahead: last.punch, broke_m3: +(mid.stat - last.stat).toFixed(2) }, debris_counted_at_rest: before.dLive, issues: last.issues, steps: last.steps, frames: last.f });
        need(start.gpu, 'no GPU world: ' + start.warn); need(start.worker === (cs.worker !== false), `engine ${start.worker ? 'in a worker' : 'inline'}`);
        need(start.selfTest && start.selfTest.ok, 'the GPU self-check failed: ' + JSON.stringify(start.selfTest));
        need(worst < 0.05, `volume changed by ${worst.toFixed(3)} m3`);
        need(before.stat - last.stat > 1, `only ${(before.stat - last.stat).toFixed(2)} m3 left the structure`);
        need(Math.abs(drawn(last) - tot(last)) < 0.02, `drawn ${drawn(last).toFixed(2)} m3 of ${tot(last).toFixed(2)}`); need(behind <= 1, `drawn behind what exists for ${behind} samples running`);
        need(last.capped + last.pFar + last.bFar + last.pCrushed + last.dust + last.lost === 0, 'debris removed: ' + JSON.stringify(res.removed));
        need(last.balls === 1 && ballMoved > 5, `the ball: ${last.balls} live, moved ${ballMoved.toFixed(1)} m`); need(last.punch[0] > 0 && mid.stat - last.stat > 0.05, `the ball broke ${(mid.stat - last.stat).toFixed(2)} m3 (look-ahead ${last.punch})`);
        need(before.dLive === 0, `${before.dLive} pieces counted as debris before anything was done`);
        need(!last.issues, 'the page complains: ' + last.issues + ' · ' + end.alert.slice(0, 200));
        if (cs.budget) need(start.debrisMax === cs.budget, `budget ${start.debrisMax}`);
      } else {
        // the page must say it, in play: the banner shown, with this trouble in it; and the state (key I) must hold it
        const bricks = [S[0] ? S[0].bricks : 0, last.bricks];
        Object.assign(res, { gpu: start.gpu, said: end.alert.slice(0, 260), issues: last.issues, start_card: start.warn.slice(0, 160), bricks, state: { off_because: end.state.off_because, device_lost: end.state.device_lost, self_test: end.state.self_test } });
        need((last.issues || '').split(',').includes(cs.key), `nothing on screen about "${cs.key}" (issues: ${last.issues || 'none'})`);
        need(end.alert.length > 20, 'the banner is not shown');
        if (cs.key === 'gpu') { need(!start.gpu, 'a GPU world was made'); need(!!end.state.off_because, 'the state does not say why'); need(bricks[1] !== bricks[0], 'the blast changed nothing'); }
        if (cs.key === 'lost') need(!!end.state.device_lost, 'the state does not say the device was lost');
      }
      res.pass = !why.length;
    } catch (e) { res.why.push('the check itself failed: ' + e.message.split('\n')[0]); }
    await b.close();
    fs.writeFileSync(path.join(out, `play_${name}.json`), JSON.stringify({ ...res, date: new Date().toISOString(), config: cfg }, null, 1));
    const { why, ...shown } = res;
    console.log(`${res.pass ? 'PASS' : 'FAIL'} play ${name}: ${JSON.stringify(shown)}`); for (const w of why) console.log('  FAIL ' + w);
    if (!res.pass) failed = true;
  }
  process.exit(failed ? 1 : 0);
})();
