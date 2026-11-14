// ---- lab scenarios (the lab itself is further down): a scenario queued before the reload sets the building
const LAB_SCENARIOS = {
  collapse: { kind: 0, storeys: 4, frames: 600, title: 'concrete frame, 4 storeys: the ground storey cut, collapse, settle', script: [[0, 'cut']], expect: { roofDrop: 2, settle: true } },
  ball: { kind: 0, storeys: 3, frames: 360, title: 'concrete frame, 3 storeys: two steel balls into the facade', script: [[10, 'ball', [-4, 4, -30, 0.15, 0.02]], [130, 'ball', [3, 2.5, -30, -0.1, 0.0]]], expect: {} },
  masonry: { kind: 1, storeys: 3, frames: 600, title: 'masonry, 3 storeys: the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, settle: true } },
  timber: { kind: 2, storeys: 3, frames: 600, title: 'mass timber, 3 storeys: the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, settle: true } },
  mixed: { kind: 3, storeys: 5, frames: 600, title: 'mixed (brick, steel, concrete, timber), 5 storeys: the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, settle: true } },
  // the mixed building's ground storey eaten away with the 0.7 m tool, as a player does: the west wall shot through
  // spot by spot, then the west halves of the south and north fronts. It must crack and give way progressively
  // (worn joints before the roof falls), and nothing may vanish.
  nibble: (() => { const s = []; let f = 20; const shot = (o, d) => { s.push([f, 'blast', [...o, ...d, 0.7]]); f += 4; };
    for (let z = -10.8; z <= 10.9; z += 0.9) { shot([-16, 1.3, z], [1, 0, 0]); shot([-16, 1.3, z], [1, 0, 0]); }
    for (let x = -10.8; x <= 0.1; x += 0.9) { shot([x, 1.3, -16], [0, 0, 1]); shot([x, 1.3, 16], [0, 0, -1]); }
    // then the two western steel columns, through the opened wall, and the rest of the south and north fronts
    for (let k = 0; k < 4; k++) { shot([-16, 1.3, -7.5], [1, 0, 0]); shot([-16, 1.3, 7.5], [1, 0, 0]); }
    for (let x = 0.9; x <= 10.9; x += 0.9) { shot([x, 1.3, -16], [0, 0, 1]); shot([x, 1.3, 16], [0, 0, -1]); }
    // the east wall, then the eastern columns through the opened fronts: the core is left (engine: worst joint 0.85,
    // test explore_the_mixed_building_cut_away), and it must wear and give way, not stand on it for ever
    for (let z = -10.8; z <= 10.9; z += 0.9) { shot([16, 1.3, z], [-1, 0, 0]); shot([16, 1.3, z], [-1, 0, 0]); }
    for (let k = 0; k < 4; k++) { shot([7.5, 1.3, -16], [0, 0, 1]); shot([7.5, 1.3, 16], [0, 0, -1]); }
    return { kind: 3, storeys: 5, frames: f + 900, title: 'mixed, 5 storeys: the ground storey eaten away with the 0.7 m tool', script: s, expect: { gradual: true, upperLoss: 0.5 } }; })(),
  // the clean-up distance, short: a steel ball fired at the facade from 30 m goes once it has flown 15 m from where
  // it was fired (a blast into a cut building's rubble threw nothing 8 m: the scenario must throw something)
  far: { kind: 0, storeys: 3, frames: 240, far: 15, title: 'concrete frame, 3 storeys: a steel ball fired from 30 m, clean-up distance 15 m', script: [[10, 'ball', [-4, 4, -30, 0.15, 0.02]]], expect: { far: true } },
  // an 8-storey concrete frame, a steel ball of 0.4 m sent down onto the middle of a roof bay at 45 m/s: it breaks
  // through the roof, which takes most of its speed, and stops inside on a floor (13 Nov; docs/LAB.md: at 25 m/s it
  // stays in a crater on the roof, at 80 m/s it bounces 20 m up and off the building)
  tallball: { kind: 0, storeys: 8, frames: 600, title: 'concrete frame, 8 storeys: a steel ball onto the roof at 45 m/s', script: [[10, 'ball', [7.5, 45, 7.5, 0, -1.5707, 45, 0.4]]], expect: { ballStops: { above: 3.0, top: 29.5, box: [-11.5, 11.5, -11.5, 11.5] } } },
  // the debris budget (14 Nov): the collapse with at most 1,000 pieces and particles kept at once
  debriscap: { kind: 0, storeys: 4, frames: 480, debris: 1000, title: 'concrete frame, 4 storeys: the collapse with a debris budget of 1,000', script: [[0, 'cut']], expect: { debrisCap: 1000, debrisHeaviest: 300, debrisShare: 30 } },   // (300 kg: `DEBRIS_SMALL`, no big piece goes while crumbs are left; 30%: all the particles this collapse makes are about 22% of the building, docs/LAB.md 14 November)
  // the complex building (kind 4, docs/PHASE2.md): steel frame with free-piece braces, a roof of rafters and boards
  // at 26 degrees, an arcade of stone arches; the ground storey and the arcade's piers cut
  complex: { kind: 4, storeys: 3, frames: 600, title: 'complex (braced steel frame, pitched timber roof, stone arcade): the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, settle: true } },
  // the complex building left alone for 10 s: its loose contents settle on the floors, nothing fails (it came down
  // by itself in the playground: 8 joints failed at about 5 s)
  complexstand: { kind: 4, storeys: 3, frames: 600, title: 'complex: left alone', script: [], expect: { stand: true, settle: true } },
  // the frame's north-east corner column shot out at 1.5 m (four shots of the 0.7 m tool from the east)
  complexcol: { kind: 4, storeys: 3, frames: 600, title: 'complex: the north-east corner column shot out', script: [20, 24, 28, 32].map(f => [f, 'blast', [16, 1.5, 6, -1, 0, 0, 0.7]]), expect: { settle: true } },
  // a 1.5 m blast in a first-floor room (from the south, over the arcade, into the west bay: its cabinet)
  complexroom: { kind: 4, storeys: 3, frames: 600, title: 'complex: a blast in a first-floor room', script: [[20, 'blast', [-6, 5.5, -16, 0, 0, 1, 1.5]]], expect: { settle: true } },
  // the keystone of the arcade's third arch knocked out (a 0.75 m shot from the south at its middle; 0.5 m left its back
  // half: engine test the_lab_keystone_shot_takes_the_keystone_out): the loggia bears
  // on the stones' level tops and holds the two halves on their piers (engine: complex::tests, nothing fails)
  keystone: { kind: 4, storeys: 3, frames: 480, title: 'complex: the keystone shot out (third arch)', script: [[20, 'blast', [-1.5, 3.4, -16, 0, 0, 1, 0.75]]], expect: { settle: true } },
  // the city (kind 5, docs/CITY.md): 3 x 3 lots left alone; then two lots' ground storeys cut (the complex building
  // and the 7-storey concrete frame beside it), the rest of the city standing round them
  citystand: { kind: 5, storeys: 3, frames: 600, title: 'city, 3 x 3 lots: left alone', script: [], expect: { stand: true, settle: true } },
  city: { kind: 5, storeys: 3, frames: 600, title: 'city, 3 x 3 lots: two lots cut (complex and a 7-storey frame)', script: [[0, 'cut', [0, 1]]], expect: { settle: true } },
  rubble: { kind: 0, storeys: 3, frames: 720, title: 'concrete frame, 3 storeys: cut, then a blast into the rubble', script: [[0, 'cut'], [480, 'blast', [-14, 6, -14, 1, -0.45, 1, 2.5]]], expect: { settle: true } },
};
const LAB_SUITES = { quick: ['collapse', 'ball'], full: ['collapse', 'ball', 'masonry', 'timber', 'mixed', 'complex', 'complexcol', 'complexroom', 'keystone', 'rubble', 'nibble', 'far'], complex: ['complexstand', 'complex', 'complexcol', 'complexroom', 'keystone'], city: ['citystand', 'city'] };
let LABSTATE = null; try { LABSTATE = JSON.parse(localStorage.getItem('sf-lab') || 'null'); } catch (e) {}
if (LABSTATE && !(LABSTATE.queue && LABSTATE.queue.length && LAB_SCENARIOS[LABSTATE.queue[0]])) LABSTATE = null;
const LAB = { on: false, t: 0, f: 0 };
// game time: the real clock, or the lab's virtual one
const clock = () => (LAB.on ? LAB.t : performance.now());
