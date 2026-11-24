// ---- lab scenarios (the lab itself is further down): a scenario queued before the reload sets the building
const LAB_SCENARIOS = {
  collapse: { kind: 0, storeys: 4, frames: 600, title: 'concrete frame, 4 storeys: the ground storey cut, collapse, settle', script: [[0, 'cut']], expect: { roofDrop: 2, settle: true } },
  ball: { kind: 0, storeys: 3, frames: 360, title: 'concrete frame, 3 storeys: two steel balls into the facade', script: [[10, 'ball', [-4, 4, -30, 0.15, 0.02]], [130, 'ball', [3, 2.5, -30, -0.1, 0.0]]], expect: { ballOn: true } },
  masonry: { kind: 1, storeys: 3, frames: 600, title: 'masonry, 3 storeys: the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, settle: true } },
  timber: { kind: 2, storeys: 3, frames: 600, title: 'mass timber, 3 storeys: the ground storey cut', script: [[0, 'cut']], expect: { roofDrop: 3, roofDropMax: 6, settle: true } },   // (at most 6 m: the cut storey is 3.75 m, and the timber walls above crush some as they land; before the pinch test's fault of 14 Nov, 3.23 m)
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
  tallball: { kind: 0, storeys: 8, frames: 600, title: 'concrete frame, 8 storeys: a steel ball onto the roof at 45 m/s', script: [[10, 'ball', [7.5, 45, 7.5, 0, -1.5707, 45, 0.4]]], expect: { ballOn: true, ballStops: { above: 3.0, top: 29.5, box: [-11.5, 11.5, -11.5, 11.5] } } },
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
  // ---- the Quarter (kind 6, the game's level: game/src/quarter.rs; docs/QUARTER.md). Actions in a building's own
  // frame (qblast, qball, qcut: lab.js), so a building moved or turned on its lot keeps its scenarios.
  // left alone for 10 s: its loose contents settle, nothing fails
  quarter: { kind: 6, frames: 600, title: 'the Quarter: left alone', script: [], expect: { stand: true, settle: true } },
  // the temple: the three columns at its south-east corner and the next one along the front shot out at mid height
  // (two shots of the 0.75 m tool each): the corner of the entablature, the gable's end and the roof over it come down
  qtemple: { kind: 6, frames: 600, roofAt: ['temple', 5, -13, 3], title: 'the Quarter: the temple\'s south-east corner columns shot out',
    script: [[6.25, -15], [6.25, -12.5], [6.25, -10]].flatMap(([x, z], k) => [[20 + 8 * k, 'qblast', ['temple', 12, 4.0, z, -1, 0, 0, 0.75]], [24 + 8 * k, 'qblast', ['temple', 12, 4.0, z, -1, 0, 0, 0.75]]])
      .concat([[44, 'qblast', ['temple', 3.75, 4.0, -22, 0, 0, 1, 0.75]], [48, 'qblast', ['temple', 3.75, 4.0, -22, 0, 0, 1, 0.75]]]), expect: { settle: true } },
  // the aqueduct: the keystone of the middle arch shot out from both sides (0.9 m): the arch falls, and the channel over it
  // (15 s: the stones of a dry arch go one after another, the last at 9 s)
  qaqueduct: { kind: 6, frames: 900, roofAt: ['aqueduct', 0, 0, 1.5], title: 'the Quarter: the keystone of the aqueduct\'s middle arch shot out',
    script: [[20, 'qblast', ['aqueduct', 0, 7.9, -8, 0, 0, 1, 0.9]], [26, 'qblast', ['aqueduct', 0, 7.9, 8, 0, 0, -1, 0.9]]], expect: { settle: true } },
  // the white villa: three rows of its pilotis shot out at 1.5 m from the entrance's end (the five of each row, one
  // shot after another along the same line: each goes on to the next piloti standing). It holds: 30 joints fail,
  // 11 more are wearing after 15 s, and the box hangs from what is left (lab, 18 Nov). With one row gone no joint
  // fails. What brings it down has not been found: docs/QUARTER.md, "Cuts they carry".
  qvilla: { kind: 6, frames: 900, roofAt: ['villa', 0, -8, 4], title: 'the Quarter: three rows of the villa\'s pilotis shot out: it holds',
    script: [0, 1, 2].flatMap(row => [-9.5, -4.75, 0, 4.75, 9.5].map((x, k) => [20 + 6 * (5 * row + k), 'qblast', ['villa', x, 1.5, -16, 0, 0, 1, 0.5]])), expect: { roofDropMax: 0.1, settle: true } },
  // the glass house: its four north columns cut under the floor
  qglass: { kind: 6, frames: 600, roofAt: ['glass', 0, 0, 4], title: 'the Quarter: the glass house\'s four north columns cut',
    script: [-10.125, -3.375, 3.375, 10.125].map((x, k) => [20 + 6 * k, 'qblast', ['glass', x, 0.8, 9, 0, 0, -1, 0.5]]), expect: { roofDrop: 1, settle: true } },
  // the barn: the four inner arcade posts of each row shot out at 3.5 m, from the nave. It holds: 10 joints fail,
  // the arcade plates and the ties carry the roof over to the gable frames and the aisle walls (lab, 18 Nov; with
  // one row gone no joint fails). (Until 18 Nov the shots started inside the posts' line and broke nothing, and
  // the scenario passed.)
  qbarn: { kind: 6, frames: 900, roofAt: ['barn', 0, 0, 5], title: 'the Quarter: eight of the barn\'s arcade posts shot out: it holds',
    script: [-1, 1].flatMap((sx, r) => [-6.75, -2.25, 2.25, 6.75].map((z, k) => [20 + 6 * (4 * r + k), 'qblast', ['barn', 0, 3.5, z, sx, 0, 0, 0.5]])), expect: { roofDropMax: 0.1, settle: true } },
  // the water tower: three of its four legs shot out at 2 m. It holds: 8 joints fail, and the tank stays up on
  // the fourth leg, the bracing and the riser pipe under its middle, the tank empty (lab, 18 Nov; on two legs no
  // joint fails).
  qtower: { kind: 6, frames: 900, roofAt: ['tower', 0, 0, 4], title: 'the Quarter: three of the water tower\'s legs shot out: it holds',
    script: [[20, 'qblast', ['tower', 8, 2.0, -2.5, -1, 0, 0, 0.5]], [26, 'qblast', ['tower', 8, 2.0, 2.5, -1, 0, 0, 0.5]], [32, 'qblast', ['tower', -8, 2.0, -2.5, 1, 0, 0, 0.5]]], expect: { roofDropMax: 0.1, settle: true } },
  // (15 s: 26 m of chimney goes over like a tree, 7 m of its top's fall in the first 10 s)
  qchimney: { kind: 6, frames: 900, roofAt: ['chimney', 0, 0, 2], title: 'the Quarter: the chimney\'s foot blown out on one side',
    script: [[20, 'qblast', ['chimney', -8, 2.75, 0, 1, 0, 0, 1.5]]], expect: { roofDrop: 10, settle: true } },
  // the mill and the office tower: their ground storeys cut, as the test buildings' are
  qmill: { kind: 6, frames: 900, roofAt: ['mill', 0, 0, 9], title: 'the Quarter: the mill\'s ground storey cut', script: [[0, 'qcut', ['mill', -10, 10, -10, 10]]], expect: { roofDrop: 3, settle: true } },
  qoffice: { kind: 6, frames: 600, roofAt: ['office', 0, 0, 9], title: 'the Quarter: the office tower\'s ground storey cut', script: [[0, 'qcut', ['office', -10, 10, -10, 10]]], expect: { roofDrop: 2 } },
  // the ball (the game's: 0.5 m, 60 m/s, 514 kg, 925 kJ): through the market hall from the crossing where the player
  // starts; through the glass house from end to end; into the temple between two columns, at its cella's wall; and
  // a 0.8 m ball down onto the office tower's roof at 45 m/s. It goes on through what it breaks or stops in it: it
  // does not come back (ballOn).
  qballhall: { kind: 6, frames: 300, title: 'the Quarter: the ball through the market hall', script: [[10, 'ball', [20, 3, -20, -Math.PI / 4, 0, 60, 0.25]]], expect: { ballOn: true } },
  qballglass: { kind: 6, frames: 300, title: 'the Quarter: the ball through the glass house, end to end', script: [[10, 'qball', ['glass', -25, 3.0, 0.5, 1, 0, 0, 60, 0.25]]], expect: { ballOn: true } },
  qballtemple: { kind: 6, frames: 300, title: 'the Quarter: the ball into the temple', script: [[10, 'qball', ['temple', 16, 4.0, 1.25, -1, 0, 0, 60, 0.25]]], expect: { ballOn: true } },
  qballoffice: { kind: 6, frames: 480, roofAt: ['office', 0, 0, 9], title: 'the Quarter: a 0.8 m ball onto the office tower\'s roof at 45 m/s', script: [[10, 'qball', ['office', 7.5, 52, 7.5, 0, -1, 0, 45, 0.4]]], expect: { ballOn: true } },
  // the ball on the roof (24 Nov; the page's second benchmark, key J): a steel ball 5 m across, 514 t, set down at
  // rest on the office tower's roof. Nothing cut, nothing thrown: the weight of what lies at rest goes to the
  // structure once a second, and the roof fails under it (natively: 263 t already fails it, 33 t does not,
  // docs/LAB.md, "The ball on the roof"). Then every floor has the weight and a storey's fall to carry: 18 MJ on a
  // 0.25 m slab. Limit, written when it was added: the ball ends at least 30 m lower, of the 37.5 m to the ground
  // floor's slab less its own radius and the rubble it lies on.
  qroofball: { kind: 6, frames: 1500, roofAt: ['office', 7, 0, 7], title: 'the Quarter: a 514 t steel ball set down on the office tower\'s roof',
    script: [[20, 'qweight', ['office', 0, 0, 2.5]]], expect: { ballDown: 30 } },
};
const QUARTER_SUITE = ['quarter', 'qtemple', 'qaqueduct', 'qvilla', 'qglass', 'qbarn', 'qtower', 'qchimney', 'qmill', 'qoffice', 'qballhall', 'qballglass', 'qballtemple', 'qballoffice', 'qroofball'];
const LAB_SUITES = { quarter: QUARTER_SUITE, quick: ['collapse', 'ball', 'quarter'], full: ['collapse', 'ball', 'masonry', 'timber', 'mixed', 'complex', 'complexcol', 'complexroom', 'keystone', 'rubble', 'nibble', 'far'], complex: ['complexstand', 'complex', 'complexcol', 'complexroom', 'keystone'], city: ['citystand', 'city'] };
let LABSTATE = null; try { LABSTATE = JSON.parse(localStorage.getItem('sf-lab') || 'null'); } catch (e) {}
if (LABSTATE && !(LABSTATE.queue && LABSTATE.queue.length && LAB_SCENARIOS[LABSTATE.queue[0]])) LABSTATE = null;
const LAB = { on: false, t: 0, f: 0 };
// game time: the real clock, or the lab's virtual one
const clock = () => (LAB.on ? LAB.t : performance.now());
