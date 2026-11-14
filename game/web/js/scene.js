// ---- renderer
const cv = $('cv');
const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa9bccb);
scene.fog = new THREE.Fog(0xa9bccb, 80, 420);
const cam = new THREE.PerspectiveCamera(70, 1, 0.05, 1000);
cam.rotation.order = 'YXZ';
scene.add(new THREE.HemisphereLight(0xdfe8f0, 0x5a554c, 0.75));
const sun = new THREE.DirectionalLight(0xfff3e0, 0.95);
sun.position.set(-30, 60, -22); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -32, right: 32, top: 32, bottom: -32, near: 1, far: 160 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.05;   // no acne on the many small debris faces
scene.add(sun); scene.add(sun.target);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(800, 800), new THREE.MeshLambertMaterial({ color: 0x6d6a64 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const plaza = new THREE.Mesh(new THREE.PlaneGeometry(44, 44), new THREE.MeshLambertMaterial({ color: 0x8a867d }));
plaza.rotation.x = -Math.PI / 2; plaza.position.y = 0.002; plaza.receiveShadow = true; scene.add(plaza);
// the city: a paved lot under each building (the streets are the bare ground), the shadow box wide enough for a lot
function setupCity() {
  plaza.visible = false;
  for (const L of lots()) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(32, 32), plaza.material);
    m.rotation.x = -Math.PI / 2; m.position.set(L.x, 0.002, L.z); m.receiveShadow = true; scene.add(m);
  }
  Object.assign(sun.shadow.camera, { left: -48, right: 48, top: 48, bottom: -48 }); sun.shadow.camera.updateProjectionMatrix();
  scene.fog.near = 120; scene.fog.far = 600;
}
function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

