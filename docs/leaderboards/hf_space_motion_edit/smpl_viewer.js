import * as THREE from "./three/three.module.js";

const COLORS = {
  source: 0x71817d,
  target: 0xad6d00,
  tmed: 0xc7563f,
  simmotionedit: 0x315f9d,
  motionlab: 0x6e5aa5,
  motionrefit: 0x8a6f42,
  motioncanvas: 0x087d72,
  motionclr: 0x9b4d83,
};

const caseSelect = document.querySelector("#case-select");
const grid = document.querySelector("#audit-grid");
const scrub = document.querySelector("#scrub");
const playButton = document.querySelector("#play");
const cameraSelect = document.querySelector("#view");
const context = document.querySelector("#case-context");
let caseIndex = null;
let model = null;
let geometry = null;
let activeTrack = "style";
let currentCase = 0;
let currentItem = null;
let playing = true;
let time = 0;
let last = performance.now();
let yaw = 0.68;
let pitch = 0.28;
let distance = 3.7;
const views = new Map();
const caseCache = new Map();
window.__permoSmplViews = views;

function flatten(values) {
  return values.flat();
}

function decodeInt16(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Int16Array(bytes.buffer);
}

function trackData() {
  return caseIndex.tracks[activeTrack];
}

function methodOrder() {
  return Object.keys(trackData().methods);
}

function decodeMethod(key, value, payload) {
  const metadata = trackData().methods[key];
  const rootFlat = decodeInt16(value.root);
  const quaternionFlat = decodeInt16(value.quaternions);
  const root = Array.from({ length: value.frames }, (_, frame) => [
    rootFlat[frame * 3] / payload.root_scale,
    rootFlat[frame * 3 + 1] / payload.root_scale,
    rootFlat[frame * 3 + 2] / payload.root_scale,
  ]);
  const quaternions = Array.from({ length: value.frames }, (_, frame) => {
    const joints = [];
    for (let joint = 0; joint < payload.joint_count; joint += 1) {
      const offset = (frame * payload.joint_count + joint) * 4;
      joints.push([
        quaternionFlat[offset] / payload.quaternion_scale,
        quaternionFlat[offset + 1] / payload.quaternion_scale,
        quaternionFlat[offset + 2] / payload.quaternion_scale,
        quaternionFlat[offset + 3] / payload.quaternion_scale,
      ]);
    }
    while (joints.length < model.joints.length) joints.push([0, 0, 0, 1]);
    return joints;
  });
  return {
    ...metadata,
    ...value.audit,
    frames: value.frames,
    root,
    quaternions,
  };
}

function makeGeometry(data) {
  const result = new THREE.BufferGeometry();
  result.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(flatten(data.vertices), 3),
  );
  result.setAttribute(
    "skinIndex",
    new THREE.Uint16BufferAttribute(flatten(data.skin_indices), 4),
  );
  result.setAttribute(
    "skinWeight",
    new THREE.Float32BufferAttribute(flatten(data.skin_weights), 4),
  );
  result.setIndex(flatten(data.faces));
  result.computeVertexNormals();
  return result;
}

function cameraPose(camera, follow = null) {
  const target = follow || new THREE.Vector3(0, 0.92, 0);
  camera.position.set(
    target.x + distance * Math.sin(yaw) * Math.cos(pitch),
    target.y + distance * Math.sin(pitch),
    target.z + distance * Math.cos(yaw) * Math.cos(pitch),
  );
  camera.lookAt(target);
}

function makeSkeleton(mesh) {
  const bones = model.joints.map(() => new THREE.Bone());
  let root = null;
  bones.forEach((bone, index) => {
    const parent = model.parents[index];
    if (parent < 0) {
      bone.position.fromArray(model.joints[index]);
      root = bone;
    } else {
      const position = new THREE.Vector3().fromArray(model.joints[index]);
      position.sub(new THREE.Vector3().fromArray(model.joints[parent]));
      bone.position.copy(position);
      bones[parent].add(bone);
    }
  });
  mesh.add(root);
  const skeleton = new THREE.Skeleton(bones);
  mesh.bind(skeleton);
  mesh.normalizeSkinWeights();
  return { bones, skeleton };
}

function metricText(data) {
  const metrics = data.metrics;
  const metric = metrics
    ? `M2M R@3 ${Number(metrics["M2M R@3"]).toFixed(2)}`
    : `${data.representation}`;
  const fit =
    data.fit_mpjpe_mm === undefined
      ? ""
      : ` · fit ${Number(data.fit_mpjpe_mm).toFixed(1)}mm`;
  return `${metric}${fit}`;
}

function applyRenderMode(view) {
  view.mesh.visible = true;
  view.helper.visible = false;
  view.material.opacity = 1;
  view.material.transparent = false;
}

function makeView(key, data) {
  const panel = document.createElement("article");
  panel.className = "motion-panel";
  panel.innerHTML = `
    <div class="motion-head">
      <strong>${data.label}</strong>
      <span class="motion-meta">${metricText(data)}<br><b class="frame"></b></span>
    </div>`;
  grid.appendChild(panel);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8faf9);
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
  cameraPose(camera);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  panel.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x4a544f, 1.25));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
  keyLight.position.set(2.4, 4.5, 3.2);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0xb7d8ff, 0.55);
  rimLight.position.set(-2.5, 2.0, -3.0);
  scene.add(rimLight);
  scene.add(new THREE.GridHelper(5, 10, 0xc8d2ce, 0xe2e8e5));

  const material = new THREE.MeshStandardMaterial({
    color: COLORS[key] || 0x71817d,
    roughness: 0.76,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.SkinnedMesh(geometry, material);
  const skeleton = makeSkeleton(mesh);
  scene.add(mesh);
  const helper = new THREE.SkeletonHelper(mesh);
  helper.material.color.setHex(0x26332f);
  helper.material.depthTest = false;
  helper.renderOrder = 3;
  scene.add(helper);

  const resize = () => {
    const width = panel.clientWidth;
    const height = renderer.domElement.clientHeight || 260;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(panel);
  resize();

  let drag = null;
  renderer.domElement.addEventListener("pointerdown", (event) => {
    drag = [event.clientX, event.clientY];
    renderer.domElement.setPointerCapture(event.pointerId);
  });
  renderer.domElement.addEventListener("pointermove", (event) => {
    if (!drag) return;
    yaw -= (event.clientX - drag[0]) * 0.008;
    pitch = Math.max(
      -0.12,
      Math.min(1.12, pitch + (event.clientY - drag[1]) * 0.006),
    );
    drag = [event.clientX, event.clientY];
    cameraSelect.value = "orbit";
    for (const view of views.values()) cameraPose(view.camera, view.follow);
  });
  renderer.domElement.addEventListener("pointerup", () => {
    drag = null;
  });
  renderer.domElement.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      distance = Math.max(1.8, Math.min(7, distance + event.deltaY * 0.002));
      for (const view of views.values()) cameraPose(view.camera, view.follow);
    },
    { passive: false },
  );

  const view = {
    panel,
    scene,
    camera,
    renderer,
    material,
    mesh,
    helper,
    bones: skeleton.bones,
    follow: new THREE.Vector3(0, 0.92, 0),
    data,
    frameTag: panel.querySelector(".frame"),
  };
  applyRenderMode(view);
  return view;
}

function disposeViews() {
  for (const view of views.values()) {
    view.material.dispose();
    view.renderer.dispose();
  }
  views.clear();
  grid.innerHTML = "";
}

function durationFor(item) {
  return Math.max(
    ...methodOrder().map(
      (key) => item.methods[key].frames / item.methods[key].fps,
    ),
  );
}

function updateView(view, seconds) {
  const frame = Math.min(
    view.data.frames - 1,
    Math.floor(seconds * view.data.fps),
  );
  const root = view.data.root[frame];
  const quaternions = view.data.quaternions[frame];
  view.bones[0].position.set(
    model.joints[0][0] + root[0],
    model.joints[0][1] + root[1],
    model.joints[0][2] + root[2],
  );
  view.follow.set(root[0], 0.92, root[2]);
  cameraPose(view.camera, view.follow);
  for (let joint = 0; joint < view.bones.length; joint += 1) {
    view.bones[joint].quaternion.fromArray(quaternions[joint]);
  }
  view.mesh.skeleton.update();
  view.helper.updateMatrixWorld(true);
  view.frameTag.textContent = `f ${frame}/${view.data.frames - 1} · ${view.data.fps} fps`;
}

function caseContext(payload) {
  const source =
    activeTrack === "content" && payload.source_action
      ? `Source action <b>${payload.source_action}</b> → target <b>${payload.action}</b>`
      : `Target style <b>${payload.style}</b> · action <b>${payload.action}</b>`;
  context.innerHTML = `${source}<br>${payload.instruction}<br><span>${payload.target_caption}</span>`;
}

async function loadCase(index) {
  currentCase = index;
  time = 0;
  scrub.value = 0;
  disposeViews();
  grid.innerHTML = '<div class="empty">Loading synchronized case…</div>';
  const metadata = trackData().cases[index];
  const cacheKey = `${activeTrack}:${index}`;
  let payload = caseCache.get(cacheKey);
  if (!payload) {
    const response = await fetch(
      `${metadata.path}?v=${encodeURIComponent(caseIndex.revision)}`,
    );
    if (!response.ok) throw new Error(`case HTTP ${response.status}`);
    payload = await response.json();
    caseCache.set(cacheKey, payload);
  }
  currentItem = {
    ...payload,
    methods: Object.fromEntries(
      methodOrder().map((key) => [
        key,
        decodeMethod(key, payload.methods[key], payload),
      ]),
    ),
  };
  caseContext(payload);
  grid.innerHTML = "";
  for (const key of methodOrder()) {
    views.set(key, makeView(key, currentItem.methods[key]));
  }
  const duration = durationFor(currentItem);
  document.querySelector("#duration").textContent = `${duration.toFixed(2)}s`;
  renderFrame();
}

function renderFrame() {
  if (!currentItem) return;
  const duration = durationFor(currentItem);
  const bounded = duration ? time % duration : 0;
  for (const view of views.values()) {
    updateView(view, bounded);
    view.renderer.render(view.scene, view.camera);
  }
  scrub.value = duration ? Math.round((bounded / duration) * 1000) : 0;
  document.querySelector("#time-now").textContent = `${bounded.toFixed(2)}s`;
}

function populateCases() {
  caseSelect.innerHTML = trackData().cases
    .map(
      (item, index) =>
        `<option value="${index}">${String(index).padStart(4, "0")} · ${item.style} · ${item.action}</option>`,
    )
    .join("");
}

async function setTrack(track) {
  if (!caseIndex || !caseIndex.tracks[track]) return;
  activeTrack = track;
  currentCase = 0;
  populateCases();
  await loadCase(0);
}

function animate(now) {
  const delta = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (playing && currentItem) time += delta;
  renderFrame();
  requestAnimationFrame(animate);
}

caseSelect.addEventListener("change", () => loadCase(Number(caseSelect.value)));
playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
});
scrub.addEventListener("input", () => {
  if (!currentItem) return;
  time = (Number(scrub.value) / 1000) * durationFor(currentItem);
  renderFrame();
});
cameraSelect.addEventListener("change", () => {
  if (cameraSelect.value === "front") {
    yaw = 0;
    pitch = 0.22;
  } else if (cameraSelect.value === "side") {
    yaw = Math.PI / 2;
    pitch = 0.22;
  }
  for (const view of views.values()) cameraPose(view.camera, view.follow);
});
window.addEventListener("permo-track-change", (event) => {
  setTrack(event.detail.track);
});

Promise.all([
  fetch("case_index.json").then((response) => response.json()),
  fetch("smpl_model.json").then((response) => response.json()),
])
  .then(async ([index, smpl]) => {
    caseIndex = index;
    model = smpl;
    geometry = makeGeometry(model);
    populateCases();
    await loadCase(0);
    requestAnimationFrame(animate);
  })
  .catch((error) => {
    grid.innerHTML = `<div class="empty">Visualization failed: ${error.message}</div>`;
    console.error(error);
  });
