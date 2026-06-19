import * as THREE from 'three';

const SKY_HORIZON = 0xf5dcc0;
const SKY_SUNSET = 0xffb870;
const ASPHALT = 0x4a4a50;
const CONCRETE = 0x9ca0a8;
const RAIL = 0xc8c8d0;
const PLATFORM = 0xd8d2c8;
const TILE = 0xc8c0b4;
const CANOPY = 0xa8a098;
const TRAIN_STEEL = 0xd8dde8;
const TRAIN_BAND = 0x1e2430;
const MTA_RED = 0xe74c3c;
const YELLOW_LINE = 0xf5c842;

const S1 = -95;
const S2 = 0;
const S3 = 95;
const TRACK_HALF = 120;
const VIADUCT_Y = 4.2;
const TRACK_Y = VIADUCT_Y + 0.55;

interface CameraKeyframe {
  progress: number;
  pos: THREE.Vector3;
  look: THREE.Vector3;
}

const CAMERA_PATH: CameraKeyframe[] = [
  { progress: 0.0, pos: new THREE.Vector3(7, 7, S1 + 16), look: new THREE.Vector3(0, TRACK_Y + 1, S1 - 2) },
  { progress: 0.25, pos: new THREE.Vector3(5, 8, S1 + 10), look: new THREE.Vector3(0, TRACK_Y + 0.5, S1 - 1) },

  { progress: 0.26, pos: new THREE.Vector3(4, 14, S1 + 2), look: new THREE.Vector3(0, TRACK_Y, S1 + 22) },
  { progress: 0.5, pos: new THREE.Vector3(2, 32, S1 + 18), look: new THREE.Vector3(0, TRACK_Y, S1 + 58) },

  { progress: 0.51, pos: new THREE.Vector3(-16, 8, S2 + 12), look: new THREE.Vector3(-5, TRACK_Y + 1, S2) },
  { progress: 0.75, pos: new THREE.Vector3(-12, 7, S2 + 8), look: new THREE.Vector3(0, TRACK_Y + 0.5, S2 - 2) },

  { progress: 0.76, pos: new THREE.Vector3(12, 10, S3 + 20), look: new THREE.Vector3(0, TRACK_Y + 1, S3 - 4) },
  { progress: 1.0, pos: new THREE.Vector3(50, 38, S3 + 62), look: new THREE.Vector3(0, TRACK_Y, S3) },
];

function lerpKeyframes(progress: number): { pos: THREE.Vector3; look: THREE.Vector3 } {
  let a = CAMERA_PATH[0];
  let b = CAMERA_PATH[CAMERA_PATH.length - 1];
  for (let i = 0; i < CAMERA_PATH.length - 1; i++) {
    if (progress >= CAMERA_PATH[i].progress && progress <= CAMERA_PATH[i + 1].progress) {
      a = CAMERA_PATH[i];
      b = CAMERA_PATH[i + 1];
      break;
    }
  }
  const range = b.progress - a.progress || 1;
  const t = THREE.MathUtils.clamp((progress - a.progress) / range, 0, 1);
  const eased = t * t * (3 - 2 * t);
  return {
    pos: a.pos.clone().lerp(b.pos, eased),
    look: a.look.clone().lerp(b.look, eased),
  };
}

function trainZ(progress: number): number {
  if (progress <= 0.25) return S1 - 2;
  if (progress <= 0.5) {
    const t = (progress - 0.25) / 0.25;
    return THREE.MathUtils.lerp(S1 - 2, S1 + 42, t);
  }
  if (progress <= 0.75) {
    const t = (progress - 0.5) / 0.25;
    return THREE.MathUtils.lerp(S1 + 42, S2 + 1, 1 - Math.pow(1 - t, 1.8));
  }
  if (progress <= 0.92) {
    const t = (progress - 0.75) / 0.17;
    return THREE.MathUtils.lerp(S2 + 3, S3 - 6, t);
  }
  const t = (progress - 0.92) / 0.08;
  return THREE.MathUtils.lerp(S3 - 6, S3 - 2, 1 - Math.pow(1 - t, 2));
}

function isTrainMoving(p: number): boolean {
  return (p > 0.24 && p < 0.26) || (p > 0.25 && p < 0.74) || (p > 0.75 && p < 0.91);
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeSkyTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 512;
  const ctx = c.getContext('2d')!;
  const g = ctx.createLinearGradient(0, 0, 0, 512);
  g.addColorStop(0, '#3d8fd9');
  g.addColorStop(0.35, '#6db8ec');
  g.addColorStop(0.62, '#b8ddf5');
  g.addColorStop(0.82, '#f8e8c8');
  g.addColorStop(1, '#f5d4a0');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 4, 512);
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeSign(text: string, w = 4): THREE.Mesh {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 96;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#1a1a22';
  ctx.fillRect(0, 0, 512, 96);
  ctx.fillStyle = '#f5c842';
  ctx.font = 'bold 34px Georgia, serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, 256, 58);
  return new THREE.Mesh(
    new THREE.PlaneGeometry(w, w * 0.22),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }),
  );
}

function buildBench(x: number, z: number, baseY: number): THREE.Group {
  const g = new THREE.Group();
  const seat = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 0.12, 0.55),
    new THREE.MeshStandardMaterial({ color: 0x7a6555 }),
  );
  seat.position.y = 0.52;
  seat.castShadow = true;
  const legMat = new THREE.MeshStandardMaterial({ color: 0x666666, metalness: 0.6 });
  [[-0.7, -0.2], [0.7, -0.2], [-0.7, 0.2], [0.7, 0.2]].forEach(([lx, lz]) => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.5, 0.08), legMat);
    leg.position.set(lx, 0.25, lz);
    g.add(leg);
  });
  g.add(seat);
  g.position.set(x, baseY, z);
  return g;
}

type RiderBehavior = 'watch' | 'phone' | 'pace' | 'tap' | 'listen';

interface EmotionBubble {
  sprite: THREE.Sprite;
  slot: number;
  baseY: number;
}

const FEATURED_MIDTOWN: { x: number; z: number; behavior: RiderBehavior; emoji: string }[] = [
  { x: -5.6, z: -5, behavior: 'watch', emoji: '⏰' },
  { x: 5.8, z: 4, behavior: 'phone', emoji: '📱' },
  { x: -4.8, z: 3, behavior: 'pace', emoji: '🤔' },
  { x: 6.0, z: -3, behavior: 'listen', emoji: '👂' },
  { x: -5.4, z: 0, behavior: 'tap', emoji: '😤' },
  { x: 5.5, z: 6, behavior: 'watch', emoji: '😩' },
  { x: -6.2, z: 6, behavior: 'phone', emoji: '💭' },
  { x: 4.6, z: -6, behavior: 'pace', emoji: '👀' },
];

function makeEmojiBubble(emoji: string): THREE.Sprite {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d')!;

  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.strokeStyle = 'rgba(240, 192, 64, 0.55)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(14, 10, size - 28, size - 36, 22);
  ctx.fill();
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(size / 2 - 10, size - 28);
  ctx.lineTo(size / 2, size - 8);
  ctx.lineTo(size / 2 + 10, size - 28);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.font = '52px "Segoe UI Emoji", "Apple Color Emoji", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(emoji, size / 2, size / 2 - 4);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.85, 0.85, 1);
  sprite.renderOrder = 999;
  sprite.visible = false;
  return sprite;
}

function attachEmojiBubble(rider: THREE.Group, emoji: string, slot: number): EmotionBubble {
  const r = rider.userData.rider as RiderData;
  const sprite = makeEmojiBubble(emoji);
  const baseY = r.head.position.y + 0.58;
  sprite.position.y = baseY;
  rider.add(sprite);
  return { sprite, slot, baseY };
}

function animateEmotionBubbles(bubbles: EmotionBubble[], time: number, waitPhase: number): void {
  bubbles.forEach((b) => {
    const stagger = b.slot * 0.07;
    const appear = THREE.MathUtils.clamp((waitPhase - stagger) / 0.12, 0, 1);
    const mat = b.sprite.material as THREE.SpriteMaterial;
    mat.opacity = appear;
    b.sprite.visible = appear > 0.03;

    const bob = Math.sin(time * 2.8 + b.slot * 0.9) * 0.07 * appear;
    const pulse = 0.72 + Math.sin(time * 3.2 + b.slot) * 0.07;
    b.sprite.position.y = b.baseY + bob;
    b.sprite.scale.set(pulse * appear, pulse * appear, 1);
  });
}

interface RiderData {
  behavior: RiderBehavior;
  head: THREE.Mesh;
  baseX: number;
  baseZ: number;
  phase: number;
  arm?: THREE.Mesh;
}

function buildPassenger(
  x: number,
  z: number,
  baseY: number,
  seed: number,
  behavior: RiderBehavior = 'watch',
): THREE.Group {
  const g = new THREE.Group();
  const rng = mulberry32(seed);
  const colors = [0xe74c3c, 0x3498db, 0x2ecc71, 0x9b59b6, 0xf39c12, 0x1abc9c, 0xe67e22];
  const bodyMat = new THREE.MeshStandardMaterial({
    color: colors[Math.floor(rng() * colors.length)],
  });
  const skinMat = new THREE.MeshStandardMaterial({ color: 0xe8b896 });
  const sitting = behavior === 'listen' || behavior === 'tap' || rng() > 0.55;
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.26, sitting ? 0.7 : 1.0, 6),
    bodyMat,
  );
  body.position.y = sitting ? 0.85 : 1.05;
  body.castShadow = true;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 6, 6), skinMat);
  head.position.y = sitting ? 1.28 : 1.68;
  head.castShadow = true;
  g.add(body, head);

  let arm: THREE.Mesh | undefined;
  if (behavior === 'phone') {
    arm = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.35, 0.08),
      new THREE.MeshStandardMaterial({ color: skinMat.color }),
    );
    arm.position.set(0.28, sitting ? 1.1 : 1.35, 0.12);
    arm.rotation.x = -0.8;
    g.add(arm);
  }

  g.position.set(x, baseY, z);
  g.rotation.y = (rng() - 0.5) * 0.5;

  const riderData: RiderData = {
    behavior,
    head,
    baseX: x,
    baseZ: z,
    phase: seed * 0.1,
    arm,
  };
  g.userData.rider = riderData;
  return g;
}

function animateMidtownRiders(riders: THREE.Group[], time: number, intensity: number): void {
  riders.forEach((g) => {
    const r = g.userData.rider as RiderData | undefined;
    if (!r) return;
    const t = time + r.phase;

    switch (r.behavior) {
      case 'watch':
        r.head.rotation.x = -0.35 + Math.sin(t * 1.2) * 0.15;
        g.rotation.y = Math.sin(t * 0.6) * 0.25 + 0.3;
        break;
      case 'phone':
        r.head.rotation.x = -0.45 + Math.sin(t * 2) * 0.08;
        if (r.arm) r.arm.rotation.z = Math.sin(t * 1.5) * 0.06;
        break;
      case 'pace':
        g.position.x = r.baseX + Math.sin(t * 1.1) * 0.2 * intensity;
        g.position.z = r.baseZ + Math.cos(t * 0.9) * 0.12 * intensity;
        break;
      case 'tap':
        g.position.y = TRACK_Y + Math.abs(Math.sin(t * 4)) * 0.04 * intensity;
        r.head.rotation.y = Math.sin(t * 0.8) * 0.12;
        break;
      case 'listen':
        r.head.rotation.x = Math.sin(t * 0.5) * 0.1;
        g.rotation.y = Math.sin(t * 0.4) * 0.08;
        break;
    }
  });
}

function buildStation(
  z: number,
  name: string,
  opts: {
    passengers: number;
    midtownRiders?: THREE.Group[];
    emotionBubbles?: EmotionBubble[];
  },
): THREE.Group {
  const g = new THREE.Group();
  g.position.z = z;
  const platW = 7;
  const platLen = 22;
  const platMat = new THREE.MeshStandardMaterial({ color: PLATFORM, roughness: 0.75 });
  const tileMat = new THREE.MeshStandardMaterial({ color: TILE });

  [-1, 1].forEach((side) => {
    const px = side * 5.2;
    const platform = new THREE.Mesh(new THREE.BoxGeometry(platW, 0.35, platLen), platMat);
    platform.position.set(px, TRACK_Y + 0.18, 0);
    platform.receiveShadow = true;
    g.add(platform);

    const tiles = new THREE.Mesh(new THREE.BoxGeometry(platW - 0.3, 0.04, platLen - 0.4), tileMat);
    tiles.position.set(px, TRACK_Y + 0.38, 0);
    g.add(tiles);

    const yLine = new THREE.Mesh(
      new THREE.BoxGeometry(platW - 0.5, 0.03, 0.12),
      new THREE.MeshStandardMaterial({
        color: YELLOW_LINE,
        emissive: YELLOW_LINE,
        emissiveIntensity: 0.35,
      }),
    );
    yLine.position.set(px, TRACK_Y + 0.42, side > 0 ? -platLen / 2 + 1.2 : platLen / 2 - 1.2);
    g.add(yLine);

    const roof = new THREE.Mesh(
      new THREE.BoxGeometry(platW + 0.5, 0.2, platLen + 1),
      new THREE.MeshStandardMaterial({ color: CANOPY }),
    );
    roof.position.set(px, TRACK_Y + 4.2, 0);
    g.add(roof);

    for (let pz = -9; pz <= 9; pz += 9) {
      const post = new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 3.8, 0.3),
        new THREE.MeshStandardMaterial({ color: CONCRETE }),
      );
      post.position.set(px + side * 3.2, TRACK_Y + 2.1, pz);
      g.add(post);
    }
  });

  const sign = makeSign(name);
  sign.position.set(-5.2, TRACK_Y + 3.5, -8);
  sign.rotation.y = Math.PI / 2;
  g.add(sign);

  [[-5.2, -6], [-5.2, 2], [5.2, -2], [5.2, 6]].forEach(([bx, bz]) =>
    g.add(buildBench(bx, bz, TRACK_Y)),
  );

  const rng = mulberry32(z * 997);

  if (z === S2 && opts.emotionBubbles) {
    FEATURED_MIDTOWN.forEach((f, i) => {
      const rider = buildPassenger(f.x, f.z, TRACK_Y, i * 17 + z, f.behavior);
      opts.emotionBubbles!.push(attachEmojiBubble(rider, f.emoji, i));
      g.add(rider);
      opts.midtownRiders?.push(rider);
    });
    const extra = Math.max(0, opts.passengers - FEATURED_MIDTOWN.length);
    for (let i = 0; i < extra; i++) {
      const side = rng() > 0.5 ? 5.2 : -5.2;
      const rider = buildPassenger(
        side + (rng() - 0.5) * 2,
        (rng() - 0.5) * 14,
        TRACK_Y,
        i * 31 + z + 500,
        'watch',
      );
      g.add(rider);
      opts.midtownRiders?.push(rider);
    }
  } else {
    for (let i = 0; i < opts.passengers; i++) {
      const side = rng() > 0.5 ? 5.2 : -5.2;
      const rider = buildPassenger(
        side + (rng() - 0.5) * 2,
        (rng() - 0.5) * 14,
        TRACK_Y,
        i * 31 + z,
        'watch',
      );
      g.add(rider);
      if (opts.midtownRiders && z === S2) opts.midtownRiders.push(rider);
    }
  }

  return g;
}

function buildBuilding(x: number, z: number, w: number, d: number, h: number, color: number): THREE.Group {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, roughness: 0.85 }),
  );
  body.position.y = h / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  const winMat = new THREE.MeshStandardMaterial({
    color: 0xfff8e0,
    emissive: 0xffe8a0,
    emissiveIntensity: 0.4,
  });
  const rows = Math.floor(h / 2.5);
  const cols = Math.floor(w / 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (mulberry32(x * 100 + z + r * cols + c)() < 0.35) continue;
      const win = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 1.2), winMat);
      win.position.set(-w / 2 + 1.2 + c * 2, 1.5 + r * 2.5, d / 2 + 0.02);
      g.add(win);
    }
  }
  g.position.set(x, 0, z);
  return g;
}

function buildUrbanBlock(scene: THREE.Scene, rng: () => number): void {
  const asphalt = new THREE.Mesh(
    new THREE.PlaneGeometry(900, 900),
    new THREE.MeshStandardMaterial({ color: ASPHALT, roughness: 0.92 }),
  );
  asphalt.rotation.x = -Math.PI / 2;
  asphalt.receiveShadow = true;
  scene.add(asphalt);

  const buildingColors = [0xe8e4dc, 0xc8d0d8, 0xd8ccc0, 0xb8c8d8, 0xf0ece4, 0xa8b0b8, 0xd4c8b8];
  for (let iz = -8; iz <= 8; iz++) {
    for (const side of [-1, 1]) {
      const bx = side * (22 + rng() * 18);
      const bz = iz * 28 + (rng() - 0.5) * 12;
      if (Math.abs(bz) < 12 && Math.abs(bx) < 30) continue;
      const h = 12 + rng() * 35;
      const w = 10 + rng() * 14;
      const d = 10 + rng() * 12;
      scene.add(
        buildBuilding(
          bx,
          bz,
          w,
          d,
          h,
          buildingColors[Math.floor(rng() * buildingColors.length)],
        ),
      );
    }
  }

  for (let z = -TRACK_HALF; z < TRACK_HALF; z += 35) {
    [-1, 1].forEach((side) => {
      const lamp = new THREE.Group();
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.08, 0.1, 5, 6),
        new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.7 }),
      );
      pole.position.y = 2.5;
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 8, 8),
        new THREE.MeshStandardMaterial({
          color: 0xfff8e0,
          emissive: 0xffe8a0,
          emissiveIntensity: 0.8,
        }),
      );
      bulb.position.y = 5.1;
      lamp.add(pole, bulb);
      lamp.position.set(side * 16, 0, z);
      scene.add(lamp);
    });
  }
}

function buildSubwayTrain(): { train: THREE.Group; wheels: THREE.Mesh[]; headlight: THREE.SpotLight } {
  const train = new THREE.Group();
  const steelMat = new THREE.MeshStandardMaterial({
    color: TRAIN_STEEL,
    metalness: 0.85,
    roughness: 0.18,
  });
  const bandMat = new THREE.MeshStandardMaterial({ color: TRAIN_BAND, metalness: 0.5, roughness: 0.4 });
  const redMat = new THREE.MeshStandardMaterial({ color: MTA_RED, metalness: 0.3, roughness: 0.5 });
  const winMat = new THREE.MeshStandardMaterial({
    color: 0x2a3548,
    metalness: 0.9,
    roughness: 0.08,
  });
  const winLit = new THREE.MeshStandardMaterial({
    color: 0xe8f4ff,
    emissive: 0xb8d8f8,
    emissiveIntensity: 0.35,
  });
  const underMat = new THREE.MeshStandardMaterial({ color: 0x2a2a30 });

  const carLen = 6.8;
  const carCount = 4;
  const totalLen = carLen * carCount - 0.4;

  for (let c = 0; c < carCount; c++) {
    const car = new THREE.Group();
    const cz = -totalLen / 2 + carLen / 2 + c * (carLen - 0.12);
    const cl = carLen - 0.08;

    const body = new THREE.Mesh(new THREE.BoxGeometry(2.7, 2.15, cl), steelMat);
    body.position.y = 1.4;
    body.castShadow = true;
    car.add(body);

    const band = new THREE.Mesh(new THREE.BoxGeometry(2.72, 0.55, cl), bandMat);
    band.position.y = 1.15;
    car.add(band);

    const redStripe = new THREE.Mesh(new THREE.BoxGeometry(2.73, 0.12, cl), redMat);
    redStripe.position.y = 0.88;
    car.add(redStripe);

    const roof = new THREE.Mesh(new THREE.BoxGeometry(2.55, 0.25, cl - 0.15), underMat);
    roof.position.y = 2.52;
    car.add(roof);

    if (c === 1 || c === 2) {
      const panto = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.5, 1.2), underMat);
      panto.position.set(0, 2.75, 0);
      car.add(panto);
    }

    const nWin = 5;
    for (let w = 0; w < nWin; w++) {
      const wz = -cl / 2 + 0.75 + w * ((cl - 1.5) / (nWin - 1));
      [-1.36, 1.36].forEach((wx) => {
        const frame = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.78, 1.05), winMat);
        frame.position.set(wx, 1.62, wz);
        car.add(frame);
        const glass = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.7, 0.95), winLit);
        glass.position.set(wx * 0.98, 1.62, wz);
        car.add(glass);
      });
      if (w < nWin - 1 && w % 2 === 0) {
        const door = new THREE.Mesh(new THREE.BoxGeometry(2.72, 1.6, 0.08), bandMat);
        door.position.set(0, 1.25, wz + 0.55);
        car.add(door);
      }
    }

    if (c < carCount - 1) {
      const bellows = new THREE.Mesh(new THREE.BoxGeometry(2.5, 1.7, 0.4), bandMat);
      bellows.position.set(0, 1.3, cl / 2 - 0.02);
      car.add(bellows);
    }

    car.position.z = cz;
    train.add(car);
  }

  const nose = new THREE.Group();
  const noseZ = -totalLen / 2 - 0.85;

  const noseMain = new THREE.Mesh(
    new THREE.SphereGeometry(1.38, 16, 12),
    steelMat,
  );
  noseMain.scale.set(1, 1, 0.55);
  noseMain.position.set(0, 1.42, 0.2);
  noseMain.castShadow = true;
  nose.add(noseMain);

  const noseBand = new THREE.Mesh(new THREE.BoxGeometry(2.72, 0.55, 0.9), bandMat);
  noseBand.position.set(0, 1.15, 0.35);
  nose.add(noseBand);

  const windshield = new THREE.Mesh(
    new THREE.BoxGeometry(2.3, 0.9, 0.06),
    new THREE.MeshStandardMaterial({
      color: 0x88b8e8,
      metalness: 0.95,
      roughness: 0.02,
      transparent: true,
      opacity: 0.75,
    }),
  );
  windshield.position.set(0, 1.7, 0.72);
  nose.add(windshield);

  const mta = new THREE.Mesh(
    new THREE.CircleGeometry(0.38, 16),
    new THREE.MeshStandardMaterial({ color: MTA_RED, emissive: MTA_RED, emissiveIntensity: 0.15 }),
  );
  mta.position.set(0, 2.2, 0.55);
  nose.add(mta);

  const mtaText = new THREE.Mesh(
    new THREE.PlaneGeometry(0.3, 0.3),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
  mtaText.position.set(0, 2.2, 0.56);
  nose.add(mtaText);

  nose.position.z = noseZ;
  train.add(nose);

  const wheels: THREE.Mesh[] = [];
  [-1.18, 1.18].forEach((x) => {
    for (let wz = -totalLen / 2 + 1; wz < totalLen / 2; wz += 3.2) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(0.36, 0.36, 0.18, 16),
        new THREE.MeshStandardMaterial({ color: 0x222228, metalness: 0.5 }),
      );
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, 0.36, wz);
      train.add(wheel);
      wheels.push(wheel);
    }
  });

  const headlight = new THREE.SpotLight(0xfff8e0, 0, 40, Math.PI / 6, 0.4);
  headlight.position.set(0, 1.3, noseZ + 1);
  headlight.target.position.set(0, 1, noseZ - 20);
  train.add(headlight);
  train.add(headlight.target);

  return { train, wheels, headlight };
}

function buildSpeedLines(count: number): THREE.Group {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.12,
  });
  for (let i = 0; i < count; i++) {
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.04, 2 + Math.random() * 4), mat);
    line.position.set((Math.random() - 0.5) * 8, TRACK_Y - 1 + Math.random() * 0.5, (Math.random() - 0.5) * 30);
    line.rotation.x = -Math.PI / 2;
    g.add(line);
  }
  return g;
}

export interface ThreeSubwayScene {
  setProgress: (p: number) => void;
  resize: () => void;
  dispose: () => void;
}

export function initThreeFallback(canvas: HTMLCanvasElement): ThreeSubwayScene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const skyTex = makeSkyTexture();
  scene.background = skyTex;
  scene.fog = new THREE.Fog(SKY_HORIZON, 100, 420);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 800);

  scene.add(new THREE.HemisphereLight(0x9ec8f0, 0x606068, 0.85));
  scene.add(new THREE.AmbientLight(0xfff8f0, 0.35));

  const sun = new THREE.DirectionalLight(0xfff8e8, 1.6);
  sun.position.set(60, 90, 40);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.far = 220;
  const sr = 120;
  sun.shadow.camera.left = -sr;
  sun.shadow.camera.right = sr;
  sun.shadow.camera.top = sr;
  sun.shadow.camera.bottom = -sr;
  scene.add(sun);

  buildUrbanBlock(scene, mulberry32(42));

  const viaductMat = new THREE.MeshStandardMaterial({ color: CONCRETE, roughness: 0.82 });
  const viaduct = new THREE.Mesh(
    new THREE.BoxGeometry(7, 1.4, TRACK_HALF * 2 + 50),
    viaductMat,
  );
  viaduct.position.set(0, VIADUCT_Y, 0);
  viaduct.castShadow = true;
  viaduct.receiveShadow = true;
  scene.add(viaduct);

  for (let z = -TRACK_HALF; z <= TRACK_HALF; z += 20) {
    [-3.2, 3.2].forEach((x) => {
      const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.8, VIADUCT_Y, 0.8), viaductMat);
      pillar.position.set(x, VIADUCT_Y / 2, z);
      pillar.castShadow = true;
      scene.add(pillar);
    });
  }

  const railMat = new THREE.MeshStandardMaterial({ color: RAIL, metalness: 0.75, roughness: 0.2 });
  const trackLen = TRACK_HALF * 2 + 50;
  [-1.45, 1.45].forEach((x) => {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.18, trackLen), railMat);
    rail.position.set(x, TRACK_Y + 0.1, 0);
    scene.add(rail);
  });

  const tieMat = new THREE.MeshStandardMaterial({ color: 0x5a4030 });
  for (let z = -TRACK_HALF; z <= TRACK_HALF; z += 1.5) {
    const tie = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.1, 0.35), tieMat);
    tie.position.set(0, TRACK_Y + 0.02, z);
    scene.add(tie);
  }

  const midtownRiders: THREE.Group[] = [];
  const emotionBubbles: EmotionBubble[] = [];
  scene.add(buildStation(S1, 'Departure', { passengers: 8 }));
  scene.add(buildStation(S2, 'Midtown', { passengers: 20, midtownRiders, emotionBubbles }));
  scene.add(buildStation(S3, 'Terminal', { passengers: 12 }));

  const { train, wheels, headlight } = buildSubwayTrain();
  train.position.set(0, TRACK_Y - 0.05, S1 - 2);
  scene.add(train);

  const speedLines = buildSpeedLines(24);
  speedLines.visible = false;
  train.add(speedLines);

  let progress = 0;
  let wheelSpin = 0;
  let animationId = 0;
  let time = 0;
  const camPos = new THREE.Vector3();
  const camLook = new THREE.Vector3();
  const targetPos = new THREE.Vector3();
  const targetLook = new THREE.Vector3();

  function setProgress(p: number): void {
    progress = p;
    train.position.z = trainZ(p);

    const cam = lerpKeyframes(p);
    targetPos.copy(cam.pos);
    targetLook.copy(cam.look);

    const sunsetT = THREE.MathUtils.clamp((p - 0.72) / 0.28, 0, 1);
    sun.color.lerpColors(new THREE.Color(0xfff8e8), new THREE.Color(0xffa050), sunsetT);
    sun.intensity = 1.6 + sunsetT * 0.4;
    scene.fog!.color.lerpColors(new THREE.Color(SKY_HORIZON), new THREE.Color(SKY_SUNSET), sunsetT);
  }

  function resize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  function animate(): void {
    animationId = requestAnimationFrame(animate);
    time += 0.016;
    const moving = isTrainMoving(progress);

    camPos.lerp(targetPos, 0.08);
    camLook.lerp(targetLook, 0.08);
    if (moving) {
      camPos.x += Math.sin(time * 1.2) * 0.04;
      camPos.y += Math.sin(time * 1.8) * 0.02;
    }
    camera.position.copy(camPos);
    camera.lookAt(camLook);

    const speed = moving ? 0.12 + progress * 0.06 : 0.005;
    wheelSpin += speed;
    wheels.forEach((w) => {
      w.rotation.x = wheelSpin;
    });

    if (moving) {
      train.position.y = TRACK_Y - 0.05 + Math.sin(time * 22) * 0.025;
      train.rotation.x = Math.sin(time * 18) * 0.004;
      headlight.intensity = 2.5 + Math.sin(time * 8) * 0.3;
      speedLines.visible = true;
      speedLines.children.forEach((child, i) => {
        child.position.z += 0.8 + progress * 0.5;
        if (child.position.z > 18) child.position.z = -18 - i * 2;
      });
    } else {
      train.position.y = THREE.MathUtils.lerp(train.position.y, TRACK_Y - 0.05, 0.1);
      train.rotation.x *= 0.9;
      headlight.intensity = THREE.MathUtils.lerp(headlight.intensity, 0.4, 0.05);
      speedLines.visible = false;
    }

    const waitPhase = THREE.MathUtils.clamp((progress - 0.48) / 0.06, 0, 1)
      * (1 - THREE.MathUtils.clamp((progress - 0.76) / 0.06, 0, 1));
    if (waitPhase > 0) {
      animateMidtownRiders(midtownRiders, time, waitPhase);
      animateEmotionBubbles(emotionBubbles, time, waitPhase);
    } else {
      emotionBubbles.forEach((b) => {
        b.sprite.visible = false;
        (b.sprite.material as THREE.SpriteMaterial).opacity = 0;
      });
    }

    renderer.render(scene, camera);
  }

  resize();
  setProgress(0);
  camPos.copy(CAMERA_PATH[0].pos);
  camLook.copy(CAMERA_PATH[0].look);
  animate();

  const onResize = () => resize();
  window.addEventListener('resize', onResize);
  canvas.classList.add('is-ready');

  return {
    setProgress,
    resize,
    dispose: () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', onResize);
      skyTex.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          mats.forEach((m) => m.dispose());
        }
        if (obj instanceof THREE.Sprite) {
          obj.material.map?.dispose();
          obj.material.dispose();
        }
      });
    },
  };
}
