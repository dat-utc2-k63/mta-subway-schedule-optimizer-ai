import { Application } from '@splinetool/runtime';

export type ScrollProgressCallback = (progress: number) => void;

const DEFAULT_SCENE = '/spline-scene.splinecode';

let app: Application | null = null;
let threeScene: { setProgress: (p: number) => void; resize: () => void; dispose: () => void } | null = null;
let ready = false;
let usingFallback = false;
const progressListeners = new Set<ScrollProgressCallback>();

export function onScrollProgress(cb: ScrollProgressCallback): () => void {
  progressListeners.add(cb);
  return () => progressListeners.delete(cb);
}

export function setScrollProgress(progress: number): void {
  const clamped = Math.max(0, Math.min(1, progress));
  if (app && ready) {
    try {
      app.setVariable('scrollProgress', clamped);
    } catch {
      // Scene may use a different variable name — see SPLINE_SCENE.md
    }
    try {
      app.emitEvent('mouseHover', 'ScrollController');
    } catch {
      // Optional event target in Spline scene
    }
  }
  if (threeScene) {
    threeScene.setProgress(clamped);
  }
  progressListeners.forEach((cb) => cb(clamped));
}

export async function initSpline(canvas: HTMLCanvasElement): Promise<boolean> {
  const statusEl = document.getElementById('spline-status');
  const placeholder = document.getElementById('spline-placeholder');

  const sceneUrl =
    import.meta.env.VITE_SPLINE_SCENE_URL?.trim() || DEFAULT_SCENE;

  const useSplineOnly = import.meta.env.VITE_FORCE_SPLINE === 'true';

  if (!useSplineOnly) {
    try {
      const head = await fetch(sceneUrl, { method: 'HEAD' });
      if (!head.ok) throw new Error('Scene file not found');
    } catch {
      return await initFallback(canvas, placeholder, statusEl);
    }
  }

  try {
    if (statusEl) statusEl.textContent = 'Loading 3D scene…';

    app = new Application(canvas);
    await app.load(sceneUrl);

    ready = true;
    canvas.classList.add('is-ready');

    if (placeholder) {
      placeholder.classList.add('is-hidden');
    }

    return true;
  } catch (err) {
    console.warn('[spline] Scene not loaded — using Three.js fallback.', err);
    return await initFallback(canvas, placeholder, statusEl);
  }
}

function initFallback(
  canvas: HTMLCanvasElement,
  placeholder: HTMLElement | null,
  statusEl: HTMLElement | null,
): Promise<boolean> {
  usingFallback = true;
  ready = true;

  if (placeholder) placeholder.classList.add('is-hidden');
  if (statusEl) statusEl.textContent = 'Loading subway scene…';

  return import('./scene-three').then(({ initThreeFallback }) => {
    threeScene = initThreeFallback(canvas);
    if (statusEl) statusEl.textContent = '';
    return false;
  });
}

export function destroySpline(): void {
  threeScene?.dispose();
  threeScene = null;
  if (app) {
    app.dispose();
    app = null;
  }
  ready = false;
  usingFallback = false;
}

export function isUsingFallback(): boolean {
  return usingFallback;
}
