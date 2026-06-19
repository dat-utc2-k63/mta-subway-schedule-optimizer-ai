import { initScrollEngine } from './scroll';
import { destroySpline, initSpline } from './spline';
import {
  initSectionReveals,
  initFormDemo,
  initDashboardLink,
} from './sections';

async function main(): Promise<void> {
  initDashboardLink();

  const canvas = document.getElementById('spline-canvas') as HTMLCanvasElement | null;
  if (canvas) {
    await initSpline(canvas);
  }

  initScrollEngine();
  initSectionReveals();
  initFormDemo();

  window.addEventListener('beforeunload', () => destroySpline());
}

main().catch(console.error);
