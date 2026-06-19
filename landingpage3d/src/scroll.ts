import Lenis from 'lenis';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { setScrollProgress } from './spline';

gsap.registerPlugin(ScrollTrigger);

export interface ScrollEngine {
  lenis: Lenis | null;
  destroy: () => void;
}

export function initScrollEngine(): ScrollEngine {
  const reducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)',
  ).matches;

  const progressBar = document.getElementById('progress-bar');
  const scrollRoot = document.getElementById('scroll-root');
  const navbar = document.getElementById('navbar');

  if (!scrollRoot) {
    return { lenis: null, destroy: () => {} };
  }

  let lenis: Lenis | null = null;

  if (!reducedMotion) {
    lenis = new Lenis({
      duration: 1.2,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      wheelMultiplier: 1,
      touchMultiplier: 2,
    });

    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => {
      lenis?.raf(time * 1000);
    });
    gsap.ticker.lagSmoothing(0);
  }

  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const href = link.getAttribute('href');
      if (!href) return;
      const target = document.querySelector(href);
      if (!target) return;
      if (lenis) {
        lenis.scrollTo(target as HTMLElement, { offset: 0, duration: 1.5 });
      } else {
        target.scrollIntoView({ behavior: 'auto' });
      }
    });
  });

  ScrollTrigger.create({
    trigger: scrollRoot,
    start: 'top top',
    end: 'bottom bottom',
    scrub: reducedMotion ? false : 0.5,
    onUpdate: (self) => {
      setScrollProgress(self.progress);
      if (progressBar) {
        progressBar.style.width = `${self.progress * 100}%`;
      }
    },
  });

  ScrollTrigger.create({
    trigger: scrollRoot,
    start: 'top top',
    end: '+=200',
    onEnter: () => navbar?.classList.add('is-visible'),
    onLeaveBack: () => navbar?.classList.remove('is-visible'),
  });

  ScrollTrigger.create({
    trigger: scrollRoot,
    start: 'top top',
    end: 'bottom bottom',
    onUpdate: (self) => {
      if (self.progress > 0.02) {
        navbar?.classList.add('is-scrolled');
      } else {
        navbar?.classList.remove('is-scrolled');
      }
    },
  });

  return {
    lenis,
    destroy: () => {
      lenis?.destroy();
      ScrollTrigger.getAll().forEach((t) => t.kill());
    },
  };
}
