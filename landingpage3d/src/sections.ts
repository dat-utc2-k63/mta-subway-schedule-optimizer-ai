import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export function initSectionReveals(): void {
  const reducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)',
  ).matches;

  document.querySelectorAll('[data-reveal]').forEach((panel) => {
    const splits = panel.querySelectorAll('[data-split] .line-inner');
    const fades = panel.querySelectorAll('[data-fade]');

    if (reducedMotion) {
      splits.forEach((el) => gsap.set(el, { y: 0 }));
      fades.forEach((el) => gsap.set(el, { opacity: 1, y: 0 }));
      panel.classList.add('is-visible');
      return;
    }

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: panel.closest('.section') ?? panel,
        start: 'top 75%',
        end: 'top 25%',
        toggleActions: 'play none none reverse',
      },
    });

    if (splits.length) {
      tl.to(splits, {
        y: 0,
        duration: 0.9,
        stagger: 0.08,
        ease: 'power3.out',
      });
    }

    if (fades.length) {
      tl.to(
        fades,
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          stagger: 0.1,
          ease: 'power2.out',
        },
        splits.length ? '-=0.5' : 0,
      );
    }

    ScrollTrigger.create({
      trigger: panel.closest('.section') ?? panel,
      start: 'top 80%',
      onEnter: () => panel.classList.add('is-visible'),
      onLeaveBack: () => panel.classList.remove('is-visible'),
    });
  });
}

export function initFormDemo(): void {
  const submitBtn = document.getElementById('form-submit-demo');
  const toast = document.getElementById('form-toast');

  submitBtn?.addEventListener('click', () => {
    if (toast) {
      toast.hidden = false;
      window.setTimeout(() => {
        toast.hidden = true;
      }, 4000);
    }
  });
}

export function initDashboardLink(): void {
  const cta = document.getElementById('cta-dashboard') as HTMLAnchorElement | null;
  if (!cta) return;

  const dashboardUrl = import.meta.env.VITE_DASHBOARD_URL?.trim() || './optimizer.html';
  cta.href = dashboardUrl;
  if (dashboardUrl.startsWith('http')) {
    cta.target = '_blank';
    cta.rel = 'noopener noreferrer';
  } else {
    cta.removeAttribute('target');
    cta.removeAttribute('rel');
  }
}
