/* ============================================================
   main.js — Awwwards-level scroll-driven landing page engine
   
   Architecture:
   1. Lenis smooth scroll → GSAP ScrollTrigger bridge
   2. Dynamic SVG track with draw-in reveal
   3. Train on MotionPath with elastic lag
   4. Staggered text line-reveal (mask slide-up)
   5. Station pulse micro-interactions
   ============================================================ */

(function () {
  'use strict';

  gsap.registerPlugin(ScrollTrigger, MotionPathPlugin);

  /* ─── DOM refs ─── */
  const wrapper     = document.getElementById('page-wrapper');
  const trackSVG    = document.getElementById('track-svg');
  const trackPath   = document.getElementById('track-path');
  const trackGhost  = document.getElementById('track-ghost');
  const trackBed    = document.getElementById('track-bed');
  const trackCenter = document.getElementById('track-center');
  const trackLeft   = document.getElementById('track-rail-left');
  const trackRight  = document.getElementById('track-rail-right');
  const tiesGroup   = document.getElementById('ties-group');
  const stationDots = document.getElementById('station-dots');
  const train       = document.getElementById('train');
  const navbar      = document.getElementById('navbar');
  const progressBar = document.getElementById('progress-bar');

  /* ─── Config ─── */
  const STATION_FRACTIONS = [0.02, 0.22, 0.44, 0.66, 0.90];
  const TRACK_LEAD = 0.08; // draw-in leads train by this fraction

  /* ─── State ─── */
  let pathLength = 0;
  let stationElements = []; // SVG groups for each station
  let activeStation = -1;

  /* ============================================================
     1. LENIS SMOOTH SCROLL
     ============================================================ */
  function initLenis() {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      wheelMultiplier: 1,
      touchMultiplier: 2,
    });

    // Bridge Lenis → GSAP
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => {
      lenis.raf(time * 1000);
    });
    gsap.ticker.lagSmoothing(0);

    // Handle anchor clicks with Lenis
    document.querySelectorAll('a[href^="#"]').forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) lenis.scrollTo(target, { offset: 0, duration: 1.5 });
      });
    });

    return lenis;
  }

  /* ============================================================
     2. BUILD DYNAMIC S-CURVE TRACK
     ============================================================ */
  function buildTrack() {
    const W = wrapper.offsetWidth;
    const H = wrapper.offsetHeight;

    trackSVG.setAttribute('viewBox', `0 0 ${W} ${H}`);

    const cx = W / 2;
    const amplitude = Math.min(W * 0.22, 300);
    const sectionH = H / 5;
    const startY = sectionH * 0.5;
    const endY = H - sectionH * 0.15;
    const seg = (endY - startY) / 4;
    const cp = seg * 0.55;

    const d = [
      `M ${cx} ${startY}`,
      `C ${cx} ${startY + cp}, ${cx + amplitude} ${startY + seg - cp}, ${cx + amplitude} ${startY + seg}`,
      `C ${cx + amplitude} ${startY + seg + cp}, ${cx - amplitude} ${startY + 2*seg - cp}, ${cx - amplitude} ${startY + 2*seg}`,
      `C ${cx - amplitude} ${startY + 2*seg + cp}, ${cx + amplitude} ${startY + 3*seg - cp}, ${cx + amplitude} ${startY + 3*seg}`,
      `C ${cx + amplitude} ${startY + 3*seg + cp}, ${cx} ${endY - cp}, ${cx} ${endY}`,
    ].join(' ');

    // Set all path copies
    trackPath.setAttribute('d', d);
    trackGhost.setAttribute('d', d);
    trackBed.setAttribute('d', d);
    trackCenter.setAttribute('d', d);

    pathLength = trackPath.getTotalLength();

    // Build parallel rails
    buildRails();

    // Setup draw-in dasharray for bed, center, and rails
    setupDrawIn(trackBed, pathLength);
    setupDrawIn(trackCenter, pathLength);
    setupDrawIn(trackLeft, pathLength);
    setupDrawIn(trackRight, pathLength);

    return d;
  }

  function setupDrawIn(el, len) {
    const totalLen = el.getTotalLength ? el.getTotalLength() : len;
    el.style.strokeDasharray = totalLen;
    el.style.strokeDashoffset = totalLen;
  }

  /* ─── Parallel rails ─── */
  function buildRails() {
    const gauge = 11;
    const step = 3;
    let leftD = '', rightD = '';

    for (let i = 0; i <= pathLength; i += step) {
      const pt  = trackPath.getPointAtLength(i);
      const pt2 = trackPath.getPointAtLength(Math.min(i + 1, pathLength));
      const angle = Math.atan2(pt2.y - pt.y, pt2.x - pt.x);
      const perp = angle + Math.PI / 2;

      const cmd = i === 0 ? 'M' : 'L';
      leftD  += `${cmd}${(pt.x + Math.cos(perp) * gauge).toFixed(1)} ${(pt.y + Math.sin(perp) * gauge).toFixed(1)} `;
      rightD += `${cmd}${(pt.x - Math.cos(perp) * gauge).toFixed(1)} ${(pt.y - Math.sin(perp) * gauge).toFixed(1)} `;
    }

    trackLeft.setAttribute('d', leftD);
    trackRight.setAttribute('d', rightD);
  }

  /* ─── Railroad ties ─── */
  function drawTies() {
    tiesGroup.innerHTML = '';
    const spacing = 26;
    const halfLen = 16;

    for (let i = 0; i < pathLength; i += spacing) {
      const pt  = trackPath.getPointAtLength(i);
      const pt2 = trackPath.getPointAtLength(Math.min(i + 1, pathLength));
      const angle = Math.atan2(pt2.y - pt.y, pt2.x - pt.x);
      const perp = angle + Math.PI / 2;

      const tie = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tie.setAttribute('x1', pt.x + Math.cos(perp) * halfLen);
      tie.setAttribute('y1', pt.y + Math.sin(perp) * halfLen);
      tie.setAttribute('x2', pt.x - Math.cos(perp) * halfLen);
      tie.setAttribute('y2', pt.y - Math.sin(perp) * halfLen);
      tie.setAttribute('stroke', '#d5cec3');
      tie.setAttribute('stroke-width', '3');
      tie.setAttribute('stroke-linecap', 'round');
      tie.setAttribute('opacity', '0');          // start invisible
      tie.dataset.dist = i;                      // for draw-in
      tiesGroup.appendChild(tie);
    }
  }

  /* ─── Station dots ─── */
  function drawStationDots() {
    stationDots.innerHTML = '';
    stationElements = [];

    STATION_FRACTIONS.forEach((frac, i) => {
      const pt = trackPath.getPointAtLength(frac * pathLength);
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('data-station-index', i);

      // Pulse ring (animated via JS when train arrives)
      const pulse = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      pulse.setAttribute('cx', pt.x);
      pulse.setAttribute('cy', pt.y);
      pulse.setAttribute('r', '14');
      pulse.setAttribute('fill', 'none');
      pulse.setAttribute('stroke', '#c0392b');
      pulse.setAttribute('stroke-width', '2');
      pulse.setAttribute('opacity', '0');
      pulse.classList.add('station-pulse');
      g.appendChild(pulse);

      // Outer ring
      const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      outer.setAttribute('cx', pt.x);
      outer.setAttribute('cy', pt.y);
      outer.setAttribute('r', '14');
      outer.setAttribute('fill', 'none');
      outer.setAttribute('stroke', '#c0392b');
      outer.setAttribute('stroke-width', '1.5');
      outer.setAttribute('opacity', '0.15');
      g.appendChild(outer);

      // Fill circle
      const fill = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      fill.setAttribute('cx', pt.x);
      fill.setAttribute('cy', pt.y);
      fill.setAttribute('r', '8');
      fill.setAttribute('fill', '#f6f3ee');
      fill.setAttribute('stroke', '#c0392b');
      fill.setAttribute('stroke-width', '2');
      fill.setAttribute('opacity', '0.5');
      g.appendChild(fill);

      // Center dot
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', pt.x);
      dot.setAttribute('cy', pt.y);
      dot.setAttribute('r', '3.5');
      dot.setAttribute('fill', '#c0392b');
      dot.setAttribute('opacity', '0.6');
      g.appendChild(dot);

      stationDots.appendChild(g);
      stationElements.push({ g, pulse, outer, fill, dot, frac, pt });
    });
  }

  /* ============================================================
     3. TRACK DRAW-IN + TRAIN MOTION (scroll-driven)
     ============================================================ */
  function setupScrollEngine() {
    const trainW = train.offsetWidth;
    const trainH = train.offsetHeight;
    const ties = tiesGroup.querySelectorAll('line');

    // Place train at start
    const startPt = trackPath.getPointAtLength(0);
    train.style.left = (startPt.x - trainW / 2) + 'px';
    train.style.top  = (startPt.y - trainH / 2) + 'px';
    requestAnimationFrame(() => train.classList.add('is-ready'));

    // Proxy for smooth train (the train "lags" behind the draw-in)
    const trainProxy = { progress: 0 };
    const drawProxy  = { progress: 0 };

    // Draw-in animation (slightly ahead of train)
    gsap.to(drawProxy, {
      progress: 1,
      ease: 'none',
      scrollTrigger: {
        trigger: wrapper,
        start: 'top top',
        end: 'bottom bottom',
        scrub: 0.3,   // faster scrub — draw-in leads
      },
      onUpdate: () => {
        const revealProgress = Math.min(drawProxy.progress + TRACK_LEAD, 1);
        const revealDist = revealProgress * pathLength;

        // Draw-in: reduce dashoffset
        const bedLen = trackBed.getTotalLength ? trackBed.getTotalLength() : pathLength;
        trackBed.style.strokeDashoffset = bedLen * (1 - revealProgress);

        const centerLen = trackCenter.getTotalLength();
        trackCenter.style.strokeDashoffset = centerLen * (1 - revealProgress);

        const leftLen = trackLeft.getTotalLength();
        trackLeft.style.strokeDashoffset = leftLen * (1 - revealProgress);

        const rightLen = trackRight.getTotalLength();
        trackRight.style.strokeDashoffset = rightLen * (1 - revealProgress);

        // Reveal ties up to draw-in point
        ties.forEach((tie) => {
          const dist = parseFloat(tie.dataset.dist);
          if (dist <= revealDist) {
            tie.setAttribute('opacity', '0.3');
          } else {
            tie.setAttribute('opacity', '0');
          }
        });
      },
    });

    // Train motion — with elastic lag (higher scrub = more inertia)
    gsap.to(trainProxy, {
      progress: 1,
      ease: 'none',
      scrollTrigger: {
        trigger: wrapper,
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1.8,    // heavier scrub = physical momentum/lag
      },
      onUpdate: () => {
        const p = trainProxy.progress;
        const d = p * pathLength;
        const pt = trackPath.getPointAtLength(d);

        // Tangent for rotation
        const nextD = Math.min(d + 4, pathLength);
        const pt2 = trackPath.getPointAtLength(nextD);
        const angle = Math.atan2(pt2.y - pt.y, pt2.x - pt.x) * (180 / Math.PI);

        train.style.left = (pt.x - trainW / 2) + 'px';
        train.style.top  = (pt.y - trainH / 2) + 'px';
        train.style.transform = `rotate(${angle}deg)`;

        // Progress bar
        progressBar.style.width = (p * 100) + '%';

        // Station proximity check → pulse
        checkStationProximity(p);
      },
    });
  }

  /* ============================================================
     4. STATION PULSE MICRO-INTERACTIONS
     ============================================================ */
  function checkStationProximity(trainProgress) {
    const threshold = 0.025;

    stationElements.forEach((station, i) => {
      const dist = Math.abs(trainProgress - station.frac);

      if (dist < threshold && activeStation !== i) {
        activeStation = i;
        triggerStationPulse(station);
      }

      // Visual: increase opacity when train is near
      const proximity = Math.max(0, 1 - dist * 8);
      station.fill.setAttribute('opacity', 0.5 + proximity * 0.5);
      station.dot.setAttribute('opacity', 0.6 + proximity * 0.4);
      station.outer.setAttribute('opacity', 0.15 + proximity * 0.35);

      // Scale the station when close
      if (proximity > 0.3) {
        const s = 1 + proximity * 0.3;
        station.g.setAttribute('transform', `translate(${station.pt.x * (1 - s)} ${station.pt.y * (1 - s)}) scale(${s})`);
      } else {
        station.g.removeAttribute('transform');
      }
    });
  }

  function triggerStationPulse(station) {
    const { pulse, pt } = station;

    // Create a new pulse ring each time
    const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    ring.setAttribute('cx', pt.x);
    ring.setAttribute('cy', pt.y);
    ring.setAttribute('r', '14');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', '#c0392b');
    ring.setAttribute('stroke-width', '2');
    ring.setAttribute('opacity', '0.6');
    station.g.insertBefore(ring, station.g.firstChild);

    gsap.to(ring, {
      attr: { r: 35, 'stroke-width': 0.5 },
      opacity: 0,
      duration: 1.2,
      ease: 'power2.out',
      onComplete: () => ring.remove(),
    });

    // Second delayed ring
    setTimeout(() => {
      const ring2 = ring.cloneNode();
      ring2.setAttribute('r', '14');
      ring2.setAttribute('opacity', '0.4');
      station.g.insertBefore(ring2, station.g.firstChild);
      gsap.to(ring2, {
        attr: { r: 28, 'stroke-width': 0.5 },
        opacity: 0,
        duration: 1,
        ease: 'power2.out',
        onComplete: () => ring2.remove(),
      });
    }, 200);
  }

  /* ============================================================
     5. TEXT REVEAL — Staggered line mask slide-up
     ============================================================ */
  function setupTextReveals() {
    // --- Hero entrance (immediate, not scroll-driven) ---
    const heroContent = document.querySelector('[data-reveal="hero"]');
    if (heroContent) {
      const heroLines = heroContent.querySelectorAll('.line-inner');
      const heroFades = heroContent.querySelectorAll('[data-fade]');

      // Staggered line reveal
      gsap.to(heroLines, {
        y: '0%',
        duration: 1.2,
        stagger: 0.12,
        ease: 'power4.out',
        delay: 0.5,
      });

      // Fade elements
      gsap.to(heroFades, {
        opacity: 1,
        y: 0,
        duration: 1,
        stagger: 0.15,
        ease: 'power3.out',
        delay: 1.1,
      });
    }

    // --- Scroll-driven reveals for other sections ---
    document.querySelectorAll('[data-reveal]:not([data-reveal="hero"])').forEach((content) => {
      const lines = content.querySelectorAll('.line-inner');
      const fades = content.querySelectorAll('[data-fade]');
      const section = content.closest('.section');

      // Line mask reveals
      if (lines.length) {
        ScrollTrigger.create({
          trigger: section,
          start: 'top 70%',
          onEnter: () => {
            gsap.to(lines, {
              y: '0%',
              duration: 1,
              stagger: 0.1,
              ease: 'power4.out',
            });
          },
          onLeaveBack: () => {
            gsap.set(lines, { y: '110%' });
          },
        });
      }

      // Fade reveals (text paragraphs, illustrations, etc)
      if (fades.length) {
        ScrollTrigger.create({
          trigger: section,
          start: 'top 65%',
          onEnter: () => {
            gsap.to(fades, {
              opacity: 1,
              y: 0,
              duration: 0.9,
              stagger: 0.12,
              ease: 'power3.out',
            });
          },
          onLeaveBack: () => {
            gsap.set(fades, { opacity: 0, y: 24 });
          },
        });
      }
    });
  }

  /* ============================================================
     6. STAGGERED CARD & COUNTER ANIMATIONS
     ============================================================ */
  function setupCards() {
    gsap.utils.toArray('[data-stagger]').forEach((el, i) => {
      gsap.from(el, {
        opacity: 0, y: 30, duration: 0.7,
        delay: (i % 3) * 0.12,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 88%',
          toggleActions: 'play none none reverse',
        },
      });
    });
  }

  function setupCounters() {
    document.querySelectorAll('[data-counter]').forEach((el) => {
      const target = parseFloat(el.dataset.counter);
      const prefix = el.dataset.prefix || '';
      const suffix = el.dataset.suffix || '';
      const dec    = parseInt(el.dataset.decimals || '0', 10);
      const obj    = { val: 0 };

      gsap.to(obj, {
        val: target,
        duration: 1.5,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 88%',
          toggleActions: 'play none none reverse',
        },
        onUpdate: () => {
          el.textContent = prefix + (dec > 0 ? obj.val.toFixed(dec) : Math.round(obj.val)) + suffix;
        },
      });
    });
  }

  /* ============================================================
     7. NAVBAR
     ============================================================ */
  function setupNavbar() {
    // Show navbar after hero entrance
    gsap.to(navbar, {
      opacity: 1,
      y: 0,
      duration: 0.8,
      delay: 1.8,
      ease: 'power3.out',
      onComplete: () => navbar.classList.add('is-visible'),
    });

    // Frosted glass on scroll
    ScrollTrigger.create({
      trigger: 'body',
      start: '100px top',
      onEnter:     () => navbar.classList.add('scrolled'),
      onLeaveBack: () => navbar.classList.remove('scrolled'),
    });
  }

  /* ============================================================
     8. HERO PARALLAX
     ============================================================ */
  function setupHeroParallax() {
    const heroMap = document.querySelector('.hero-map-svg');
    if (heroMap) {
      gsap.to(heroMap, {
        y: 100,
        ease: 'none',
        scrollTrigger: {
          trigger: '.section--hero',
          start: 'top top',
          end: 'bottom top',
          scrub: true,
        },
      });
    }
  }

  /* ============================================================
     RESIZE HANDLER
     ============================================================ */
  let resizeRAF;
  function handleResize() {
    cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(() => {
      buildTrack();
      drawTies();
      drawStationDots();
      ScrollTrigger.refresh();
    });
  }

  /* ============================================================
     INIT
     ============================================================ */
  function init() {
    // 1. Smooth scroll
    initLenis();

    // 2. Build track geometry
    buildTrack();
    drawTies();
    drawStationDots();

    // 3. Scroll engine (track draw-in + train)
    setupScrollEngine();

    // 4. Content reveals
    setupTextReveals();
    setupCards();
    setupCounters();

    // 5. UI
    setupNavbar();
    setupHeroParallax();

    // 6. Resize
    window.addEventListener('resize', handleResize);
  }

  // Wait for full layout
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => requestAnimationFrame(init));
  } else {
    requestAnimationFrame(init);
  }
})();
