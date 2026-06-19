# MTA 3D Landing Page

Scroll-driven 3D landing experience for the **NYC Subway Scheduling Optimization** thesis project. A fixed Spline canvas shows a vintage subway journey while HTML overlays present key messages.

## Quick start

```bash
cd landingpage3d
npm install
npm run dev
```

Open [http://localhost:5174](http://localhost:5174).

## Build for production

```bash
npm run build
npm run preview
```

Output is in `dist/`.

## 3D scene

By default the page renders a **built-in Three.js scene**: vintage subway train, three stations, tracks, passengers, and cost-props — all driven by scroll (camera + train movement).

To replace it with a custom Spline scene, add `public/spline-scene.splinecode` or set `VITE_SPLINE_SCENE_URL`. See **[SPLINE_SCENE.md](./SPLINE_SCENE.md)** for Spline specs.

### Hosted scene URL

```bash
# .env.local
VITE_SPLINE_SCENE_URL=https://prod.spline.design/your-scene/scene.splinecode
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_SPLINE_SCENE_URL` | `/spline-scene.splinecode` | Spline scene URL or local path |
| `VITE_DASHBOARD_URL` | `/optimizer` | Optimizer dashboard (FastAPI route, separate from landing `/`) |

## Architecture

- **Vite** + TypeScript — build tooling
- **@splinetool/runtime** — 3D scene
- **Lenis** + **GSAP ScrollTrigger** — smooth scroll and `scrollProgress` (0→1)
- **HTML/CSS overlays** — typography, stats, hourglass, demo feedback form

## Sections (scroll map)

| Progress | Content |
|----------|---------|
| 0 – 0.25 | Hero — project title & subtitle |
| 0.25 – 0.5 | Operational cost optimization + stat chips |
| 0.5 – 0.75 | Wait time optimization + animated hourglass |
| 0.75 – 1.0 | Journey summary + demo feedback form |

## Feedback form

Display only — clicking **Submit Feedback** shows a toast: *"Demo form — submissions are not collected."*

## Related

- 2D scroll landing: [`../landing/`](../landing/)
- Optimizer dashboard (FastAPI): [`../api.py`](../api.py) → `/optimizer`
