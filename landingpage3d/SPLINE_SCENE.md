# Spline Scene Specification — MTA 3D Landing

This document describes how to build the 3D scene in [Spline](https://spline.design) so it syncs with scroll via the `scrollProgress` variable (0 → 1).

## Export

1. Build the scene in Spline Editor following the specs below.
2. **Export → Web** → download `.splinecode`.
3. Place the file at `public/spline-scene.splinecode`.
4. Alternatively set `VITE_SPLINE_SCENE_URL` to a hosted Spline URL.

## Scene Variable (required)

Create a **Number** variable in Spline:

| Name | Type | Range | Driven by |
|------|------|-------|-----------|
| `scrollProgress` | Number | 0 – 1 | Web runtime (`app.setVariable`) |

Bind all camera and train animations to this single variable using **States** or **Events → On Variable Change**.

## Color Palette

| Element | Hex | Notes |
|---------|-----|-------|
| Station walls | `#8a8278` | Warm gray stone |
| Floor tiles | `#6b6560` | Muted brown-gray |
| Train body | `#303030` | Dark charcoal |
| Warm accent light | `#f0c040` | Low intensity point/area lights |
| Platform edge | `#b8af9f` | Safety stripe |
| Ambient | `#4a4743` | Warm gray fill |

## Assets

| Asset | Count | Style |
|-------|-------|-------|
| Classic NYC subway car | 1 | Low-poly, 3–4 windows per side |
| Station module | 3 | Arched ceiling, vintage tile columns |
| Track segment | 1 | Curved path between stations |
| Passenger figures | 8–12 | Low-poly, muted clothing |
| Savings props (optional) | 2–3 | Coin stack, minimal bar chart |
| Vintage clock (optional) | 1 | Platform wall decoration |

Free sources: Sketchfab (CC0 subway models), Spline community library, or block-out primitives in Spline.

## Camera Keyframes (`scrollProgress`)

| Progress | Section | Camera | Train |
|----------|---------|--------|-------|
| **0.00** | Hero — Departure | Isometric ~35°, warm lit Station 1 | Parked at Station 1 |
| **0.25** | Transition | Begin rotate toward oblique top-down | Starts rolling |
| **0.50** | Cost optimization | Angled top-down (~55°), follow train | Cruising mid-track |
| **0.65** | Approach Station 2 | Dolly toward platform, zoom in | Decelerating |
| **0.75** | Wait time | Close-up waiting hall, passengers visible | Slow stop at Station 2 |
| **0.90** | Approach Terminal | Pull back wide | Accelerate lightly |
| **1.00** | Terminal | Static panoramic overview, Station 3 | Stopped at final platform |

### Spline setup tips

1. Create a **Camera** object with 6–8 **States** keyed to `scrollProgress` thresholds.
2. Use **Follow Path** on the train along a spline curve matching the track.
3. Map train path position: `0` at 0.0, `0.35` at 0.5, `0.7` at 0.75, `1.0` at 1.0.
4. Passenger group: opacity 0 below 0.45; idle bob animation active 0.5–0.8.
5. Cost props (coins/charts): rise from platform 0.25–0.5.

## Lighting

- 1 **Directional Light** — warm white, soft shadows, angle from upper-left.
- 2 **Area Lights** at Station 1 — color `#f0c040`, intensity low.
- **Ambient** — `#6b6560` at ~0.4 intensity for vintage mood.

## Optional Event Target

If using Events instead of (or alongside) variables, create an empty object named `ScrollController` and listen for runtime events. The bridge in `src/spline.ts` emits `mouseHover` on this target as a hook.

## Performance

- Keep total poly count under ~80k for mobile.
- Bake lighting where possible.
- Export with **Compress** enabled in Spline.
- Test on 1080p and 768px widths.

## Verification

After export:

```bash
cd landingpage3d
npm run dev
```

Scroll the page — camera and train should move smoothly with page scroll. If nothing moves, confirm the variable is named exactly `scrollProgress` (case-sensitive).
