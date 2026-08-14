# venule

A browser-based vein-contrast visualizer. Static HTML/CSS/JS, zero dependencies,
zero build step, no backend. All processing happens on-device.

**venule is not a medical device.** It is an educational visualizer. It must not
be used to guide injections, blood draws, or any clinical decision — keep that
framing (start-card small print, landing "Limits" section, footer) intact if you
fork or redistribute.

## Run locally

```
python3 serve.py          # http://localhost:4179
```

Camera capture requires a secure context: `localhost` works, plain `http://` on
a LAN IP does not. For phones, deploy to HTTPS (below) or use `Open a photo`.

## Deploy

Live at **https://jellocello30.github.io/venule/** (GitHub Pages, repo
`JelloCello30/venule`, deploy-from-branch `main` / root). Push to `main` to
redeploy. Pages serves over HTTPS, so phone cameras work out of the box. A
custom domain can be added in the repo's Pages settings.

## How it works

Pipeline per frame, in [js/vision.js](js/vision.js):

1. **Green channel** — hemoglobin absorbs most strongly at 540–580 nm, so
   near-surface veins are a few percent darker in green.
2. **3×3 box denoise** — CLAHE amplifies sensor noise; a light blur first buys
   a lot of clarity for almost no detail.
3. **CLAHE** — 8×N tiles, clip factor 1.2–5.5 (the *Contrast* slider), bilinear
   blending between tile LUTs.
4. **Frangi vesselness** (Veins/Split modes) — Hessian eigenvalues at σ ≈
   {1.5, 3, 5} (scaled to processing width), β = 0.5, structureness scale `c`
   set adaptively per scale from a subsampled Hessian-norm pre-scan. Dark-ridge
   case only (λ₂ > 0).
5. **Composite** — smoothstep-thresholded overlay (the *Sensitivity* slider) in
   the accent color over a desaturated base; display normalization uses a
   decaying running max so it doesn't flicker.

Modes: **Reveal** (default; computational photography: motion-adaptive
temporal frame stacking + soft-clipped unsharp amplification of the stacked
RGB at vein scales, composited as a hard-light detail layer over crisp
native-res video, skin-gated), **Veins** (teal dark-ridge overlay),
**Labels** (adds numbered tags via connected-component analysis of the
thresholded map — hysteresis flood fill, elongated components only, throttled
to every 12th frame live so tags don't jitter), **Structures** (adds the
bright-ridge case of the same Frangi filter in amber — raised, tendon-like
ridges), **Pulse** (remote photoplethysmography: per-pixel temporal band-pass
via dual EMAs at 120px width, energy map overlaid in coral, heart rate from
autocorrelation of the spatial-mean signal — live feed only), **Split**
(original | processed with a draggable divider). Artery *outlines* are
deliberately absent: arteries sit below visible-light penetration depth, so a
camera image contains no artery-shape signal to detect — pulsation is the
part that survives, and Pulse mode shows exactly that.

[js/app.js](js/app.js) handles camera plumbing and adapts processing resolution
(256–512 px wide) to hold ~30 fps on the current machine. Frozen frames and
photos re-render only when a slider changes.

## Verify

Open `test/test.html` on the local server. It draws a synthetic arm (skin-tone
base + mottling + noise + faint vein strokes at known positions), runs the real
pipeline, and reports on-vein vs off-vein response. PASS thresholds are printed
with the numbers.

## Limits (also stated on the landing page)

Visible light scatters within ~1–2 mm of tissue: deep veins are not in the
image and no software can recover them. Contrast falls with darker skin tones,
dim light, and subcutaneous fat. Clinical devices solve this with near-infrared
hardware; venule will happily process an IR-sensitive USB camera's feed via the
same `getUserMedia` path, which improves things considerably.

## License

MIT — see [LICENSE](LICENSE).
