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

Two pipelines share the buffers, in [js/vision.js](js/vision.js):

**Reveal** (default) — no detection at all. Motion-compensated temporal
stacking (see below) then local-contrast amplification of the stacked RGB,
composited over crisp native-res video as a hard-light detail layer.

**Trace** — detection on the spectral ratio map (see "How Trace avoids
hair"), skin-gated by a union of a MediaPipe `selfie_multiclass` category
mask (in a worker) and a YCbCr colour gate, then CLAHE, then multi-scale
Frangi vesselness (σ ≈ {1.4, 2.6, 4.2} scaled to processing width, β = 0.5,
dark-ridge case only), 99th-percentile normalization, smoothstep threshold,
and hysteresis connected components for the numbered tags.

Modes: **Reveal** (default; computational photography: motion-compensated
temporal frame stacking — coarse-to-fine global translation alignment plus
exposure normalization, so handheld shake still stacks — then soft-clipped
unsharp amplification of the stacked RGB at vein scales, composited as a
hard-light detail layer over crisp native-res video, skin-gated; press and
hold the stage to peek at the raw feed), **Trace** (detection + numbered
tags in one view), **Pulse** (rPPG: per-pixel temporal band-pass via dual
EMAs at 120px width, energy map in coral, heart rate from autocorrelation —
live feed only), **Split** (raw | processed, draggable divider).

**How Trace avoids hair.** The detector input is the spectral ratio
B/((R+G)/2), not luminance. Neutral absorbers — hair, shadows, creases, ink,
dim light — scale R, G and B together, which leaves that ratio unchanged, so
they cannot produce a ridge. A vein can't be neutral: blue light doesn't
reach vein depth, so it dims R and G alone and the ratio rises. A gate keeps
only pixels deviating vein-ward from their local surroundings, then Frangi
vesselness traces the surviving dark ridges. Measured 588x more overlay on
veins than on hair strands in `test/test.html`, which includes hair strands
as explicit false-positive controls. Artery *outlines* are absent by
design: arteries sit below visible-light penetration depth, so the image
contains no artery-shape signal — pulsation is the part that survives, and
Pulse mode shows exactly that.

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
