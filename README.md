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

Grounded in published skin optics rather than tuned heuristics.

**Veins** — melanin/hemoglobin separation (Tsumura et al., *Image-based skin
color and texture analysis/synthesis*, SIGGRAPH 2003). In optical-density
space skin colour is a linear mix of melanin, hemoglobin and shading along
three physically distinct absorbance directions, so one 3x3 solve recovers
each independently. Consequences that matter:

* neutral absorbers — hair, shadow, ink, dim light, exposure drift — scale
  R, G and B together and land wholly in the *shading* term
* skin tone lands in the *melanin* term and stops competing with the signal
* what remains in the *hemoglobin* term is blood, i.e. the vein

That hemoglobin map is motion-compensated temporally stacked (night-mode
style), band-passed at vein scales, weighted by Frangi-style tubularity
computed **on the hemoglobin map** (run on brightness, a ridge filter cannot
tell a vein from a hair — that was the old failure), and amplified with a
self-calibrating gain (95th-percentile of blood detail → fixed on-screen
amplitude, so the view neither vanishes nor blows out across skin tones,
lighting and cameras). Composited as a hard-light layer over crisp
native-resolution video.

Measured in `test/test.html`: veins amplify 10x more than hair (hair ends up
quieter than plain skin), vein signal survives 181x stronger than an
illumination gradient, and 99% of it survives on much darker skin.

**Pulse** — fingertip contact PPG: finger over the lens, torch on, green
channel, autocorrelation. This replaced a non-contact (rPPG) attempt that
never locked; contact PPG on a phone validates against ECG at r = 0.997 /
RMSE ~1 bpm, while non-contact needs a face, a tripod and clean light to
reach a fraction of that. Verified at 69.2 bpm against a 69 bpm reference,
and it refuses to report a number when the lens is not covered.

**Compare** — raw | processed with a draggable divider. Press and hold the
image anywhere to peek at the raw feed.

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
