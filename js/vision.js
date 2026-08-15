/* venule — vision pipeline.
 *
 * Plain ES2017, zero dependencies. Everything runs on typed arrays at a
 * reduced processing resolution so it holds real-time rates in plain JS.
 *
 * The vein view (Reveal) is grounded in melanin/hemoglobin separation
 * (Tsumura et al., SIGGRAPH 2003): in optical-density space, skin colour
 * decomposes into melanin, hemoglobin and shading along physically
 * distinct directions, so one 3x3 solve isolates blood from skin tone,
 * shading and neutral absorbers such as hair. That hemoglobin map is
 * temporally stacked (motion-compensated, night-mode style) and its local
 * contrast amplified over crisp native-resolution video.
 *
 * Pulse is fingertip contact PPG (finger on the lens, torch on), the
 * method validated against ECG at r = 0.997 on phones.
 */
(function () {
  'use strict';

  var V = {};

  /* ---------------- buffers ---------------- */

  V.makeBuffers = function (w, h) {
    var n = w * h;
    return {
      w: w, h: h,
      g8: new Uint8ClampedArray(n),   // green channel, denoised (stack aligner)
      f0: new Float32Array(n),        // scratch (blur result)
      f1: new Float32Array(n),        // scratch (blur temp)
      hemo: new Float32Array(n),      // hemoglobin concentration map
      shade: new Float32Array(n),     // neutral/shading term (hair-edge guard)
      vw: new Float32Array(n),        // vessel-shape weight 0..1
      mask: new Float32Array(n),      // soft skin mask, 0..1
      normHist: new Uint32Array(256), // histogram scratch (luminance gate)
      avgR: new Float32Array(n),      // temporal frame stack (noise ↓ √N)
      avgG: new Float32Array(n),
      avgB: new Float32Array(n),
      mean: new Float32Array(n),      // local-mean scratch for Reveal
      ds1: new Float32Array(Math.max(1, w >> 2) * Math.max(1, h >> 2)), // alignment pyramids
      ds2: new Float32Array(Math.max(1, w >> 2) * Math.max(1, h >> 2))
    };
  };

  V.makeState = function () {
    return { stacked: false, motion: 0, gain: 0 };
  };

  /* ---------------- channel extraction + denoise ---------------- */

  // 3x3 box blur on the green channel. CLAHE amplifies sensor noise hard,
  // so a light denoise first costs little detail and buys a lot of clarity.
  function extractGreenDenoised(rgba, bufs) {
    var w = bufs.w, h = bufs.h, f0 = bufs.f0, f1 = bufs.f1, g8 = bufs.g8;
    var i, j, x, y, n = w * h;
    for (i = 0, j = 0; i < n; i++, j += 4) f0[i] = rgba[j + 1];
    for (y = 0; y < h; y++) {
      var r = y * w;
      f1[r] = (f0[r] * 2 + f0[r + 1]) / 3;
      for (x = 1; x < w - 1; x++) f1[r + x] = (f0[r + x - 1] + f0[r + x] + f0[r + x + 1]) / 3;
      f1[r + w - 1] = (f0[r + w - 2] + f0[r + w - 1] * 2) / 3;
    }
    for (x = 0; x < w; x++) g8[x] = (f1[x] * 2 + f1[x + w]) / 3;
    for (y = 1; y < h - 1; y++) {
      var a = (y - 1) * w, b = y * w, c = (y + 1) * w;
      for (x = 0; x < w; x++) g8[b + x] = (f1[a + x] + f1[b + x] + f1[c + x]) / 3;
    }
    var lb = (h - 1) * w, pb = (h - 2) * w;
    for (x = 0; x < w; x++) g8[lb + x] = (f1[pb + x] + f1[lb + x] * 2) / 3;
  }

  /* ---------------- hemoglobin separation ---------------- */

  // Tsumura et al., "Image-based skin color and texture analysis/synthesis
  // by extracting hemoglobin and melanin information in the skin"
  // (SIGGRAPH 2003) — the established basis for pulling blood out of a
  // plain colour photograph.
  //
  // In optical-density space (OD = -log intensity) skin colour is a linear
  // mix of three physically distinct directions:
  //
  //   melanin     absorbs increasingly toward blue
  //   hemoglobin  absorbs strongly in green (540-580 nm)
  //   shading     scales all channels equally, i.e. (1,1,1)
  //
  // Because the directions differ, one 3x3 solve recovers each
  // concentration independently. That is what makes the vein signal
  // separable in principle rather than by tuning:
  //
  //   * shading, hair, ink, dim light and exposure drift are NEUTRAL —
  //     they move along (1,1,1) and land wholly in the shading term
  //   * skin tone lands in the melanin term, so it stops competing with
  //     the blood signal (the reason naive methods fail on darker skin)
  //   * what remains in the hemoglobin term is blood — that is the vein
  var MEL = [0.4143, 0.3570, 0.8372];   // melanin absorbance, RGB
  var HEM = [0.2988, 0.6838, 0.6657];   // hemoglobin absorbance, RGB
  var HEM_ROW = null;   // row of inv([mel hem shade]) yielding hemoglobin
  var SHADE_ROW = null; // …and the row yielding the neutral/shading term

  (function solveRows() {
    // columns: melanin, hemoglobin, shading; rows: R, G, B
    var a = MEL[0], b = HEM[0], c = 1;
    var d = MEL[1], e = HEM[1], f = 1;
    var g = MEL[2], hh = HEM[2], i = 1;
    var det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
    HEM_ROW = [
      -(d * i - f * g) / det,
      (a * i - c * g) / det,
      -(a * f - c * d) / det
    ];
    SHADE_ROW = [
      (d * hh - e * g) / det,
      -(a * hh - b * g) / det,
      (a * e - b * d) / det
    ];
  })();

  // 8-bit code value → optical density. A LUT because a log() per channel
  // per pixel is not affordable at 30 fps in plain JS.
  var OD_LUT = new Float32Array(256);
  for (var odI = 0; odI < 256; odI++) OD_LUT[odI] = -Math.log((odI + 1) / 256);

  // Relative hemoglobin concentration (higher = more blood). Accepts
  // interleaved RGBA bytes, or three Float32 planes so Reveal can run it on
  // the temporally stacked (noise-reduced) frame.
  // `shadeDst` is optional: the neutral term, used to spot hair and shadow
  // edges (see the suppression step in analyzeReveal).
  function hemoglobinMap(rgba, pr, pg, pb, dst, n, shadeDst) {
    var k0 = HEM_ROW[0], k1 = HEM_ROW[1], k2 = HEM_ROW[2];
    var s0 = SHADE_ROW[0], s1 = SHADE_ROW[1], s2 = SHADE_ROW[2];
    var i, j, dR, dG, dB, R, G, B;
    if (rgba) {
      for (i = 0, j = 0; i < n; i++, j += 4) {
        dR = OD_LUT[rgba[j]]; dG = OD_LUT[rgba[j + 1]]; dB = OD_LUT[rgba[j + 2]];
        dst[i] = k0 * dR + k1 * dG + k2 * dB;
        if (shadeDst) shadeDst[i] = s0 * dR + s1 * dG + s2 * dB;
      }
    } else {
      for (i = 0; i < n; i++) {
        R = pr[i]; G = pg[i]; B = pb[i];
        R = R < 0 ? 0 : R > 255 ? 255 : R;
        G = G < 0 ? 0 : G > 255 ? 255 : G;
        B = B < 0 ? 0 : B > 255 ? 255 : B;
        dR = OD_LUT[R | 0]; dG = OD_LUT[G | 0]; dB = OD_LUT[B | 0];
        dst[i] = k0 * dR + k1 * dG + k2 * dB;
        if (shadeDst) shadeDst[i] = s0 * dR + s1 * dG + s2 * dB;
      }
    }
  }
  V.hemoglobinMap = hemoglobinMap; // used by the test harness

  /* ---------------- skin mask ---------------- */

  // Without this, the amplifier happily boosts fabric folds,
  // wood grain and shadow edges across the whole frame. A soft YCbCr skin
  // gate keeps the overlay on skin; the app also uses the returned coverage
  // to suppress the overlay entirely when almost nothing skin-like is
  // visible (pointing the camera at a room).
  function trap(x, a, b, c, d) {
    if (x <= a || x >= d) return 0;
    if (x < b) return (x - a) / (b - a);
    if (x > c) return (d - x) / (d - c);
    return 1;
  }

  // 3x3 box blur, Float32 in/out (softens the mask's edges)
  function box3f(src, dst, tmp, w, h) {
    var x, y;
    for (y = 0; y < h; y++) {
      var r = y * w;
      tmp[r] = (src[r] * 2 + src[r + 1]) / 3;
      for (x = 1; x < w - 1; x++) tmp[r + x] = (src[r + x - 1] + src[r + x] + src[r + x + 1]) / 3;
      tmp[r + w - 1] = (src[r + w - 2] + src[r + w - 1] * 2) / 3;
    }
    for (x = 0; x < w; x++) dst[x] = (tmp[x] * 2 + tmp[x + w]) / 3;
    for (y = 1; y < h - 1; y++) {
      var a = (y - 1) * w, b = y * w, c = (y + 1) * w;
      for (x = 0; x < w; x++) dst[b + x] = (tmp[a + x] + tmp[b + x] + tmp[c + x]) / 3;
    }
    var lb = (h - 1) * w, pb = (h - 2) * w;
    for (x = 0; x < w; x++) dst[lb + x] = (tmp[pb + x] + tmp[lb + x] * 2) / 3;
  }

  // separable 3x3 min filter; out-of-bounds counts as 0 so frame borders
  // erode too
  function erode3(src, dst, tmp, w, h) {
    var x, y;
    for (y = 0; y < h; y++) {
      var r = y * w;
      for (x = 0; x < w; x++) {
        var a = x > 0 ? src[r + x - 1] : 0;
        var b = src[r + x];
        var c = x < w - 1 ? src[r + x + 1] : 0;
        var m = a < b ? a : b;
        tmp[r + x] = c < m ? c : m;
      }
    }
    for (y = 0; y < h; y++) {
      var r2 = y * w, up = y > 0 ? r2 - w : -1, dn = y < h - 1 ? r2 + w : -1;
      for (x = 0; x < w; x++) {
        var a2 = up >= 0 ? tmp[up + x] : 0;
        var b2 = tmp[r2 + x];
        var c2 = dn >= 0 ? tmp[dn + x] : 0;
        var m2 = a2 < b2 ? a2 : b2;
        dst[r2 + x] = c2 < m2 ? c2 : m2;
      }
    }
  }

  // Fills bufs.mask; returns hard-coverage fraction. The mask is the UNION
  // of two independent detectors, because each fails alone: the YCbCr color
  // gate breaks under unusual white balance, and MediaPipe's selfie model —
  // trained on people, not close-ups — often misses a faceless hand or
  // forearm filling the frame. Either one finding skin is enough; the
  // luminance gate and flattenNonSkin clean up whatever the union admits.
  // Uses f0/f1 as scratch — call BEFORE vesselness(), which reuses them.
  function buildSkinMask(rgba, bufs, cat, cw, ch) {
    var w = bufs.w, h = bufs.h, n = w * h, f0 = bufs.f0;
    var i, j, m;
    // 1) color-space score
    for (i = 0, j = 0; i < n; i++, j += 4) {
      var r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      var y = 0.299 * r + 0.587 * g + 0.114 * b;
      var cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      var cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
      // soft trapezoids around the classic YCbCr skin cluster; the Cr low
      // edge sits above warm-tinted shadows on pale fabric (measured ~135
      // on real photos vs ≥150 for lit skin)
      m = trap(cr, 135, 143, 172, 184) * trap(cb, 70, 80, 126, 135);
      if (y < 40) m *= y / 40; // too dark to carry signal either way
      f0[i] = m;
    }
    // 2) union with the neural body-skin/face-skin categories
    // (0=background 1=hair 2=body-skin 3=face-skin 4=clothes 5=others)
    if (cat && cw > 0) {
      if (cw === w && ch === h) {
        for (i = 0; i < n; i++) {
          if ((cat[i] === 2 || cat[i] === 3) && f0[i] < 1) f0[i] = 1;
        }
      } else {
        for (var yy = 0; yy < h; yy++) {
          var srow = ((yy * ch / h) | 0) * cw, drow = yy * w;
          for (var x = 0; x < w; x++) {
            var c = cat[srow + ((x * cw / w) | 0)];
            if ((c === 2 || c === 3) && f0[drow + x] < 1) f0[drow + x] = 1;
          }
        }
      }
    }
    var count = 0;
    for (i = 0; i < n; i++) { if (f0[i] > 0.5) count++; }
    // shrink so the limb's silhouette edge — itself a strong dark ridge —
    // falls outside the mask (iterations scale with resolution), then
    // soften the boundary
    var iters = Math.max(3, Math.round(w / 128));
    var a2 = f0, b2 = bufs.mask;
    for (var e = 0; e < iters; e++) {
      erode3(a2, b2, bufs.f1, w, h);
      var t2 = a2; a2 = b2; b2 = t2;
    }
    if (a2 !== bufs.mask) bufs.mask.set(a2);
    box3f(bufs.mask, bufs.mask, bufs.f1, w, h);
    return count / n;
  }

  // Shadows read as dark tubes too (finger gaps, the crease where an arm
  // meets the sheet), but they are far darker than lit skin while veins
  // are only a few percent darker. Gate the mask by brightness relative
  // to the skin region's median green level.
  // reads luminance straight from RGB: g8 now carries the opponent map
  function luminanceGateFromRGB(rgba, bufs, loF, hiF) {
    var n = bufs.w * bufs.h, mask = bufs.mask;
    var hist = bufs.normHist;
    hist.fill(0);
    var count = 0, i, j;
    for (i = 0, j = 0; i < n; i++, j += 4) {
      if (mask[i] > 0.5) {
        var L = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) | 0;
        hist[L > 255 ? 255 : L]++;
        count++;
      }
    }
    if (count < 100) return;
    var half = count / 2, acc = 0, med = 128;
    for (i = 0; i < 256; i++) { acc += hist[i]; if (acc >= half) { med = i; break; } }
    // veins sit at ~85-95% of the skin median; rim shading at a limb's
    // silhouette sits well below — gate between them
    var lo = med * loF, hi = med * hiF;
    var invSpan = 1 / Math.max(hi - lo, 1);
    for (i = 0, j = 0; i < n; i++, j += 4) {
      if (mask[i] <= 0) continue;
      var g = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
      if (g <= lo) mask[i] = 0;
      else if (g < hi) mask[i] *= (g - lo) * invSpan;
    }
  }

  /* ---------------- vessel shape weighting ---------------- */

  // Frangi-style tubularity, computed on the HEMOGLOBIN map rather than on
  // brightness. This is the whole difference from the detector that was
  // removed: run on luminance, a ridge filter cannot tell a vein from a
  // hair or a wrinkle; run on hemoglobin, hair and shading are already
  // gone, so what is left to be tube-shaped is blood. Used as a soft
  // weight on the amplifier — never as a "this is a vessel" claim — so
  // blood that is merely blotchy still shows, just less loudly than blood
  // arranged in a line.
  //
  // Veins are HIGH hemoglobin, i.e. bright ridges, so the sign convention
  // is the opposite of the classical dark-vessel formulation.
  function vesselWeight(hemo, dst, tmp, tmp2, w, h) {
    var n = w * h, i;
    var scales = [Math.max(1, Math.round(w / 260)), Math.max(2, Math.round(w / 130))];
    for (i = 0; i < n; i++) dst[i] = 0;
    var vmax = 0;
    for (var s = 0; s < scales.length; s++) {
      var rad = scales[s];
      boxBlurRunning(hemo, tmp, tmp2, w, h, rad);
      boxBlurRunning(tmp, tmp, tmp2, w, h, rad);   // 2 boxes ≈ Gaussian
      var s2 = rad * rad;
      for (var y = 1; y < h - 1; y++) {
        var row = y * w;
        for (var x = 1; x < w - 1; x++) {
          var k = row + x, c = tmp[k];
          var hxx = (tmp[k - 1] - 2 * c + tmp[k + 1]) * s2;
          var hyy = (tmp[k - w] - 2 * c + tmp[k + w]) * s2;
          var hxy = (tmp[k + w + 1] + tmp[k - w - 1] - tmp[k - w + 1] - tmp[k + w - 1]) * 0.25 * s2;
          var half = (hxx - hyy) * 0.5;
          var disc = Math.sqrt(half * half + hxy * hxy);
          var mn = (hxx + hyy) * 0.5;
          var l1 = mn + disc, l2 = mn - disc;
          var lamA, lamB;
          if ((l1 < 0 ? -l1 : l1) >= (l2 < 0 ? -l2 : l2)) { lamB = l1; lamA = l2; }
          else { lamB = l2; lamA = l1; }
          if (lamB >= 0) continue;               // bright ridge only
          var rb = lamA / lamB;
          var ss = lamA * lamA + lamB * lamB;
          var vv = Math.exp(-rb * rb * 2) * ss;  // β = 0.5; structureness raw
          if (vv > dst[k]) {
            dst[k] = vv;
            if (vv > vmax) vmax = vv;
          }
        }
      }
    }
    // normalize to 0..1 with a soft knee so weak-but-real vessels survive
    var inv = vmax > 1e-20 ? 1 / vmax : 0;
    for (i = 0; i < n; i++) {
      var t = dst[i] * inv;
      // sharp knee: diffuse mottling scores low here and stays quiet,
      // line-shaped blood saturates toward 1
      dst[i] = t <= 0 ? 0 : t / (t + 0.05);
    }
  }

  /* ---------------- Reveal: stack + amplify ---------------- */

  // O(n) box blur via running sums (radius-independent cost)
  function boxBlurRunning(src, dst, tmp, w, h, r) {
    var x, y, i, acc;
    var normH = 1 / (2 * r + 1);
    for (y = 0; y < h; y++) {
      var row = y * w;
      acc = src[row] * (r + 1);
      for (x = 1; x <= r; x++) acc += src[row + (x < w ? x : w - 1)];
      for (x = 0; x < w; x++) {
        var add = x + r + 1; if (add >= w) add = w - 1;
        var sub = x - r; if (sub < 0) sub = 0;
        tmp[row + x] = acc * normH;
        acc += src[row + add] - src[row + sub];
      }
    }
    for (x = 0; x < w; x++) {
      acc = tmp[x] * (r + 1);
      for (y = 1; y <= r; y++) acc += tmp[(y < h ? y : h - 1) * w + x];
      for (y = 0; y < h; y++) {
        var addY = y + r + 1; if (addY >= h) addY = h - 1;
        var subY = y - r; if (subY < 0) subY = 0;
        dst[y * w + x] = acc * normH;
        acc += tmp[addY * w + x] - tmp[subY * w + x];
      }
    }
  }

  // 4x downsample (block mean) for the coarse alignment pass
  function downsample4(getPx, w, h, out, sw, sh) {
    for (var y = 0; y < sh; y++) {
      var y0 = y * 4;
      for (var x = 0; x < sw; x++) {
        var x0 = x * 4, acc = 0, cnt = 0;
        for (var yy = y0; yy < y0 + 4 && yy < h; yy++) {
          for (var xx = x0; xx < x0 + 4 && xx < w; xx++) { acc += getPx(yy * w + xx); cnt++; }
        }
        out[y * sw + x] = acc / cnt;
      }
    }
  }

  // dst = src translated by (dx, dy) with edge clamping
  function copyShift(src, dst, w, h, dx, dy) {
    for (var y = 0; y < h; y++) {
      var sy = y + dy;
      if (sy < 0) sy = 0; else if (sy >= h) sy = h - 1;
      var srow = sy * w, drow = y * w;
      for (var x = 0; x < w; x++) {
        var sx = x + dx;
        if (sx < 0) sx = 0; else if (sx >= w) sx = w - 1;
        dst[drow + x] = src[srow + sx];
      }
    }
  }

  // Temporal stacking with global motion compensation. Handheld phone +
  // living subject means pure EMA either smears or never accumulates; a
  // coarse-to-fine translation search aligns the stack to each new frame
  // (real night modes do the same, plus rotation we can live without).
  // Exposure is normalized first so auto-exposure breathing doesn't read
  // as motion and flush the stack.
  function updateStack(rgba, bufs, state, still) {
    var w = bufs.w, h = bufs.h, n = w * h;
    var aR = bufs.avgR, aG = bufs.avgG, aB = bufs.avgB, g8 = bufs.g8;
    var i, j, x, y;
    if (still) {
      // a frozen frame KEEPS the stack it accumulated live — re-copying
      // would throw away exactly the noise reduction the user froze to study
      if (!state.stacked) {
        for (i = 0, j = 0; i < n; i++, j += 4) {
          aR[i] = rgba[j]; aG[i] = rgba[j + 1]; aB[i] = rgba[j + 2];
        }
        state.stacked = true;
      }
      state.motion = 0;
      return;
    }
    if (!state.stacked) {
      for (i = 0, j = 0; i < n; i++, j += 4) {
        aR[i] = rgba[j]; aG[i] = rgba[j + 1]; aB[i] = rgba[j + 2];
      }
      state.stacked = true;
      state.motion = 0;
      return;
    }

    var sw = Math.max(1, w >> 2), sh = Math.max(1, h >> 2);
    var dsS = bufs.ds1, dsF = bufs.ds2;
    downsample4(function (p) { return aG[p]; }, w, h, dsS, sw, sh);
    downsample4(function (p) { return g8[p]; }, w, h, dsF, sw, sh);

    // exposure ratio stack/frame, clamped — applied to the incoming frame
    var mS = 0, mF = 0, sn = sw * sh;
    for (i = 0; i < sn; i++) { mS += dsS[i]; mF += dsF[i]; }
    var expo = mF > 1 ? mS / mF : 1;
    if (expo < 0.85) expo = 0.85; else if (expo > 1.18) expo = 1.18;
    for (i = 0; i < sn; i++) dsF[i] *= expo;

    // coarse translation search at 1/4 resolution (±4 cells = ±16 px)
    var R = 4, best = Infinity, bdx = 0, bdy = 0;
    for (var dy = -R; dy <= R; dy++) {
      for (var dx = -R; dx <= R; dx++) {
        var sad = 0, cnt = 0;
        for (y = R; y < sh - R; y += 2) {
          var fr = y * sw, sr = (y + dy) * sw + dx;
          for (x = R; x < sw - R; x += 2) {
            var d = dsF[fr + x] - dsS[sr + x];
            sad += d < 0 ? -d : d;
            cnt++;
          }
        }
        sad /= cnt;
        if (sad < best) { best = sad; bdx = dx; bdy = dy; }
      }
    }

    // refine at full resolution around the coarse winner (±3 px)
    var cx = bdx * 4, cy = bdy * 4;
    var bestF = Infinity, fdx = cx, fdy = cy;
    var M = 20; // stay away from clamped borders
    for (var ry = cy - 3; ry <= cy + 3; ry++) {
      for (var rx = cx - 3; rx <= cx + 3; rx++) {
        var sadF = 0, cntF = 0;
        for (y = M; y < h - M; y += 3) {
          var frow = y * w, srow = (y + ry) * w + rx;
          for (x = M; x < w - M; x += 3) {
            var dF = g8[frow + x] * expo - aG[srow + x];
            sadF += dF < 0 ? -dF : dF;
            cntF++;
          }
        }
        sadF /= cntF;
        if (sadF < bestF) { bestF = sadF; fdx = rx; fdy = ry; }
      }
    }
    state.motion = bestF; // residual AFTER alignment — true scene change

    if (bestF > 26) {
      // scene replaced (new arm position, camera swung) — restart cleanly
      for (i = 0, j = 0; i < n; i++, j += 4) {
        aR[i] = rgba[j]; aG[i] = rgba[j + 1]; aB[i] = rgba[j + 2];
      }
      return;
    }

    var alpha = bestF > 10 ? 0.5 : bestF > 4 ? 0.22 : 0.08;
    var chans = [aR, aG, aB];
    for (var c = 0; c < 3; c++) {
      var av = chans[c];
      if (fdx !== 0 || fdy !== 0) {
        copyShift(av, bufs.mean, w, h, fdx, fdy);
        for (i = 0, j = c; i < n; i++, j += 4) {
          av[i] = bufs.mean[i] + alpha * (rgba[j] * expo - bufs.mean[i]);
        }
      } else {
        for (i = 0, j = c; i < n; i++, j += 4) {
          av[i] += alpha * (rgba[j] * expo - av[i]);
        }
      }
    }
  }

  // Reveal writes a hard-light detail layer: 128 = neutral (no change to
  // the crisp base video), above/below = amplified local color deviation.
  // Veins pop as themselves — darker, greener — rather than as painted
  // annotations, and the layer upscales cleanly over native-res video.
  V.analyzeReveal = function (rgba, outRGBA, bufs, state, params) {
    var w = bufs.w, h = bufs.h, n = w * h;

    extractGreenDenoised(rgba, bufs); // g8 = green, used by the stack aligner
    updateStack(rgba, bufs, state, params.still);
    var skinFrac = buildSkinMask(rgba, bufs, params.catMask, params.catW || 0, params.catH || 0);
    luminanceGateFromRGB(rgba, bufs, 0.40, 0.65);

    var cov = (skinFrac - 0.03) / 0.05;
    if (cov < 0) cov = 0; else if (cov > 1) cov = 1;
    var mask = bufs.mask, mean = bufs.mean;
    var hemo = bufs.hemo;

    // Hemoglobin concentration from the STACKED (noise-reduced) frame.
    // Shading, hair and skin tone have already been separated out by the
    // 3x3 solve, so what is amplified below is blood and nothing else.
    var shade = bufs.shade;
    hemoglobinMap(null, bufs.avgR, bufs.avgG, bufs.avgB, hemo, n, shade);

    // Neutral-edge suppression. A hair or a shadow edge is a sharp step in
    // the SHADING term; sensor noise and demosaicing let a little of any
    // sharp step bleed into the other terms. Where the neutral channel has
    // a strong local step, trust the hemoglobin detail less. Veins are
    // invisible in the neutral channel, so this costs them nothing.
    boxBlurRunning(shade, mean, bufs.f1, w, h, Math.max(2, Math.round(4 * w / 384)));
    for (var q = 0; q < n; q++) {
      var sd = shade[q] - mean[q];
      if (sd < 0) sd = -sd;
      shade[q] = 1 / (1 + sd * 26);   // 1 = clean skin, →0 at a hard edge
    }

    // Two passes of fine smoothing: pores and single-pixel grain are
    // ~1px, veins are 3-10px. Killing the former is what stops real skin
    // texture from being amplified alongside blood.
    box3f(hemo, bufs.f0, bufs.f1, w, h);
    box3f(bufs.f0, bufs.f0, bufs.f1, w, h);
    var r = Math.max(4, Math.round(10 * w / 384));
    boxBlurRunning(hemo, mean, bufs.f1, w, h, r);

    // Self-calibrating gain. Hemoglobin contrast varies enormously with
    // skin tone, lighting and camera, so a fixed multiplier either does
    // nothing or blows the image out (it blew it out). Instead measure the
    // 95th percentile of blood detail actually present on skin and scale
    // so that maps to a fixed on-screen amplitude — the display always
    // uses its full range, whatever the input contrast.
    var det = bufs.f0, i, j;
    for (i = 0; i < n; i++) det[i] -= mean[i];

    // Vessel-shape weighting FIRST, so the gain below calibrates on the
    // signal that is actually displayed. (Calibrating before weighting
    // silently blanked images whose vessel weights were mostly small.)
    var vw = bufs.vw;
    vesselWeight(hemo, vw, bufs.f1, mean, w, h);
    for (i = 0; i < n; i++) {
      det[i] *= mask[i] * shade[i] * (0.12 + 0.88 * vw[i] * vw[i]);
    }

    var dmax = 0;
    for (i = 0; i < n; i++) {
      if (mask[i] > 0.5) {
        var ad = det[i] < 0 ? -det[i] : det[i];
        if (ad > dmax) dmax = ad;
      }
    }
    var p95 = dmax;
    if (dmax > 1e-9) {
      var hist = bufs.normHist;
      hist.fill(0);
      var sc = 255 / dmax, cnt = 0;
      for (i = 0; i < n; i++) {
        if (mask[i] > 0.5) {
          var b2 = (det[i] < 0 ? -det[i] : det[i]) * sc | 0;
          hist[b2 > 255 ? 255 : b2]++;
          cnt++;
        }
      }
      if (cnt > 200) {
        var target = cnt * 0.05, acc = 0, bin = 255;
        for (; bin > 0; bin--) { acc += hist[bin]; if (acc >= target) break; }
        p95 = Math.max(bin / 255 * dmax, dmax * 0.04);
      }
    }
    var want = 34 + params.strength * 62;       // target 8-bit amplitude
    var gain = want / Math.max(p95, 1e-6);
    // temporal smoothing so live video doesn't pump frame to frame
    state.gain = (params.still || !state.gain) ? gain : state.gain * 0.85 + gain * 0.15;
    gain = state.gain * cov;

    for (i = 0, j = 0; i < n; i++, j += 4) {
      var d = det[i] * gain;
      // soft knee instead of a hard clamp: keeps midtone boost, avoids the
      // clipped "metallic etching" look
      d = d / (1 + (d < 0 ? -d : d) * 0.02);
      // more blood than surrounding skin → darker and cooler, which is how
      // a vein actually looks; hard-light keeps non-skin untouched at 128
      var v = 128 - d;
      outRGBA[j] = v;
      outRGBA[j + 1] = v - d * 0.10;
      outRGBA[j + 2] = v - d * 0.28;
      outRGBA[j + 3] = 255;
    }
    return { skinFrac: skinFrac, motion: state.motion };
  };

  /* ---------------- contact PPG (Pulse) ---------------- */

  // Fingertip photoplethysmography: finger over the lens, torch on.
  //
  // This replaces an earlier non-contact (rPPG) attempt that never locked
  // in practice. The literature is unambiguous about why: contact PPG on a
  // phone validates against ECG at r = 0.997 / RMSE ≈ 1 bpm, because the
  // finger is pressed against the sensor, the torch supplies constant
  // illumination, and the whole frame is one signal. Non-contact rPPG needs
  // a face, a tripod and clean light to reach a fraction of that. When a
  // reliable method exists, use it.
  //
  // Green channel: hemoglobin absorbs 540-580 nm most strongly, so green
  // carries the largest pulsatile modulation.
  var PPG_CAP = 900;   // ~30 s at 30 fps

  V.makePulseState = function () {
    return {
      raw: new Float32Array(PPG_CAP),
      t: new Float32Array(PPG_CAP),
      n: 0,
      dc: 0,           // slow baseline (finger pressure, drift)
      ac: 0,           // smoothed |AC| — signal strength
      covered: false,
      lastT: 0
    };
  };

  // One frame → one sample. Returns the current frame's mean green and a
  // coverage verdict so the UI can coach the user.
  V.pulseSample = function (st, rgba, n, tMs) {
    var sumG = 0, sumR = 0, i, j;
    // stride 4 pixels: a covered lens is spatially uniform, so a sparse
    // mean is as good as a full one and much cheaper
    var count = 0;
    for (i = 0, j = 0; i < n; i += 4, j += 16) {
      sumR += rgba[j];
      sumG += rgba[j + 1];
      count++;
    }
    var meanG = sumG / count, meanR = sumR / count;

    // A finger on the lens with the torch on is dark-red and saturated in
    // red: red high, green low. That signature is what "covered" means.
    var covered = meanR > 45 && meanG < meanR * 0.62;
    st.covered = covered;

    if (!st.lastT || tMs - st.lastT > 700 || !covered) {
      // (re)start: seed the baseline AT the current level. Letting it ramp
      // up from zero injects a long decaying transient that swamps the
      // pulse and makes autocorrelation lock onto the shortest lag.
      st.n = 0;
      st.dc = meanG;
      st.ac = 0;
      st.lastT = tMs;
      return { meanG: meanG, covered: covered, strength: 0 };
    }
    var dt = Math.min(0.2, Math.max(0.005, (tMs - st.lastT) / 1000));
    st.lastT = tMs;

    // remove the slow baseline; what is left is the pulse waveform
    var aDC = 1 - Math.exp(-dt / 1.5);
    st.dc += aDC * (meanG - st.dc);
    var acVal = st.dc - meanG;          // inverted: more blood = less light
    var aAC = 1 - Math.exp(-dt / 1.0);
    st.ac += aAC * (Math.abs(acVal) - st.ac);

    var k = st.n % PPG_CAP;
    st.raw[k] = acVal;
    st.t[k] = tMs / 1000;
    st.n++;
    return { meanG: meanG, covered: true, strength: st.ac };
  };

  // Autocorrelation over the recent window → bpm + confidence.
  V.pulseBpm = function (st) {
    var have = Math.min(st.n, PPG_CAP);
    if (have < 64) return { bpm: 0, conf: 0, secs: 0 };
    var m = have, i;
    var s = new Float32Array(m);
    var start = st.n - m;
    for (i = 0; i < m; i++) s[i] = st.raw[(start + i) % PPG_CAP];
    var t0 = st.t[start % PPG_CAP], t1 = st.t[(st.n - 1) % PPG_CAP];
    var span = t1 - t0;
    if (span < 2.5) return { bpm: 0, conf: 0, secs: span };
    var dt = span / (m - 1);

    var mean = 0;
    for (i = 0; i < m; i++) mean += s[i];
    mean /= m;
    var norm = 0;
    for (i = 0; i < m; i++) { s[i] -= mean; norm += s[i] * s[i]; }
    if (norm < 1e-9) return { bpm: 0, conf: 0, secs: span };

    var lagMin = Math.max(2, Math.round(0.33 / dt));   // 180 bpm
    var lagMax = Math.min(m - 8, Math.round(1.5 / dt)); // 40 bpm
    var bestLag = 0, bestR = 0;
    for (var lag = lagMin; lag <= lagMax; lag++) {
      var acc = 0;
      for (i = 0; i + lag < m; i++) acc += s[i] * s[i + lag];
      var r = acc / norm;
      if (r > bestR) { bestR = r; bestLag = lag; }
    }
    if (!bestLag) return { bpm: 0, conf: 0, secs: span };
    // parabolic interpolation around the peak for sub-sample precision
    return { bpm: 60 / (bestLag * dt), conf: bestR, secs: span };
  };

  // Waveform for the on-screen trace: newest `count` samples, normalized
  // to -1..1. Seeing your own pulse beat is the proof it is working.
  V.pulseWave = function (st, count) {
    var have = Math.min(st.n, PPG_CAP, count);
    var out = new Float32Array(have);
    if (!have) return out;
    var start = st.n - have, i, mx = 1e-6;
    for (i = 0; i < have; i++) {
      var v = st.raw[(start + i) % PPG_CAP];
      out[i] = v;
      if (Math.abs(v) > mx) mx = Math.abs(v);
    }
    for (i = 0; i < have; i++) out[i] /= mx;
    return out;
  };

  window.VenuleVision = V;
})();
