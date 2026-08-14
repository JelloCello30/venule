/* venule — vision pipeline.
 *
 * Plain ES2017, zero dependencies. Everything runs on typed arrays at a
 * reduced processing resolution so it holds real-time rates in plain JS.
 *
 * Pipeline: green channel → light denoise → CLAHE → (veins/structures modes)
 * multi-scale Frangi vesselness → composite. Green is used because hemoglobin
 * absorbs most strongly around 540–580 nm, so near-surface veins are a few
 * percent darker there than surrounding skin. Dark tubular structures read as
 * vein-like; bright tubular structures read as raised, tendon-like ridges.
 */
(function () {
  'use strict';

  var V = {};

  /* ---------------- buffers ---------------- */

  V.makeBuffers = function (w, h) {
    var n = w * h;
    return {
      w: w, h: h,
      g8: new Uint8ClampedArray(n),   // green channel, denoised
      eq: new Uint8ClampedArray(n),   // CLAHE output
      f0: new Float32Array(n),        // scratch (blur result)
      f1: new Float32Array(n),        // scratch (blur temp)
      vess: new Float32Array(n),      // dark-ridge vesselness (veins)
      vessS: new Float32Array(n),     // …temporally smoothed
      alpha: new Float32Array(n),     // …overlay opacity after threshold
      vessB: new Float32Array(n),     // bright-ridge response (tendon-like)
      vessBS: new Float32Array(n),
      alphaB: new Float32Array(n),
      seen: new Uint8Array(n),        // component labeling scratch
      stack: new Int32Array(n),
      pix: new Int32Array(n),         // pixels of the component being traced
      mask: new Float32Array(n),      // soft skin mask, 0..1
      normHist: new Uint32Array(256), // percentile-normalization scratch
      hist: null, lut: null           // sized lazily by clahe()
    };
  };

  V.makeState = function () {
    return { norm: 0, normB: 0 };
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

  /* ---------------- skin mask ---------------- */

  // Without this, the ridge filter happily highlights hair, fabric folds,
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

  // Fills bufs.mask; returns hard-coverage fraction (mask > 0.5).
  // Uses f0/f1 as scratch — call BEFORE vesselness(), which reuses them.
  function skinMask(rgba, bufs) {
    var w = bufs.w, h = bufs.h, n = w * h, f0 = bufs.f0;
    var count = 0;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      var y = 0.299 * r + 0.587 * g + 0.114 * b;
      var cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
      var cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
      // soft trapezoids around the classic YCbCr skin cluster; the Cr low
      // edge sits above warm-tinted shadows on pale fabric (measured ~135
      // on real photos vs ≥150 for lit skin)
      var m = trap(cr, 135, 143, 172, 184) * trap(cb, 70, 80, 126, 135);
      if (y < 40) m *= y / 40; // too dark to carry signal either way
      f0[i] = m;
      if (m > 0.5) count++;
    }
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
  function luminanceGate(bufs) {
    var n = bufs.w * bufs.h, g8 = bufs.g8, mask = bufs.mask;
    var hist = bufs.normHist;
    hist.fill(0);
    var count = 0, i;
    for (i = 0; i < n; i++) {
      if (mask[i] > 0.5) { hist[g8[i]]++; count++; }
    }
    if (count < 100) return;
    var half = count / 2, acc = 0, med = 128;
    for (i = 0; i < 256; i++) { acc += hist[i]; if (acc >= half) { med = i; break; } }
    // veins sit at ~85-95% of the skin median; rim shading at a limb's
    // silhouette sits well below — gate between them
    var lo = med * 0.45, hi = med * 0.72;
    var invSpan = 1 / Math.max(hi - lo, 1);
    for (i = 0; i < n; i++) {
      if (mask[i] <= 0) continue;
      var g = g8[i];
      if (g <= lo) mask[i] = 0;
      else if (g < hi) mask[i] *= (g - lo) * invSpan;
    }
  }

  // The decisive anti-silhouette step: rewrite everything outside the mask
  // to the skin's median tone BEFORE the ridge filter runs. A boundary the
  // filter never sees cannot fire — this is what keeps arm outlines, sheet
  // folds and finger-gap shadows out of the overlay even though the
  // filter's support is wider than any practical mask erosion.
  function flattenNonSkin(bufs) {
    var n = bufs.w * bufs.h, eq = bufs.eq, mask = bufs.mask;
    var hist = bufs.normHist;
    hist.fill(0);
    var count = 0, i;
    for (i = 0; i < n; i++) {
      if (mask[i] > 0.5) { hist[eq[i]]++; count++; }
    }
    var med = 128;
    if (count >= 100) {
      var half = count / 2, acc = 0;
      for (i = 0; i < 256; i++) { acc += hist[i]; if (acc >= half) { med = i; break; } }
    }
    for (i = 0; i < n; i++) {
      var m = mask[i];
      if (m < 1) eq[i] = med + (eq[i] - med) * m;
    }
  }

  /* ---------------- CLAHE ---------------- */

  var TILES_X = 8;

  function clahe(bufs, clipFactor) {
    var w = bufs.w, h = bufs.h, src = bufs.g8, dst = bufs.eq;
    var tilesX = TILES_X;
    var tilesY = Math.max(2, Math.round(tilesX * h / w));
    var tileW = Math.ceil(w / tilesX), tileH = Math.ceil(h / tilesY);
    var nTiles = tilesX * tilesY;

    if (!bufs.hist || bufs.hist.length !== nTiles * 256) {
      bufs.hist = new Uint32Array(nTiles * 256);
      bufs.lut = new Uint8Array(nTiles * 256);
    }
    var hist = bufs.hist, lut = bufs.lut;
    hist.fill(0);

    var x, y, t, i;
    for (y = 0; y < h; y++) {
      var ty = Math.min(tilesY - 1, (y / tileH) | 0);
      var rowBase = ty * tilesX, row = y * w;
      for (x = 0; x < w; x++) {
        var tx = Math.min(tilesX - 1, (x / tileW) | 0);
        hist[(rowBase + tx) * 256 + src[row + x]]++;
      }
    }

    for (t = 0; t < nTiles; t++) {
      var base = t * 256, total = 0;
      for (i = 0; i < 256; i++) total += hist[base + i];
      if (total === 0) continue;
      var clipLimit = Math.max(1, (clipFactor * total / 256) | 0);
      var excess = 0;
      for (i = 0; i < 256; i++) {
        var v = hist[base + i];
        if (v > clipLimit) { excess += v - clipLimit; hist[base + i] = clipLimit; }
      }
      var bonus = (excess / 256) | 0, rem = excess - bonus * 256;
      if (bonus > 0) for (i = 0; i < 256; i++) hist[base + i] += bonus;
      for (i = 0; i < rem; i++) hist[base + i]++;
      var c = 0, scale = 255 / total;
      for (i = 0; i < 256; i++) {
        c += hist[base + i];
        var m = (c * scale + 0.5) | 0;
        lut[base + i] = m > 255 ? 255 : m;
      }
    }

    // bilinear blend between the four surrounding tile mappings
    for (y = 0; y < h; y++) {
      var gy = y / tileH - 0.5;
      var ty0 = Math.floor(gy), fy = gy - ty0, ty1 = ty0 + 1;
      if (ty0 < 0) { ty0 = 0; ty1 = 0; fy = 0; }
      if (ty1 >= tilesY) { ty1 = tilesY - 1; if (ty0 > ty1) ty0 = ty1; if (ty0 === ty1) fy = 0; }
      var rA = ty0 * tilesX, rB = ty1 * tilesX, row2 = y * w;
      for (x = 0; x < w; x++) {
        var gx = x / tileW - 0.5;
        var tx0 = Math.floor(gx), fx = gx - tx0, tx1 = tx0 + 1;
        if (tx0 < 0) { tx0 = 0; tx1 = 0; fx = 0; }
        if (tx1 >= tilesX) { tx1 = tilesX - 1; if (tx0 > tx1) tx0 = tx1; if (tx0 === tx1) fx = 0; }
        var vv = src[row2 + x];
        var a2 = lut[(rA + tx0) * 256 + vv], b2 = lut[(rA + tx1) * 256 + vv];
        var c2 = lut[(rB + tx0) * 256 + vv], d2 = lut[(rB + tx1) * 256 + vv];
        dst[row2 + x] = (a2 * (1 - fx) + b2 * fx) * (1 - fy) + (c2 * (1 - fx) + d2 * fx) * fy;
      }
    }
  }

  /* ---------------- Frangi vesselness ---------------- */

  function gaussianKernel(sigma) {
    var radius = Math.max(1, Math.round(sigma * 2.5));
    var k = new Float32Array(radius * 2 + 1), sum = 0, i;
    for (i = -radius; i <= radius; i++) {
      var v = Math.exp(-(i * i) / (2 * sigma * sigma));
      k[i + radius] = v; sum += v;
    }
    for (i = 0; i < k.length; i++) k[i] /= sum;
    return { k: k, radius: radius };
  }

  var kernelCache = {};
  function getKernel(sigma) {
    var key = sigma.toFixed(3);
    if (!kernelCache[key]) kernelCache[key] = gaussianKernel(sigma);
    return kernelCache[key];
  }

  // src: Uint8 or Float32; dst, tmp: Float32. dst === src is safe.
  function blurSeparable(src, dst, tmp, w, h, kern) {
    var k = kern.k, radius = kern.radius, taps = k.length;
    var x, y, i, acc, xx;
    // horizontal → tmp: clamped edges, branch-free interior
    for (y = 0; y < h; y++) {
      var row = y * w, xEnd = w - radius;
      for (x = 0; x < w && x < radius; x++) {
        acc = 0;
        for (i = -radius; i <= radius; i++) {
          xx = x + i;
          if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
          acc += src[row + xx] * k[i + radius];
        }
        tmp[row + x] = acc;
      }
      for (x = radius; x < xEnd; x++) {
        acc = 0;
        var base = row + x - radius;
        for (i = 0; i < taps; i++) acc += src[base + i] * k[i];
        tmp[row + x] = acc;
      }
      for (x = xEnd > radius ? xEnd : radius; x < w; x++) {
        acc = 0;
        for (i = -radius; i <= radius; i++) {
          xx = x + i;
          if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
          acc += src[row + xx] * k[i + radius];
        }
        tmp[row + x] = acc;
      }
    }
    // vertical → dst: accumulate whole rows so reads stay sequential
    for (y = 0; y < h; y++) {
      var drow = y * w;
      for (x = 0; x < w; x++) dst[drow + x] = 0;
      for (i = -radius; i <= radius; i++) {
        var yy = y + i;
        if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        var srow = yy * w, kv = k[i + radius];
        for (x = 0; x < w; x++) dst[drow + x] += tmp[srow + x] * kv;
      }
    }
  }

  // Frangi 1998, beta = 0.5, structureness scale c set adaptively per scale
  // from a subsampled pre-scan of the Hessian norm. Dark ridges (λ2 > 0) are
  // vein-like; bright ridges (λ2 < 0) are raised, tendon-like structures.
  var INV_TWO_BETA2 = 2.0; // 1 / (2 * 0.5^2)

  function vesselness(bufs, sigmas, wantBright) {
    var w = bufs.w, h = bufs.h, out = bufs.vess, outB = bufs.vessB;
    var blurred = bufs.f0, tmp = bufs.f1;
    out.fill(0);
    if (wantBright) outB.fill(0);
    var maxD = 0, maxB = 0;
    var prevSigma = 0;

    for (var s = 0; s < sigmas.length; s++) {
      var sigma = sigmas[s];
      // incremental blurring: sigma_total^2 = sigma_prev^2 + sigma_step^2
      var step = prevSigma === 0 ? sigma : Math.sqrt(sigma * sigma - prevSigma * prevSigma);
      blurSeparable(prevSigma === 0 ? bufs.eq : blurred, blurred, tmp, w, h, getKernel(step));
      prevSigma = sigma;

      var s2 = sigma * sigma;
      var x, y, i2;

      // pre-scan (every 3rd pixel) for the structureness scale c
      var maxS2 = 1e-6;
      for (y = 1; y < h - 1; y += 3) {
        for (x = 1; x < w - 1; x += 3) {
          i2 = y * w + x;
          var cc = blurred[i2];
          var xx2 = (blurred[i2 - 1] - 2 * cc + blurred[i2 + 1]) * s2;
          var yy2 = (blurred[i2 - w] - 2 * cc + blurred[i2 + w]) * s2;
          var xy2 = (blurred[i2 + w + 1] + blurred[i2 - w - 1] - blurred[i2 - w + 1] - blurred[i2 + w - 1]) * 0.25 * s2;
          var m2 = xx2 * xx2 + yy2 * yy2 + 2 * xy2 * xy2;
          if (m2 > maxS2) maxS2 = m2;
        }
      }
      var invTwoC2 = 2 / maxS2; // c = 0.5 * sqrt(maxS2)

      for (y = 1; y < h - 1; y++) {
        var row = y * w;
        for (x = 1; x < w - 1; x++) {
          var i = row + x;
          var c = blurred[i];
          var hxx = (blurred[i - 1] - 2 * c + blurred[i + 1]) * s2;
          var hyy = (blurred[i - w] - 2 * c + blurred[i + w]) * s2;
          var hxy = (blurred[i + w + 1] + blurred[i - w - 1] - blurred[i - w + 1] - blurred[i + w - 1]) * 0.25 * s2;
          var half = (hxx - hyy) * 0.5;
          var disc = Math.sqrt(half * half + hxy * hxy);
          var mean = (hxx + hyy) * 0.5;
          var l1 = mean + disc, l2 = mean - disc;
          var lamA, lamB; // |lamA| <= |lamB|
          if ((l1 < 0 ? -l1 : l1) >= (l2 < 0 ? -l2 : l2)) { lamB = l1; lamA = l2; }
          else { lamB = l2; lamA = l1; }
          if (lamB === 0) continue;
          var rb = lamA / lamB;
          var ss = lamA * lamA + lamB * lamB;
          var vv = Math.exp(-rb * rb * INV_TWO_BETA2) * (1 - Math.exp(-ss * invTwoC2));
          if (lamB > 0) {
            if (vv > out[i]) { out[i] = vv; if (vv > maxD) maxD = vv; }
          } else if (wantBright) {
            if (vv > outB[i]) { outB[i] = vv; if (vv > maxB) maxB = vv; }
          }
        }
      }
    }
    return { maxD: maxD, maxB: maxB };
  }

  // 99th percentile of the response inside the skin mask. Normalizing by
  // the frame max is fragile — one strong edge (a sleeve, a table) crushes
  // every vein below threshold. The percentile tracks vein cores instead.
  function percentileNorm(vals, mask, n, frameMax, hist) {
    if (frameMax <= 1e-9) return 1e-4;
    hist.fill(0);
    var count = 0, s = 255 / frameMax;
    for (var i = 0; i < n; i++) {
      if (mask[i] > 0.5 && vals[i] > 0) {
        var b = (vals[i] * s) | 0;
        hist[b > 255 ? 255 : b]++;
        count++;
      }
    }
    if (count < 200) return frameMax;
    var target = count * 0.01, acc = 0, bin = 255;
    for (; bin > 0; bin--) { acc += hist[bin]; if (acc >= target) break; }
    return Math.max(bin / 255 * frameMax, frameMax * 0.05, 1e-4);
  }

  // EMA + normalize + smoothstep threshold, gated by the skin mask and the
  // global coverage scale → overlay opacity in [0, 0.92]
  function thresholdPass(raw, smoothed, alpha, mask, n, norm, sensitivity, keep, covScale) {
    var t = 0.65 * (1 - sensitivity) + 0.06;
    var invSpan = 1 / Math.max(1 - t, 1e-3);
    var invMax = 1 / Math.max(norm, 1e-6);
    var gain = 0.92 * covScale;
    for (var i = 0; i < n; i++) {
      var sm = smoothed[i] * keep + raw[i] * (1 - keep);
      smoothed[i] = sm;
      var a = (sm * invMax - t) * invSpan;
      if (a <= 0) { alpha[i] = 0; continue; }
      if (a > 1) a = 1;
      a = a * a * (3 - 2 * a); // smoothstep: soft toe and shoulder
      alpha[i] = a * gain * mask[i];
    }
  }

  /* ---------------- connected components (Labels mode) ---------------- */

  // Hysteresis flood fill over the vein opacity map: components seed at
  // strong pixels and grow through weak-but-connected ones (8-connected),
  // so a vein whose response dips locally still labels as one structure.
  // Returns the largest elongated components: {x, y, area, len} in
  // processing-resolution coordinates.
  function findComponents(bufs) {
    var w = bufs.w, h = bufs.h, alpha = bufs.alpha;
    var seen = bufs.seen, stack = bufs.stack;
    seen.fill(0);
    var HI = 0.28, LO = 0.09;
    var scale = w / 384;
    var minArea = 100 * scale * scale, minLen = 36 * scale;
    var comps = [];
    var n = w * h;
    var sp = 0;
    function push(idx) {
      if (!seen[idx] && alpha[idx] > LO) { seen[idx] = 1; stack[sp++] = idx; }
    }
    for (var i0 = 0; i0 < n; i0++) {
      if (seen[i0] || alpha[i0] <= HI) continue;
      sp = 0;
      stack[sp++] = i0; seen[i0] = 1;
      var area = 0, sx = 0, sy = 0;
      var minX = w, maxX = 0, minY = h, maxY = 0;
      while (sp > 0) {
        var j = stack[--sp];
        var x = j % w, y = (j / w) | 0;
        bufs.pix[area++] = j; sx += x; sy += y;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        var xm = x > 0, xp = x < w - 1;
        if (xm) push(j - 1);
        if (xp) push(j + 1);
        if (y > 0) { push(j - w); if (xm) push(j - w - 1); if (xp) push(j - w + 1); }
        if (y < h - 1) { push(j + w); if (xm) push(j + w - 1); if (xp) push(j + w + 1); }
      }
      var len = Math.max(maxX - minX, maxY - minY);
      // len^2/area ≈ length/width for a stroke: keep elongated shapes only
      if (area >= minArea && len >= minLen && len * len >= 4 * area) {
        // anchor the tag at the component pixel nearest the centroid, so it
        // sits on the structure even when the shape curves
        var cx = sx / area, cy = sy / area;
        var bestD = Infinity, bx = cx, by = cy;
        for (var q = 0; q < area; q++) {
          var pj = bufs.pix[q];
          var px = pj % w, py = (pj / w) | 0;
          var dd = (px - cx) * (px - cx) + (py - cy) * (py - cy);
          if (dd < bestD) { bestD = dd; bx = px; by = py; }
        }
        comps.push({ x: bx, y: by, area: area, len: len });
      }
    }
    comps.sort(function (a, b) { return b.area - a.area; });
    return comps.slice(0, 9);
  }

  /* ---------------- rendering ---------------- */

  var ACC_R = 70, ACC_G = 235, ACC_B = 200;    // vein-like: teal
  var RID_R = 255, RID_G = 176, RID_B = 66;    // tendon-like ridge: amber

  // Grayscale CLAHE view (Enhance mode replaces the video entirely).
  V.renderEnhance = function (bufs, outRGBA) {
    var eq = bufs.eq, n = bufs.w * bufs.h;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var v = eq[i];
      outRGBA[j] = v; outRGBA[j + 1] = v; outRGBA[j + 2] = v; outRGBA[j + 3] = 255;
    }
  };

  // Alpha-carrying overlay: the app drawImages this over crisp full-res
  // video, so detection quality and display quality are decoupled.
  V.renderOverlay = function (bufs, outRGBA, withBright) {
    var alpha = bufs.alpha, alphaB = bufs.alphaB, n = bufs.w * bufs.h;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var a = alpha[i];
      if (withBright && alphaB[i] > a) {
        outRGBA[j] = RID_R; outRGBA[j + 1] = RID_G; outRGBA[j + 2] = RID_B;
        outRGBA[j + 3] = (alphaB[i] * 255) | 0;
      } else {
        outRGBA[j] = ACC_R; outRGBA[j + 1] = ACC_G; outRGBA[j + 2] = ACC_B;
        outRGBA[j + 3] = (a * 255) | 0;
      }
    }
  };

  /* ---------------- main entry ---------------- */

  // params: { mode: 'enhance' | 'veins' | 'structures', strength: 0..1,
  //           sensitivity: 0..1, still: bool, labels: bool }
  // Fills bufs.eq (always) and bufs.alpha/alphaB (overlay modes); returns
  // { labels: [...]|null, skinFrac: 0..1 }.
  V.analyze = function (rgba, bufs, state, params) {
    var w = bufs.w, h = bufs.h, n = w * h;

    extractGreenDenoised(rgba, bufs);
    clahe(bufs, 1.2 + params.strength * 4.3);

    if (params.mode === 'enhance') {
      return { labels: null, skinFrac: 1 };
    }

    var skinFrac = skinMask(rgba, bufs); // must run before vesselness (scratch reuse)
    luminanceGate(bufs);
    flattenNonSkin(bufs);
    var wantBright = params.mode === 'structures';
    // physical vein widths are resolution-independent: scale sigmas with w
    var k = w / 384;
    var res = vesselness(bufs, [1.4 * k, 2.6 * k, 4.2 * k], wantBright);
    var keep = params.still ? 0 : 0.35;
    // almost no skin in frame → fade the whole overlay out instead of
    // decorating the furniture
    var cov = (skinFrac - 0.03) / 0.05;
    if (cov < 0) cov = 0; else if (cov > 1) cov = 1;

    var p = percentileNorm(bufs.vess, bufs.mask, n, res.maxD, bufs.normHist);
    state.norm = (params.still || !state.norm) ? p : state.norm * 0.75 + p * 0.25;
    thresholdPass(bufs.vess, bufs.vessS, bufs.alpha, bufs.mask, n, state.norm, params.sensitivity, keep, cov);

    if (wantBright) {
      var pB = percentileNorm(bufs.vessB, bufs.mask, n, res.maxB, bufs.normHist);
      state.normB = (params.still || !state.normB) ? pB : state.normB * 0.75 + pB * 0.25;
      thresholdPass(bufs.vessB, bufs.vessBS, bufs.alphaB, bufs.mask, n, state.normB, params.sensitivity, keep, cov);
    }

    return { labels: params.labels ? findComponents(bufs) : null, skinFrac: skinFrac };
  };

  /* ---------------- pulse (remote photoplethysmography) ---------------- */

  // Blood volume changes with each heartbeat shift skin brightness by a
  // fraction of a percent. Per-pixel temporal band-pass (difference of two
  // EMAs, ~0.7–3 Hz passband) exposes it; squared and smoothed it becomes a
  // "where does this skin pulse" map. This shows pulsation strength — it
  // does NOT locate arteries, which lie below visible-light depth.
  var TAU_FAST = 0.25, TAU_SLOW = 1.2, TAU_ENERGY = 1.5; // seconds
  var SIG_CAP = 512;

  V.makePulseState = function (w, h) {
    var n = w * h;
    return {
      w: w, h: h,
      fast: new Float32Array(n),
      slow: new Float32Array(n),
      energy: new Float32Array(n),
      eMax: 1e-6,
      primed: false,
      lastT: 0,
      sig: new Float32Array(SIG_CAP),   // spatial-mean band-passed signal
      sigT: new Float32Array(SIG_CAP),  // sample timestamps, seconds
      sigN: 0                           // total samples written (ring)
    };
  };

  // rgba: current frame at pulse resolution; tMs: caller-supplied clock so
  // replays and tests stay deterministic.
  V.pulseUpdate = function (st, rgba, tMs) {
    var n = st.w * st.h;
    var i, j, g;
    if (!st.primed || tMs - st.lastT > 500) {
      // (re)prime after a gap: EMAs snap to the frame, energy restarts
      for (i = 0, j = 0; i < n; i++, j += 4) {
        g = rgba[j + 1];
        st.fast[i] = g; st.slow[i] = g; st.energy[i] = 0;
      }
      st.eMax = 1e-6;
      st.sigN = 0;
      st.primed = true;
      st.lastT = tMs;
      return;
    }
    var dt = Math.min(0.1, Math.max(0.01, (tMs - st.lastT) / 1000));
    st.lastT = tMs;
    var af = 1 - Math.exp(-dt / TAU_FAST);
    var as = 1 - Math.exp(-dt / TAU_SLOW);
    var ae = 1 - Math.exp(-dt / TAU_ENERGY);
    var sum = 0, eMax = st.eMax * 0.995;
    for (i = 0, j = 0; i < n; i++, j += 4) {
      g = rgba[j + 1];
      var f = st.fast[i] + af * (g - st.fast[i]);
      var s = st.slow[i] + as * (g - st.slow[i]);
      st.fast[i] = f; st.slow[i] = s;
      var bp = f - s;
      sum += bp;
      var e = st.energy[i] + ae * (bp * bp - st.energy[i]);
      st.energy[i] = e;
      if (e > eMax) eMax = e;
    }
    st.eMax = Math.max(eMax, 1e-6);
    var k = st.sigN % SIG_CAP;
    st.sig[k] = sum / n;
    st.sigT[k] = tMs / 1000;
    st.sigN++;
  };

  // Autocorrelation of the mean band-passed signal → beats per minute.
  // Returns { bpm, conf }; conf < 0.3 means "don't show a number".
  V.pulseBpm = function (st) {
    var have = Math.min(st.sigN, SIG_CAP);
    if (have < 90) return { bpm: 0, conf: 0 };
    var m = have, i;
    var s = new Float32Array(m);
    var start = st.sigN - m;
    for (i = 0; i < m; i++) s[i] = st.sig[(start + i) % SIG_CAP];
    var t0 = st.sigT[start % SIG_CAP], t1 = st.sigT[(st.sigN - 1) % SIG_CAP];
    var span = t1 - t0;
    if (span < 3) return { bpm: 0, conf: 0 };
    var dt = span / (m - 1);
    var mean = 0;
    for (i = 0; i < m; i++) mean += s[i];
    mean /= m;
    var norm = 0;
    for (i = 0; i < m; i++) { s[i] -= mean; norm += s[i] * s[i]; }
    if (norm < 1e-9) return { bpm: 0, conf: 0 };
    var lagMin = Math.max(2, Math.round(0.4 / dt));   // 150 bpm
    var lagMax = Math.min(m - 10, Math.round(1.5 / dt)); // 40 bpm
    var bestLag = 0, bestR = 0;
    for (var lag = lagMin; lag <= lagMax; lag++) {
      var acc = 0;
      for (i = 0; i + lag < m; i++) acc += s[i] * s[i + lag];
      var r = acc / norm;
      if (r > bestR) { bestR = r; bestLag = lag; }
    }
    if (!bestLag) return { bpm: 0, conf: 0 };
    return { bpm: 60 / (bestLag * dt), conf: bestR };
  };

  // Coral heat overlay: alpha carries the map, so a plain drawImage
  // composites it over any base layer.
  V.pulseCompose = function (st, outRGBA, gain) {
    var n = st.w * st.h;
    var invMax = 1 / st.eMax;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var a = st.energy[i] * invMax * gain;
      if (a > 1) a = 1;
      a = a * a * (3 - 2 * a);
      outRGBA[j] = 255; outRGBA[j + 1] = 96; outRGBA[j + 2] = 92;
      outRGBA[j + 3] = (a * 216) | 0;
    }
  };

  window.VenuleVision = V;
})();
