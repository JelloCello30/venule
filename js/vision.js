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
      hist: null, lut: null           // sized lazily by clahe()
    };
  };

  V.makeState = function () {
    return { dispMax: 1e-4, dispMaxB: 1e-4 };
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

  // EMA + normalize + smoothstep threshold → overlay opacity in [0, 0.92]
  function thresholdPass(raw, smoothed, alpha, n, dispMax, sensitivity, keep) {
    var t = 0.65 * (1 - sensitivity) + 0.06;
    var invSpan = 1 / Math.max(1 - t, 1e-3);
    var invMax = 1 / dispMax;
    for (var i = 0; i < n; i++) {
      var sm = smoothed[i] * keep + raw[i] * (1 - keep);
      smoothed[i] = sm;
      var a = (sm * invMax - t) * invSpan;
      if (a <= 0) { alpha[i] = 0; continue; }
      if (a > 1) a = 1;
      a = a * a * (3 - 2 * a); // smoothstep: soft toe and shoulder
      alpha[i] = a * 0.92;
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

  /* ---------------- composite ---------------- */

  var ACC_R = 70, ACC_G = 235, ACC_B = 200;    // vein-like: teal
  var RID_R = 255, RID_G = 176, RID_B = 66;    // tendon-like ridge: amber

  function composeEnhance(bufs, outRGBA) {
    var eq = bufs.eq, n = bufs.w * bufs.h;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      var v = eq[i];
      outRGBA[j] = v; outRGBA[j + 1] = v; outRGBA[j + 2] = v; outRGBA[j + 3] = 255;
    }
  }

  function composeOverlay(bufs, srcRGBA, outRGBA, withBright) {
    var alpha = bufs.alpha, alphaB = bufs.alphaB, n = bufs.w * bufs.h;
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      // base: original, half-desaturated and dimmed so the overlay reads
      var r = srcRGBA[j], g = srcRGBA[j + 1], b = srcRGBA[j + 2];
      var m = (r + g + b) * 0.3333;
      r = (r + (m - r) * 0.55) * 0.78;
      g = (g + (m - g) * 0.55) * 0.78;
      b = (b + (m - b) * 0.55) * 0.78;
      var a = alpha[i];
      if (a > 0) {
        r += (ACC_R - r) * a; g += (ACC_G - g) * a; b += (ACC_B - b) * a;
      }
      if (withBright) {
        var ab = alphaB[i];
        if (ab > 0) {
          r += (RID_R - r) * ab; g += (RID_G - g) * ab; b += (RID_B - b) * ab;
        }
      }
      outRGBA[j] = r; outRGBA[j + 1] = g; outRGBA[j + 2] = b; outRGBA[j + 3] = 255;
    }
  }

  /* ---------------- main entry ---------------- */

  // params: { mode: 'enhance' | 'veins' | 'structures', strength: 0..1,
  //           sensitivity: 0..1, still: bool, labels: bool }
  // Returns an array of labeled components when params.labels is set,
  // otherwise null.
  V.process = function (rgba, outRGBA, bufs, state, params) {
    var w = bufs.w, h = bufs.h, n = w * h;

    extractGreenDenoised(rgba, bufs);
    clahe(bufs, 1.2 + params.strength * 4.3);

    if (params.mode === 'enhance') {
      composeEnhance(bufs, outRGBA);
      return null;
    }

    var wantBright = params.mode === 'structures';
    // physical vein widths are resolution-independent: scale sigmas with w
    var k = w / 384;
    var res = vesselness(bufs, [1.5 * k, 3.0 * k, 5.0 * k], wantBright);
    var keep = params.still ? 0 : 0.35;

    state.dispMax = Math.max(res.maxD, state.dispMax * 0.9, 1e-4);
    thresholdPass(bufs.vess, bufs.vessS, bufs.alpha, n, state.dispMax, params.sensitivity, keep);

    if (wantBright) {
      state.dispMaxB = Math.max(res.maxB, state.dispMaxB * 0.9, 1e-4);
      thresholdPass(bufs.vessB, bufs.vessBS, bufs.alphaB, n, state.dispMaxB, params.sensitivity, keep);
    }
    composeOverlay(bufs, rgba, outRGBA, wantBright);

    return params.labels ? findComponents(bufs) : null;
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
