/* venule — app shell: camera, render loop, controls. */
(function () {
  'use strict';

  var V = window.VenuleVision;

  var els = {
    startCard: document.getElementById('start-card'),
    errorCard: document.getElementById('error-card'),
    errorText: document.getElementById('error-text'),
    stage: document.getElementById('stage'),
    canvas: document.getElementById('view'),
    labels: document.getElementById('labels'),
    legend: document.getElementById('legend'),
    video: document.getElementById('cam'),
    hud: document.getElementById('hud'),
    readout: document.getElementById('readout'),
    controls: document.getElementById('controls'),
    modeBtns: Array.prototype.slice.call(document.querySelectorAll('[data-mode]')),
    strength: document.getElementById('strength'),
    sensitivity: document.getElementById('sensitivity'),
    sensRow: document.getElementById('sens-row'),
    btnFreeze: document.getElementById('btn-freeze'),
    btnSave: document.getElementById('btn-save'),
    btnPhoto: document.getElementById('btn-photo'),
    btnFlip: document.getElementById('btn-flip'),
    btnLive: document.getElementById('btn-live'),
    fileInput: document.getElementById('file-input'),
    divider: document.getElementById('divider')
  };

  var ctx = els.canvas.getContext('2d', { willReadFrequently: false });
  var ctxL = els.labels.getContext('2d');

  var PROC_WIDTHS = [256, 320, 384, 448, 512];

  /* ---------------- neural skin segmentation ---------------- */
  // MediaPipe selfie_multiclass gives per-pixel body-skin/face-skin
  // categories, on-device. Loads in the background; until it's ready (or
  // if the CDN is unreachable) vision.js falls back to its color mask.
  var seg = { inst: null, status: 'loading', mask: null, maskW: 0, maskH: 0, busy: false, lastTs: 0 };

  function initSegmenter() {
    var tries = 0;
    var timer = setInterval(function () {
      var mp = window.__venuleSeg;
      if (mp === undefined && ++tries < 150) return; // module still loading
      clearInterval(timer);
      if (!mp) { seg.status = 'unavailable'; return; }
      mp.FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      ).then(function (fileset) {
        return mp.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: {
            // CPU delegate on purpose: the GPU delegate scrambles category
            // ids on iOS Safari (google-ai-edge/mediapipe#6142)
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
          },
          runningMode: 'VIDEO',
          outputCategoryMask: true,
          outputConfidenceMasks: false
        });
      }).then(function (inst) {
        seg.inst = inst;
        seg.status = 'ready';
      }).catch(function () {
        seg.status = 'unavailable';
      });
    }, 100);
  }

  function updateSegMask() {
    if (!seg.inst || seg.busy) return;
    seg.busy = true;
    var ts = Math.max(seg.lastTs + 1, Math.round(performance.now()));
    seg.lastTs = ts;
    try {
      seg.inst.segmentForVideo(app.procCanvas, ts, function (result) {
        var cm = result.categoryMask;
        if (cm) {
          var arr = cm.getAsUint8Array();
          if (!seg.mask || seg.mask.length !== arr.length) seg.mask = new Uint8Array(arr.length);
          seg.mask.set(arr);
          seg.maskW = cm.width;
          seg.maskH = cm.height;
        }
        result.close();
        seg.busy = false;
        // stills render on demand: repaint once now that the mask exists
        if (app.phase === 'photo' || app.frozen) app.needsRender = true;
      });
    } catch (e) {
      seg.status = 'unavailable';
      seg.busy = false;
    }
  }

  var app = {
    phase: 'start',          // start | live | photo | error
    mode: 'reveal',          // reveal | veins | labels | structures | pulse | split
    facing: 'environment',
    stream: null,
    frozen: false,
    procIdx: 2,
    bufs: null,
    state: V.makeState(),
    procCanvas: document.createElement('canvas'),
    procCtx: null,
    outCanvas: document.createElement('canvas'),
    outCtx: null,
    outImage: null,
    skinFrac: 1,
    photoBitmap: null,
    split: 0.5,
    lastLabels: null,
    labelTick: 0,
    pulseState: null,
    pulseCanvas: document.createElement('canvas'),
    pulseCtx: null,
    heatCanvas: document.createElement('canvas'),
    heatCtx: null,
    heatImg: null,
    bpmShown: 0,
    stillCanvas: document.createElement('canvas'),
    stillCtx: null,
    segTick: 0,
    stillSegDone: false,
    motion: 0,
    compareRaw: false,
    blackTick: 0,
    blackCount: 0,
    debug: /[?&]debug=1/.test(location.search),
    frameMs: [],
    frameCount: 0,
    fps: 0,
    lastT: 0,
    needsRender: true,       // for frozen/photo: re-render only on change
    multiCam: false,
    rafId: 0
  };
  app.procCtx = app.procCanvas.getContext('2d', { willReadFrequently: true });
  app.outCtx = app.outCanvas.getContext('2d');
  app.pulseCtx = app.pulseCanvas.getContext('2d', { willReadFrequently: true });
  app.heatCtx = app.heatCanvas.getContext('2d');
  app.stillCtx = app.stillCanvas.getContext('2d');

  var PULSE_W = 120;

  function syncPulseSize() {
    var ph = Math.max(16, Math.round(PULSE_W / sourceAspect() / 2) * 2);
    if (app.pulseCanvas.width === PULSE_W && app.pulseCanvas.height === ph && app.pulseState) return;
    app.pulseCanvas.width = PULSE_W;
    app.pulseCanvas.height = ph;
    app.heatCanvas.width = PULSE_W;
    app.heatCanvas.height = ph;
    app.pulseState = V.makePulseState(PULSE_W, ph);
    app.heatImg = new ImageData(PULSE_W, ph);
  }

  /* ---------------- sizing ---------------- */

  function setProcSize(w, h) {
    // keep dimensions even and sane
    w = Math.max(64, w | 0); h = Math.max(64, h | 0);
    if (app.bufs && app.bufs.w === w && app.bufs.h === h) return;
    app.procCanvas.width = w;
    app.procCanvas.height = h;
    app.outCanvas.width = w;
    app.outCanvas.height = h;
    app.bufs = V.makeBuffers(w, h);
    app.state = V.makeState();
    app.outImage = new ImageData(w, h);
    // labels are stored in processing-resolution coordinates: stale ones
    // would draw offset until the recompute throttle ticks over
    app.lastLabels = null;
    app.labelTick = 0;
  }

  // The display canvas holds the source at (capped) native resolution —
  // detection runs at the smaller processing size and is drawn on top, so
  // the picture stays crisp no matter what the adaptive resolution does.
  function syncDisplaySize() {
    var sw, sh, isPhoto = app.phase === 'photo' && app.photoBitmap;
    if (isPhoto) { sw = app.photoBitmap.width; sh = app.photoBitmap.height; }
    else if (frozenSource()) { sw = app.stillCanvas.width; sh = app.stillCanvas.height; }
    else if (els.video.videoWidth) { sw = els.video.videoWidth; sh = els.video.videoHeight; }
    else return;
    var cap = 1600;
    var sc = Math.min(1, cap / sw);
    var W = Math.round(sw * sc / 2) * 2, H = Math.round(sh * sc / 2) * 2;
    if (els.canvas.width !== W || els.canvas.height !== H) {
      els.canvas.width = W;
      els.canvas.height = H;
      layoutView();
    }
  }

  // Contain-fit the display canvas into the viewport.
  function layoutView() {
    if (!els.canvas.width) return;
    var vw = window.innerWidth, vh = window.innerHeight;
    if (!vw || !vh) {
      // hidden/zero-sized viewport: fall back to the stylesheet's auto sizing
      els.canvas.style.width = '';
      els.canvas.style.height = '';
      return;
    }
    var ar = els.canvas.width / els.canvas.height;
    var w = Math.min(vw, vh * ar);
    els.canvas.style.width = w + 'px';
    els.canvas.style.height = (w / ar) + 'px';
  }
  window.addEventListener('resize', layoutView);

  function sourceAspect() {
    // a pending photoBitmap wins regardless of phase — openPhoto sizes
    // buffers before the phase flips to 'photo'
    if (app.photoBitmap) {
      return app.photoBitmap.width / app.photoBitmap.height;
    }
    if (els.video.videoWidth) return els.video.videoWidth / els.video.videoHeight;
    return 4 / 3;
  }

  function syncProcSize() {
    var w = PROC_WIDTHS[app.procIdx];
    var h = Math.round(w / sourceAspect() / 2) * 2;
    setProcSize(w, h);
  }

  /* ---------------- phases ---------------- */

  function show(phase) {
    app.phase = phase;
    els.startCard.hidden = phase !== 'start';
    els.errorCard.hidden = phase !== 'error';
    els.stage.classList.toggle('active', phase === 'live' || phase === 'photo');
    els.controls.hidden = !(phase === 'live' || phase === 'photo');
    els.hud.hidden = !(phase === 'live' || phase === 'photo');
    els.btnLive.hidden = phase !== 'photo';
    els.btnFreeze.hidden = phase === 'photo';
    els.btnFlip.hidden = phase !== 'live' || !app.multiCam;
    updateModeUI();
    layoutView();
  }

  function fail(message) {
    stopStream();
    els.errorText.textContent = message;
    show('error');
  }

  /* ---------------- camera ---------------- */

  function stopStream() {
    if (app.stream) {
      app.stream.getTracks().forEach(function (t) { t.stop(); });
      app.stream = null;
    }
  }

  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      fail('This browser does not support camera capture. You can still open a photo below.');
      return;
    }
    stopStream();
    var constraints = {
      audio: false,
      video: {
        facingMode: { ideal: app.facing },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    };
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      app.stream = stream;
      els.video.srcObject = stream;
      return els.video.play();
    }).then(function () {
      return navigator.mediaDevices.enumerateDevices();
    }).then(function (devices) {
      app.multiCam = devices.filter(function (d) { return d.kind === 'videoinput'; }).length > 1;
      app.photoBitmap = null;
      setFrozen(false);
      setupTorch();
      syncProcSize();
      showHint('back of your hand · bright light · hold still');
      show('live');
      startLoop();
    }).catch(function (err) {
      var name = err && err.name;
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        fail('Camera access was blocked. Allow it in your browser’s site permissions and reload — or open a photo instead.');
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        fail('No camera was found on this device. You can open a photo instead.');
      } else {
        fail('The camera could not be started (' + (name || 'unknown error') + '). You can open a photo instead.');
      }
    });
  }

  /* ---------------- render loop ---------------- */

  // The still source while frozen; null means live video.
  function frozenSource() {
    return (app.frozen && app.stillCanvas.width) ? app.stillCanvas : null;
  }

  // A frozen frame must be a real snapshot: the <video> keeps playing, so
  // re-renders (slider tweaks, mode switches) would silently re-sample it.
  function setFrozen(v) {
    if (v && app.phase === 'live' && els.video.videoWidth) {
      app.stillCanvas.width = els.video.videoWidth;
      app.stillCanvas.height = els.video.videoHeight;
      app.stillCtx.drawImage(els.video, 0, 0);
    }
    app.frozen = v;
    if (v) app.stillSegDone = false;
    if (!v) { app.stillCanvas.width = 0; app.stillCanvas.height = 0; }
    els.btnFreeze.textContent = v ? 'Resume' : 'Freeze';
    app.needsRender = true;
    updateReadout();
  }

  function drawSourceToProc() {
    var pc = app.procCanvas, pctx = app.procCtx;
    var still = frozenSource();
    if (app.phase === 'photo' && app.photoBitmap) {
      pctx.drawImage(app.photoBitmap, 0, 0, pc.width, pc.height);
    } else if (still) {
      pctx.drawImage(still, 0, 0, pc.width, pc.height);
    } else {
      pctx.drawImage(els.video, 0, 0, pc.width, pc.height);
    }
  }

  function pipelineModeFor(mode) {
    return mode === 'structures' ? 'structures' : 'veins'; // veins, labels
  }

  function renderPulseOverlay() {
    syncPulseSize();
    app.pulseCtx.drawImage(els.video, 0, 0, app.pulseCanvas.width, app.pulseCanvas.height);
    var frame = app.pulseCtx.getImageData(0, 0, app.pulseCanvas.width, app.pulseCanvas.height);
    V.pulseUpdate(app.pulseState, frame.data, performance.now());
    // sensitivity slider doubles as overlay gain here
    var gain = 0.6 + parseFloat(els.sensitivity.value) * 2.4;
    V.pulseCompose(app.pulseState, app.heatImg.data, gain);
    app.heatCtx.putImageData(app.heatImg, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(app.heatCanvas, 0, 0, els.canvas.width, els.canvas.height);
  }

  function renderFrame() {
    var t0 = performance.now();
    syncDisplaySize();
    var w = app.bufs.w, h = app.bufs.h;
    var dw = els.canvas.width, dh = els.canvas.height;
    var src = (app.phase === 'photo' && app.photoBitmap) ? app.photoBitmap
      : (frozenSource() || els.video);

    if (app.mode === 'pulse') {
      // pulse needs no vesselness analysis: crisp video + heat overlay
      ctx.drawImage(src, 0, 0, dw, dh);
      if (app.phase === 'live' && !app.frozen) renderPulseOverlay();
      els.divider.hidden = true;
      drawLabels();
      return performance.now() - t0;
    }

    if (app.compareRaw) {
      // press-and-hold raw peek: just the source, nothing drawn over it
      ctx.drawImage(src, 0, 0, dw, dh);
      els.divider.hidden = true;
      ctxL.setTransform(1, 0, 0, 1, 0, 0);
      ctxL.clearRect(0, 0, els.labels.width, els.labels.height);
      return performance.now() - t0;
    }

    drawSourceToProc();
    var img = app.procCtx.getImageData(0, 0, w, h);

    var still = app.phase === 'photo' || app.frozen;
    // refresh the neural mask: every other live frame, once per still
    if (!still) {
      if (app.segTick++ % 2 === 0) updateSegMask();
    } else if (!app.stillSegDone && seg.inst) {
      app.stillSegDone = true;
      updateSegMask();
    }

    // black-frame watchdog: a live feed that renders but delivers no light
    // is otherwise indistinguishable from "app is broken"
    if (app.phase === 'live' && ++app.blackTick % 60 === 0) {
      var lum = 0, npx = 0;
      for (var bi = 1; bi < img.data.length; bi += 163 * 4 + 1) { lum += img.data[bi]; npx++; }
      if (lum / npx < 4) {
        if (++app.blackCount >= 2) showToast('camera is delivering black frames — reload the page, or try another browser');
      } else {
        app.blackCount = 0;
      }
    }

    if (app.mode === 'reveal' || app.mode === 'split') {
      // computational-photography path: temporal stack + in-place
      // amplification, composited as a hard-light detail layer over the
      // crisp base (128 = neutral, so non-skin is untouched)
      var resR = V.analyzeReveal(img.data, app.outImage.data, app.bufs, app.state, {
        strength: parseFloat(els.strength.value),
        still: still,
        catMask: seg.mask,
        catW: seg.maskW,
        catH: seg.maskH
      });
      app.skinFrac = resR.skinFrac;
      app.motion = resR.motion;
      app.outCtx.putImageData(app.outImage, 0, 0);
      ctx.drawImage(src, 0, 0, dw, dh);
      ctx.imageSmoothingEnabled = true;
      ctx.globalCompositeOperation = 'hard-light';
      if (app.mode === 'split') {
        var cut = Math.round(dw * app.split);
        ctx.save();
        ctx.beginPath();
        ctx.rect(cut, 0, dw - cut, dh);
        ctx.clip();
        ctx.drawImage(app.outCanvas, 0, 0, dw, dh);
        ctx.restore();
      } else {
        ctx.drawImage(app.outCanvas, 0, 0, dw, dh);
      }
      ctx.globalCompositeOperation = 'source-over';
    } else {
      // annotation path: veins / labels / structures
      var wantLabels = app.mode === 'labels' && (still || app.labelTick++ % 12 === 0);
      var res = V.analyze(img.data, app.bufs, app.state, {
        mode: pipelineModeFor(app.mode),
        strength: parseFloat(els.strength.value),
        sensitivity: parseFloat(els.sensitivity.value),
        still: still,
        labels: wantLabels,
        catMask: seg.mask,
        catW: seg.maskW,
        catH: seg.maskH
      });
      if (wantLabels) app.lastLabels = res.labels;
      app.skinFrac = res.skinFrac;

      ctx.drawImage(src, 0, 0, dw, dh);
      V.renderOverlay(app.bufs, app.outImage.data, app.mode === 'structures');
      app.outCtx.putImageData(app.outImage, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.fillStyle = 'rgba(6,8,10,0.16)';
      ctx.fillRect(0, 0, dw, dh);
      ctx.drawImage(app.outCanvas, 0, 0, dw, dh);
    }
    els.divider.hidden = app.mode !== 'split';
    drawLabels();

    return performance.now() - t0;
  }

  // roundRect is missing on older Safari
  function pillPath(c, x, y, w, h, r) {
    if (c.roundRect) { c.roundRect(x, y, w, h, r); return; }
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function drawLabels() {
    var cw = els.canvas.clientWidth, ch = els.canvas.clientHeight;
    if (!cw) return;
    var dpr = window.devicePixelRatio || 1;
    var pw = Math.round(cw * dpr), ph = Math.round(ch * dpr);
    if (els.labels.width !== pw || els.labels.height !== ph) {
      els.labels.width = pw; els.labels.height = ph;
    }
    ctxL.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctxL.clearRect(0, 0, cw, ch);
    if (app.mode !== 'labels' || !app.lastLabels) return;

    var sx = cw / app.bufs.w, sy = ch / app.bufs.h;
    ctxL.font = '600 11px ui-monospace, Menlo, monospace';
    ctxL.textBaseline = 'middle';
    for (var i = 0; i < app.lastLabels.length; i++) {
      var c = app.lastLabels[i];
      var x = c.x * sx, y = c.y * sy;
      var tag = 'V' + (i + 1);
      var tw = ctxL.measureText(tag).width;
      ctxL.fillStyle = '#46e0b4';
      ctxL.beginPath(); ctxL.arc(x, y, 3, 0, Math.PI * 2); ctxL.fill();
      ctxL.strokeStyle = 'rgba(70,224,180,0.6)'; ctxL.lineWidth = 1;
      ctxL.beginPath(); ctxL.moveTo(x + 3, y - 3); ctxL.lineTo(x + 12, y - 12); ctxL.stroke();
      ctxL.fillStyle = 'rgba(12,14,16,0.82)';
      ctxL.beginPath();
      var bx = x + 12, by = y - 22;
      pillPath(ctxL, bx, by, tw + 12, 19, 6);
      ctxL.fill();
      ctxL.strokeStyle = 'rgba(255,255,255,0.18)';
      ctxL.stroke();
      ctxL.fillStyle = '#e9ebe7';
      ctxL.fillText(tag, bx + 6, by + 10);
    }
  }

  function adaptResolution(ms) {
    app.frameMs.push(ms);
    if (app.frameMs.length < 40) return;
    var avg = app.frameMs.reduce(function (a, b) { return a + b; }, 0) / app.frameMs.length;
    app.frameMs.length = 0;
    if (avg > 34 && app.procIdx > 0) { app.procIdx--; syncProcSize(); }
    else if (avg < 16 && app.procIdx < PROC_WIDTHS.length - 1) { app.procIdx++; syncProcSize(); }
  }

  function updateReadout() {
    if (!app.bufs) return; // nothing sized yet (fresh page, pre-camera)
    var txt = app.bufs.w + 'px · ' + app.fps + ' fps' + (app.frozen ? ' · frozen' : '');
    if (app.mode !== 'pulse' && app.skinFrac < 0.05 &&
        (app.phase === 'live' || app.phase === 'photo')) {
      txt += ' · point at skin';
    }
    if (app.mode === 'reveal' && app.phase === 'live' && !app.frozen) {
      if (app.motion > 6) txt += ' · hold still to stack';
      else if (app.motion < 3 && app.state && app.state.stacked) txt += ' · stacked';
    }
    if (app.debug) {
      txt += ' · ' + (seg.status === 'ready' ? 'ai-mask' : 'color-mask (' + seg.status + ')') +
        ' · skin ' + Math.round(app.skinFrac * 100) + '%';
    }
    if (app.mode === 'pulse' && app.phase === 'live') {
      if (app.frozen) {
        txt += ' · pulse paused';
      } else {
        var est = app.pulseState ? V.pulseBpm(app.pulseState) : { conf: 0 };
        if (est.conf >= 0.3) {
          // display smoothing so the number doesn't twitch
          app.bpmShown = app.bpmShown ? app.bpmShown * 0.7 + est.bpm * 0.3 : est.bpm;
          txt += ' · ~' + Math.round(app.bpmShown) + ' bpm';
        } else {
          app.bpmShown = 0;
          txt += ' · measuring — hold still';
        }
      }
    }
    els.readout.textContent = txt;
  }

  function loop(t) {
    app.rafId = requestAnimationFrame(loop);
    if (app.phase !== 'live' && app.phase !== 'photo') return;

    var live = app.phase === 'live' && !app.frozen;
    if (!live && !app.needsRender) return;
    app.needsRender = false;

    if (live && els.video.readyState < 2) return;
    if (live) syncProcSize(); // catches late videoWidth changes

    var ms = renderFrame();
    // pulse and raw-peek frames are cheap and don't use the adaptive tier —
    // measuring them would ratchet procIdx up and stall on return
    if (live && app.mode !== 'pulse' && !app.compareRaw) adaptResolution(ms);

    app.frameCount++;
    if (t - app.lastT > 500) {
      app.fps = live ? Math.round(app.frameCount * 1000 / (t - app.lastT)) : 0;
      app.frameCount = 0;
      app.lastT = t;
      updateReadout();
    }
  }

  function startLoop() {
    cancelAnimationFrame(app.rafId);
    app.lastT = performance.now();
    app.frameCount = 0;
    app.needsRender = true;
    app.rafId = requestAnimationFrame(loop);
  }

  /* ---------------- photo mode ---------------- */

  function openPhoto(file) {
    createImageBitmap(file).then(function (bmp) {
      stopStream();
      // the stream just died, and the torch with it — drop its stale button
      torchOn = false;
      document.getElementById('btn-torch').hidden = true;
      app.photoBitmap = bmp;
      app.stillSegDone = false;
      setFrozen(false);
      if (app.mode === 'pulse') app.mode = 'veins';
      // stills can afford the top resolution
      app.procIdx = PROC_WIDTHS.length - 1;
      syncProcSize();
      // same-size photo swaps skip setProcSize's reset, and the display
      // normalization must not carry one image's peak into the next
      app.state = V.makeState();
      app.lastLabels = null;
      app.labelTick = 0;
      show('photo');
      // stills normalize directly (no EMA), so the first render is converged
      app.needsRender = true;
      startLoop();
      updateReadout();
    }).catch(function () {
      fail('That image could not be read. Try a JPEG or PNG.');
    });
  }

  /* ---------------- controls ---------------- */

  function updateModeUI() {
    els.modeBtns.forEach(function (b) {
      b.classList.toggle('on', b.dataset.mode === app.mode);
      // pulse needs a live feed: a still has no heartbeat to band-pass
      if (b.dataset.mode === 'pulse') b.disabled = app.phase === 'photo';
    });
    els.sensRow.hidden = app.mode === 'reveal' || app.mode === 'split';
    els.legend.hidden = app.mode !== 'structures';
    els.divider.hidden = app.mode !== 'split';
  }

  function setMode(m) {
    if (m === 'pulse' && app.phase === 'photo') return;
    app.mode = m;
    if (app.state) app.state.stacked = false; // fresh stack, no stale ghosts
    app.needsRender = true;
    updateModeUI();
    updateReadout();
  }

  els.modeBtns.forEach(function (b) {
    b.addEventListener('click', function () { setMode(b.dataset.mode); });
  });

  [els.strength, els.sensitivity].forEach(function (s) {
    s.addEventListener('input', function () { app.needsRender = true; });
  });

  els.btnFreeze.addEventListener('click', function () {
    setFrozen(!app.frozen);
  });

  els.btnSave.addEventListener('click', function () {
    els.canvas.toBlob(function (blob) {
      if (!blob) return;
      var a = document.createElement('a');
      var d = new Date();
      function p(x) { return String(x).padStart(2, '0'); }
      a.download = 'venule-' + d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) +
        '-' + p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds()) + '.png';
      a.href = URL.createObjectURL(blob);
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
    }, 'image/png');
  });

  els.btnPhoto.addEventListener('click', function () { els.fileInput.click(); });
  document.getElementById('btn-photo-start').addEventListener('click', function () { els.fileInput.click(); });
  document.getElementById('btn-photo-error').addEventListener('click', function () { els.fileInput.click(); });

  els.fileInput.addEventListener('change', function () {
    if (els.fileInput.files && els.fileInput.files[0]) openPhoto(els.fileInput.files[0]);
    els.fileInput.value = '';
  });

  els.btnFlip.addEventListener('click', function () {
    app.facing = app.facing === 'environment' ? 'user' : 'environment';
    startCamera();
  });

  els.btnLive.addEventListener('click', function () {
    app.photoBitmap = null;
    app.procIdx = 2;
    startCamera();
  });

  document.getElementById('btn-start').addEventListener('click', startCamera);
  document.getElementById('btn-retry').addEventListener('click', startCamera);

  /* split divider drag */
  function setSplit(clientX) {
    var r = els.canvas.getBoundingClientRect();
    app.split = Math.min(0.95, Math.max(0.05, (clientX - r.left) / r.width));
    els.divider.style.left = (app.split * 100) + '%';
    app.needsRender = true;
  }
  var dragging = false;
  els.stage.addEventListener('pointerdown', function (e) {
    if (app.mode === 'split') {
      dragging = true;
      els.stage.setPointerCapture(e.pointerId);
      setSplit(e.clientX);
      return;
    }
    // press-and-hold anywhere shows the raw feed (before/after, like photo
    // editors); release returns to the processed view. Capture the pointer:
    // a mouse released over the fixed control bar would otherwise never
    // deliver pointerup back here and the peek would stick on.
    if (app.mode !== 'pulse' && e.button === 0) {
      try { els.stage.setPointerCapture(e.pointerId); } catch (err) { /* inactive pointer */ }
      app.compareRaw = true;
      app.needsRender = true;
    }
  });
  els.stage.addEventListener('pointermove', function (e) { if (dragging) setSplit(e.clientX); });
  function endPointer() {
    dragging = false;
    if (app.compareRaw) { app.compareRaw = false; app.needsRender = true; }
  }
  els.stage.addEventListener('pointerup', endPointer);
  els.stage.addEventListener('pointercancel', endPointer);
  els.stage.addEventListener('lostpointercapture', endPointer);
  els.divider.style.left = '50%';

  /* drag & drop a photo anywhere */
  document.addEventListener('dragover', function (e) { e.preventDefault(); });
  document.addEventListener('drop', function (e) {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) openPhoto(e.dataTransfer.files[0]);
  });

  /* keyboard */
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === '1') setMode('reveal');
    if (e.key === '2') setMode('veins');
    if (e.key === '3') setMode('labels');
    if (e.key === '4') setMode('structures');
    if (e.key === '5') setMode('pulse');
    if (e.key === '6') setMode('split');
    if (e.key === 'f' && app.phase === 'live') els.btnFreeze.click();
    if (e.key === 's') els.btnSave.click();
  });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
      cancelAnimationFrame(app.rafId);
      endPointer(); // a mouse released while hidden never sends pointerup
    } else {
      startLoop();
    }
  });

  function showToast(msg) {
    var t = document.getElementById('errtoast');
    if (t) { t.textContent = msg; t.hidden = false; }
  }

  var hintTimer = 0;
  function showHint(msg) {
    var el2 = document.getElementById('hint');
    if (!el2) return;
    el2.textContent = msg;
    el2.hidden = false;
    el2.classList.remove('fade');
    clearTimeout(hintTimer);
    hintTimer = setTimeout(function () { el2.classList.add('fade'); }, 4500);
  }

  // torch (flashlight) — Android Chrome exposes it via track capabilities;
  // hidden everywhere else
  var torchOn = false;
  function setupTorch() {
    var btn = document.getElementById('btn-torch');
    torchOn = false;
    btn.hidden = true;
    if (!app.stream) return;
    var track = app.stream.getVideoTracks()[0];
    if (!track || !track.getCapabilities) return;
    var caps = track.getCapabilities();
    if (!caps || !caps.torch) return;
    btn.hidden = false;
    btn.textContent = 'Light';
    btn.onclick = function () {
      torchOn = !torchOn;
      track.applyConstraints({ advanced: [{ torch: torchOn }] }).then(function () {
        btn.textContent = torchOn ? 'Light ·on' : 'Light';
      }).catch(function () { btn.hidden = true; });
    };
  }

  // surface hard failures on screen — a silent white/black page is
  // undebuggable from a phone
  window.addEventListener('error', function (e) {
    showToast('error: ' + (e.message || 'unknown'));
  });

  initSegmenter();
  show('start');
})();
