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

  var app = {
    phase: 'start',          // start | live | photo | error
    mode: 'veins',           // enhance | veins | split
    facing: 'environment',
    stream: null,
    frozen: false,
    procIdx: 2,
    bufs: null,
    state: V.makeState(),
    procCanvas: document.createElement('canvas'),
    procCtx: null,
    outImage: null,
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
    frameMs: [],
    frameCount: 0,
    fps: 0,
    lastT: 0,
    needsRender: true,       // for frozen/photo: re-render only on change
    multiCam: false,
    rafId: 0
  };
  app.procCtx = app.procCanvas.getContext('2d', { willReadFrequently: true });
  app.pulseCtx = app.pulseCanvas.getContext('2d', { willReadFrequently: true });
  app.heatCtx = app.heatCanvas.getContext('2d');

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
    els.canvas.width = w;
    els.canvas.height = h;
    app.bufs = V.makeBuffers(w, h);
    app.state = V.makeState();
    app.outImage = new ImageData(w, h);
  }

  function sourceAspect() {
    if (app.phase === 'photo' && app.photoBitmap) {
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
        width: { ideal: 1280 },
        height: { ideal: 720 }
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
      app.frozen = false;
      app.photoBitmap = null;
      syncProcSize();
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

  function drawSourceToProc() {
    var pc = app.procCanvas, pctx = app.procCtx;
    if (app.phase === 'photo' && app.photoBitmap) {
      pctx.drawImage(app.photoBitmap, 0, 0, pc.width, pc.height);
    } else {
      pctx.drawImage(els.video, 0, 0, pc.width, pc.height);
    }
  }

  function pipelineModeFor(mode) {
    if (mode === 'enhance' || mode === 'pulse') return 'enhance';
    if (mode === 'structures') return 'structures';
    return 'veins'; // veins, labels, split
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
    var w = app.bufs.w, h = app.bufs.h;

    drawSourceToProc();
    var img = app.procCtx.getImageData(0, 0, w, h);

    var still = app.phase === 'photo' || app.frozen;
    // component labeling jitters if refreshed every frame; throttle it live
    var wantLabels = app.mode === 'labels' && (still || app.labelTick++ % 12 === 0);
    var found = V.process(img.data, app.outImage.data, app.bufs, app.state, {
      mode: pipelineModeFor(app.mode),
      strength: parseFloat(els.strength.value),
      sensitivity: parseFloat(els.sensitivity.value),
      still: still,
      labels: wantLabels
    });
    if (wantLabels) app.lastLabels = found;

    ctx.putImageData(app.outImage, 0, 0);

    if (app.mode === 'pulse' && app.phase === 'live' && !app.frozen) {
      renderPulseOverlay();
    }

    if (app.mode === 'split') {
      // original feed left of the divider, processed right
      var cut = Math.round(w * app.split);
      if (cut > 0) {
        var src = (app.phase === 'photo' && app.photoBitmap) ? app.photoBitmap : els.video;
        var sw = src.videoWidth || src.width, sh = src.videoHeight || src.height;
        ctx.drawImage(src, 0, 0, sw * app.split, sh, 0, 0, cut, h);
      }
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
    var txt = app.bufs.w + 'px · ' + app.fps + ' fps' + (app.frozen ? ' · frozen' : '');
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
    if (live) adaptResolution(ms);

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
      app.photoBitmap = bmp;
      app.frozen = false;
      if (app.mode === 'pulse') app.mode = 'veins';
      // stills can afford the top resolution
      app.procIdx = PROC_WIDTHS.length - 1;
      syncProcSize();
      show('photo');
      // run a few passes so the adaptive normalization settles
      var img; drawSourceToProc();
      img = app.procCtx.getImageData(0, 0, app.bufs.w, app.bufs.h);
      for (var i = 0; i < 2; i++) {
        V.process(img.data, app.outImage.data, app.bufs, app.state, {
          mode: pipelineModeFor(app.mode), strength: parseFloat(els.strength.value),
          sensitivity: parseFloat(els.sensitivity.value), still: true
        });
      }
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
    els.sensRow.hidden = app.mode === 'enhance';
    els.legend.hidden = app.mode !== 'structures';
    els.divider.hidden = app.mode !== 'split';
  }

  function setMode(m) {
    if (m === 'pulse' && app.phase === 'photo') return;
    app.mode = m;
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
    app.frozen = !app.frozen;
    app.needsRender = true;
    els.btnFreeze.textContent = app.frozen ? 'Resume' : 'Freeze';
    updateReadout();
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
    if (app.mode !== 'split') return;
    dragging = true;
    els.stage.setPointerCapture(e.pointerId);
    setSplit(e.clientX);
  });
  els.stage.addEventListener('pointermove', function (e) { if (dragging) setSplit(e.clientX); });
  els.stage.addEventListener('pointerup', function () { dragging = false; });
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
    if (e.key === '1') setMode('enhance');
    if (e.key === '2') setMode('veins');
    if (e.key === '3') setMode('labels');
    if (e.key === '4') setMode('structures');
    if (e.key === '5') setMode('pulse');
    if (e.key === '6') setMode('split');
    if (e.key === 'f' && app.phase === 'live') els.btnFreeze.click();
    if (e.key === 's') els.btnSave.click();
  });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) cancelAnimationFrame(app.rafId);
    else startLoop();
  });

  show('start');
})();
