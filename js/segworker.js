/* venule — neural skin segmentation, off the main thread.
 *
 * MediaPipe's WASM inference is synchronous in whatever thread calls it;
 * on a phone CPU that's 50-150 ms of jank per call if run on the main
 * thread. Here it runs in a module worker: the app posts an ImageBitmap,
 * we post back a transferable category mask.
 *
 * CPU delegate on purpose: the GPU delegate scrambles category ids on
 * iOS Safari (google-ai-edge/mediapipe#6142).
 *
 * This is a CLASSIC worker (no {type:'module'}): MediaPipe's WASM loader
 * calls importScripts(), which module workers prohibit. The ESM API is
 * loaded with dynamic import(), which classic workers allow.
 */
let segmenter = null;

(async () => {
  try {
    const m = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    const fileset = await m.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    segmenter = await m.ImageSegmenter.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
      },
      runningMode: 'IMAGE',
      outputCategoryMask: true,
      outputConfidenceMasks: false
    });
    postMessage({ type: 'ready' });
  } catch (e) {
    postMessage({ type: 'fail', err: String(e && e.message || e) });
  }
})();

onmessage = (ev) => {
  const bmp = ev.data && ev.data.bitmap;
  if (!segmenter || !bmp) {
    if (bmp && bmp.close) bmp.close();
    postMessage({ type: 'mask', mask: null });
    return;
  }
  try {
    segmenter.segment(bmp, (result) => {
      let mask = null, w = 0, h = 0;
      const cm = result.categoryMask;
      if (cm) {
        mask = new Uint8Array(cm.getAsUint8Array());
        w = cm.width;
        h = cm.height;
      }
      result.close();
      if (bmp.close) bmp.close();
      if (mask) postMessage({ type: 'mask', mask, w, h }, [mask.buffer]);
      else postMessage({ type: 'mask', mask: null });
    });
  } catch (e) {
    if (bmp.close) bmp.close();
    postMessage({ type: 'mask', mask: null });
  }
};
