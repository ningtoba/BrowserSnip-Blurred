import * as ort from 'onnxruntime-web';
import type { DetectionBox } from '@/types';

const SCRFD_INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const SCORE_THRESH = 0.9;
const NMS_THRESH = 0.4;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function nms(dets: { x1: number; y1: number; x2: number; y2: number; score: number }[], thresh: number): typeof dets {
  if (dets.length === 0) return [];
  dets.sort((a, b) => b.score - a.score);

  const keep: typeof dets = [];
  const suppressed = new Uint8Array(dets.length);

  for (let i = 0; i < dets.length; i++) {
    if (suppressed[i]) continue;
    keep.push(dets[i]);
    const a = dets[i];
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);

    for (let j = i + 1; j < dets.length; j++) {
      if (suppressed[j]) continue;
      const b = dets[j];
      const xx1 = Math.max(a.x1, b.x1);
      const yy1 = Math.max(a.y1, b.y1);
      const xx2 = Math.min(a.x2, b.x2);
      const yy2 = Math.min(a.y2, b.y2);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      const iou = inter / (areaA + areaB - inter);
      if (iou > thresh) suppressed[j] = 1;
    }
  }
  return keep;
}

export async function detectFacesSCRFD(
  imageData: ImageData,
  origWidth: number,
  origHeight: number
): Promise<DetectionBox[]> {
  const session = (await import('@/lib/engine/session')).getSession('scrfd');
  if (!session) throw new Error('SCRFD session not initialized');

  const modelW = SCRFD_INPUT_SIZE;
  const modelH = SCRFD_INPUT_SIZE;
  const scaleX = origWidth / modelW;
  const scaleY = origHeight / modelH;

  // Resize to 640×640 and convert to CHW
  const src = new OffscreenCanvas(origWidth, origHeight);
  src.getContext('2d')!.putImageData(imageData, 0, 0);
  const dst = new OffscreenCanvas(modelW, modelH);
  const dstCtx = dst.getContext('2d', { willReadFrequently: true })!;
  dstCtx.drawImage(src, 0, 0, modelW, modelH);
  const resized = dstCtx.getImageData(0, 0, modelW, modelH);

  // CHW format, RGB, raw pixel values [0, 255]
  const planeSize = modelW * modelH;
  const chw = new Float32Array(3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    chw[i] = resized.data[i * 4];                     // R
    chw[planeSize + i] = resized.data[i * 4 + 1];     // G
    chw[2 * planeSize + i] = resized.data[i * 4 + 2]; // B
  }

  const tensor = new ort.Tensor('float32', chw, [1, 3, modelH, modelW]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);

  console.debug('[SCRFD] output names:', Object.keys(results));

  const allDets: { x1: number; y1: number; x2: number; y2: number; score: number }[] = [];

  for (let li = 0; li < STRIDES.length; li++) {
    const stride = STRIDES[li];
    const featH = Math.floor(modelH / stride);
    const featW = Math.floor(modelW / stride);
    const numPositions = featH * featW;

    const scoreKey = `score_${stride}`;
    const bboxKey = `bbox_${stride}`;
    const scoreTensor = results[scoreKey];
    const bboxTensor = results[bboxKey];
    if (!scoreTensor || !bboxTensor) {
      console.warn(`[SCRFD] missing output: ${scoreKey} or ${bboxKey}`);
      continue;
    }

    const scoreData = (await scoreTensor.getData()) as Float32Array;
    const bboxData = (await bboxTensor.getData()) as Float32Array;

    console.debug(`[SCRFD] stride ${stride}: score shape=[${scoreTensor.dims}], bbox shape=[${bboxTensor.dims}], numPositions=${numPositions}`);

    // SCRFD anchor layout: all anchor-0 positions, then all anchor-1 positions
    // Each anchor center = (col * stride, row * stride) in model space
    for (let a = 0; a < NUM_ANCHORS; a++) {
      const anchorOffset = a * numPositions;
      for (let pos = 0; pos < numPositions; pos++) {
        const idx = anchorOffset + pos;
        const score = sigmoid(scoreData[idx]);
        if (score < SCORE_THRESH) continue;

        const col = pos % featW;
        const row = Math.floor(pos / featW);
        const cx = col * stride;
        const cy = row * stride;

        // SCRFD bbox: distances from anchor center to [left, top, right, bottom]
        // NOT SSD-style (cx, cy, w, h with exp). These are direct pixel distances.
        const dl = bboxData[idx * 4] * stride;
        const dt = bboxData[idx * 4 + 1] * stride;
        const dr = bboxData[idx * 4 + 2] * stride;
        const db = bboxData[idx * 4 + 3] * stride;

        const x1 = Math.max(0, (cx - dl) * scaleX);
        const y1 = Math.max(0, (cy - dt) * scaleY);
        const x2 = Math.min(origWidth, (cx + dr) * scaleX);
        const y2 = Math.min(origHeight, (cy + db) * scaleY);

        if (x2 > x1 && y2 > y1) {
          allDets.push({ x1, y1, x2, y2, score });
        }
      }
    }

    scoreTensor.dispose();
    bboxTensor.dispose();
    if (results[`kps_${stride}`]) results[`kps_${stride}`].dispose();
  }

  // NMS
  console.debug('[SCRFD] pre-NMS:', allDets.length, 'detections');
  if (allDets.length > 0) {
    const sample = allDets[0];
    console.debug('[SCRFD] sample det:', {
      x1: sample.x1.toFixed(1), y1: sample.y1.toFixed(1),
      x2: sample.x2.toFixed(1), y2: sample.y2.toFixed(1),
      score: sample.score.toFixed(3),
      w: (sample.x2 - sample.x1).toFixed(1),
      h: (sample.y2 - sample.y1).toFixed(1),
    });
  }

  const kept = nms(allDets, NMS_THRESH);
  console.debug('[SCRFD] post-NMS:', kept.length);

  // Filter: aspect ratio (faces are roughly square) and minimum size
  return kept
    .filter((d) => {
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;
      if (w < 10 || h < 10) return false;
      const aspect = w / h;
      return aspect > 0.4 && aspect < 2.5;
    })
    .map((d) => ({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, confidence: d.score }));
}
