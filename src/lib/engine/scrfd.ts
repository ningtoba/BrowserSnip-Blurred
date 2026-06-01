import * as ort from 'onnxruntime-web';
import type { DetectionBox } from '@/types';

const SCRFD_INPUT_SIZE = 640;
const STRIDES = [8, 16, 32];
const SCORE_THRESH = 0.6;
const NMS_THRESH = 0.3;

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
    chw[i] = resized.data[i * 4];
    chw[planeSize + i] = resized.data[i * 4 + 1];
    chw[2 * planeSize + i] = resized.data[i * 4 + 2];
  }

  const tensor = new ort.Tensor('float32', chw, [1, 3, modelH, modelW]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);

  const allDets: { x1: number; y1: number; x2: number; y2: number; score: number }[] = [];

  for (let li = 0; li < STRIDES.length; li++) {
    const stride = STRIDES[li];
    const featH = Math.floor(modelH / stride);
    const featW = Math.floor(modelW / stride);
    const numPositions = featH * featW;

    const scoreTensor = results[`score_${stride}`];
    const bboxTensor = results[`bbox_${stride}`];
    if (!scoreTensor || !bboxTensor) {
      console.warn(`[SCRFD] missing outputs for stride ${stride}`);
      continue;
    }

    const scoreData = (await scoreTensor.getData()) as Float32Array;
    const bboxData = (await bboxTensor.getData()) as Float32Array;
    const totalAnchors = scoreData.length;

    // Flat indexing: iterate all anchors, derive position from index
    for (let idx = 0; idx < totalAnchors; idx++) {
      const score = scoreData[idx]; // model outputs probabilities directly
      if (score < SCORE_THRESH) continue;

      // Derive grid position from flat index
      // Try: idx = pos * numAnchors + anchorIdx (interleaved)
      // The position in the feature map determines the anchor center
      const pos = Math.floor(idx / Math.ceil(totalAnchors / numPositions));
      const col = pos % featW;
      const row = Math.floor(pos / featW);
      const cx = col * stride;
      const cy = row * stride;

      // SCRFD bbox: distances from anchor center to [left, top, right, bottom]
      const dl = bboxData[idx * 4] * stride;
      const dt = bboxData[idx * 4 + 1] * stride;
      const dr = bboxData[idx * 4 + 2] * stride;
      const db = bboxData[idx * 4 + 3] * stride;

      const x1 = Math.max(0, (cx - dl) * scaleX);
      const y1 = Math.max(0, (cy - dt) * scaleY);
      const x2 = Math.min(origWidth, (cx + dr) * scaleX);
      const y2 = Math.min(origHeight, (cy + db) * scaleY);

      if (x2 > x1 && y2 > y1 && x2 - x1 > 5 && y2 - y1 > 5) {
        allDets.push({ x1, y1, x2, y2, score });
      }
    }

    scoreTensor.dispose();
    bboxTensor.dispose();
    if (results[`kps_${stride}`]) results[`kps_${stride}`].dispose();
  }

  const kept = nms(allDets, NMS_THRESH);

  return kept
    .filter((d) => {
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;
      // Minimum face size: reject tiny detections (likely false positives)
      if (w < 30 || h < 30) return false;
      // Aspect ratio: faces are roughly square
      const aspect = w / h;
      return aspect > 0.5 && aspect < 2.0;
    })
    .map((d) => ({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, confidence: d.score }));
}
