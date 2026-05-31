import * as ort from 'onnxruntime-web';
import { getGPUDevice } from '@/lib/webgpu/context';
import type { DetectionBox } from '@/types';

const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const NMS_THRESHOLD = 0.4;
const DET_THRESHOLD = 0.5;
const INPUT_MEAN = 127.5;
const INPUT_STD = 128.0;

interface SCRFDDetection {
  x1: number; y1: number; x2: number; y2: number;
  confidence: number;
  keypoints: [number, number][];
}

function generateAnchorCenters(h: number, w: number, stride: number): Float32Array {
  const total = h * w * NUM_ANCHORS;
  const centers = new Float32Array(total * 2);
  let idx = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const cx = (x + 0.5) * stride;
      const cy = (y + 0.5) * stride;
      for (let a = 0; a < NUM_ANCHORS; a++) {
        centers[idx * 2] = cx;
        centers[idx * 2 + 1] = cy;
        idx++;
      }
    }
  }
  return centers;
}

function distance2bbox(
  centers: Float32Array,
  bboxPreds: Float32Array,
  maxW: number,
  maxH: number
): Float32Array {
  const n = centers.length / 2;
  const bboxes = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const cx = centers[i * 2];
    const cy = centers[i * 2 + 1];
    const l = bboxPreds[i * 4];
    const t = bboxPreds[i * 4 + 1];
    const r = bboxPreds[i * 4 + 2];
    const b = bboxPreds[i * 4 + 3];

    bboxes[i * 4] = Math.max(0, Math.min(cx - l, maxW));
    bboxes[i * 4 + 1] = Math.max(0, Math.min(cy - t, maxH));
    bboxes[i * 4 + 2] = Math.max(0, Math.min(cx + r, maxW));
    bboxes[i * 4 + 3] = Math.max(0, Math.min(cy + b, maxH));
  }
  return bboxes;
}

function distance2kps(
  centers: Float32Array,
  kpsPreds: Float32Array,
  maxW: number,
  maxH: number
): Float32Array {
  const n = centers.length / 2;
  const kps = new Float32Array(n * 10);
  for (let i = 0; i < n; i++) {
    const cx = centers[i * 2];
    const cy = centers[i * 2 + 1];
    for (let k = 0; k < 5; k++) {
      const kx = cx + kpsPreds[i * 10 + k * 2];
      const ky = cy + kpsPreds[i * 10 + k * 2 + 1];
      kps[i * 10 + k * 2] = Math.max(0, Math.min(kx, maxW));
      kps[i * 10 + k * 2 + 1] = Math.max(0, Math.min(ky, maxH));
    }
  }
  return kps;
}

function nms(dets: Float32Array, thresh: number): number[] {
  const n = dets.length / 5;
  if (n === 0) return [];

  const x1 = new Float32Array(n);
  const y1 = new Float32Array(n);
  const x2 = new Float32Array(n);
  const y2 = new Float32Array(n);
  const scores = new Float32Array(n);
  const areas = new Float32Array(n);
  const order = new Uint32Array(n);

  for (let i = 0; i < n; i++) {
    x1[i] = dets[i * 5];
    y1[i] = dets[i * 5 + 1];
    x2[i] = dets[i * 5 + 2];
    y2[i] = dets[i * 5 + 3];
    scores[i] = dets[i * 5 + 4];
    areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
    order[i] = i;
  }

  // Sort by score descending
  order.sort((a, b) => scores[b] - scores[a]);

  const keep: number[] = [];
  let remaining = n;

  for (let i = 0; i < n && remaining > 0; i++) {
    const idx = order[i];
    if (scores[idx] < 0) continue;
    keep.push(idx);

    const xx1 = x1[idx];
    const yy1 = y1[idx];
    const xx2 = x2[idx];
    const yy2 = y2[idx];
    const area = areas[idx];

    for (let j = i + 1; j < n; j++) {
      const oIdx = order[j];
      if (scores[oIdx] < 0) continue;

      const ox1 = Math.max(xx1, x1[oIdx]);
      const oy1 = Math.max(yy1, y1[oIdx]);
      const ox2 = Math.min(xx2, x2[oIdx]);
      const oy2 = Math.min(yy2, y2[oIdx]);

      const w = Math.max(0, ox2 - ox1 + 1);
      const h = Math.max(0, oy2 - oy1 + 1);
      const inter = w * h;
      const ovr = inter / (area + areas[oIdx] - inter);

      if (ovr > thresh) {
        scores[oIdx] = -1;
        remaining--;
      }
    }
    remaining--;
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

  // Preprocess: resize preserving ratio, pad to model input size
  const imRatio = origHeight / origWidth;
  const modelW = 640;
  const modelH = 640;
  let newW: number, newH: number;
  if (imRatio > 1) {
    newH = modelH;
    newW = Math.round(newH / imRatio);
  } else {
    newW = modelW;
    newH = Math.round(newW * imRatio);
  }
  const detScale = newH / origHeight;

  // Create padded input image
  const canvas = new OffscreenCanvas(modelW, modelH);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgb(0, 0, 0)';
  ctx.fillRect(0, 0, modelW, modelH);

  const src = new OffscreenCanvas(origWidth, origHeight);
  src.getContext('2d')!.putImageData(imageData, 0, 0);
  ctx.drawImage(src, 0, 0, newW, newH);

  const inputImageData = ctx.getImageData(0, 0, modelW, modelH);

  // BGR + normalize: (pixel - 127.5) / 128.0 → CHW
  const chw = new Float32Array(3 * modelW * modelH);
  const planeSize = modelW * modelH;
  for (let i = 0; i < planeSize; i++) {
    chw[2 * planeSize + i] = (inputImageData.data[i * 4] - INPUT_MEAN) / INPUT_STD;     // R→B
    chw[planeSize + i] = (inputImageData.data[i * 4 + 1] - INPUT_MEAN) / INPUT_STD;      // G→G
    chw[i] = (inputImageData.data[i * 4 + 2] - INPUT_MEAN) / INPUT_STD;                  // B→R
  }

  const tensor = new ort.Tensor('float32', chw, [1, 3, modelH, modelW]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);

  // Collect detections across all stride levels
  const allDets: number[] = [];
  const allKps: number[] = [];
  const fmc = STRIDES.length;

  for (let idx = 0; idx < fmc; idx++) {
    const stride = STRIDES[idx];

    const scoreOut = results[session.outputNames[idx]];
    const bboxOut = results[session.outputNames[idx + fmc]];
    const kpsOut = results[session.outputNames[idx + fmc * 2]];

    const scoreData = (await scoreOut.getData()) as Float32Array;
    let scores: Float32Array;
    if (scoreOut.dims.length === 4) {
      scores = new Float32Array(scoreData.buffer, scoreData.byteOffset, scoreOut.dims[1] * scoreOut.dims[2] * scoreOut.dims[3]);
    } else {
      scores = scoreData;
    }

    const bboxData = (await bboxOut.getData()) as Float32Array;
    let bboxPreds: Float32Array;
    if (bboxOut.dims.length === 4) {
      bboxPreds = new Float32Array(bboxData.buffer, bboxData.byteOffset, bboxOut.dims[1] * bboxOut.dims[2] * bboxOut.dims[3]);
    } else {
      bboxPreds = bboxData;
    }

    const kpsData = (await kpsOut.getData()) as Float32Array;
    let kpsPreds: Float32Array;
    if (kpsOut.dims.length === 4) {
      kpsPreds = new Float32Array(kpsData.buffer, kpsData.byteOffset, kpsOut.dims[1] * kpsOut.dims[2] * kpsOut.dims[3]);
    } else {
      kpsPreds = kpsData;
    }

    const h = Math.floor(modelH / stride);
    const w = Math.floor(modelW / stride);
    const centers = generateAnchorCenters(h, w, stride);

    // Reshape predictions: (2, H, W) → (H*W*2,) and (8, H, W) → (H*W*2, 4)
    const n = h * w * NUM_ANCHORS;

    // Flatten scores and filter
    const flatScores = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      flatScores[i] = scores[i];
    }

    // Reshape bbox preds: (8, H, W) → (H*W*2, 4)
    const bboxFlat = new Float32Array(n * 4);
    for (let i = 0; i < h * w; i++) {
      const row = Math.floor(i / w);
      const col = i % w;
      for (let a = 0; a < NUM_ANCHORS; a++) {
        const dst = (i * NUM_ANCHORS + a) * 4;
        for (let c = 0; c < 4; c++) {
          bboxFlat[dst + c] = bboxPreds[(a * 4 + c) * h * w + row * w + col] * stride;
        }
      }
    }

    // Reshape kps preds: (20, H, W) → (H*W*2, 10)
    const kpsFlat = new Float32Array(n * 10);
    for (let i = 0; i < h * w; i++) {
      const row = Math.floor(i / w);
      const col = i % w;
      for (let a = 0; a < NUM_ANCHORS; a++) {
        const dst = (i * NUM_ANCHORS + a) * 10;
        for (let c = 0; c < 10; c++) {
          kpsFlat[dst + c] = kpsPreds[(a * 10 + c) * h * w + row * w + col] * stride;
        }
      }
    }

    const bboxes = distance2bbox(centers, bboxFlat, modelW, modelH);
    const kpss = distance2kps(centers, kpsFlat, modelW, modelH);

    for (let i = 0; i < n; i++) {
      if (flatScores[i] >= DET_THRESHOLD) {
        allDets.push(
          bboxes[i * 4] / detScale,
          bboxes[i * 4 + 1] / detScale,
          bboxes[i * 4 + 2] / detScale,
          bboxes[i * 4 + 3] / detScale,
          flatScores[i],
        );
        for (let k = 0; k < 10; k++) {
          allKps.push(kpss[i * 10 + k] / detScale);
        }
      }
    }

    scoreOut.dispose();
    bboxOut.dispose();
    kpsOut.dispose();
  }

  if (allDets.length === 0) return [];

  // Sort by score and apply NMS
  const detArr = new Float32Array(allDets);
  const sortOrder = Array.from({ length: detArr.length / 5 }, (_, i) => i);
  sortOrder.sort((a, b) => detArr[b * 5 + 4] - detArr[a * 5 + 4]);

  const sorted = new Float32Array(allDets.length);
  for (let i = 0; i < sortOrder.length; i++) {
    const src = sortOrder[i] * 5;
    const dst = i * 5;
    sorted[dst] = detArr[src];
    sorted[dst + 1] = detArr[src + 1];
    sorted[dst + 2] = detArr[src + 2];
    sorted[dst + 3] = detArr[src + 3];
    sorted[dst + 4] = detArr[src + 4];
  }

  const keep = nms(sorted, NMS_THRESHOLD);

  return keep.map((idx) => ({
    x1: sorted[idx * 5],
    y1: sorted[idx * 5 + 1],
    x2: sorted[idx * 5 + 2],
    y2: sorted[idx * 5 + 3],
    confidence: sorted[idx * 5 + 4],
  }));
}
