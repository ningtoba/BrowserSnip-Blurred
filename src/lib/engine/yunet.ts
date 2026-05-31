import * as ort from 'onnxruntime-web';
import type { DetectionBox } from '@/types';
import { YUNET_INPUT_SIZE, YUNET_CONF_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K } from '@/lib/constants';

const STRIDES = [8, 16, 32];
const INPUT_MEAN = 127.5;
const INPUT_STD = 128.0;

function generateGridCenters(h: number, w: number, stride: number): Float32Array {
  const n = h * w;
  const centers = new Float32Array(n * 2);
  let idx = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      centers[idx * 2] = x;
      centers[idx * 2 + 1] = y;
      idx++;
    }
  }
  return centers;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function nmsYunet(dets: Float32Array, thresh: number): number[] {
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

export async function detectFacesYuNet(
  imageData: ImageData,
  origWidth: number,
  origHeight: number
): Promise<DetectionBox[]> {
  const session = (await import('@/lib/engine/session')).getSession('yunet');
  if (!session) throw new Error('YuNet session not initialized');

  const modelW = YUNET_INPUT_SIZE;
  const modelH = YUNET_INPUT_SIZE;

  // Letterbox: resize preserving aspect ratio, pad to model input size
  const imRatio = origHeight / origWidth;
  let newW: number, newH: number;
  if (imRatio > 1) {
    newH = modelH;
    newW = Math.round(newH / imRatio);
  } else {
    newW = modelW;
    newH = Math.round(newW * imRatio);
  }
  const detScale = newH / origHeight;

  // Create padded canvas
  const canvas = new OffscreenCanvas(modelW, modelH);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgb(0, 0, 0)';
  ctx.fillRect(0, 0, modelW, modelH);

  const src = new OffscreenCanvas(origWidth, origHeight);
  src.getContext('2d')!.putImageData(imageData, 0, 0);
  ctx.drawImage(src, 0, 0, newW, newH);

  const inputImageData = ctx.getImageData(0, 0, modelW, modelH);

  // BGR normalize + CHW: (pixel - 127.5) / 128.0
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

  const fmc = STRIDES.length;
  const allDets: number[] = [];

  for (let idx = 0; idx < fmc; idx++) {
    const stride = STRIDES[idx];

    const clsOut = results[session.outputNames[idx]];
    const objOut = results[session.outputNames[idx + fmc]];
    const bboxOut = results[session.outputNames[idx + fmc * 2]];
    const kpsOut = results[session.outputNames[idx + fmc * 3]];

    const clsData = (await clsOut.getData()) as Float32Array;
    const objData = (await objOut.getData()) as Float32Array;
    const bboxData = (await bboxOut.getData()) as Float32Array;
    const kpsData = (await kpsOut.getData()) as Float32Array;

    const h = Math.floor(modelH / stride);
    const w = Math.floor(modelW / stride);
    const n = h * w;

    for (let i = 0; i < n; i++) {
      const cls = clamp(clsData[i], 0, 1);
      const obj = clamp(objData[i], 0, 1);
      const score = Math.sqrt(cls * obj);

      if (score < YUNET_CONF_THRESHOLD) continue;

      const col = i % w;
      const row = Math.floor(i / w);

      // Decode bbox: center + log_size → (x1, y1, w, h)
      const dx = bboxData[i * 4];
      const dy = bboxData[i * 4 + 1];
      const dw = bboxData[i * 4 + 2];
      const dh = bboxData[i * 4 + 3];

      const cx = (col + dx) * stride;
      const cy = (row + dy) * stride;
      const bw = Math.exp(dw) * stride;
      const bh = Math.exp(dh) * stride;

      const x1 = (cx - bw / 2) / detScale;
      const y1 = (cy - bh / 2) / detScale;
      const x2 = (cx + bw / 2) / detScale;
      const y2 = (cy + bh / 2) / detScale;

      allDets.push(
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(origWidth, x2),
        Math.min(origHeight, y2),
        score,
      );
    }

    clsOut.dispose();
    objOut.dispose();
    bboxOut.dispose();
    kpsOut.dispose();
  }

  if (allDets.length === 0) return [];

  // Sort by score descending
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

  const keep = nmsYunet(sorted, YUNET_NMS_THRESHOLD);

  const result = keep.map((i) => ({
    x1: sorted[i * 5],
    y1: sorted[i * 5 + 1],
    x2: sorted[i * 5 + 2],
    y2: sorted[i * 5 + 3],
    confidence: sorted[i * 5 + 4],
  }));

  // Limit to top K
  if (result.length > YUNET_TOP_K) {
    return result.slice(0, YUNET_TOP_K);
  }

  return result;
}
