import * as ort from 'onnxruntime-web';
import type { DetectionBox } from '@/types';
import { YUNET_INPUT_SIZE, YUNET_CONF_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K } from '@/lib/constants';
import { getGPUDevice } from '@/lib/webgpu/context';

const STRIDES = [8, 16, 32];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
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

async function preprocessCPU(imageData: ImageData): Promise<Float32Array> {
  const mw = YUNET_INPUT_SIZE;
  const planeSize = mw * mw;
  const chw = new Float32Array(3 * planeSize);

  // RGB channel order, raw pixel values (matching OpenCV blobFromImage with scale=1.0, mean=Scalar(), swapRB=true)
  for (let i = 0; i < planeSize; i++) {
    chw[i] = imageData.data[i * 4];                     // R
    chw[planeSize + i] = imageData.data[i * 4 + 1];     // G
    chw[2 * planeSize + i] = imageData.data[i * 4 + 2]; // B
  }

  return chw;
}

async function preprocessGPU(imageData: ImageData): Promise<Float32Array> {
  const { preprocessCHW } = await import('@/lib/webgpu/preprocess');
  return preprocessCHW(imageData, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE);
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

  // Direct resize to 640×640 (matching OpenCV FaceDetectorYN resize behavior)
  const src = new OffscreenCanvas(origWidth, origHeight);
  src.getContext('2d')!.putImageData(imageData, 0, 0);
  const dst = new OffscreenCanvas(modelW, modelH);
  const dstCtx = dst.getContext('2d', { willReadFrequently: true })!;
  dstCtx.drawImage(src, 0, 0, modelW, modelH);
  const resized = dstCtx.getImageData(0, 0, modelW, modelH);

  const detScale = modelH / origHeight;

  // Preprocess: RGB CHW with raw pixel values
  let chw: Float32Array;
  const gpuDev = await getGPUDevice();
  if (gpuDev) {
    try {
      chw = await preprocessGPU(resized);
    } catch {
      chw = await preprocessCPU(resized);
    }
  } else {
    chw = await preprocessCPU(resized);
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

    const clsData = (await clsOut.getData()) as Float32Array;
    const objData = (await objOut.getData()) as Float32Array;
    const bboxData = (await bboxOut.getData()) as Float32Array;

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
        Math.max(0, Math.min(origWidth, x1)),
        Math.max(0, Math.min(origHeight, y1)),
        Math.max(0, Math.min(origWidth, x2)),
        Math.max(0, Math.min(origHeight, y2)),
        score,
      );
    }

    clsOut.dispose();
    objOut.dispose();
    bboxOut.dispose();
  }

  if (allDets.length === 0) return [];

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

  const keep = nms(sorted, YUNET_NMS_THRESHOLD);

  const result = keep.map((i) => ({
    x1: sorted[i * 5],
    y1: sorted[i * 5 + 1],
    x2: sorted[i * 5 + 2],
    y2: sorted[i * 5 + 3],
    confidence: sorted[i * 5 + 4],
  }));

  return result.length > YUNET_TOP_K ? result.slice(0, YUNET_TOP_K) : result;
}
