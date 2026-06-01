import * as ort from 'onnxruntime-web';
import {
  imageDataToRGB,
  bgrToFloat32CHW,
  cropImageData,
  resizeImageData,
} from '@/lib/utils/image';
import { runMFN } from '@/lib/engine/session';
import { getGPUDevice } from '@/lib/webgpu/context';
import { l2Normalize } from '@/lib/utils/math';
import { FACE_INPUT_SIZE, FACE_EXPAND_RATIO } from '@/lib/constants';
import type { DetectionBox } from '@/types';

const MFN_MEAN = [127.5, 127.5, 127.5];
const MFN_STD = [127.5, 127.5, 127.5];

export function expandBbox(
  bbox: DetectionBox,
  frameW: number,
  frameH: number
): DetectionBox {
  const w = bbox.x2 - bbox.x1;
  const h = bbox.y2 - bbox.y1;
  const expandW = w * FACE_EXPAND_RATIO;
  const expandH = h * FACE_EXPAND_RATIO;

  return {
    x1: Math.max(0, Math.round(bbox.x1 - expandW)),
    y1: Math.max(0, Math.round(bbox.y1 - expandH)),
    x2: Math.min(frameW, Math.round(bbox.x2 + expandW)),
    y2: Math.min(frameH, Math.round(bbox.y2 + expandH)),
    confidence: bbox.confidence,
  };
}

function preprocessOneFace(
  frameImageData: ImageData,
  bbox: DetectionBox,
  frameW: number,
  frameH: number,
  dst: Float32Array,
  offset: number
): boolean {
  const expanded = expandBbox(bbox, frameW, frameH);
  const w = expanded.x2 - expanded.x1;
  const h = expanded.y2 - expanded.y1;
  if (w < 10 || h < 10) return false;

  const cropped = cropImageData(frameImageData, expanded.x1, expanded.y1, w, h);
  const resized = resizeImageData(cropped, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
  const rgb = imageDataToRGB(resized);

  // Normalize and write directly into the batch tensor at the given offset
  const planeSize = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    dst[offset + i] = (rgb[i * 3] - MFN_MEAN[0]) / MFN_STD[0];
    dst[offset + planeSize + i] = (rgb[i * 3 + 1] - MFN_MEAN[1]) / MFN_STD[1];
    dst[offset + 2 * planeSize + i] = (rgb[i * 3 + 2] - MFN_MEAN[2]) / MFN_STD[2];
  }
  return true;
}

/**
 * Batch face recognition: preprocess all face crops into a single tensor,
 * run one MFN inference, split the output into individual embeddings.
 * This is dramatically faster than calling recognizeFace() per face.
 */
export async function recognizeFacesBatch(
  faces: { frameData: ImageData; bbox: DetectionBox; frameW: number; frameH: number }[]
): Promise<(Float32Array | null)[]> {
  if (faces.length === 0) return [];

  const entry = (await import('@/lib/engine/session')).getSession('mfn');
  if (!entry) throw new Error('MFN session not initialized');

  const planeSize = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
  const batchSize = faces.length;
  const batchData = new Float32Array(batchSize * 3 * planeSize);

  // Preprocess all faces into the batch tensor
  const validMask: boolean[] = [];
  for (let i = 0; i < batchSize; i++) {
    const f = faces[i];
    const ok = preprocessOneFace(f.frameData, f.bbox, f.frameW, f.frameH, batchData, i * 3 * planeSize);
    validMask.push(ok);
  }

  // Run single batched inference
  const tensor = new ort.Tensor('float32', batchData, [batchSize, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[entry.inputNames[0]] = tensor;
  const results = await entry.run(feeds);
  const output = results[entry.outputNames[0]];
  const outputData = (await output.getData()) as Float32Array;
  tensor.dispose();
  output.dispose();

  // Split output into individual embeddings
  const embedDim = outputData.length / batchSize;
  const embeddings: (Float32Array | null)[] = [];
  for (let i = 0; i < batchSize; i++) {
    if (!validMask[i]) {
      embeddings.push(null);
    } else {
      const emb = new Float32Array(embedDim);
      emb.set(outputData.subarray(i * embedDim, (i + 1) * embedDim));
      embeddings.push(l2Normalize(emb));
    }
  }

  return embeddings;
}

export async function recognizeFace(
  frameImageData: ImageData,
  bbox: DetectionBox,
  frameW: number,
  frameH: number
): Promise<Float32Array | null> {
  const results = await recognizeFacesBatch([{ frameData: frameImageData, bbox, frameW, frameH }]);
  return results[0];
}
