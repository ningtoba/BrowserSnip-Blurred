import * as ort from 'onnxruntime-web';
import {
  imageDataToRGB,
  bgrToFloat32CHW,
  cropImageData,
  resizeImageData,
} from '@/lib/utils/image';
import { runMFN } from '@/lib/engine/session';
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

  const planeSize = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    dst[offset + i] = (rgb[i * 3] - MFN_MEAN[0]) / MFN_STD[0];
    dst[offset + planeSize + i] = (rgb[i * 3 + 1] - MFN_MEAN[1]) / MFN_STD[1];
    dst[offset + 2 * planeSize + i] = (rgb[i * 3 + 2] - MFN_MEAN[2]) / MFN_STD[2];
  }
  return true;
}

/**
 * Batch face recognition: preprocess all face crops, then run MFN inference
 * on each (MFN expects batch size 1, so we loop but preprocess is batched).
 */
export async function recognizeFacesBatch(
  faces: { frameData: ImageData; bbox: DetectionBox; frameW: number; frameH: number }[]
): Promise<(Float32Array | null)[]> {
  if (faces.length === 0) return [];

  const planeSize = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
  const tensorSize = 3 * planeSize;

  // Preprocess all faces into individual tensors
  const preprocessed: { data: Float32Array; index: number }[] = [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const data = new Float32Array(tensorSize);
    const ok = preprocessOneFace(f.frameData, f.bbox, f.frameW, f.frameH, data, 0);
    if (ok) preprocessed.push({ data, index: i });
  }

  // Run MFN inference on each face (batch size 1)
  const results: (Float32Array | null)[] = new Array(faces.length).fill(null);
  for (const { data, index } of preprocessed) {
    const embedding = await runMFN(data);
    results[index] = l2Normalize(new Float32Array(embedding));
  }

  return results;
}

export async function recognizeFace(
  frameImageData: ImageData,
  bbox: DetectionBox,
  frameW: number,
  frameH: number
): Promise<Float32Array | null> {
  const expanded = expandBbox(bbox, frameW, frameH);
  const w = expanded.x2 - expanded.x1;
  const h = expanded.y2 - expanded.y1;
  if (w < 10 || h < 10) return null;

  const cropped = cropImageData(frameImageData, expanded.x1, expanded.y1, w, h);
  const resized = resizeImageData(cropped, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
  const rgb = imageDataToRGB(resized);
  const data = bgrToFloat32CHW(rgb, FACE_INPUT_SIZE, FACE_INPUT_SIZE, MFN_MEAN, MFN_STD);

  const embedding = await runMFN(data);
  return l2Normalize(embedding);
}
