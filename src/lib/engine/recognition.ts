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

export function preprocessMFN(cropImageData: ImageData): Float32Array {
  const resized = resizeImageData(cropImageData, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
  const rgb = imageDataToRGB(resized);
  return bgrToFloat32CHW(rgb, FACE_INPUT_SIZE, FACE_INPUT_SIZE, MFN_MEAN, MFN_STD);
}

export async function recognizeFace(
  frameImageData: ImageData,
  bbox: DetectionBox,
  frameW: number,
  frameH: number
): Promise<Float32Array | null> {
  try {
    const expanded = expandBbox(bbox, frameW, frameH);
    const w = expanded.x2 - expanded.x1;
    const h = expanded.y2 - expanded.y1;

    if (w < 10 || h < 10) return null;

    const cropped = cropImageData(
      frameImageData,
      expanded.x1,
      expanded.y1,
      w,
      h
    );

    const tensor = preprocessMFN(cropped);
    const embedding = await runMFN(tensor);
    return l2Normalize(embedding);
  } catch {
    return null;
  }
}
