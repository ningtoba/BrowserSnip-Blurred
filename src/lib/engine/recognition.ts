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

async function preprocessMFN_GPU(cropImageData: ImageData): Promise<ort.Tensor> {
  const { preprocessMFN } = await import('@/lib/webgpu/preprocess');
  return preprocessMFN(cropImageData);
}

async function preprocessMFN_CPU(cropImageData: ImageData): Promise<ort.Tensor> {
  const resized = resizeImageData(cropImageData, FACE_INPUT_SIZE, FACE_INPUT_SIZE);
  const rgb = imageDataToRGB(resized);
  const data = bgrToFloat32CHW(rgb, FACE_INPUT_SIZE, FACE_INPUT_SIZE, MFN_MEAN, MFN_STD);
  return new ort.Tensor('float32', data, [1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE]);
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

    let tensor: ort.Tensor;
    const gpuDev = await getGPUDevice();
    if (gpuDev) {
      try {
        tensor = await preprocessMFN_GPU(cropped);
      } catch {
        tensor = await preprocessMFN_CPU(cropped);
      }
    } else {
      tensor = await preprocessMFN_CPU(cropped);
    }

    const embedding = await runMFN(tensor);
    tensor.dispose();
    return l2Normalize(embedding);
  } catch {
    return null;
  }
}
