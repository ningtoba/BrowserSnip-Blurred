import * as ort from 'onnxruntime-web';
import { runYOLO } from '@/lib/engine/session';
import { getGPUDevice } from '@/lib/webgpu/context';
import { YOLO_INPUT_SIZE, MAX_DETECTIONS_PER_FRAME } from '@/lib/constants';
import type { DetectionBox } from '@/types';

async function preprocessYOLO_GPU(imageData: ImageData) {
  const { preprocessYOLO } = await import('@/lib/webgpu/preprocess');
  return preprocessYOLO(imageData);
}

async function preprocessYOLO_CPU(
  imageData: ImageData
): Promise<{ tensor: ort.Tensor; scale: number; padLeft: number; padTop: number }> {
  const sw = imageData.width;
  const sh = imageData.height;
  const targetW = YOLO_INPUT_SIZE;
  const targetH = YOLO_INPUT_SIZE;
  const scale = Math.min(targetW / sw, targetH / sh);
  const newW = Math.round(sw * scale);
  const newH = Math.round(sh * scale);
  const padLeft = Math.floor((targetW - newW) / 2);
  const padTop = Math.floor((targetH - newH) / 2);

  const canvas = new OffscreenCanvas(targetW, targetH);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetW, targetH);
  const src = new OffscreenCanvas(sw, sh);
  src.getContext('2d')!.putImageData(imageData, 0, 0);
  ctx.drawImage(src, padLeft, padTop, newW, newH);

  const bitmap = await createImageBitmap(canvas);
  const tensor = await ort.Tensor.fromImage(bitmap, {
    dataType: 'float32',
    tensorFormat: 'RGB',
    tensorLayout: 'NCHW',
    norm: { bias: 0, mean: 255 },
  });
  bitmap.close();

  return { tensor, scale, padLeft, padTop };
}

export function postprocessYOLO(
  boxes: DetectionBox[],
  origWidth: number,
  origHeight: number,
  scale: number,
  padLeft: number,
  padTop: number
): DetectionBox[] {
  const result: DetectionBox[] = [];
  for (const box of boxes) {
    let { x1, y1, x2, y2 } = box;
    x1 = (x1 - padLeft) / scale;
    y1 = (y1 - padTop) / scale;
    x2 = (x2 - padLeft) / scale;
    y2 = (y2 - padTop) / scale;
    x1 = Math.max(0, Math.min(x1, origWidth));
    y1 = Math.max(0, Math.min(y1, origHeight));
    x2 = Math.max(0, Math.min(x2, origWidth));
    y2 = Math.max(0, Math.min(y2, origHeight));
    result.push({ x1, y1, x2, y2, confidence: box.confidence });
  }
  return result.slice(0, MAX_DETECTIONS_PER_FRAME);
}

export async function detectFaces(
  imageData: ImageData,
  origWidth: number,
  origHeight: number
): Promise<DetectionBox[]> {
  // Try GPU preprocessing first (zero-copy fromGpuBuffer), fall back to CPU
  let tensor: ort.Tensor;
  let scale: number;
  let padLeft: number;
  let padTop: number;

  const gpuDev = await getGPUDevice();
  if (gpuDev) {
    try {
      const result = await preprocessYOLO_GPU(imageData);
      tensor = result.tensor;
      scale = result.scale;
      padLeft = result.padLeft;
      padTop = result.padTop;
    } catch {
      const result = await preprocessYOLO_CPU(imageData);
      tensor = result.tensor;
      scale = result.scale;
      padLeft = result.padLeft;
      padTop = result.padTop;
    }
  } else {
    const result = await preprocessYOLO_CPU(imageData);
    tensor = result.tensor;
    scale = result.scale;
    padLeft = result.padLeft;
    padTop = result.padTop;
  }

  const boxes = await runYOLO(tensor);
  tensor.dispose();
  return postprocessYOLO(boxes, origWidth, origHeight, scale, padLeft, padTop);
}
