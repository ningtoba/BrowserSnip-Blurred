import { imageDataToRGB, rgbToFloat32CHW, letterboxImageData } from '@/lib/utils/image';
import { runYOLO } from '@/lib/engine/session';
import { YOLO_INPUT_SIZE, MAX_DETECTIONS_PER_FRAME } from '@/lib/constants';
import type { DetectionBox } from '@/types';

export function preprocessYOLO(
  imageData: ImageData
): { tensor: Float32Array; scale: number; padLeft: number; padTop: number } {
  const { data: letterboxed, scale, padLeft, padTop } = letterboxImageData(
    imageData,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    114,
    114,
    114
  );

  const rgb = imageDataToRGB(letterboxed);
  const tensor = rgbToFloat32CHW(rgb, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

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
  const { tensor, scale, padLeft, padTop } = preprocessYOLO(imageData);
  const boxes = await runYOLO(tensor);
  return postprocessYOLO(boxes, origWidth, origHeight, scale, padLeft, padTop);
}
