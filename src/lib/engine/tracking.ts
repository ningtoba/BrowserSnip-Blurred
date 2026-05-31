import type { DetectionBox, FaceDetection } from '@/types';

export function computeIOU(a: DetectionBox, b: DetectionBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  if (x2 <= x1 || y2 <= y1) return 0;

  const inter = (x2 - x1) * (y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

export function matchByIOU(
  prev: DetectionBox[],
  curr: DetectionBox[],
  iouThreshold: number = 0.3
): Map<number, number> {
  const matches = new Map<number, number>();
  const used = new Set<number>();

  for (let i = 0; i < prev.length; i++) {
    let bestIOU = iouThreshold;
    let bestJ = -1;
    for (let j = 0; j < curr.length; j++) {
      if (used.has(j)) continue;
      const iou = computeIOU(prev[i], curr[j]);
      if (iou > bestIOU) {
        bestIOU = iou;
        bestJ = j;
      }
    }
    if (bestJ >= 0) {
      matches.set(i, bestJ);
      used.add(bestJ);
    }
  }

  return matches;
}

export function interpolateBbox(
  a: DetectionBox,
  b: DetectionBox,
  t: number
): DetectionBox {
  return {
    x1: a.x1 + (b.x1 - a.x1) * t,
    y1: a.y1 + (b.y1 - a.y1) * t,
    x2: a.x2 + (b.x2 - a.x2) * t,
    y2: a.y2 + (b.y2 - a.y2) * t,
    confidence: a.confidence + (b.confidence - a.confidence) * t,
  };
}

export function interpolateDetections(
  prevFrame: DetectionBox[],
  nextFrame: DetectionBox[],
  prevDetections: FaceDetection[],
  nextDetections: FaceDetection[],
  currentIndex: number,
  prevIndex: number,
  nextIndex: number
): { boxes: DetectionBox[]; detections: FaceDetection[] } {
  const t = (currentIndex - prevIndex) / (nextIndex - prevIndex);

  // Match prev to next via IOU to maintain identity consistency
  const matchMap = matchByIOU(prevFrame, nextFrame, 0.3);

  const boxes: DetectionBox[] = [];
  const detections: FaceDetection[] = [];

  for (const [pi, ni] of matchMap) {
    const interpBox = interpolateBbox(prevFrame[pi], nextFrame[ni], t);
    boxes.push(interpBox);

    // Use the nearest real detection's embedding
    const sourceDet = t < 0.5 ? prevDetections[pi] : nextDetections[ni];
    detections.push({
      ...interpBox,
      frameIndex: currentIndex,
      frameTimestamp: sourceDet.frameTimestamp,
      embedding: sourceDet.embedding,
      clusterId: sourceDet.clusterId,
    });
  }

  return { boxes, detections };
}
