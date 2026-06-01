import type { DetectionBox } from '@/types';
import { computeIOU } from './tracking';

/**
 * One-to-one greedy matching: sort all valid pairs by IOU descending,
 * assign best-first with strict one-to-one constraint.
 * This is simpler than Hungarian and sufficient for face tracking.
 */
export function matchDetectionsToTracks(
  detections: DetectionBox[],
  predictedBoxes: DetectionBox[],
  iouThreshold: number = 0.1
): { detIdx: number; trackIdx: number }[] {
  const pairs: { detIdx: number; trackIdx: number; iou: number }[] = [];

  for (let di = 0; di < detections.length; di++) {
    for (let ti = 0; ti < predictedBoxes.length; ti++) {
      const iou = computeIOU(detections[di], predictedBoxes[ti]);
      if (iou >= iouThreshold) {
        pairs.push({ detIdx: di, trackIdx: ti, iou });
      }
    }
  }

  // Sort by IOU descending — best matches first
  pairs.sort((a, b) => b.iou - a.iou);

  const usedDets = new Set<number>();
  const usedTracks = new Set<number>();
  const matches: { detIdx: number; trackIdx: number }[] = [];

  for (const p of pairs) {
    if (usedDets.has(p.detIdx) || usedTracks.has(p.trackIdx)) continue;
    matches.push({ detIdx: p.detIdx, trackIdx: p.trackIdx });
    usedDets.add(p.detIdx);
    usedTracks.add(p.trackIdx);
  }

  return matches;
}
