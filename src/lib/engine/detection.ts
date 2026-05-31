import { detectFacesYuNet } from '@/lib/engine/yunet';
import type { DetectionBox } from '@/types';

export async function detectFaces(
  imageData: ImageData,
  origWidth: number,
  origHeight: number
): Promise<DetectionBox[]> {
  return detectFacesYuNet(imageData, origWidth, origHeight);
}
