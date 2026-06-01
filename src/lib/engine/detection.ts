import { detectFacesSCRFD } from '@/lib/engine/scrfd';
import type { DetectionBox } from '@/types';

export async function detectFaces(
  imageData: ImageData,
  origWidth: number,
  origHeight: number
): Promise<DetectionBox[]> {
  return detectFacesSCRFD(imageData, origWidth, origHeight);
}
