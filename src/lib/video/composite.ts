import { detectFaces } from '@/lib/engine/detection';
import { recognizeFace } from '@/lib/engine/recognition';
import { matchDetectionsToIdentities } from '@/lib/engine/clustering';
import { applyBlurToFrame } from '@/lib/engine/blur';
import type { FaceDetection, FaceIdentity, BlurConfig } from '@/types';
import type { FFmpeg } from '@ffmpeg/ffmpeg';

export async function processFrameDetections(
  imageData: ImageData,
  frameIndex: number,
  frameTimestamp: number,
  origWidth: number,
  origHeight: number
): Promise<FaceDetection[]> {
  const boxes = await detectFaces(imageData, origWidth, origHeight);
  return boxes.map((box) => ({
    ...box,
    frameIndex,
    frameTimestamp,
  }));
}

export async function processFrameEmbeddings(
  detections: FaceDetection[],
  frameImageData: ImageData,
  frameW: number,
  frameH: number
): Promise<void> {
  for (const det of detections) {
    const embedding = await recognizeFace(frameImageData, det, frameW, frameH);
    if (embedding) {
      det.embedding = embedding;
    }
  }
}

export async function processAndWriteBatch(
  videoFile: File,
  timestamps: number[],
  identities: FaceIdentity[],
  selectedIds: Set<number>,
  blurConfig: BlurConfig,
  origWidth: number,
  origHeight: number,
  ffmpeg: FFmpeg,
  batchIndex: number,
  onProgress?: (frameIndex: number, total: number) => void,
  signal?: AbortSignal
): Promise<number> {
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;

  const url = URL.createObjectURL(videoFile);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
    video.load();
  });

  const canvas = document.createElement('canvas');
  canvas.width = origWidth;
  canvas.height = origHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  let frameCount = 0;
  const total = timestamps.length;

  for (let i = 0; i < total; i++) {
    if (signal?.aborted) break;

    const timestamp = timestamps[i];
    video.currentTime = timestamp;

    await new Promise<void>((resolve) => {
      video.addEventListener('seeked', () => resolve(), { once: true });
    });

    ctx.drawImage(video, 0, 0, origWidth, origHeight);
    const imageData = ctx.getImageData(0, 0, origWidth, origHeight);

    const boxes = await detectFaces(imageData, origWidth, origHeight);

    const detections: FaceDetection[] = boxes.map((box) => ({
      ...box,
      frameIndex: i,
      frameTimestamp: timestamp,
    }));

    for (const det of detections) {
      const embedding = await recognizeFace(imageData, det, origWidth, origHeight);
      if (embedding) det.embedding = embedding;
    }

    const matchMap = matchDetectionsToIdentities(detections, identities);

    const targetIndices = new Set<number>();
    for (const [detIdx, clusterId] of matchMap) {
      if (selectedIds.has(clusterId)) {
        targetIndices.add(detIdx);
      }
    }

    let frameImageData = imageData;
    if (targetIndices.size > 0) {
      frameImageData = await applyBlurToFrame(imageData, boxes, targetIndices, blurConfig.type);
    }

    const pngName = `frame_${String(batchIndex * 300 + frameCount + 1).padStart(4, '0')}.png`;
    const blob = await canvasToPNGBlob(frameImageData);
    const buf = new Uint8Array(await blob.arrayBuffer());
    await ffmpeg.writeFile(pngName, buf);

    frameCount++;
    onProgress?.(i + 1, total);
  }

  URL.revokeObjectURL(url);
  video.remove();

  return frameCount;
}

async function canvasToPNGBlob(imageData: ImageData): Promise<Blob> {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(imageData, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}
