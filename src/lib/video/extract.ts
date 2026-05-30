import { SAMPLE_FPS } from '@/lib/constants';

export function generateSampleTimestamps(
  duration: number,
  sampleFps: number = SAMPLE_FPS
): number[] {
  const timestamps: number[] = [];
  const margin = 0.1;
  const effectiveDuration = Math.max(0, duration - margin * 2);
  const interval = 1 / sampleFps;
  for (let t = margin; t < duration - margin; t += interval) {
    timestamps.push(t);
  }
  return timestamps;
}

export function generateAllFrameTimestamps(
  duration: number,
  fps: number
): number[] {
  const timestamps: number[] = [];
  const frameInterval = 1 / fps;
  for (let t = 0; t < duration; t += frameInterval) {
    timestamps.push(Math.min(t, duration - 0.001));
  }
  return timestamps;
}

export async function extractFramesAtTimestamps(
  videoFile: File,
  timestamps: number[],
  onProgress?: (i: number, total: number) => void,
  signal?: AbortSignal
): Promise<ImageData[]> {
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.crossOrigin = 'anonymous';

  const url = URL.createObjectURL(videoFile);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    const onLoaded = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      resolve();
    };
    const onError = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      reject(new Error('Failed to load video'));
    };
    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('error', onError);
    video.load();
  });

  const total = timestamps.length;
  const frames: ImageData[] = [];

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d')!;

  for (let i = 0; i < total; i++) {
    if (signal?.aborted) break;

    const timestamp = timestamps[i];
    video.currentTime = timestamp;

    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      };
      video.addEventListener('seeked', onSeeked);
    });

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    frames.push(imageData);

    onProgress?.(i + 1, total);
  }

  URL.revokeObjectURL(url);
  video.remove();

  return frames;
}

export async function getVideoMetadata(
  videoFile: File
): Promise<{ width: number; height: number; duration: number; fps: number }> {
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;

  const url = URL.createObjectURL(videoFile);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    const onLoaded = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      resolve();
    };
    const onError = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
      reject(new Error('Failed to load video'));
    };
    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('error', onError);
    video.load();
  });

  const metadata = {
    width: video.videoWidth,
    height: video.videoHeight,
    duration: video.duration,
    fps: 30,
  };

  video.remove();
  URL.revokeObjectURL(url);

  return metadata;
}
