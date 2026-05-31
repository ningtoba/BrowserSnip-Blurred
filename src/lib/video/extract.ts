import { SAMPLE_FPS } from '@/lib/constants';

export function generateSampleTimestamps(
  duration: number,
  sampleFps: number = SAMPLE_FPS
): number[] {
  const timestamps: number[] = [];
  const margin = 0.1;
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
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
    video.load();
  });

  const total = timestamps.length;
  const frames: ImageData[] = [];
  const canvas = new OffscreenCanvas(video.videoWidth, video.videoHeight);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  for (let i = 0; i < total; i++) {
    if (signal?.aborted) break;

    video.currentTime = timestamps[i];
    await new Promise<void>((resolve) => {
      video.addEventListener('seeked', () => resolve(), { once: true });
    });

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    onProgress?.(i + 1, total);
  }

  URL.revokeObjectURL(url);
  video.remove();
  return frames;
}

export async function extractFramesStreaming(
  videoFile: File,
  onFrame: (imageData: ImageData, timestamp: number, index: number) => Promise<boolean>,
  signal?: AbortSignal,
  playbackRate: number = 0.5
): Promise<number> {
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.crossOrigin = 'anonymous';
  video.playsInline = true;

  const url = URL.createObjectURL(videoFile);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
    video.load();
  });

  const w = video.videoWidth;
  const h = video.videoHeight;
  video.playbackRate = playbackRate;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  let frameIndex = 0;
  let stopped = false;

  const processLoop = new Promise<number>((resolve) => {
    const cb: VideoFrameRequestCallback = (_now, _metadata) => {
      if (signal?.aborted || stopped || video.ended) {
        resolve(frameIndex);
        return;
      }

      ctx.drawImage(video, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);

      onFrame(imageData, video.currentTime, frameIndex)
        .then((keepGoing) => {
          if (!keepGoing || signal?.aborted || video.ended) {
            stopped = true;
            video.pause();
            resolve(frameIndex);
            return;
          }
          frameIndex++;
          updateProgressThrottled(frameIndex, video.duration, video.currentTime);
          video.requestVideoFrameCallback(cb);
        })
        .catch(() => {
          video.pause();
          resolve(frameIndex);
        });
    };

    video.requestVideoFrameCallback(cb);
    video.play().catch(() => {
      video.muted = true;
      video.play().catch(() => resolve(frameIndex));
    });
  });

  const result = await processLoop;

  video.pause();
  URL.revokeObjectURL(url);
  video.remove();

  return result;
}

export async function extractFramesSeeking(
  videoFile: File,
  onFrame: (imageData: ImageData, timestamp: number, index: number) => Promise<boolean>,
  signal?: AbortSignal
): Promise<number> {
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.crossOrigin = 'anonymous';

  const url = URL.createObjectURL(videoFile);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
    video.load();
  });

  const w = video.videoWidth;
  const h = video.videoHeight;
  const duration = video.duration;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const frameInterval = 1 / 30;
  const totalFrames = Math.ceil(duration / frameInterval);

  let frameIndex = 0;
  for (let ts = 0; ts < duration && frameIndex < totalFrames; ts += frameInterval) {
    if (signal?.aborted) break;

    const tSeek = performance.now();
    video.currentTime = Math.min(ts, duration - 0.001);
    await new Promise<void>((resolve) => {
      video.addEventListener('seeked', () => resolve(), { once: true });
    });
    const seekMs = performance.now() - tSeek;

    ctx.drawImage(video, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    if (frameIndex % 100 === 0) console.debug(`[extract] frame ${frameIndex}: seek=${seekMs.toFixed(1)}ms`);

    const keepGoing = await onFrame(imageData, video.currentTime, frameIndex);
    if (!keepGoing) break;
    frameIndex++;
  }

  URL.revokeObjectURL(url);
  video.remove();
  return frameIndex;
}

let lastProgressUpdate = 0;

function updateProgressThrottled(
  frameIndex: number,
  duration: number,
  currentTime: number
): void {
  const now = performance.now();
  if (now - lastProgressUpdate < 200) return;
  lastProgressUpdate = now;
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
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
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
