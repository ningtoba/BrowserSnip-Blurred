import type { DetectionBox, BlurType } from '@/types';
import { PIXELATE_BLOCK_SIZE } from '@/lib/constants';
import { getGPUDevice } from '@/lib/webgpu/context';

let gpuAvailable: boolean | null = null;

async function checkGPU(): Promise<boolean> {
  if (gpuAvailable !== null) return gpuAvailable;
  const dev = await getGPUDevice();
  gpuAvailable = dev !== null;
  return gpuAvailable;
}

// ── CPU implementation (always works) ──

function cpuPixelateBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox,
  blockSize: number = PIXELATE_BLOCK_SIZE
): void {
  const x = Math.round(bbox.x1);
  const y = Math.round(bbox.y1);
  const w = Math.round(bbox.x2 - bbox.x1);
  const h = Math.round(bbox.y2 - bbox.y1);
  if (w <= 0 || h <= 0) return;

  // Use canvas drawImage scaling — no getImageData/putImageData,
  // so compositing works correctly at edges with no black background.
  const smallW = Math.max(1, Math.round(w / blockSize));
  const smallH = Math.max(1, Math.round(h / blockSize));

  // Create a tiny canvas with the averaged colors
  const tmp = new OffscreenCanvas(smallW, smallH);
  const tmpCtx = tmp.getContext('2d')!;
  tmpCtx.drawImage(ctx.canvas as any, x, y, w, h, 0, 0, smallW, smallH);

  // Draw it back scaled up with nearest-neighbor (pixelated)
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp as any, 0, 0, smallW, smallH, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}

function cpuEyebarBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox
): void {
  const fw = bbox.x2 - bbox.x1, fh = bbox.y2 - bbox.y1;
  const barY = bbox.y1 + fh * 0.22;
  const barH = Math.max(10, fh * 0.2);
  const barW = fw * 0.9;
  const barX = bbox.x1 + fw * 0.05;
  const radius = Math.min(4, barH / 2);

  ctx.fillStyle = '#000000';
  ctx.beginPath();
  ctx.moveTo(barX + radius, barY);
  ctx.lineTo(barX + barW - radius, barY);
  ctx.quadraticCurveTo(barX + barW, barY, barX + barW, barY + radius);
  ctx.lineTo(barX + barW, barY + barH - radius);
  ctx.quadraticCurveTo(barX + barW, barY + barH, barX + barW - radius, barY + barH);
  ctx.lineTo(barX + radius, barY + barH);
  ctx.quadraticCurveTo(barX, barY + barH, barX, barY + barH - radius);
  ctx.lineTo(barX, barY + radius);
  ctx.quadraticCurveTo(barX, barY, barX + radius, barY);
  ctx.closePath();
  ctx.fill();
}

// ── GPU-accelerated blur ──

async function gpuPixelateBlur(
  imageData: ImageData, bbox: DetectionBox, blockSize: number
): Promise<ImageData> {
  const { gpuPixelateBlur: gpuFn } = await import('@/lib/webgpu/blur');
  return gpuFn(imageData, bbox, blockSize);
}

async function gpuEyebarBlur(
  imageData: ImageData, bbox: DetectionBox
): Promise<ImageData> {
  const { gpuEyebarBlur: gpuFn } = await import('@/lib/webgpu/blur');
  return gpuFn(imageData, bbox);
}

// ── Unified blur application ──

export function applyPixelateBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox,
  blockSize?: number
): void {
  cpuPixelateBlur(ctx, bbox, blockSize);
}

export function applyEyeBarBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox
): void {
  cpuEyebarBlur(ctx, bbox);
}

export async function applyBlurToFrame(
  sourceImageData: ImageData,
  detections: DetectionBox[],
  targetIndices: Set<number>,
  blurType: BlurType
): Promise<ImageData> {
  const useGPU = await checkGPU();

  if (useGPU) {
    try {
      const { gpuApplyBlur } = await import('@/lib/webgpu/blur');
      return await gpuApplyBlur(sourceImageData, detections, targetIndices, blurType);
    } catch {
      // GPU failed, fall through to CPU
    }
  }

  // CPU path
  const w = sourceImageData.width;
  const h = sourceImageData.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.putImageData(sourceImageData, 0, 0);

  for (const idx of targetIndices) {
    const bbox = detections[idx];
    if (blurType === 'pixelate') {
      cpuPixelateBlur(ctx, bbox);
    } else {
      cpuEyebarBlur(ctx, bbox);
    }
  }

  return ctx.getImageData(0, 0, w, h);
}
