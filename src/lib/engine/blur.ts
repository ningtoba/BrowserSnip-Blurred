import type { DetectionBox, BlurType } from '@/types';
import { PIXELATE_BLOCK_SIZE } from '@/lib/constants';

export function applyPixelateBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox,
  blockSize: number = PIXELATE_BLOCK_SIZE
): void {
  const x = Math.round(bbox.x1);
  const y = Math.round(bbox.y1);
  const w = Math.round(bbox.x2 - bbox.x1);
  const h = Math.round(bbox.y2 - bbox.y1);

  if (w <= 0 || h <= 0) return;

  const imageData = ctx.getImageData(x, y, w, h);
  const data = imageData.data;

  const smallW = Math.max(1, Math.floor(w / blockSize));
  const smallH = Math.max(1, Math.floor(h / blockSize));

  const blurred = new Uint8ClampedArray(w * h * 4);

  for (let by = 0; by < smallH; by++) {
    for (let bx = 0; bx < smallW; bx++) {
      let r = 0, g = 0, b = 0, count = 0;

      const startX = bx * blockSize;
      const startY = by * blockSize;
      const endX = Math.min(startX + blockSize, w);
      const endY = Math.min(startY + blockSize, h);

      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          const idx = (py * w + px) * 4;
          r += data[idx];
          g += data[idx + 1];
          b += data[idx + 2];
          count++;
        }
      }

      const avgR = Math.round(r / count);
      const avgG = Math.round(g / count);
      const avgB = Math.round(b / count);

      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          const idx = (py * w + px) * 4;
          blurred[idx] = avgR;
          blurred[idx + 1] = avgG;
          blurred[idx + 2] = avgB;
          blurred[idx + 3] = data[idx + 3];
        }
      }
    }
  }

  const blurredImageData = new ImageData(blurred, w, h);
  ctx.putImageData(blurredImageData, x, y);
}

export function applyEyeBarBlur(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  bbox: DetectionBox
): void {
  const faceW = bbox.x2 - bbox.x1;
  const faceH = bbox.y2 - bbox.y1;

  const eyeY = bbox.y1 + faceH * 0.28;
  const barH = Math.max(6, faceH * 0.12);
  const barW = faceW * 0.85;
  const barX = bbox.x1 + faceW * 0.075;

  ctx.fillStyle = '#000000';
  ctx.beginPath();
  const radius = Math.min(3, barH / 2);
  ctx.moveTo(barX + radius, eyeY);
  ctx.lineTo(barX + barW - radius, eyeY);
  ctx.quadraticCurveTo(barX + barW, eyeY, barX + barW, eyeY + radius);
  ctx.lineTo(barX + barW, eyeY + barH - radius);
  ctx.quadraticCurveTo(barX + barW, eyeY + barH, barX + barW - radius, eyeY + barH);
  ctx.lineTo(barX + radius, eyeY + barH);
  ctx.quadraticCurveTo(barX, eyeY + barH, barX, eyeY + barH - radius);
  ctx.lineTo(barX, eyeY + radius);
  ctx.quadraticCurveTo(barX, eyeY, barX + radius, eyeY);
  ctx.closePath();
  ctx.fill();
}

export function applyBlurToFrame(
  sourceImageData: ImageData,
  detections: DetectionBox[],
  targetIndices: Set<number>,
  blurType: BlurType
): ImageData {
  const w = sourceImageData.width;
  const h = sourceImageData.height;

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(sourceImageData, 0, 0);

  for (const idx of targetIndices) {
    const bbox = detections[idx];
    if (blurType === 'pixelate') {
      applyPixelateBlur(ctx, bbox);
    } else {
      applyEyeBarBlur(ctx, bbox);
    }
  }

  return ctx.getImageData(0, 0, w, h);
}
