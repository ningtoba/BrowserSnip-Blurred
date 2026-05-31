export function imageDataToRGB(imageData: ImageData): Uint8Array {
  const { data, width, height } = imageData;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgb[i * 3] = data[i * 4];
    rgb[i * 3 + 1] = data[i * 4 + 1];
    rgb[i * 3 + 2] = data[i * 4 + 2];
  }
  return rgb;
}

export function rgbToFloat32CHW(
  rgb: Uint8Array,
  width: number,
  height: number
): Float32Array {
  const result = new Float32Array(3 * width * height);
  const planeSize = width * height;
  for (let i = 0; i < planeSize; i++) {
    result[i] = rgb[i * 3] / 255;
    result[planeSize + i] = rgb[i * 3 + 1] / 255;
    result[2 * planeSize + i] = rgb[i * 3 + 2] / 255;
  }
  return result;
}

export function bgrToFloat32CHW(
  rgb: Uint8Array,
  width: number,
  height: number,
  mean: number[],
  std: number[]
): Float32Array {
  const result = new Float32Array(3 * width * height);
  const planeSize = width * height;
  for (let i = 0; i < planeSize; i++) {
    result[2 * planeSize + i] = (rgb[i * 3] - mean[0]) / std[0];
    result[planeSize + i] = (rgb[i * 3 + 1] - mean[1]) / std[1];
    result[i] = (rgb[i * 3 + 2] - mean[2]) / std[2];
  }
  return result;
}

export function resizeImageData(
  source: ImageData,
  targetW: number,
  targetH: number
): ImageData {
  const canvas = new OffscreenCanvas(targetW, targetH);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const srcCanvas = new OffscreenCanvas(source.width, source.height);
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(source, 0, 0);
  ctx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  return ctx.getImageData(0, 0, targetW, targetH);
}

export function letterboxImageData(
  source: ImageData,
  targetW: number,
  targetH: number,
  fillR: number,
  fillG: number,
  fillB: number
): { data: ImageData; scale: number; padLeft: number; padTop: number } {
  const sw = source.width;
  const sh = source.height;
  const scale = Math.min(targetW / sw, targetH / sh);
  const newW = Math.round(sw * scale);
  const newH = Math.round(sh * scale);
  const padLeft = Math.floor((targetW - newW) / 2);
  const padTop = Math.floor((targetH - newH) / 2);

  const canvas = new OffscreenCanvas(targetW, targetH);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = `rgb(${fillR},${fillG},${fillB})`;
  ctx.fillRect(0, 0, targetW, targetH);

  const srcCanvas = new OffscreenCanvas(source.width, source.height);
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(source, 0, 0);
  ctx.drawImage(srcCanvas, padLeft, padTop, newW, newH);

  return {
    data: ctx.getImageData(0, 0, targetW, targetH),
    scale,
    padLeft,
    padTop,
  };
}

export function cropImageData(
  source: ImageData,
  x: number,
  y: number,
  w: number,
  h: number
): ImageData {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const srcCanvas = new OffscreenCanvas(source.width, source.height);
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(source, 0, 0);
  ctx.drawImage(srcCanvas, x, y, w, h, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}
