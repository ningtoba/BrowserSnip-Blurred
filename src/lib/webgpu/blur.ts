import { getGPUDevice } from '@/lib/webgpu/context';
import { PIXELATE_SHADER, EYEBAR_SHADER } from '@/lib/webgpu/shaders';
import type { DetectionBox, BlurType } from '@/types';
import { PIXELATE_BLOCK_SIZE } from '@/lib/constants';

let pixelatePipeline: GPUComputePipeline | null = null;
let eyebarPipeline: GPUComputePipeline | null = null;
let sampler: GPUSampler | null = null;

// Cached resources — reused across frames
let cachedWidth = 0;
let cachedHeight = 0;
let inputTex: GPUTexture | null = null;
let outputTex: GPUTexture | null = null;
let readBuf: GPUBuffer | null = null;
let paramsBuf: GPUBuffer | null = null;

async function ensurePipelines(): Promise<boolean> {
  const dev = await getGPUDevice();
  if (!dev) return false;

  if (!pixelatePipeline) {
    const mod = dev.createShaderModule({ code: PIXELATE_SHADER });
    pixelatePipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: 'main' },
    });
  }

  if (!eyebarPipeline) {
    const mod = dev.createShaderModule({ code: EYEBAR_SHADER });
    eyebarPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: 'main' },
    });
  }

  if (!sampler) {
    sampler = dev.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
    });
  }

  return true;
}

function ensureTextures(dev: GPUDevice, w: number, h: number): void {
  if (cachedWidth === w && cachedHeight === h && inputTex && outputTex && readBuf && paramsBuf) return;

  inputTex?.destroy();
  outputTex?.destroy();
  readBuf?.destroy();
  paramsBuf?.destroy();

  inputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  outputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  readBuf = dev.createBuffer({
    size: w * h * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  paramsBuf = dev.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  cachedWidth = w;
  cachedHeight = h;
}

async function readBackResult(dev: GPUDevice, w: number, h: number): Promise<ImageData> {
  await readBuf!.mapAsync(GPUMapMode.READ);
  const resultData = new Uint8ClampedArray(readBuf!.getMappedRange().slice(0));
  const result = new ImageData(resultData, w, h);
  readBuf!.unmap();

  // Recreate readBuf after unmap (mapped buffers can't be reused)
  readBuf!.destroy();
  readBuf = dev.createBuffer({
    size: w * h * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return result;
}

function uploadInputAndParams(dev: GPUDevice, imageData: ImageData, paramsData: Float32Array): void {
  const w = imageData.width;
  const h = imageData.height;

  dev.queue.writeTexture(
    { texture: inputTex! },
    imageData.data,
    { bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  dev.queue.writeBuffer(paramsBuf!, 0, paramsData);
}

function runComputeAndCopy(dev: GPUDevice, pipeline: GPUComputePipeline, w: number, h: number): GPUCommandEncoder {
  const bindGroup = dev.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex!.createView() },
      { binding: 1, resource: outputTex!.createView() },
      { binding: 2, resource: { buffer: paramsBuf! } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
  pass.end();

  encoder.copyTextureToBuffer(
    { texture: outputTex! },
    { buffer: readBuf!, bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  dev.queue.submit([encoder.finish()]);

  return encoder;
}

export async function gpuPixelateBlur(
  inputImageData: ImageData,
  bbox: DetectionBox,
  blockSize: number = PIXELATE_BLOCK_SIZE
): Promise<ImageData> {
  const dev = await getGPUDevice();
  if (!dev || !(await ensurePipelines())) throw new Error('WebGPU not available');

  const w = inputImageData.width;
  const h = inputImageData.height;
  ensureTextures(dev, w, h);

  const faceX = Math.round(bbox.x1);
  const faceY = Math.round(bbox.y1);
  const faceW = Math.round(bbox.x2 - bbox.x1);
  const faceH = Math.round(bbox.y2 - bbox.y1);

  uploadInputAndParams(dev, inputImageData, new Float32Array([faceX, faceY, faceW, faceH, blockSize]));
  runComputeAndCopy(dev, pixelatePipeline!, w, h);

  return readBackResult(dev, w, h);
}

export async function gpuEyebarBlur(
  inputImageData: ImageData,
  bbox: DetectionBox
): Promise<ImageData> {
  const dev = await getGPUDevice();
  if (!dev || !(await ensurePipelines())) throw new Error('WebGPU not available');

  const w = inputImageData.width;
  const h = inputImageData.height;
  const faceW = bbox.x2 - bbox.x1;
  const faceH = bbox.y2 - bbox.y1;
  const barY = bbox.y1 + faceH * 0.28;
  const barH = Math.max(6, faceH * 0.12);
  const barW = faceW * 0.85;
  const barX = bbox.x1 + faceW * 0.075;

  ensureTextures(dev, w, h);
  uploadInputAndParams(dev, inputImageData, new Float32Array([barX, barY, barW, barH, Math.min(3, barH / 2)]));
  runComputeAndCopy(dev, eyebarPipeline!, w, h);

  return readBackResult(dev, w, h);
}

export async function gpuApplyBlur(
  imageData: ImageData,
  detections: DetectionBox[],
  targetIndices: Set<number>,
  blurType: BlurType
): Promise<ImageData> {
  let current = imageData;
  for (const idx of targetIndices) {
    const bbox = detections[idx];
    try {
      if (blurType === 'pixelate') {
        current = await gpuPixelateBlur(current, bbox);
      } else {
        current = await gpuEyebarBlur(current, bbox);
      }
    } catch {
      throw new Error('GPU blur failed');
    }
  }
  return current;
}
