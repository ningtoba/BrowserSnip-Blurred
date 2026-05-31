import { getGPUDevice } from '@/lib/webgpu/context';
import { PIXELATE_SHADER, EYEBAR_SHADER } from '@/lib/webgpu/shaders';
import type { DetectionBox, BlurType } from '@/types';
import { PIXELATE_BLOCK_SIZE } from '@/lib/constants';

let pixelatePipeline: GPUComputePipeline | null = null;
let eyebarPipeline: GPUComputePipeline | null = null;
let sampler: GPUSampler | null = null;

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

export async function gpuPixelateBlur(
  inputImageData: ImageData,
  bbox: DetectionBox,
  blockSize: number = PIXELATE_BLOCK_SIZE
): Promise<ImageData> {
  const dev = await getGPUDevice();
  if (!dev || !(await ensurePipelines())) {
    throw new Error('WebGPU not available');
  }

  const w = inputImageData.width;
  const h = inputImageData.height;

  // Create input texture
  const inputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  dev.queue.writeTexture(
    { texture: inputTex },
    inputImageData.data,
    { bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  // Create output texture
  const outputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // Face region params
  const faceX = Math.round(bbox.x1);
  const faceY = Math.round(bbox.y1);
  const faceW = Math.round(bbox.x2 - bbox.x1);
  const faceH = Math.round(bbox.y2 - bbox.y1);

  const paramsBuf = dev.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(
    paramsBuf,
    0,
    new Float32Array([faceX, faceY, faceW, faceH, blockSize]),
  );

  const bindGroup = dev.createBindGroup({
    layout: pixelatePipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex.createView() },
      { binding: 1, resource: outputTex.createView() },
      { binding: 2, resource: { buffer: paramsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pixelatePipeline!);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
  pass.end();

  // Read result back
  const readBuf = dev.createBuffer({
    size: w * h * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer(
    { texture: outputTex },
    { buffer: readBuf, bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  dev.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const resultData = new Uint8ClampedArray(readBuf.getMappedRange());
  const result = new ImageData(resultData.slice(0), w, h);
  readBuf.unmap();

  inputTex.destroy();
  outputTex.destroy();

  return result;
}

export async function gpuEyebarBlur(
  inputImageData: ImageData,
  bbox: DetectionBox
): Promise<ImageData> {
  const dev = await getGPUDevice();
  if (!dev || !(await ensurePipelines())) {
    throw new Error('WebGPU not available');
  }

  const w = inputImageData.width;
  const h = inputImageData.height;
  const faceW = bbox.x2 - bbox.x1;
  const faceH = bbox.y2 - bbox.y1;

  const barY = bbox.y1 + faceH * 0.28;
  const barH = Math.max(6, faceH * 0.12);
  const barW = faceW * 0.85;
  const barX = bbox.x1 + faceW * 0.075;

  const inputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  dev.queue.writeTexture(
    { texture: inputTex },
    inputImageData.data,
    { bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  const outputTex = dev.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const paramsBuf = dev.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(
    paramsBuf,
    0,
    new Float32Array([barX, barY, barW, barH, Math.min(3, barH / 2)]),
  );

  const bindGroup = dev.createBindGroup({
    layout: eyebarPipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex.createView() },
      { binding: 1, resource: outputTex.createView() },
      { binding: 2, resource: { buffer: paramsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(eyebarPipeline!);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
  pass.end();

  const readBuf = dev.createBuffer({
    size: w * h * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer(
    { texture: outputTex },
    { buffer: readBuf, bytesPerRow: w * 4, rowsPerImage: h },
    [w, h],
  );

  dev.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const resultData = new Uint8ClampedArray(readBuf.getMappedRange());
  const result = new ImageData(resultData.slice(0), w, h);
  readBuf.unmap();

  inputTex.destroy();
  outputTex.destroy();

  return result;
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
      // GPU blur failed — the calling code should fall back to CPU
      throw new Error('GPU blur failed');
    }
  }
  return current;
}
