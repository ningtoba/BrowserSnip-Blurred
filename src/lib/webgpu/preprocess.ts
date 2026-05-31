import * as ort from 'onnxruntime-web';
import { getGPUDevice } from '@/lib/webgpu/context';
import { NORMALIZE_CHW_SHADER } from '@/lib/webgpu/shaders';
import { YOLO_INPUT_SIZE, FACE_INPUT_SIZE } from '@/lib/constants';

let yoloPipeline: GPUComputePipeline | null = null;
let mfnPipeline: GPUComputePipeline | null = null;
let sampler: GPUSampler | null = null;

async function ensurePipelines(dev: GPUDevice): Promise<void> {
  if (!yoloPipeline) {
    const mod = dev.createShaderModule({ code: NORMALIZE_CHW_SHADER });
    yoloPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: 'main' },
    });
  }
  if (!sampler) {
    sampler = dev.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }
}

interface NormalizeParams {
  srcW: number; srcH: number;
  dstW: number; dstH: number;
  scale: number; padLeft: number; padTop: number;
  meanR: number; meanG: number; meanB: number;
  stdR: number; stdG: number; stdB: number;
}

function letterboxParams(
  srcW: number, srcH: number, dstW: number, dstH: number
): { scale: number; padLeft: number; padTop: number } {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return {
    scale,
    padLeft: Math.floor((dstW - newW) / 2),
    padTop: Math.floor((dstH - newH) / 2),
  };
}

async function runNormalizeShader(
  imageData: ImageData,
  dstW: number,
  dstH: number,
  normParams: { mean: [number, number, number]; std: [number, number, number] }
): Promise<{ gpuBuffer: GPUBuffer; scale: number; padLeft: number; padTop: number }> {
  const dev = await getGPUDevice();
  if (!dev) throw new Error('GPU preprocessing unavailable');
  await ensurePipelines(dev);

  const srcW = imageData.width;
  const srcH = imageData.height;
  const { scale, padLeft, padTop } = letterboxParams(srcW, srcH, dstW, dstH);

  // Upload input ImageData to GPU texture
  const inputTex = dev.createTexture({
    size: [srcW, srcH],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  dev.queue.writeTexture(
    { texture: inputTex },
    imageData.data,
    { bytesPerRow: srcW * 4, rowsPerImage: srcH },
    [srcW, srcH],
  );

  // Output buffer: CHW float32 layout
  const outputSize = dstW * dstH * 3 * 4; // 3 channels × float32
  const outputBuf = dev.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Uniform params
  const paramsData = new Float32Array([
    srcW, srcH, dstW, dstH, scale, padLeft, padTop,
    normParams.mean[0], normParams.mean[1], normParams.mean[2],
    normParams.std[0], normParams.std[1], normParams.std[2],
  ]);
  const paramsBuf = dev.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(paramsBuf, 0, paramsData);

  const bindGroup = dev.createBindGroup({
    layout: yoloPipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex.createView() },
      { binding: 1, resource: sampler! },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(yoloPipeline!);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dstW / 16), Math.ceil(dstH / 16));
  pass.end();
  dev.queue.submit([encoder.finish()]);

  inputTex.destroy();
  paramsBuf.destroy();

  return { gpuBuffer: outputBuf, scale, padLeft, padTop };
}

export async function preprocessYOLO(
  imageData: ImageData
): Promise<{
  tensor: ort.Tensor;
  scale: number;
  padLeft: number;
  padTop: number;
}> {
  const { gpuBuffer, scale, padLeft, padTop } = await runNormalizeShader(
    imageData,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    { mean: [0, 0, 0], std: [255, 255, 255] },  // normalize to [0, 1]
  );

  const tensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: 'float32',
    dims: [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE],
    dispose: () => gpuBuffer.destroy(),
  });

  return { tensor, scale, padLeft, padTop };
}

export async function preprocessMFN(
  faceImageData: ImageData
): Promise<ort.Tensor> {
  const { gpuBuffer } = await runNormalizeShader(
    faceImageData,
    FACE_INPUT_SIZE,
    FACE_INPUT_SIZE,
    { mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5] },  // MFN: (pixel - 127.5) / 127.5
  );

  const tensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: 'float32',
    dims: [1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE],
    dispose: () => gpuBuffer.destroy(),
  });

  return tensor;
}
