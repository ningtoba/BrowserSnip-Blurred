import * as ort from 'onnxruntime-web';
import { getGPUDevice } from '@/lib/webgpu/context';
import { NORMALIZE_CHW_SHADER } from '@/lib/webgpu/shaders';
import { YUNET_INPUT_SIZE, FACE_INPUT_SIZE } from '@/lib/constants';

let pipeline: GPUComputePipeline | null = null;
let sampler: GPUSampler | null = null;

async function ensurePipeline(dev: GPUDevice): Promise<GPUComputePipeline> {
  if (!pipeline) {
    const mod = dev.createShaderModule({ code: NORMALIZE_CHW_SHADER });
    pipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: 'main' },
    });
  }
  if (!sampler) {
    sampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }
  return pipeline;
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
): Promise<Float32Array> {
  const dev = await getGPUDevice();
  if (!dev) throw new Error('GPU not available');

  const pipe = await ensurePipeline(dev);
  const srcW = imageData.width;
  const srcH = imageData.height;
  const { scale, padLeft, padTop } = letterboxParams(srcW, srcH, dstW, dstH);

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

  const outputSize = dstW * dstH * 3 * 4;
  const outputBuf = dev.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

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
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTex.createView() },
      { binding: 1, resource: sampler! },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dstW / 16), Math.ceil(dstH / 16));
  pass.end();
  dev.queue.submit([encoder.finish()]);

  inputTex.destroy();
  paramsBuf.destroy();

  // Read back result from GPU
  const readBuf = dev.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readEncoder = dev.createCommandEncoder();
  readEncoder.copyBufferToBuffer(outputBuf, 0, readBuf, 0, outputSize);
  dev.queue.submit([readEncoder.finish()]);
  outputBuf.destroy();

  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();

  return result;
}

export async function preprocessCHW(
  imageData: ImageData,
  dstW: number,
  dstH: number
): Promise<Float32Array> {
  // srcW==dstW and srcH==dstH: already resized, just do CHW conversion
  // mean=[0,0,0], std=[1,1,1] → raw pixel values [0,255] matching OpenCV blobFromImage
  return runNormalizeShader(
    imageData, dstW, dstH,
    { mean: [0, 0, 0], std: [1, 1, 1] },
  );
}

export async function preprocessYuNet(
  imageData: ImageData
): Promise<{
  tensor: ort.Tensor;
  scale: number;
  padLeft: number;
  padTop: number;
}> {
  const { scale, padLeft, padTop } = letterboxParams(
    imageData.width, imageData.height, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE,
  );
  const data = await runNormalizeShader(
    imageData, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE,
    { mean: [0, 0, 0], std: [255, 255, 255] },
  );
  return {
    tensor: new ort.Tensor('float32', data, [1, 3, YUNET_INPUT_SIZE, YUNET_INPUT_SIZE]),
    scale, padLeft, padTop,
  };
}

export async function preprocessMFN(
  faceImageData: ImageData
): Promise<ort.Tensor> {
  const data = await runNormalizeShader(
    faceImageData, FACE_INPUT_SIZE, FACE_INPUT_SIZE,
    { mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5] },
  );
  return new ort.Tensor('float32', data, [1, 3, FACE_INPUT_SIZE, FACE_INPUT_SIZE]);
}
