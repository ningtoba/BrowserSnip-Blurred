import { getGPUDevice } from '@/lib/webgpu/context';
import { NORMALIZE_CHW_EXTERNAL_SHADER } from '@/lib/webgpu/shaders';
import { YOLO_INPUT_SIZE } from '@/lib/constants';
import * as ort from 'onnxruntime-web';

let zeroCopyPipeline: GPUComputePipeline | null = null;
let zcSampler: GPUSampler | null = null;

async function ensureZeroCopyPipeline(dev: GPUDevice): Promise<void> {
  if (!zeroCopyPipeline) {
    const mod = dev.createShaderModule({ code: NORMALIZE_CHW_EXTERNAL_SHADER });
    zeroCopyPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: 'main' },
    });
  }
  if (!zcSampler) {
    zcSampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }
}

export async function preprocessFromVideoElement(
  video: HTMLVideoElement,
  srcW: number,
  srcH: number
): Promise<{
  tensor: ort.Tensor;
  scale: number;
  padLeft: number;
  padTop: number;
}> {
  const dev = await getGPUDevice();
  if (!dev) throw new Error('GPU not available for zero-copy preprocessing');

  await ensureZeroCopyPipeline(dev);

  const dstW = YOLO_INPUT_SIZE;
  const dstH = YOLO_INPUT_SIZE;
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padLeft = Math.floor((dstW - newW) / 2);
  const padTop = Math.floor((dstH - newH) / 2);

  // Zero-copy: import video frame directly as GPU external texture
  const externalTex = dev.importExternalTexture({
    source: video,
    colorSpace: 'srgb',
  });

  const outputSize = dstW * dstH * 3 * 4;
  const outputBuf = dev.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramsData = new Float32Array([
    srcW, srcH, dstW, dstH, scale, padLeft, padTop,
    0, 0, 0,     // mean for [0,1] normalization
    255, 255, 255, // std
  ]);
  const paramsBuf = dev.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(paramsBuf, 0, paramsData);

  const bindGroup = dev.createBindGroup({
    layout: zeroCopyPipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: externalTex },  // texture_external
      { binding: 1, resource: zcSampler! },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(zeroCopyPipeline!);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dstW / 16), Math.ceil(dstH / 16));
  pass.end();
  dev.queue.submit([encoder.finish()]);

  paramsBuf.destroy();

  const tensor = ort.Tensor.fromGpuBuffer(outputBuf, {
    dataType: 'float32',
    dims: [1, 3, dstW, dstH],
    dispose: () => outputBuf.destroy(),
  });

  return { tensor, scale, padLeft, padTop };
}
