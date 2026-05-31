import * as ort from 'onnxruntime-web';

let device: GPUDevice | null = null;
let isOrtDevice = false;

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (device) return device;

  // Try ORT's device first (enables fromGpuBuffer zero-copy)
  try {
    device = await ort.env.webgpu.device;
    isOrtDevice = true;
    return device;
  } catch {
    // ORT has no device — create our own for standalone GPU ops
  }

  if (typeof navigator.gpu === 'undefined') return null;

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    device = await adapter.requestDevice();
    isOrtDevice = false;
    return device;
  } catch {
    return null;
  }
}

export function usesOrtDevice(): boolean {
  return isOrtDevice;
}

export function getDevice(): GPUDevice | null {
  return device;
}
