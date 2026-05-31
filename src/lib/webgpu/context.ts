let device: GPUDevice | null = null;

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (device) return device;
  if (typeof navigator.gpu === 'undefined') return null;

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    device = await adapter.requestDevice();
    return device;
  } catch {
    return null;
  }
}

export function getDevice(): GPUDevice | null {
  return device;
}
