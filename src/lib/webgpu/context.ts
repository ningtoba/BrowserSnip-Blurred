let device: GPUDevice | null = null;
let queue: GPUQueue | null = null;

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (device) return device;
  if (typeof navigator.gpu === 'undefined') return null;

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
      },
    });
    queue = device.queue;
    return device;
  } catch {
    return null;
  }
}

export function getGPUQueue(): GPUQueue | null {
  return queue;
}

export function getDevice(): GPUDevice | null {
  return device;
}
