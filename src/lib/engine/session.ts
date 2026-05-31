import * as ort from 'onnxruntime-web';
import { MODELS } from '@/lib/constants';

type ModelName = 'yunet' | 'mfn';

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'webgpu' | 'wasm';
}

const sessions = new Map<ModelName, SessionEntry>();
const loading = new Map<ModelName, Promise<void>>();
let generation = 0;

ort.env.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency || 4);

function getConfig(name: ModelName) {
  const config = MODELS.find((m) => m.name === name);
  if (!config) throw new Error(`Unknown model: ${name}`);
  return config;
}

export function hasWebGPU(): boolean {
  return typeof navigator.gpu !== 'undefined';
}

async function fetchModelBuffer(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(
      `Model file not found at ${url}. Place the ONNX model in public/models/. ` +
      `See README.md for instructions.`
    );
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 1024) {
    throw new Error(
      `File at ${url} is too small (${buffer.byteLength} bytes) to be an ONNX model.`
    );
  }
  return buffer;
}

async function createWebGPUSession(modelBuffer: ArrayBuffer): Promise<ort.InferenceSession> {
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: [
      {
        name: 'webgpu',
        validationMode: 'disabled',
      },
    ],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    preferredOutputLocation: 'gpu-buffer',
    intraOpNumThreads: Math.max(2, navigator.hardwareConcurrency || 4),
  });

  return session;
}

async function createWasmSession(modelBuffer: ArrayBuffer): Promise<ort.InferenceSession> {
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    intraOpNumThreads: Math.max(2, navigator.hardwareConcurrency || 4),
  });

  return session;
}

export async function initSession(name: ModelName, signal?: AbortSignal): Promise<void> {
  if (sessions.has(name)) return;

  // If already loading, wait for the existing promise
  const existing = loading.get(name);
  if (existing) {
    try {
      await existing;
      return;
    } catch {
      // Previous load failed — will retry below
    }
  }

  const config = getConfig(name);
  const gen = generation;
  const abortController = new AbortController();
  const linkedSignal = signal
    ? abortController.signal
    : abortController.signal;

  // Link external signal
  if (signal) {
    signal.addEventListener('abort', () => abortController.abort(), { once: true });
  }

  const promise = (async () => {
    const buffer = await fetchModelBuffer(config.url, linkedSignal);

    // Check generation — if disposeAll was called, abort
    if (gen !== generation) {
      return;
    }

    let session: ort.InferenceSession;
    let backend: 'webgpu' | 'wasm';

    if (hasWebGPU()) {
      try {
        session = await createWebGPUSession(buffer);
        backend = 'webgpu';
      } catch (err) {
        console.warn(`WebGPU init failed for ${name}, falling back to WASM:`, err instanceof Error ? err.message : err);
        session = await createWasmSession(buffer);
        backend = 'wasm';
      }
    } else {
      session = await createWasmSession(buffer);
      backend = 'wasm';
    }

    // Final generation check before storing
    if (gen !== generation) {
      session.release();
      return;
    }

    sessions.set(name, { session, backend });
  })();

  loading.set(name, promise);

  try {
    await promise;
  } finally {
    loading.delete(name);
  }
}

export function getBackend(name: ModelName): 'webgpu' | 'wasm' | null {
  return sessions.get(name)?.backend ?? null;
}

export function getSession(name: ModelName): ort.InferenceSession | null {
  return sessions.get(name)?.session ?? null;
}

export async function runMFN(
  input: Float32Array | ort.Tensor
): Promise<Float32Array> {
  const entry = sessions.get('mfn');
  if (!entry) throw new Error('MFN session not initialized');

  const tensor: ort.Tensor = input instanceof ort.Tensor
    ? input
    : new ort.Tensor('float32', input as Float32Array, [1, 3, 112, 112]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[entry.session.inputNames[0]] = tensor;

  const results = await entry.session.run(feeds);
  const output = results[entry.session.outputNames[0]];

  const data = await output.getData();
  const floatData = new Float32Array(data as Float32Array);
  output.dispose();

  return floatData;
}

export function isReady(name: ModelName): boolean {
  return sessions.has(name);
}

export async function disposeAll(): Promise<void> {
  generation++;
  for (const [, entry] of sessions) {
    try {
      entry.session.release();
    } catch { /* ignore release errors */ }
  }
  sessions.clear();
  loading.clear();
}
