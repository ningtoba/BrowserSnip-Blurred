import * as ort from 'onnxruntime-web';
import { MODELS } from '@/lib/constants';

type ModelName = 'scrfd' | 'mfn';

interface SessionEntry {
  session: ort.InferenceSession;
  backend: 'webgpu' | 'wasm';
}

const sessions = new Map<ModelName, SessionEntry>();
const loading = new Map<ModelName, Promise<void>>();

ort.env.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency || 4);

function getConfig(name: ModelName) {
  const config = MODELS.find((m) => m.name === name);
  if (!config) throw new Error(`Unknown model: ${name}`);
  return config;
}

export function hasWebGPU(): boolean {
  return typeof navigator.gpu !== 'undefined';
}

async function fetchModelBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
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

async function createWebGPUSession(modelBuffer: ArrayBuffer, inputShape: number[]): Promise<ort.InferenceSession> {
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

  // Warm up: compile shaders and cache compute pipelines
  const inputSize = inputShape.reduce((a, b) => a * b, 1);
  const dummyData = new Float32Array(inputSize);
  const dummyTensor = new ort.Tensor('float32', dummyData, inputShape);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = dummyTensor;
  await session.run(feeds);

  return session;
}

async function createWasmSession(modelBuffer: ArrayBuffer, inputShape: number[]): Promise<ort.InferenceSession> {
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    intraOpNumThreads: Math.max(2, navigator.hardwareConcurrency || 4),
  });

  // Warm up
  const inputSize = inputShape.reduce((a, b) => a * b, 1);
  const dummyData = new Float32Array(inputSize);
  const dummyTensor = new ort.Tensor('float32', dummyData, inputShape);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = dummyTensor;
  await session.run(feeds);

  return session;
}

export async function initSession(name: ModelName): Promise<void> {
  if (sessions.has(name)) return;
  if (loading.has(name)) return loading.get(name)!;

  const config = getConfig(name);
  const promise = (async () => {
    const buffer = await fetchModelBuffer(config.url);

    let session: ort.InferenceSession;
    let backend: 'webgpu' | 'wasm';

    if (hasWebGPU()) {
      try {
        session = await createWebGPUSession(buffer, config.inputShape);
        backend = 'webgpu';
      } catch (err) {
        console.warn(`WebGPU init failed for ${name}, falling back to WASM:`, err);
        session = await createWasmSession(buffer, config.inputShape);
        backend = 'wasm';
      }
    } else {
      session = await createWasmSession(buffer, config.inputShape);
      backend = 'wasm';
    }

    sessions.set(name, { session, backend });
  })();

  loading.set(name, promise);
  await promise;
  loading.delete(name);
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
  for (const [, entry] of sessions) {
    entry.session.release();
  }
  sessions.clear();
  loading.clear();
}
