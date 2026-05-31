import * as ort from 'onnxruntime-web';
import type { DetectionBox } from '@/types';
import { MODELS, DETECTION_CONFIDENCE } from '@/lib/constants';

type ModelName = 'yolo' | 'mfn';

const sessions = new Map<ModelName, ort.InferenceSession>();
const loading = new Map<ModelName, Promise<void>>();

ort.env.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency || 4);

function getConfig(name: ModelName) {
  const config = MODELS.find((m) => m.name === name);
  if (!config) throw new Error(`Unknown model: ${name}`);
  return config;
}

async function fetchModelBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Model file not found at ${url}. Place the ONNX model in public/models/. ` +
      `See README.md for instructions on obtaining the required models.`
    );
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('text/html')) {
    throw new Error(
      `Model file not found at ${url} (received HTML instead of binary). ` +
      `Place the ONNX model in public/models/. See README.md.`
    );
  }

  const buffer = await response.arrayBuffer();

  if (buffer.byteLength < 1024) {
    throw new Error(
      `File at ${url} is too small (${buffer.byteLength} bytes) to be an ONNX model. ` +
      `Ensure you have placed the actual ONNX file, not a placeholder.`
    );
  }

  return buffer;
}

export async function initSession(name: ModelName): Promise<void> {
  if (sessions.has(name)) return;
  if (loading.has(name)) return loading.get(name)!;

  const config = getConfig(name);
  const promise = (async () => {
    const buffer = await fetchModelBuffer(config.url);

    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      intraOpNumThreads: Math.max(2, navigator.hardwareConcurrency || 4),
    });

    sessions.set(name, session);
  })();

  loading.set(name, promise);
  await promise;
  loading.delete(name);
}

export async function runYOLO(input: Float32Array): Promise<DetectionBox[]> {
  const session = sessions.get('yolo');
  if (!session) throw new Error('YOLO session not initialized');

  const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  const data = output.data as Float32Array;
  const dims = output.dims;

  const numDetections = dims[1];
  const boxes: DetectionBox[] = [];

  for (let i = 0; i < numDetections; i++) {
    const offset = i * 6;
    const x1 = data[offset];
    const y1 = data[offset + 1];
    const x2 = data[offset + 2];
    const y2 = data[offset + 3];
    const confidence = data[offset + 4];

    if (confidence < DETECTION_CONFIDENCE) continue;

    boxes.push({ x1, y1, x2, y2, confidence });
  }

  return boxes;
}

export async function runMFN(input: Float32Array): Promise<Float32Array> {
  const session = sessions.get('mfn');
  if (!session) throw new Error('MFN session not initialized');

  const tensor = new ort.Tensor('float32', input, [1, 3, 112, 112]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  return new Float32Array(output.data as Float32Array);
}

export function isReady(name: ModelName): boolean {
  return sessions.has(name);
}

export async function disposeAll(): Promise<void> {
  for (const [, session] of sessions) {
    session.release();
  }
  sessions.clear();
  loading.clear();
}
