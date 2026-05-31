import { useState, useEffect, useCallback } from 'react';
import { initSession, disposeAll, hasWebGPU, getBackend } from '@/lib/engine/session';
import { MODELS } from '@/lib/constants';

interface ONNXState {
  gpuAccelerated: boolean;
  modelsReady: boolean;
  loadingMessage: string;
  loadingPercent: number;
}

export function useONNX(): ONNXState {
  const [state, setState] = useState<ONNXState>({
    gpuAccelerated: hasWebGPU(),
    modelsReady: false,
    loadingMessage: hasWebGPU()
      ? 'Checking WebGPU support...'
      : 'WebGPU not available, using CPU fallback...',
    loadingPercent: 0,
  });

  const loadModels = useCallback(async () => {
    const totalModels = MODELS.length;
    let loaded = 0;

    for (const model of MODELS) {
      const modelLabel = model.name === 'yunet' ? 'face detection' : 'face recognition';
      setState((s) => ({
        ...s,
        loadingMessage: `Downloading ${modelLabel} model (${model.sizeMB} MB)...`,
        loadingPercent: (loaded / totalModels) * 60,
      }));

      try {
        await initSession(model.name as 'yunet' | 'mfn');
      } catch (err) {
        setState((s) => ({
          ...s,
          loadingMessage: `Failed to load model: ${err instanceof Error ? err.message : 'Unknown error'}`,
        }));
        return;
      }

      loaded++;
      setState((s) => ({
        ...s,
        loadingPercent: (loaded / totalModels) * 60,
      }));
    }

    const detBackend = getBackend('yunet');
    setState((s) => ({
      ...s,
      gpuAccelerated: detBackend === 'webgpu',
      loadingMessage: detBackend === 'webgpu'
        ? 'Initializing WebGPU inference engine...'
        : 'Initializing CPU inference engine (slower)...',
      loadingPercent: 80,
    }));

    await new Promise((r) => setTimeout(r, 300));

    setState((s) => ({
      ...s,
      loadingMessage: 'Ready',
      loadingPercent: 100,
      modelsReady: true,
    }));
  }, []);

  useEffect(() => {
    loadModels();
    return () => {
      disposeAll();
    };
  }, [loadModels]);

  return state;
}
