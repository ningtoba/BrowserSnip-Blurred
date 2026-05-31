import { useState, useEffect, useCallback } from 'react';
import { initSession, disposeAll, hasWebGPU } from '@/lib/engine/session';
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
    loadingMessage: 'Loading AI models...',
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

    const gpuOk = hasWebGPU();
    setState((s) => ({
      ...s,
      gpuAccelerated: gpuOk,
      loadingMessage: gpuOk
        ? 'Ready — blur accelerated by WebGPU'
        : 'Ready — CPU only',
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
