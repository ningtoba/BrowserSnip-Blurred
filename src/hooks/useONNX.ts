import { useState, useEffect, useCallback } from 'react';
import { initSession, isReady, disposeAll } from '@/lib/engine/session';
import { MODELS } from '@/lib/constants';

interface ONNXState {
  webGPUSupported: boolean;
  modelsReady: boolean;
  loadingMessage: string;
  loadingPercent: number;
}

export function useONNX(): ONNXState {
  const [state, setState] = useState<ONNXState>({
    webGPUSupported: typeof navigator.gpu !== 'undefined',
    modelsReady: false,
    loadingMessage: 'Checking WebGPU support...',
    loadingPercent: 0,
  });

  const loadModels = useCallback(async () => {
    if (!state.webGPUSupported) return;

    const totalModels = MODELS.length;
    let loaded = 0;

    for (const model of MODELS) {
      setState((s) => ({
        ...s,
        loadingMessage: `Downloading ${model.name === 'yolo' ? 'face detection' : 'face recognition'} model (${model.sizeMB} MB)...`,
        loadingPercent: (loaded / totalModels) * 60,
      }));

      try {
        await initSession(model.name as 'yolo' | 'mfn');
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

    setState((s) => ({
      ...s,
      loadingMessage: 'Initializing WebGPU inference engine...',
      loadingPercent: 80,
    }));

    await new Promise((r) => setTimeout(r, 300));

    setState((s) => ({
      ...s,
      loadingMessage: 'Ready',
      loadingPercent: 100,
      modelsReady: true,
    }));
  }, [state.webGPUSupported]);

  useEffect(() => {
    loadModels();
    return () => {
      disposeAll();
    };
  }, [loadModels]);

  return state;
}
