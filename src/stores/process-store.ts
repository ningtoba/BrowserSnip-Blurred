import { create } from 'zustand';
import type {
  FaceDetection,
  FaceIdentity,
  BlurType,
  BlurConfig,
  PipelinePhase,
  PipelineProgress,
} from '@/types';

interface ProcessState {
  phase: PipelinePhase;
  progress: PipelineProgress;
  allDetections: FaceDetection[];
  identities: FaceIdentity[];
  identityThumbnails: Map<number, string>;
  selectedIdentities: Set<number>;
  blurConfig: BlurConfig;
  isProcessing: boolean;
  outputBlob: Blob | null;
  outputUrl: string | null;
  error: string | null;
  logs: string[];

  setPhase: (phase: PipelinePhase) => void;
  updateProgress: (p: Partial<PipelineProgress>) => void;
  setDetectionsAndIdentities: (
    detections: FaceDetection[],
    identities: FaceIdentity[]
  ) => void;
  setIdentityThumbnails: (thumbnails: Map<number, string>) => void;
  toggleIdentity: (id: number) => void;
  setBlurType: (type: BlurType) => void;
  startProcessing: () => void;
  setOutput: (blob: Blob, url: string) => void;
  setError: (error: string) => void;
  appendLog: (line: string) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>((set, get) => ({
  phase: 'idle',
  progress: {
    phase: 'idle',
    phaseDescription: '',
    phasePercent: 0,
    overallPercent: 0,
  },
  allDetections: [],
  identities: [],
  identityThumbnails: new Map(),
  selectedIdentities: new Set(),
  blurConfig: { type: 'pixelate', pixelSize: 15, selectedIdentities: [] },
  isProcessing: false,
  outputBlob: null,
  outputUrl: null,
  error: null,
  logs: [],

  setPhase: (phase) =>
    set((s) => ({
      phase,
      progress: {
        ...s.progress,
        phase,
      },
    })),

  updateProgress: (p) =>
    set((s) => ({
      progress: { ...s.progress, ...p },
    })),

  setDetectionsAndIdentities: (allDetections, identities) =>
    set({ allDetections, identities }),

  setIdentityThumbnails: (identityThumbnails) =>
    set({ identityThumbnails }),

  toggleIdentity: (id) =>
    set((s) => {
      const next = new Set(s.selectedIdentities);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return { selectedIdentities: next };
    }),

  setBlurType: (type) =>
    set((s) => ({ blurConfig: { ...s.blurConfig, type } })),

  startProcessing: () =>
    set({
      isProcessing: true,
      progress: {
        phase: 'processing-frames',
        phaseDescription: '',
        phasePercent: 0,
        overallPercent: 0,
      },
      error: null,
      outputBlob: null,
      outputUrl: null,
      logs: [],
    }),

  setOutput: (blob, url) => {
    const prev = get().outputUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      isProcessing: false,
      phase: 'done',
      outputBlob: blob,
      outputUrl: url,
    });
  },

  setError: (error) =>
    set({ isProcessing: false, error }),

  appendLog: (line) =>
    set((s) => ({ logs: [...s.logs, line] })),

  reset: () => {
    const prev = get().outputUrl;
    if (prev) URL.revokeObjectURL(prev);

    set({
      phase: 'idle',
      progress: {
        phase: 'idle',
        phaseDescription: '',
        phasePercent: 0,
        overallPercent: 0,
      },
      allDetections: [],
      identities: [],
      identityThumbnails: new Map(),
      selectedIdentities: new Set(),
      isProcessing: false,
      outputBlob: null,
      outputUrl: null,
      error: null,
      logs: [],
    });
  },
}));
