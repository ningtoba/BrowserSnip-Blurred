import { create } from 'zustand';
import type { VideoMetadata } from '@/types';

interface FileState {
  file: File | null;
  metadata: VideoMetadata | null;
  isLargeFile: boolean;
  probing: boolean;

  setFile: (file: File | null) => void;
  setMetadata: (meta: VideoMetadata | null) => void;
  setProbing: (probing: boolean) => void;
  reset: () => void;
}

export const useFileStore = create<FileState>((set) => ({
  file: null,
  metadata: null,
  isLargeFile: false,
  probing: false,

  setFile: (file) =>
    set({
      file,
      isLargeFile: (file?.size ?? 0) > 500 * 1024 * 1024,
      metadata: null,
    }),

  setMetadata: (meta) => set({ metadata: meta }),

  setProbing: (probing) => set({ probing }),

  reset: () =>
    set({
      file: null,
      metadata: null,
      isLargeFile: false,
      probing: false,
    }),
}));
