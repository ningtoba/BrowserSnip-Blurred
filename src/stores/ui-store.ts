import { create } from 'zustand';

interface UIState {
  showLogMonitor: boolean;
  toggleLogMonitor: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  showLogMonitor: false,
  toggleLogMonitor: () => set((s) => ({ showLogMonitor: !s.showLogMonitor })),
}));
