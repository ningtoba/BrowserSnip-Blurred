import type { ONNXModelConfig, PipelinePhase } from '@/types';

export const SAMPLE_FPS = 1;
export const CLUSTER_SIMILARITY_THRESHOLD = 0.55;
export const FACE_EXPAND_RATIO = 0.2;
export const FACE_INPUT_SIZE = 112;
export const YUNET_INPUT_SIZE = 640;
export const YUNET_CONF_THRESHOLD = 0.3;
export const YUNET_NMS_THRESHOLD = 0.35;
export const YUNET_TOP_K = 5000;
export const PIXELATE_BLOCK_SIZE = 15;
export const BATCH_SIZE = 300;
export const DETECT_EVERY_N_FRAMES = 2;
export const MAX_DETECTIONS_PER_FRAME = 20;

export const MODELS: ONNXModelConfig[] = [
  {
    name: 'yunet',
    url: '/models/face_detection_yunet_2023mar.onnx',
    sizeMB: 0.23,
    inputShape: [1, 3, 640, 640],
  },
  {
    name: 'mfn',
    url: '/models/w600k_mbf.onnx',
    sizeMB: 10,
    inputShape: [1, 3, 112, 112],
  },
];

export const PHASE_WEIGHTS: Record<PipelinePhase, number> = {
  'idle': 0,
  'loading-models': 15,
  'extracting-frames': 10,
  'detecting-faces': 25,
  'recognizing-faces': 25,
  'clustering': 5,
  'waiting-selection': 0,
  'processing-frames': 15,
  'reconstructing': 5,
  'done': 0,
};

export const PHASE_DESCRIPTIONS: Record<PipelinePhase, string> = {
  'idle': '',
  'loading-models': 'Downloading AI models...',
  'extracting-frames': 'Extracting sample frames...',
  'detecting-faces': 'Detecting faces with AI...',
  'recognizing-faces': 'Analyzing face identities...',
  'clustering': 'Grouping faces by identity...',
  'waiting-selection': 'Select faces to blur',
  'processing-frames': 'Applying blur to all frames...',
  'reconstructing': 'Encoding final video...',
  'done': 'Done',
};
