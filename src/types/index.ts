export interface VideoMetadata {
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  fileSize: number;
  fileName: string;
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface FaceDetection extends DetectionBox {
  frameIndex: number;
  frameTimestamp: number;
  embedding?: Float32Array;
  clusterId?: number;
}

export interface FaceIdentity {
  id: number;
  representativeFace: number;
  faces: FaceDetection[];
  averageEmbedding: Float32Array;
}

export type BlurType = 'pixelate' | 'eye-bar';

export interface BlurConfig {
  type: BlurType;
  pixelSize: number;
  selectedIdentities: number[];
}

export type PipelinePhase =
  | 'idle'
  | 'loading-models'
  | 'extracting-frames'
  | 'detecting-faces'
  | 'recognizing-faces'
  | 'clustering'
  | 'waiting-selection'
  | 'processing-frames'
  | 'reconstructing'
  | 'done';

export interface PipelineProgress {
  phase: PipelinePhase;
  phaseDescription: string;
  phasePercent: number;
  overallPercent: number;
  detail?: string;
}

export interface ONNXModelConfig {
  name: string;
  url: string;
  sizeMB: number;
  inputShape: number[];
}
