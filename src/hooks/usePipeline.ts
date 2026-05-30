import { useCallback, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import {
  generateSampleTimestamps,
  generateAllFrameTimestamps,
  extractFramesAtTimestamps,
  getVideoMetadata,
} from '@/lib/video/extract';
import { processFrameDetections, processFrameEmbeddings } from '@/lib/video/composite';
import { clusterFaces } from '@/lib/engine/clustering';
import { reconstructVideo } from '@/lib/video/reconstruct';
import { SAMPLE_FPS, PHASE_WEIGHTS, BATCH_SIZE } from '@/lib/constants';
import type { FaceDetection, FaceIdentity, PipelinePhase } from '@/types';
import { getFFmpeg } from '@/lib/ffmpeg/core';
import { reconstructCommand } from '@/lib/ffmpeg/commands';
import { detectFaces } from '@/lib/engine/detection';
import { recognizeFace } from '@/lib/engine/recognition';
import { matchDetectionsToIdentities } from '@/lib/engine/clustering';
import { applyBlurToFrame } from '@/lib/engine/blur';

function computeOverallPercent(
  phase: PipelinePhase,
  phasePercent: number
): number {
  let total = 0;
  const phases: PipelinePhase[] = [
    'loading-models',
    'extracting-frames',
    'detecting-faces',
    'recognizing-faces',
    'clustering',
    'waiting-selection',
    'processing-frames',
    'reconstructing',
  ];

  for (const p of phases) {
    if (p === phase) {
      total += (PHASE_WEIGHTS[p] * phasePercent) / 100;
      break;
    }
    total += PHASE_WEIGHTS[p];
    if (p === 'waiting-selection') break;
  }

  return Math.min(Math.round(total), 100);
}

async function canvasToPNGBlob(imageData: ImageData): Promise<Blob> {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(imageData, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

export function usePipeline() {
  const abortRef = useRef<AbortController | null>(null);

  const setPhase = useProcessStore((s) => s.setPhase);
  const updateProgress = useProcessStore((s) => s.updateProgress);
  const setDetectionsAndIdentities = useProcessStore((s) => s.setDetectionsAndIdentities);
  const setIdentityThumbnails = useProcessStore((s) => s.setIdentityThumbnails);
  const startProcessing = useProcessStore((s) => s.startProcessing);
  const setError = useProcessStore((s) => s.setError);
  const setOutput = useProcessStore((s) => s.setOutput);

  const startScan = useCallback(
    async (file: File) => {
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;

      try {
        const meta = await getVideoMetadata(file);
        useFileStore.getState().setMetadata({
          duration: meta.duration,
          width: meta.width,
          height: meta.height,
          fps: meta.fps,
          codec: 'h264',
          fileSize: file.size,
          fileName: file.name,
        });

        // Phase 1: Extract sample frames
        setPhase('extracting-frames');
        const sampleTimestamps = generateSampleTimestamps(meta.duration, SAMPLE_FPS);
        updateProgress({
          phaseDescription: `Extracting ${sampleTimestamps.length} sample frames...`,
          phasePercent: 0,
          overallPercent: computeOverallPercent('extracting-frames', 0),
        });

        const sampleFrames = await extractFramesAtTimestamps(
          file,
          sampleTimestamps,
          (i, total) => {
            const pct = Math.round((i / total) * 100);
            updateProgress({
              phaseDescription: `Extracting sample frames... (${i}/${total})`,
              phasePercent: pct,
              overallPercent: computeOverallPercent('extracting-frames', pct),
              detail: `Frame ${i}/${total}`,
            });
          },
          signal
        );

        if (signal.aborted) return;

        // Phase 2: Detect faces
        setPhase('detecting-faces');
        const allDetections: FaceDetection[] = [];

        for (let i = 0; i < sampleFrames.length; i++) {
          if (signal.aborted) return;

          const dets = await processFrameDetections(
            sampleFrames[i],
            i,
            sampleTimestamps[i],
            meta.width,
            meta.height
          );

          allDetections.push(...dets);

          const pct = Math.round(((i + 1) / sampleFrames.length) * 100);
          updateProgress({
            phaseDescription: `Detecting faces... (${i + 1}/${sampleFrames.length})`,
            phasePercent: pct,
            overallPercent: computeOverallPercent('detecting-faces', pct),
            detail: `Frame ${i + 1}/${sampleFrames.length} — ${allDetections.length} faces found`,
          });
        }

        if (signal.aborted) return;

        // Phase 3: Recognize faces
        setPhase('recognizing-faces');
        const validDetections = allDetections.filter((d) => {
          const w = d.x2 - d.x1;
          const h = d.y2 - d.y1;
          return w >= 10 && h >= 10;
        });

        for (let i = 0; i < validDetections.length; i++) {
          if (signal.aborted) return;

          const frameData = sampleFrames[validDetections[i].frameIndex];
          await processFrameEmbeddings([validDetections[i]], frameData, meta.width, meta.height);

          const pct = Math.round(((i + 1) / validDetections.length) * 100);
          updateProgress({
            phaseDescription: `Analyzing faces... (${i + 1}/${validDetections.length})`,
            phasePercent: pct,
            overallPercent: computeOverallPercent('recognizing-faces', pct),
            detail: `Face ${i + 1}/${validDetections.length}`,
          });
        }

        if (signal.aborted) return;

        // Phase 4: Cluster
        setPhase('clustering');
        updateProgress({
          phaseDescription: 'Grouping faces by identity...',
          phasePercent: 50,
          overallPercent: computeOverallPercent('clustering', 50),
        });

        const identities = clusterFaces(validDetections);

        // Generate thumbnails
        const thumbnails = new Map<number, string>();
        for (const identity of identities) {
          const rep = validDetections[identity.representativeFace];
          const frameData = sampleFrames[rep.frameIndex];
          const w = rep.x2 - rep.x1;
          const h = rep.y2 - rep.y1;
          if (w > 0 && h > 0) {
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d')!;
            const srcCanvas = document.createElement('canvas');
            srcCanvas.width = frameData.width;
            srcCanvas.height = frameData.height;
            const srcCtx = srcCanvas.getContext('2d')!;
            srcCtx.putImageData(frameData, 0, 0);
            ctx.drawImage(srcCanvas, rep.x1, rep.y1, w, h, 0, 0, w, h);
            thumbnails.set(identity.id, canvas.toDataURL('image/jpeg', 0.7));
          }
        }

        setDetectionsAndIdentities(validDetections, identities);
        setIdentityThumbnails(thumbnails);

        updateProgress({
          phaseDescription: `${identities.length} identities found`,
          phasePercent: 100,
          overallPercent: computeOverallPercent('clustering', 100),
          detail: identities.length > 0
            ? `Found ${identities.length} unique people`
            : 'No faces detected',
        });

        setPhase('waiting-selection');
      } catch (err) {
        if (!signal.aborted) {
          setError(err instanceof Error ? err.message : 'Scan failed');
        }
      }
    },
    [setPhase, updateProgress, setDetectionsAndIdentities, setIdentityThumbnails, setError]
  );

  const processAndExport = useCallback(async () => {
    const state = useProcessStore.getState();
    const file = useFileStore.getState().file;
    const metadata = useFileStore.getState().metadata;

    if (!file || !metadata) return;

    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    try {
      startProcessing();

      const allTimestamps = generateAllFrameTimestamps(metadata.duration, metadata.fps);
      const totalFrames = allTimestamps.length;
      const selectedIds = state.selectedIdentities;
      const blurConfig = state.blurConfig;
      const identities = state.identities;

      // Process frames in batches
      const numBatches = Math.ceil(totalFrames / BATCH_SIZE);
      const ffmpeg = await getFFmpeg();

      let globalFrameCount = 0;

      const video = document.createElement('video');
      video.preload = 'auto';
      video.muted = true;

      const url = URL.createObjectURL(file);
      video.src = url;

      await new Promise<void>((resolve, reject) => {
        video.addEventListener('loadedmetadata', () => resolve(), { once: true });
        video.addEventListener('error', () => reject(new Error('Failed to load video')), { once: true });
        video.load();
      });

      const canvas = document.createElement('canvas');
      canvas.width = metadata.width;
      canvas.height = metadata.height;
      const ctx = canvas.getContext('2d')!;

      for (let batch = 0; batch < numBatches; batch++) {
        if (signal.aborted) break;

        const batchStart = batch * BATCH_SIZE;
        const batchEnd = Math.min(batchStart + BATCH_SIZE, totalFrames);
        let batchFrameCount = 0;

        for (let i = batchStart; i < batchEnd; i++) {
          if (signal.aborted) break;

          const timestamp = allTimestamps[i];
          video.currentTime = timestamp;

          await new Promise<void>((resolve) => {
            video.addEventListener('seeked', () => resolve(), { once: true });
          });

          ctx.drawImage(video, 0, 0, metadata.width, metadata.height);
          const imageData = ctx.getImageData(0, 0, metadata.width, metadata.height);

          const boxes = await detectFaces(imageData, metadata.width, metadata.height);

          const detections: FaceDetection[] = boxes.map((box) => ({
            ...box,
            frameIndex: i,
            frameTimestamp: timestamp,
          }));

          for (const det of detections) {
            const embedding = await recognizeFace(imageData, det, metadata.width, metadata.height);
            if (embedding) det.embedding = embedding;
          }

          const matchMap = matchDetectionsToIdentities(detections, identities);

          const targetIndices = new Set<number>();
          for (const [detIdx, clusterId] of matchMap) {
            if (selectedIds.has(clusterId)) {
              targetIndices.add(detIdx);
            }
          }

          let frameImageData = imageData;
          if (targetIndices.size > 0) {
            frameImageData = applyBlurToFrame(imageData, boxes, targetIndices, blurConfig.type);
          }

          const pngName = `frame_${String(globalFrameCount + 1).padStart(4, '0')}.png`;
          const blob = await canvasToPNGBlob(frameImageData);
          const buf = new Uint8Array(await blob.arrayBuffer());
          await ffmpeg.writeFile(pngName, buf);

          globalFrameCount++;
          batchFrameCount++;

          const overallPct = computeOverallPercent(
            'processing-frames',
            Math.round((globalFrameCount / totalFrames) * 100)
          );
          updateProgress({
            phaseDescription: `Processing frames... (${globalFrameCount}/${totalFrames})`,
            phasePercent: Math.round((globalFrameCount / totalFrames) * 100),
            overallPercent: overallPct,
            detail: `Frame ${globalFrameCount}/${totalFrames}`,
          });
        }
      }

      URL.revokeObjectURL(url);
      video.remove();

      if (signal.aborted) return;

      // Reconstruct
      setPhase('reconstructing');
      updateProgress({
        phaseDescription: 'Encoding final video...',
        phasePercent: 0,
        overallPercent: computeOverallPercent('reconstructing', 0),
      });

      const blob = await reconstructVideo(globalFrameCount, metadata, (desc, pct) => {
        updateProgress({
          phaseDescription: desc,
          phasePercent: pct,
          overallPercent: computeOverallPercent('reconstructing', pct),
        });
      });

      const outputUrl = URL.createObjectURL(blob);
      setOutput(blob, outputUrl);
    } catch (err) {
      if (!signal.aborted) {
        setError(err instanceof Error ? err.message : 'Processing failed');
      }
    }
  }, [startProcessing, setOutput, setError, setPhase, updateProgress]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { startScan, processAndExport, cancel };
}
