import { useCallback, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import {
  generateSampleTimestamps,
  extractFramesAtTimestamps,
  extractFramesStreaming,
  extractFramesZeroCopy,
  getVideoMetadata,
} from '@/lib/video/extract';
import { clusterFaces, matchDetectionsToIdentities } from '@/lib/engine/clustering';
import { reconstructVideo } from '@/lib/video/reconstruct';
import { detectFaces, postprocessYOLO } from '@/lib/engine/detection';
import { recognizeFace } from '@/lib/engine/recognition';
import { applyBlurToFrame } from '@/lib/engine/blur';
import { SAMPLE_FPS, PHASE_WEIGHTS, BATCH_SIZE, DETECT_EVERY_N_FRAMES, DETECTION_CONFIDENCE } from '@/lib/constants';
import type { FaceDetection, DetectionBox, FaceIdentity, PipelinePhase } from '@/types';
import { getFFmpeg } from '@/lib/ffmpeg/core';

function computeOverallPercent(phase: PipelinePhase, phasePercent: number): number {
  let total = 0;
  const phases: PipelinePhase[] = [
    'loading-models', 'extracting-frames', 'detecting-faces',
    'recognizing-faces', 'clustering', 'waiting-selection',
    'processing-frames', 'reconstructing',
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
          duration: meta.duration, width: meta.width, height: meta.height,
          fps: meta.fps, codec: 'h264', fileSize: file.size, fileName: file.name,
        });

        setPhase('extracting-frames');
        const sampleTimestamps = generateSampleTimestamps(meta.duration, SAMPLE_FPS);
        updateProgress({
          phaseDescription: `Extracting ${sampleTimestamps.length} sample frames...`,
          phasePercent: 0,
          overallPercent: computeOverallPercent('extracting-frames', 0),
        });

        const sampleFrames = await extractFramesAtTimestamps(
          file, sampleTimestamps,
          (i, total) => updateProgress({
            phaseDescription: `Extracting sample frames... (${i}/${total})`,
            phasePercent: Math.round((i / total) * 100),
            overallPercent: computeOverallPercent('extracting-frames', Math.round((i / total) * 100)),
            detail: `Frame ${i}/${total}`,
          }),
          signal,
        );

        if (signal.aborted) return;

        // Detect + recognize in one pass to minimize GPU context switches
        setPhase('detecting-faces');
        const allDetections: FaceDetection[] = [];
        for (let i = 0; i < sampleFrames.length; i++) {
          if (signal.aborted) return;
          const boxes = await detectFaces(sampleFrames[i], meta.width, meta.height);
          for (const box of boxes) {
            allDetections.push({ ...box, frameIndex: i, frameTimestamp: sampleTimestamps[i] });
          }
          if (i % 5 === 0 || i === sampleFrames.length - 1) {
            updateProgress({
              phaseDescription: `Detecting faces... (${i + 1}/${sampleFrames.length})`,
              phasePercent: Math.round(((i + 1) / sampleFrames.length) * 100),
              overallPercent: computeOverallPercent('detecting-faces', Math.round(((i + 1) / sampleFrames.length) * 100)),
              detail: `Frame ${i + 1}/${sampleFrames.length} — ${allDetections.length} faces found`,
            });
          }
        }
        if (signal.aborted) return;

        setPhase('recognizing-faces');
        const validDetections = allDetections.filter((d) => {
          const rw = d.x2 - d.x1, rh = d.y2 - d.y1;
          return rw >= 10 && rh >= 10;
        });

        for (let i = 0; i < validDetections.length; i++) {
          if (signal.aborted) return;
          const det = validDetections[i];
          const frameData = sampleFrames[det.frameIndex];
          det.embedding = await recognizeFace(frameData, det, meta.width, meta.height) ?? undefined;
          if (i % 10 === 0 || i === validDetections.length - 1) {
            updateProgress({
              phaseDescription: `Analyzing faces... (${i + 1}/${validDetections.length})`,
              phasePercent: Math.round(((i + 1) / validDetections.length) * 100),
              overallPercent: computeOverallPercent('recognizing-faces', Math.round(((i + 1) / validDetections.length) * 100)),
              detail: `Face ${i + 1}/${validDetections.length}`,
            });
          }
        }
        if (signal.aborted) return;

        setPhase('clustering');
        updateProgress({ phaseDescription: 'Grouping faces by identity...', phasePercent: 50, overallPercent: computeOverallPercent('clustering', 50) });
        const identities = clusterFaces(validDetections);

        const thumbnails = new Map<number, string>();
        for (const identity of identities) {
          const rep = validDetections[identity.representativeFace];
          const frameData = sampleFrames[rep.frameIndex];
          const rw = rep.x2 - rep.x1, rh = rep.y2 - rep.y1;
          if (rw > 0 && rh > 0) {
            const c = new OffscreenCanvas(rw, rh);
            const cx = c.getContext('2d')!;
            const src = new OffscreenCanvas(frameData.width, frameData.height);
            src.getContext('2d')!.putImageData(frameData, 0, 0);
            cx.drawImage(src, rep.x1, rep.y1, rw, rh, 0, 0, rw, rh);
            const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
            const reader = new FileReader();
            const dataUrl = await new Promise<string>((resolve) => {
              reader.onload = () => resolve(reader.result as string);
              reader.readAsDataURL(blob);
            });
            thumbnails.set(identity.id, dataUrl);
          }
        }

        setDetectionsAndIdentities(validDetections, identities);
        setIdentityThumbnails(thumbnails);
        updateProgress({
          phaseDescription: `${identities.length} identities found`,
          phasePercent: 100, overallPercent: computeOverallPercent('clustering', 100),
          detail: identities.length > 0 ? `Found ${identities.length} unique people` : 'No faces detected',
        });
        setPhase('waiting-selection');
      } catch (err) {
        if (!signal.aborted) setError(err instanceof Error ? err.message : 'Scan failed');
      }
    },
    [setPhase, updateProgress, setDetectionsAndIdentities, setIdentityThumbnails, setError],
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
      const selectedIds = state.selectedIdentities;
      const blurConfig = state.blurConfig;
      const identities = state.identities;
      const totalFrames = Math.ceil(metadata.duration * metadata.fps);
      const ffmpeg = await getFFmpeg();

      let globalFrameCount = 0;
      let lastBoxes: DetectionBox[] = [];
      let lastDetections: FaceDetection[] = [];
      const offscreen = new OffscreenCanvas(metadata.width, metadata.height);
      const offCtx = offscreen.getContext('2d')!;

      await extractFramesZeroCopy(
        file,
        async (video, imageData, timestamp, _index) => {
          if (signal.aborted) return false;

          const shouldDetect = globalFrameCount % DETECT_EVERY_N_FRAMES === 0;
          let boxes: DetectionBox[];
          let detections: FaceDetection[];

          if (shouldDetect) {
            // Try zero-copy GPU path: importExternalTexture from video element → GPUBuffer → ORT
            let usedZeroCopy = false;
            try {
              const { preprocessFromVideoElement } = await import('@/lib/webgpu/video');
              const { runYOLO } = await import('@/lib/engine/session');
              const zcResult = await preprocessFromVideoElement(video, metadata.width, metadata.height);
              const rawBoxes = await runYOLO(zcResult.tensor);
              zcResult.tensor.dispose();
              boxes = postprocessYOLO(rawBoxes, metadata.width, metadata.height, zcResult.scale, zcResult.padLeft, zcResult.padTop);
              usedZeroCopy = true;
            } catch {
              boxes = await detectFaces(imageData, metadata.width, metadata.height);
            }

            detections = boxes.map((box, di) => ({
              ...box, frameIndex: di, frameTimestamp: timestamp,
            }));

            // Recognition still uses ImageData path (face crops are small)
            for (const det of detections) {
              const emb = await recognizeFace(imageData, det, metadata.width, metadata.height);
              if (emb) det.embedding = emb;
            }

            lastBoxes = boxes;
            lastDetections = detections;
          } else {
            boxes = lastBoxes.map((b) => ({ ...b }));
            detections = lastDetections.map((d) => ({
              ...d, frameIndex: globalFrameCount, frameTimestamp: timestamp,
            }));
          }

          const matchMap = matchDetectionsToIdentities(detections, identities);
          const targetIndices = new Set<number>();
          for (const [detIdx, clusterId] of matchMap) {
            if (selectedIds.has(clusterId)) targetIndices.add(detIdx);
          }

          let outputData = imageData;
          if (targetIndices.size > 0) {
            outputData = await applyBlurToFrame(imageData, boxes, targetIndices, blurConfig.type);
          }

          const pngName = `frame_${String(globalFrameCount + 1).padStart(4, '0')}.png`;
          offCtx.putImageData(outputData, 0, 0);
          const blob = await offscreen.convertToBlob({ type: 'image/png' });
          await ffmpeg.writeFile(pngName, new Uint8Array(await blob.arrayBuffer()));

          globalFrameCount++;

          const pct = Math.round((globalFrameCount / totalFrames) * 100);
          updateProgress({
            phaseDescription: `Processing frames... (${globalFrameCount}/${totalFrames})`,
            phasePercent: pct,
            overallPercent: computeOverallPercent('processing-frames', pct),
            detail: `Frame ${globalFrameCount}/${totalFrames}`,
          });

          return true;
        },
        signal,
      );

      if (signal.aborted) return;

      setPhase('reconstructing');
      updateProgress({
        phaseDescription: 'Encoding final video...',
        phasePercent: 0,
        overallPercent: computeOverallPercent('reconstructing', 0),
      });

      const blob = await reconstructVideo(globalFrameCount, metadata, (desc, pct) => {
        updateProgress({
          phaseDescription: desc, phasePercent: pct,
          overallPercent: computeOverallPercent('reconstructing', pct),
        });
      });

      setOutput(blob, URL.createObjectURL(blob));
    } catch (err) {
      if (!signal.aborted) setError(err instanceof Error ? err.message : 'Processing failed');
    }
  }, [startProcessing, setOutput, setError, setPhase, updateProgress]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { startScan, processAndExport, cancel };
}
