import { useCallback, useRef } from 'react';
import { useFileStore } from '@/stores/file-store';
import { useProcessStore } from '@/stores/process-store';
import {
  generateSampleTimestamps,
  extractFramesAtTimestamps,
  extractFramesSeeking,
  getVideoMetadata,
} from '@/lib/video/extract';
import { decodeFramesWebCodecs } from '@/lib/video/decoder';
import { clusterFaces } from '@/lib/engine/clustering';
// reconstructVideoRaw inlined for batch processing
import { detectFaces } from '@/lib/engine/detection';
import { recognizeFacesBatch, recognizeFace } from '@/lib/engine/recognition';
import { cosineSimilarity } from '@/lib/utils/math';
import { applyPixelateBlur, applyEyeBarBlur, applyBlackBoxBlur } from '@/lib/engine/blur';
import { KalmanBox } from '@/lib/engine/kalman';
import { matchDetectionsToTracks } from '@/lib/engine/match';
import { computeIOU } from '@/lib/engine/tracking';
import { SAMPLE_FPS, PHASE_WEIGHTS, DETECT_EVERY_N_FRAMES } from '@/lib/constants';
import type { FaceDetection, DetectionBox, FaceIdentity, PipelinePhase, BlurType } from '@/types';
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

        setPhase('detecting-faces');
        const allDetections: FaceDetection[] = [];
        for (let i = 0; i < sampleFrames.length; i++) {
          if (signal.aborted) return;
          const boxes = await detectFaces(sampleFrames[i], meta.width, meta.height);
          console.debug(`[scan] frame ${i}: ${boxes.length} faces, boxes:`, boxes.map(b => `(${b.x1.toFixed(0)},${b.y1.toFixed(0)} ${b.x2.toFixed(0)},${b.y2.toFixed(0)} conf=${b.confidence.toFixed(2)})`).join(', '));
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

        // Batched face recognition: preprocess all faces, run one MFN inference
        updateProgress({
          phaseDescription: `Analyzing ${validDetections.length} faces (batched)...`,
          phasePercent: 10,
          overallPercent: computeOverallPercent('recognizing-faces', 10),
        });

        const batchInputs = validDetections.map((det) => ({
          frameData: sampleFrames[det.frameIndex],
          bbox: det,
          frameW: meta.width,
          frameH: meta.height,
        }));

        const embeddings = await recognizeFacesBatch(batchInputs);
        for (let i = 0; i < validDetections.length; i++) {
          validDetections[i].embedding = embeddings[i] ?? undefined;
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

  function matchToIdentitiesByIOU(
  currentBoxes: DetectionBox[],
  scanDetections: FaceDetection[],
  identities: FaceIdentity[],
  currentTimestamp: number,
): Map<number, number> {
  const matchMap = new Map<number, number>();

  // Find the closest sample frame by timestamp
  const sampleTimestamps = [...new Set(scanDetections.map((d) => d.frameTimestamp))].sort();
  if (sampleTimestamps.length === 0) return matchMap;

  let closestTs = sampleTimestamps[0];
  let minDist = Math.abs(currentTimestamp - closestTs);
  for (const ts of sampleTimestamps) {
    const dist = Math.abs(currentTimestamp - ts);
    if (dist < minDist) { minDist = dist; closestTs = ts; }
  }

  // Get identity-labeled detections from the closest sample frame
  const sampleDets = scanDetections.filter((d) => d.frameTimestamp === closestTs && d.clusterId !== undefined);
  if (sampleDets.length === 0) return matchMap;

  // Match current boxes to sample detections via IOU
  for (let ci = 0; ci < currentBoxes.length; ci++) {
    let bestIOU = 0.3;
    let bestId = -1;
    for (const sd of sampleDets) {
      const iou = computeIOU(currentBoxes[ci], sd);
      if (iou > bestIOU) {
        bestIOU = iou;
        bestId = sd.clusterId!;
      }
    }
    if (bestId >= 0) {
      matchMap.set(ci, bestId);
    }
  }

  return matchMap;
}

function cpuBlurFrame(
  imageData: ImageData,
  boxes: DetectionBox[],
  targetIndices: Set<number>,
  blurTypeMap: Map<number, BlurType>,
  indexToId: Map<number, number>,
  fallbackType: BlurType
): ImageData {
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.putImageData(imageData, 0, 0);

  for (const idx of targetIndices) {
    const bbox = boxes[idx];
    const clamped: DetectionBox = {
      x1: Math.max(0, bbox.x1),
      y1: Math.max(0, bbox.y1),
      x2: Math.min(imageData.width, bbox.x2),
      y2: Math.min(imageData.height, bbox.y2),
      confidence: bbox.confidence,
    };
    const w = clamped.x2 - clamped.x1;
    const h = clamped.y2 - clamped.y1;
    if (w < 2 || h < 2) continue;

    const id = indexToId.get(idx);
    const type = (id !== undefined ? blurTypeMap.get(id) : undefined) ?? fallbackType;

    if (type === 'pixelate') {
      applyPixelateBlur(ctx, clamped);
    } else if (type === 'eye-bar') {
      applyEyeBarBlur(ctx, clamped);
    } else {
      applyBlackBoxBlur(ctx, clamped);
    }
  }

  return ctx.getImageData(0, 0, imageData.width, imageData.height);
}

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
      const scanDetections = state.allDetections;
      let totalFrames = Math.ceil(metadata.duration * metadata.fps);
      const { width, height } = metadata;
      let fps = metadata.fps;
      const ffmpeg = await getFFmpeg();

      let globalFrameCount = 0;
      // Per-identity Kalman filter tracking.
      const identityKalman = new Map<number, import('@/lib/engine/kalman').KalmanBox>();
      // Per-identity face embeddings from scan phase (for appearance-based matching)
      const identityEmbeddings = new Map<number, Float32Array>();
      for (const id of identities) {
        if (id.averageEmbedding) identityEmbeddings.set(id.id, id.averageEmbedding);
      }
      const jpgCanvas = new OffscreenCanvas(width, height);
      const jpgCtx = jpgCanvas.getContext('2d', { willReadFrequently: true })!;

      console.time('total-processing');

      // WebCodecs primary, seek-based fallback
      const useWebCodecs = typeof VideoDecoder !== 'undefined';
      if (useWebCodecs) {
        try {
          for await (const { imageData, timestamp } of decodeFramesWebCodecs(file, signal, (count) => {
            totalFrames = count;
          }, (actualFps) => {
            fps = actualFps;
          })) {
            await processFrame(imageData, timestamp);
          }
        } catch (err) {
          console.warn('WebCodecs failed, falling back to seek-based:', err instanceof Error ? err.message : err);
          useProcessStore.getState().appendLog('WebCodecs failed — using software decoding (slower)');
          // Clean partial JPEGs and reset state
          for (let f = 1; f <= globalFrameCount; f++) {
            try { await ffmpeg.deleteFile(`frame_${String(f).padStart(4, '0')}.jpg`); } catch { /* ignore */ }
          }
          globalFrameCount = 0;
          identityKalman.clear();
          await extractFramesSeeking(file, async (imageData, timestamp) => {
            if (signal.aborted) return false;
            await processFrame(imageData, timestamp);
            return true;
          }, signal);
        }
      } else {
        useProcessStore.getState().appendLog('WebCodecs not available — using software decoding');
        await extractFramesSeeking(file, async (imageData, timestamp) => {
          if (signal.aborted) return false;
          await processFrame(imageData, timestamp);
          return true;
        }, signal);
      }

      async function processFrame(imageData: ImageData, timestamp: number) {
          if (signal.aborted) return;

          const shouldDetect = globalFrameCount % DETECT_EVERY_N_FRAMES === 0;

          if (shouldDetect) {
            const detectedBoxes = await detectFaces(imageData, width, height);

            // ── Step 1: Remove stale tracks (unmatched for too long) ──
            // When a face turns away, the Kalman filter keeps predicting blindly.
            // After 15 frames without a real detection, stop tracking to prevent
            // the blur from drifting to wrong areas (hands, background).
            const MAX_UNMATCHED_FRAMES = 15;
            for (const [id, kf] of identityKalman) {
              if (kf.getFramesSinceUpdate() > MAX_UNMATCHED_FRAMES) {
                identityKalman.delete(id);
              }
            }

            // ── Step 2: Predict positions for active tracks ──
            const trackedIds = Array.from(identityKalman.keys());
            const predictedBoxes: DetectionBox[] = [];
            for (const id of trackedIds) {
              predictedBoxes.push(identityKalman.get(id)!.predict());
            }

            // ── Step 3: One-to-one matching (greedy by IOU, best-first) ──
            const matchedDets = new Set<number>();
            const matchedIds = new Set<number>();

            if (trackedIds.length > 0 && detectedBoxes.length > 0) {
              const matches = matchDetectionsToTracks(detectedBoxes, predictedBoxes, 0.1);
              for (const m of matches) {
                const id = trackedIds[m.trackIdx];
                identityKalman.get(id)!.update(detectedBoxes[m.detIdx]);
                matchedDets.add(m.detIdx);
                matchedIds.add(id);
              }
            }

            // ── Step 4: Unmatched detections → scan-phase IOU matching ──
            for (let ci = 0; ci < detectedBoxes.length; ci++) {
              if (matchedDets.has(ci)) continue;
              let bestIOU = 0.15;
              let bestId = -1;
              for (const det of scanDetections) {
                if (det.clusterId === undefined) continue;
                if (matchedIds.has(det.clusterId)) continue;
                if (identityKalman.has(det.clusterId)) continue;
                const iou = computeIOU(detectedBoxes[ci], det);
                if (iou > bestIOU) { bestIOU = iou; bestId = det.clusterId!; }
              }
              if (bestId >= 0) {
                identityKalman.set(bestId, new KalmanBox(detectedBoxes[ci]));
                matchedIds.add(bestId);
                matchedDets.add(ci);
              }
            }

            // ── Step 5: Embedding-based re-identification ──
            // For still-unmatched detections, extract face embedding and compare
            // to scan-phase identity embeddings. This handles cases where IOU fails
            // (face moved far, camera panned, face turned and came back).
            if (identityEmbeddings.size > 0) {
              for (let ci = 0; ci < detectedBoxes.length; ci++) {
                if (matchedDets.has(ci)) continue;
                const det = detectedBoxes[ci];
                const rw = det.x2 - det.x1, rh = det.y2 - det.y1;
                if (rw < 20 || rh < 20) continue; // too small for reliable embedding

                try {
                  const embedding = await recognizeFace(imageData, det, width, height);
                  if (!embedding) continue;

                  let bestSim = 0.45; // cosine similarity threshold
                  let bestId = -1;
                  for (const [id, refEmb] of identityEmbeddings) {
                    if (matchedIds.has(id)) continue;
                    if (identityKalman.has(id)) continue;
                    const sim = cosineSimilarity(embedding, refEmb);
                    if (sim > bestSim) { bestSim = sim; bestId = id; }
                  }
                  if (bestId >= 0) {
                    identityKalman.set(bestId, new KalmanBox(det));
                    matchedIds.add(bestId);
                    matchedDets.add(ci);
                  }
                } catch { /* embedding extraction failed, skip */ }
              }
            }
          } else {
            // Non-detection frames: Kalman filter predicts positions
            for (const [, kf] of identityKalman) {
              kf.predict();
            }
          }

          // Build boxes + target indices from Kalman filter state
          const boxes: DetectionBox[] = [];
          const targetIndices = new Set<number>();
          const indexToId = new Map<number, number>();
          for (const [id, kf] of identityKalman) {
            const box = kf.getState();
            const idx = boxes.length;
            indexToId.set(idx, id);
            boxes.push(box);
            if (selectedIds.has(id)) targetIndices.add(idx);
          }

          let outputData = imageData;
          if (targetIndices.size > 0) {
            outputData = cpuBlurFrame(imageData, boxes, targetIndices,
              useProcessStore.getState().identityBlurTypes, indexToId, blurConfig.type);
          }

          const jpgName = `frame_${String(globalFrameCount + 1).padStart(4, '0')}.jpg`;
          jpgCtx.putImageData(outputData, 0, 0);
          const blob = await jpgCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
          await ffmpeg.writeFile(jpgName, new Uint8Array(await blob.arrayBuffer()));

          globalFrameCount++;

          // Throttle progress updates to avoid 859 React re-renders
          if (globalFrameCount % 10 === 0) {
            const pct = Math.min(99, Math.round((globalFrameCount / totalFrames) * 100));
            updateProgress({
              phaseDescription: `Processing frames... ${globalFrameCount}/${totalFrames}`,
              phasePercent: pct,
              overallPercent: computeOverallPercent('processing-frames', pct),
              detail: `Frame ${globalFrameCount}/${totalFrames}`,
            });
          }
      }

      console.timeEnd('total-processing');

      if (signal.aborted || globalFrameCount === 0) return;

      // Single ffmpeg pass: JPEG frames → H.264 video
      setPhase('reconstructing');
      updateProgress({
        phaseDescription: 'Encoding video...',
        phasePercent: 0,
        overallPercent: computeOverallPercent('reconstructing', 0),
      });

      await ffmpeg.exec(
        [
          '-framerate', fps.toString(),
          '-i', 'frame_%04d.jpg',
          '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
          '-pix_fmt', 'yuv420p',
          '-threads', Math.min(4, navigator.hardwareConcurrency || 2).toString(),
          '-movflags', '+faststart',
          'output.mp4',
        ],
        600_000
      );

      // Mux original audio into output video
      try {
        const inputBuf = await file.arrayBuffer();
        await ffmpeg.writeFile('input.mp4', new Uint8Array(inputBuf));
        await ffmpeg.exec(
          [
            '-i', 'output.mp4', '-i', 'input.mp4',
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart',
            'final.mp4',
          ],
          120_000
        );
        const finalData = await ffmpeg.readFile('final.mp4');
        setOutput(new Blob([finalData], { type: 'video/mp4' }), URL.createObjectURL(new Blob([finalData], { type: 'video/mp4' })));
        try { await ffmpeg.deleteFile('input.mp4'); } catch { /* ignore */ }
        try { await ffmpeg.deleteFile('final.mp4'); } catch { /* ignore */ }
      } catch {
        // Audio mux failed — use video-only output
        const data = await ffmpeg.readFile('output.mp4');
        setOutput(new Blob([data], { type: 'video/mp4' }), URL.createObjectURL(new Blob([data], { type: 'video/mp4' })));
      }

      // Clean up JPEG frames
      for (let f = 1; f <= globalFrameCount; f++) {
        try { await ffmpeg.deleteFile(`frame_${String(f).padStart(4, '0')}.jpg`); } catch { /* ignore */ }
      }
      try { await ffmpeg.deleteFile('output.mp4'); } catch { /* ignore */ }
    } catch (err) {
      if (!signal.aborted) setError(err instanceof Error ? err.message : 'Processing failed');
    }
  }, [startProcessing, setOutput, setError, setPhase, updateProgress]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { startScan, processAndExport, cancel };
}
