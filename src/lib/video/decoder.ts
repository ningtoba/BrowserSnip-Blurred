import { createFile, MP4BoxBuffer } from 'mp4box';

interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

async function* decodeAndYield(
  config: VideoDecoderConfig,
  chunks: { data: ArrayBuffer; key: boolean }[],
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const support = await VideoDecoder.isConfigSupported(config);
  if (!support.supported) throw new Error('VideoDecoder config not supported');

  let frameIndex = 0;
  let decodeError: Error | null = null;
  let resolveWaiter: (() => void) | null = null;

  const readyFrames: VideoFrame[] = [];
  function onFrame(frame: VideoFrame) {
    readyFrames.push(frame);
    resolveWaiter?.();
  }

  const decoder = new VideoDecoder({
    output: onFrame,
    error(err: Error) { decodeError = err; resolveWaiter?.(); },
  });
  decoder.configure(config);

  const canvas = new OffscreenCanvas(1, 1);
  let ctx: OffscreenCanvasRenderingContext2D | null = null;
  const frameIntervalUs = 33_333;
  let timestamp = 0;

  // Process in batches: decode N chunks, yield all resulting frames, repeat.
  // This keeps memory bounded (~N frames in flight at a time).
  const BATCH = 120;

  for (let batchStart = 0; batchStart < chunks.length; batchStart += BATCH) {
    if (signal.aborted) break;
    const batchEnd = Math.min(batchStart + BATCH, chunks.length);

    // Feed this batch of chunks to the decoder
    for (let i = batchStart; i < batchEnd; i++) {
      decoder.decode(new EncodedVideoChunk({
        type: chunks[i].key ? 'key' : 'delta',
        timestamp, duration: frameIntervalUs,
        data: chunks[i].data,
      }));
      timestamp += frameIntervalUs;
    }

    // Wait for at least one frame (or error)
    if (readyFrames.length === 0) {
      await new Promise<void>(r => { resolveWaiter = r; });
      resolveWaiter = null;
    }
    if (decodeError) break;

    // Small delay to let more frames arrive
    await new Promise(r => setTimeout(r, 5));

    // Yield all frames that have been produced so far
    while (readyFrames.length > 0) {
      const frame = readyFrames.shift()!;
      if (signal.aborted) { frame.close(); continue; }
      const w = frame.displayWidth, h = frame.displayHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      }
      if (!ctx) ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(frame, 0, 0);
      yield { imageData: ctx.getImageData(0, 0, w, h), timestamp: frameIndex / 30, index: frameIndex };
      frameIndex++;
      frame.close(); // free VideoFrame memory immediately
    }
  }

  // Flush remaining frames
  await decoder.flush();
  while (readyFrames.length > 0) {
    const frame = readyFrames.shift()!;
    if (signal.aborted) { frame.close(); continue; }
    const w = frame.displayWidth, h = frame.displayHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    }
    if (!ctx) ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(frame, 0, 0);
    yield { imageData: ctx.getImageData(0, 0, w, h), timestamp: frameIndex / 30, index: frameIndex };
    frameIndex++;
    frame.close();
  }

  decoder.close();
  if (decodeError) throw decodeError;
}

/**
 * Use ffmpeg.wasm to transcode any input to H.264 MP4.
 * Returns the MP4 file bytes.
 */
async function transcodeToH264MP4(
  ffmpeg: import('@ffmpeg/ffmpeg').FFmpeg,
  videoFile: File
): Promise<Uint8Array> {
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());
  await ffmpeg.writeFile('input.bin', inputBuf);

  console.debug('[WebCodecs] transcoding to H.264 MP4...');
  await ffmpeg.exec(
    [
      '-i', 'input.bin',
      '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
      '-an', '-movflags', '+faststart',
      'output.mp4',
    ],
    300_000
  );

  const mp4Data = await ffmpeg.readFile('output.mp4');

  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('output.mp4'); } catch { /* ignore */ }

  if (typeof mp4Data === 'string') throw new Error('Expected binary data from ffmpeg, got string');
  return mp4Data;
}

/**
 * Serialize mp4box's parsed avcC box into the AVCDecoderConfigurationRecord
 * binary format that WebCodecs expects. mp4box already parsed the media —
 * this just repackages its parsed fields into the binary record.
 */
function serializeAvcC(avcCBox: any): Uint8Array {
  const sps = avcCBox.SPS as { length: number; data: Uint8Array }[];
  const pps = avcCBox.PPS as { length: number; data: Uint8Array }[];

  let size = 5 + 1; // header + numOfSPS
  for (const s of sps) size += 2 + s.length;
  size += 1; // numOfPPS
  for (const p of pps) size += 2 + p.length;

  const buf = new Uint8Array(size);
  let off = 0;
  buf[off++] = avcCBox.configurationVersion;
  buf[off++] = avcCBox.AVCProfileIndication;
  buf[off++] = avcCBox.profile_compatibility;
  buf[off++] = avcCBox.AVCLevelIndication;
  buf[off++] = 0xFC | (avcCBox.lengthSizeMinusOne & 0x03);
  buf[off++] = sps.length & 0x1F;
  for (const s of sps) {
    buf[off++] = (s.length >> 8) & 0xFF;
    buf[off++] = s.length & 0xFF;
    buf.set(s.data, off); off += s.length;
  }
  buf[off++] = pps.length;
  for (const p of pps) {
    buf[off++] = (p.length >> 8) & 0xFF;
    buf[off++] = p.length & 0xFF;
    buf.set(p.data, off); off += p.length;
  }
  return buf;
}

/**
 * Demux an MP4 buffer using mp4box.js — the right tool for MP4 demuxing.
 * Returns the VideoDecoderConfig (with avcC description) and encoded samples.
 */
function demuxMP4(mp4Buf: ArrayBuffer): Promise<{
  config: VideoDecoderConfig;
  chunks: { data: ArrayBuffer; key: boolean }[];
  trackDuration: number;
  trackTimescale: number;
}> {
  return new Promise((resolve, reject) => {
    const file = createFile();
    const samples: { data: ArrayBuffer; key: boolean }[] = [];
    let config: VideoDecoderConfig | null = null;
    let trackDuration = 0;
    let trackTimescale = 1;
    let ready = false;

    file.onReady = (_info: any) => {
      const trak = (file as any).moov?.traks?.[0];
      const sampleEntry = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0];
      const avcCBox = sampleEntry?.avcC;

      if (!sampleEntry) { reject(new Error('No sample entry in MP4')); return; }

      const codec = sampleEntry.getCodec();
      const width = sampleEntry.getWidth?.() ?? 0;
      const height = sampleEntry.getHeight?.() ?? 0;

      // Get actual duration and timescale from the track
      trackDuration = trak?.mdia?.mdhd?.duration ?? 0;
      trackTimescale = trak?.mdia?.mdhd?.timescale ?? 1;

      config = { codec, codedWidth: width, codedHeight: height };

      if (avcCBox) {
        const avcCBytes = serializeAvcC(avcCBox);
        const descBuf = new ArrayBuffer(avcCBytes.length);
        new Uint8Array(descBuf).set(avcCBytes);
        (config as any).description = descBuf;
        console.debug('[mp4box] avcC:', descBuf.byteLength, 'bytes, codec:', codec);
      } else {
        console.warn('[mp4box] no avcC box found — decoder will fail');
      }

      const trackId = trak.tkhd.track_id;
      file.setExtractionOptions(trackId, 'video');
      file.start();
      ready = true;
    };

    file.onSamples = (_trackId: number, _user: any, s: any[]) => {
      for (const sample of s) {
        samples.push({ data: sample.data.buffer.slice(0), key: sample.is_sync });
      }
    };

    file.onError = (e: any) => reject(new Error('mp4box error: ' + (e?.message || e)));

    const buf = MP4BoxBuffer.fromArrayBuffer(mp4Buf, 0);
    (buf as any).fileStart = 0;
    file.appendBuffer(buf);
    file.flush();

    const check = setInterval(() => {
      if (ready && config && samples.length > 0) {
        clearInterval(check);
        console.debug('[mp4box] samples:', samples.length,
          'keyframes:', samples.filter(s => s.key).length,
          'codec:', config.codec,
          'duration:', trackDuration, 'timescale:', trackTimescale);
        resolve({ config, chunks: samples, trackDuration, trackTimescale });
      }
    }, 50);
    setTimeout(() => { clearInterval(check); reject(new Error('mp4box extraction timed out')); }, 30000);
  });
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal,
  onTotalFrames?: (count: number) => void,
  onActualFps?: (fps: number) => void
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();

  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());
  await ffmpeg.writeFile('input.bin', inputBuf);

  // Fast path: try codec copy first (~2s). If input is already H.264, this
  // avoids the 15s transcode. Falls back to transcode for AV1/HEVC/VP9.
  let mp4Data: Uint8Array;
  try {
    console.debug('[WebCodecs] trying fast remux (codec copy)...');
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'copy', '-an', '-movflags', '+faststart', 'output.mp4'],
      60_000
    );
    mp4Data = new Uint8Array((await ffmpeg.readFile('output.mp4')) as Uint8Array);
    await ffmpeg.deleteFile('output.mp4');

    // Verify it's actually H.264 by checking the codec via mp4box
    const testBuf = mp4Data.buffer.slice(mp4Data.byteOffset, mp4Data.byteOffset + mp4Data.byteLength);
    const testFile = createFile();
    let isH264 = false;
    testFile.onReady = (info: any) => {
      isH264 = info.videoTracks?.[0]?.codec?.startsWith('avc') ?? false;
      testFile.flush();
    };
    const testMp4Buf = MP4BoxBuffer.fromArrayBuffer(testBuf, 0);
    (testMp4Buf as any).fileStart = 0;
    testFile.appendBuffer(testMp4Buf);
    testFile.flush();
    await new Promise(r => setTimeout(r, 100));

    if (!isH264) throw new Error('Not H.264');
    console.debug('[WebCodecs] fast remux OK, H.264 confirmed');
  } catch {
    // Slow path: transcode to H.264 (~15s for 30s video)
    console.debug('[WebCodecs] transcode to H.264...');
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
       '-an', '-movflags', '+faststart', 'output.mp4'],
      300_000
    );
    mp4Data = new Uint8Array((await ffmpeg.readFile('output.mp4')) as Uint8Array);
    await ffmpeg.deleteFile('output.mp4');
  }

  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }

  console.debug('[WebCodecs] MP4:', mp4Data.length, 'bytes');
  if (mp4Data.length === 0) throw new Error('ffmpeg produced empty MP4');

  // mp4box demuxes the MP4 into codec config + encoded samples
  const mp4Buf = mp4Data.buffer.slice(mp4Data.byteOffset, mp4Data.byteOffset + mp4Data.byteLength);
  const { config, chunks, trackDuration, trackTimescale } = await demuxMP4(mp4Buf);

  // Report actual frame count and fps from mp4box (not metadata)
  onTotalFrames?.(chunks.length);
  if (trackDuration > 0 && trackTimescale > 0) {
    const actualDuration = trackDuration / trackTimescale;
    const actualFps = chunks.length / actualDuration;
    onActualFps?.(actualFps);
    console.debug('[WebCodecs] actual fps:', actualFps.toFixed(2),
      'duration:', actualDuration.toFixed(2) + 's',
      'frames:', chunks.length);
  }

  // VideoDecoder decodes the samples
  yield* decodeAndYield(config, chunks, signal);
}
